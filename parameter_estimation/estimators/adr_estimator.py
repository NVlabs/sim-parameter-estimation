# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import gym
import numpy as np
import logging

import torch

from parameter_estimation.utils.rollout_evaluation import evaluate_actions
from parameter_estimation.rewarders import DiscriminatorRewarder, StateDifferenceRewarder

from parameter_estimation.adr import SVPG
from parameter_estimation.estimators import BaseEstimator

from policy.train import run_training_episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class ADREstimator(BaseEstimator):
    """Simulation object which creates randomized environments based on specified params, 
    handles SVPG-based policy search to create envs, 
    and evaluates controller policies in those environments
    """

    """
    nagents,
    nparams,
    temperature,
    svpg_rollout_length,
    svpg_horizon,
    max_step_length,
    initial_svpg_steps,
    discriminator_batchsz=320,
    reward_scale=-1,
    """

    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs
                 ):
                 
        self.nagents = kwargs['nagents']
        self.nparams = randomized_env.randomization_space.shape[0]
        assert self.nagents > 2

        self.svpg_horizon = kwargs['svpg_horizon']
        self.initial_svpg_steps = kwargs['initial_svpg_steps']

        self.seed = seed
        self.svpg_timesteps = 0
        self.agent_timesteps = 0
        self.agent_timesteps_since_eval = 0

        state_dim = randomized_env.observation_space.shape[0]
        action_dim = randomized_env.action_space.shape[0]

        if kwargs['learned_reward']:
            self.rewarder = DiscriminatorRewarder(
                state_dim=state_dim,
                action_dim=action_dim,
                discriminator_batchsz=kwargs['discriminator_batchsz'],
                reward_scale=-1,
            )

        else:
            self.rewarder = StateDifferenceRewarder(weights=-1)

        self.learned_reward = kwargs['learned_reward']

        self.svpg = SVPG(nagents=self.nagents,
                         nparams=self.nparams,
                         max_step_length=kwargs['max_step_length'],
                         svpg_rollout_length=kwargs['svpg_rollout_length'],
                         svpg_horizon=kwargs['svpg_horizon'],
                         temperature=kwargs['temperature'],
                         kld_coefficient=0.0)

        self.simulation_instances_full_horizon = np.ones((self.nagents,
                                                          self.svpg_horizon,
                                                          self.svpg.svpg_rollout_length,
                                                          self.svpg.nparams)) * -1


    def load_trajectory(self, reference_env, reference_action_fp):
        actions = np.load(reference_action_fp)
        self.reference_actions = np.repeat(actions[:, np.newaxis], reference_env.nenvs, axis=1)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)    

    def get_parameter_values(self):
        return self.svpg.last_states

    def get_parameter_estimate(self, randomized_env):
        return np.mean(self.svpg.last_states, axis=0)

    def update_parameter_estimate(self, randomized_env, policy=None, reference_env=None):
        """Select an action based on SVPG policy, where an action is the delta in each dimension.
        Update the counts and statistics after training agent,
        rolling out policies, and calculating simulator reward.
        """
        if self.svpg_timesteps >= self.initial_svpg_steps:
            # Get sim instances from SVPG policy
            simulation_instances = self.svpg.step()

            index = self.svpg_timesteps % self.svpg_horizon
            self.simulation_instances_full_horizon[:, index, :, :] = simulation_instances

        else:
            # Creates completely randomized environment
            simulation_instances = np.ones((self.nagents,
                                            self.svpg.svpg_rollout_length,
                                            self.svpg.nparams)) * -1

        assert (self.nagents, self.svpg.svpg_rollout_length, self.svpg.nparams) == simulation_instances.shape

        # Create placeholders for trajectories
        randomized_trajectories = [[] for _ in range(self.nagents)]

        # Create placeholder for rewards
        rewards = np.zeros(simulation_instances.shape[:2])

        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))

        # Create environment instances with vectorized env, and rollout agent_policy in both
        for t in range(self.svpg.svpg_rollout_length):
            agent_timesteps_current_iteration = 0
            
            # TODO: Double check shape here
            randomized_env.randomize(randomized_values=simulation_instances[t])
            if policy is not None:
                randomized_trajectory, randomized_actions = run_training_episode(randomized_env, policy, self.agent_timesteps)
                self.reference_trajectory = evaluate_actions(reference_env, np.transpose(randomized_actions, (1, 0, 2)))
            else:
                randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)

            for i in range(self.nagents):
                agent_timesteps_current_iteration += len(randomized_trajectory[i])
                self.agent_timesteps += len(randomized_trajectory[i])
                randomized_trajectories[i].append(randomized_trajectory[i])
                simulator_reward = self.rewarder(randomized_trajectories[i][t][np.newaxis, :], self.reference_trajectory[0][np.newaxis, :])
                rewards[i][t] = simulator_reward
            
            if self.learned_reward:
                # flatten and combine all randomized and reference trajectories for discriminator
                flattened_reference = [self.reference_trajectory[i] for i in range(self.nagents)]
                flattened_reference = np.concatenate(flattened_reference)
                flattened_randomized = [randomized_trajectories[i][t] for i in range(self.nagents)]
                flattened_randomized = np.concatenate(flattened_randomized)

                # Train discriminator based on state action pairs for agent env. steps
                self.rewarder.train_discriminator(
                    flattened_reference, 
                    flattened_randomized,
                    iterations=150
                )

        # Calculate discriminator based reward, pass it back to SVPG policy
        if self.svpg_timesteps >= self.initial_svpg_steps:
            self.svpg.train(rewards)

        self.svpg_timesteps += 1

