# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.simopt.reps import REPS
from parameter_estimation.rewarders import StateDifferenceRewarder, DiscriminatorRewarder
from parameter_estimation.utils.rollout_evaluation import evaluate_actions

from policy.train import run_training_episode


class SimOptEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed
        
        nparams = self.randomized_env.randomization_space.shape[0]

        means = np.empty(nparams)

        if kwargs['random_init'] and kwargs['mean_init']:
            means[::2] = kwargs['mean_init'] * 0.5
            means[1::2] = kwargs['mean_init'] * 1.5
        elif kwargs['random_init']:
            means = np.random.random(nparams)
        elif kwargs['mean_init']:
            means = np.ones(nparams) * kwargs['mean_init']
        else:
            raise NotImplementedError

        self.mean_init = means
        self.cov_init = np.identity(nparams) * kwargs['cov_init']

        self.reps = REPS(self.mean_init, self.cov_init)

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
        self.reps_updates = kwargs['reps_updates']
        self.nenvs = self.randomized_env.nenvs
        self.agent_timesteps = 0

    def load_trajectory(self, reference_env, reference_action_fp):
        actions = np.load(reference_action_fp)
        self.reference_actions = np.repeat(actions[:, np.newaxis], self.reference_env.nenvs, axis=1)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)       
    
    def get_parameter_estimate(self, randomized_env):
        return np.clip(self.reps.current_mean, 0, 1)

    def _get_current_reps_state(self):
        return self.reps.current_mean, self.reps.current_cov

    def update_parameter_estimate(self, randomized_env, policy=None, reference_env=None):
        for _ in range(self.reps_updates):
            mean, cov = self._get_current_reps_state()
            parameter_estimates = np.random.multivariate_normal(mean, cov, self.nenvs)
            parameter_estimates = np.clip(parameter_estimates, 0, 1)

            randomized_env.randomize(randomized_values=parameter_estimates)
            if policy is not None:
                randomized_trajectory, randomized_actions = run_training_episode(randomized_env, policy, self.agent_timesteps)
                self.reference_trajectory = evaluate_actions(reference_env, np.transpose(randomized_actions, (1, 0, 2)))
            else:
                randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)

            costs = self.rewarder(randomized_trajectory, self.reference_trajectory)

            try:
                self.reps.learn(parameter_estimates, costs)
            except:
                self.reps.current_cov = self.cov_init
                self.reps.learn(parameter_estimates, costs)

            # flatten and combine all randomized and reference trajectories for discriminator
            flattened_reference = [self.reference_trajectory[i] for i in range(self.nenvs)]
            flattened_reference = np.concatenate(flattened_reference)

            flattened_randomized = [randomized_trajectory[i] for i in range(self.nenvs)]
            flattened_randomized = np.concatenate(flattened_randomized)

            self.agent_timesteps += len(flattened_randomized)    

            if self.learned_reward:
                # Train discriminator based on state action pairs for agent env. steps
                self.rewarder.train_discriminator(
                    flattened_reference, 
                    flattened_randomized,
                    iterations=150
                )

            
