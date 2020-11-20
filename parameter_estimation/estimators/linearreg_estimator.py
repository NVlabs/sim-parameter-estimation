# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.utils.rollout_evaluation import evaluate_actions
from parameter_estimation.regression import Regression

from policy.train import run_training_episode

class RegressionEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed

        state_dim = randomized_env.observation_space.shape[0]
        action_dim = randomized_env.action_space.shape[0]
        self.randomization_dim = randomized_env.randomization_space.shape[0]
        self.nenvs = randomized_env.nenvs

        self.regressor = Regression(state_dim, action_dim, self.randomization_dim)
        self.agent_timesteps = 0

    def load_trajectory(self, reference_env, reference_action_fp):
        actions = np.load(reference_action_fp)
        self.reference_actions = np.repeat(actions[:, np.newaxis], self.nenvs, axis=1)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)    
    
    def get_parameter_estimate(self, randomized_env):
        return self.regressor(self.flattened_reference) 
    
    def update_parameter_estimate(self, randomized_env, policy=None, reference_env=None):
        random_params = np.random.uniform(size=(self.nenvs, self.randomization_dim))
        randomized_env.randomize(randomized_values=random_params)

        if policy is not None:
            randomized_trajectory, _ = run_training_episode(randomized_env, policy, self.agent_timesteps)
        else:
            randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)

        random_params = np.repeat(random_params[:, :, np.newaxis], len(self.reference_actions), axis=2)
        random_params = np.transpose(random_params, (0, 2, 1))
        flattened_params = [random_params[i] for i in range(self.nenvs)]
        flattened_params = np.concatenate(flattened_params)

        flattened_randomized = [randomized_trajectory[i] for i in range(self.nenvs)]
        flattened_randomized = np.concatenate(flattened_randomized)
        
        self.regressor.train_regressor(flattened_randomized, flattened_params, iterations=10)
        self.agent_timesteps += len(flattened_randomized)