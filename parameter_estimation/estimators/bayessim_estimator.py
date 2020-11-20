# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

from parameter_estimation.estimators import BaseEstimator

from parameter_estimation.bayessim.model_helper import _get_mdntorch
from parameter_estimation.utils.rollout_evaluation import evaluate_actions

class BayesSimEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 prior_mean,
                 prior_std,
                 model_name,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed

        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.nenvs = randomized_env.nenvs

        self.nparams = randomized_env.randomization_space.shape[0]

        input_dim = randomized_env.observation_space.shape[0] * 2 + randomized_env.action_space.shape[0]
        self.model = _get_mdntorch(n_components=5, output_dim=self.nparams, input_dim=input_dim)
    
    def load_trajectory(self, reference_env, reference_action_fp):
        actions = np.load(reference_action_fp)
        self.reference_actions = np.repeat(actions[:, np.newaxis], self.nenvs, axis=1)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)    
    
    def get_parameter_estimate(self, threshold=0.005): 
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x = self.flattened_reference
        mog = self.model.predict_mog(x)[0]

        return mog.eval(np.ones(self.nparams).reshape(-1, 1) * 0.5)

        
    def update_parameter_estimate(self, randomized_env, policy=None, reference_env=None):
        random_params = np.random.uniform(size=(self.nenvs, self.nparams))
        randomized_env.randomize(randomized_values=random_params)

        randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)

        random_params = np.repeat(random_params[:, :, np.newaxis], len(self.reference_actions), axis=2)
        random_params = np.transpose(random_params, (0, 2, 1))
        flattened_params = [random_params[i] for i in range(self.nenvs)]
        flattened_params = np.concatenate(flattened_params)

        flattened_randomized = [randomized_trajectory[i] for i in range(self.nenvs)]
        flattened_randomized = np.concatenate(flattened_randomized)

        self.model.train(x_data=flattened_randomized,
                        y_data=flattened_params,
                        nepoch=10,
                        batch_size=100)