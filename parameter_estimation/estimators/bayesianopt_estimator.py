# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

from bayes_opt import BayesianOptimization, UtilityFunction

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.rewarders import StateDifferenceRewarder
from parameter_estimation.utils.rollout_evaluation import evaluate_actions

from policy.train import run_training_episode

class BayesianOptEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed

        self.statedifference_rewarder = StateDifferenceRewarder(weights=-1)

        self.nparams = randomized_env.randomization_space.shape[0]
        self.nenvs = randomized_env.nenvs

        pbounds = {}
        for i in range(self.nparams):
            pbounds[str(i)] = (0, 1)

        self.bayesianoptimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=seed,
        )

        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.registered_points = {}
        self.agent_timesteps = 0
    
    def load_trajectory(self, reference_env, reference_action_fp):
        actions = np.load(reference_action_fp)
        self.reference_actions = np.repeat(actions[:, np.newaxis], self.nenvs, axis=1)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)      
    
    def get_parameter_estimate(self, randomized_env):
        return list(self.bayesianoptimizer.max['params'].values())
    
    def update_parameter_estimate(self, randomized_env, policy=None, reference_env=None):
        parameter_estimate = self.bayesianoptimizer.suggest(self.utility)
        randomized_env.randomize(randomized_values=[list(parameter_estimate.values())])

        if policy is not None:
            randomized_trajectory, randomized_actions = run_training_episode(randomized_env, policy, self.agent_timesteps)
            self.reference_trajectory = evaluate_actions(reference_env, np.transpose(randomized_actions, (1, 0, 2)))
        else:
            randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)

        cost = self.statedifference_rewarder(randomized_trajectory, self.reference_trajectory)

        registered_cost = self.registered_points.get(tuple(parameter_estimate.values()), cost[0])
        try:
            self.bayesianoptimizer.register(params=parameter_estimate, target=registered_cost)
        except:
            pass

        self.registered_points[tuple(parameter_estimate.values())] = registered_cost

        flattened_randomized = [randomized_trajectory[i] for i in range(self.nenvs)]
        flattened_randomized = np.concatenate(flattened_randomized)
        self.agent_timesteps += len(flattened_randomized)