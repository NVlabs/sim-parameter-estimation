import numpy as np

from bayes_opt import BayesianOptimization, UtilityFunction

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.rewarders import StateDifferenceRewarder
from parameter_estimation.utils.rollout_evaluation import evaluate_actions

class BayesianOptEstimator(BaseEstimator):
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed

        self.statedifference_rewarder = StateDifferenceRewarder(weights=None)

        self.nparams = randomized_env.randomization_space.shape[0]

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
    
    def load_trajectory(self, reference_env, reference_action_fp):
        self.reference_actions = np.load(reference_action_fp)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)    
    
    def get_parameter_estimate(self, randomized_env):
        return list(self.bayesianoptimizer.max['params'].values())
    
    def update_parameter_estimate(self, randomized_env):
        parameter_estimate = self.bayesianoptimizer.suggest(self.utility)
        randomized_env.randomize(randomized_values=[list(parameter_estimate.values())])
        randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)
        cost = self.statedifference_rewarder(randomized_trajectory, self.reference_trajectory)

        registered_cost = self.registered_points.get(tuple(parameter_estimate.values()), cost[0])
        try:
            self.bayesianoptimizer.register(params=parameter_estimate, target=registered_cost)
        except:
            pass

        self.registered_points[tuple(parameter_estimate.values())] = registered_cost