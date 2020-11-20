import numpy as np

from parameter_estimation.estimators import BaseEstimator
from parameter_estimation.simopt.reps import REPS
from parameter_estimation.rewarders import StateDifferenceRewarder, DiscriminatorRewarder
from parameter_estimation.utils.rollout_evaluation import evaluate_actions


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
        means[::2] = kwargs['mean_init'] * 0.5
        means[1::2] = kwargs['mean_init'] * 1.5

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
            self.rewarder = StateDifferenceRewarder(weights=None)

        self.learned_reward = kwargs['learned_reward']
        self.reps_updates = kwargs['reps_updates']
        self.nagents = kwargs['nagents']

        
    
    def load_trajectory(self, reference_env, reference_action_fp):
        self.reference_actions = np.load(reference_action_fp)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)        
    
    def get_parameter_estimate(self, randomized_env):
        return self.reps.current_mean

    def _get_current_reps_state(self):
        return self.reps.current_mean, self.reps.current_cov

    def update_parameter_estimate(self, randomized_env):
        for _ in range(self.reps_updates):
            mean, cov = self._get_current_reps_state()
            parameter_estimates = np.random.multivariate_normal(mean, cov, self.nagents)
            parameter_estimates = np.clip(parameter_estimates, 0, 1)

            randomized_env.randomize(randomized_values=parameter_estimates)
            randomized_trajectory = evaluate_actions(randomized_env, self.reference_actions)
            
            costs = self.rewarder(randomized_trajectory, self.reference_trajectory)
            try:
                self.reps.learn(parameter_estimates, costs)
            except:
                print('Singular Matrix, resetting')
                self.reps.current_cov = self.cov_init

            if self.learned_reward:
                # flatten and combine all randomized and reference trajectories for discriminator
                flattened_randomized = [randomized_trajectory[i] for i in range(self.nagents)]
                flattened_randomized = np.concatenate(flattened_randomized)

                # Train discriminator based on state action pairs for agent env. steps
                self.rewarder.train_discriminator(
                    self.flattened_reference, 
                    flattened_randomized,
                    iterations=150
                )

            
