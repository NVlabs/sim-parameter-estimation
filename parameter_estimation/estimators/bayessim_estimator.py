import numpy as np

from parameter_estimation.estimators import BaseEstimator

from parameter_estimation.bayessim.model_helper import get_bayessim_model
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

        self.model = get_bayessim_model(model_name)
    
    def load_trajectory(self, reference_env, reference_action_fp):
        self.reference_actions = np.load(reference_action_fp)
        self.reference_trajectory = evaluate_actions(reference_env, self.reference_actions)
        self.flattened_reference = np.squeeze(self.reference_trajectory)
    
    def get_parameter_estimate(self, threshold=0.005): 

        # TODO: Where to get X?
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x = x.reshape(1, -1)
        mog = self.model.predict_mog(x)[0]
        
        if self.prior_mean is not None:
            # mog is posterior given proposal prior
            mog.prune_negligible_components(threshold=threshold)
            
            # TODO: Conform posterior to generator
            # compute posterior given prior by analytical division step
            if isinstance(self.generator.prior, dd.Uniform):
                posterior = mog / self.generator.proposal
            elif isinstance(self.generator.prior, dd.Gaussian):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented
            return posterior
        else:
            return mog  

    
    def update_parameter_estimate(self):
        self.model.train(x_data=self.flattened_reference,
                            y_data=np.ones(self.flattened_reference, self.randomized_env.randomization_space.shape[0]) * 0.5,
                            nepoch=1,
                            batch_size=100)