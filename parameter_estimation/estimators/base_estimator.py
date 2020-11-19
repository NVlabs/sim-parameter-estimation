from abc import ABC, abstractmethod

class BaseEstimator(object):
    @abstractmethod
    def __init__(self,
                 reference_env,
                 randomized_env,
                 seed,
                 **kwargs):

        self.reference_env = reference_env
        self.randomized_env = randomized_env
        self.seed = seed
    
    @abstractmethod
    def load_trajectory(self, reference_action_fp):
        pass
    
    @abstractmethod
    def get_parameter_estimate(self, randomized_env):
        pass
    
    @abstractmethod
    def update_parameter_estimate(self, randomized_env):
        pass
