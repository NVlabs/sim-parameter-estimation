import numpy as np

class StateDifferenceRewarder(object):
    def __init__(self, weights=None):
        self.weights = weights
    
    def __call__(self, samples, targets):
        difference = samples - targets
        if self.weights is None:
            return np.linalg.norm(difference, axis=(1,2))

        else:
            raise NotImplementedError