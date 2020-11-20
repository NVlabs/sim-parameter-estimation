import numpy as np

class StateDifferenceRewarder(object):
    def __init__(self, weights=1.0):
        self.weights = weights
    
    def __call__(self, samples, targets):
        difference = samples - targets
        return self.weights * np.linalg.norm(difference, axis=(1, 2))