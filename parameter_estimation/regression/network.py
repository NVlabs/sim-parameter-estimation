import torch
import torch.nn as nn

class LinearRegressor(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim, out_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear((state_dim + action_dim + state_dim), out_dim)

    # Tuple of S-A-S'
    def forward(self, x):
        return self.linear(x)