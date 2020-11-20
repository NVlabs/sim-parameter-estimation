import numpy as np

import torch
import torch.nn as nn

from .network import LinearRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regression(object):
    def __init__(self, state_dim, action_dim, randomization_dim, learning_rate=1e-3, batch_size=320):
        self.regressor = LinearRegressor(
            state_dim=state_dim,
            action_dim=action_dim,
            out_dim=randomization_dim).to(device)

        self.batch_size = batch_size

        self.regression_criterion = nn.MSELoss()
        self.regression_optimizer = torch.optim.Adam(self.regressor.parameters(), lr=learning_rate)

    def __call__(self, trajectory):
        with torch.no_grad():
            return torch.mean(self.regressor(self._trajectory2tensor(trajectory)), axis=0).cpu().numpy()

    def train_regressor(self, randomized_trajectory, labels, iterations):
        randomized_trajectory = self._trajectory2tensor(randomized_trajectory)
        labels = self._trajectory2tensor(labels)
        for _ in range(iterations):
            randind = np.random.randint(0, len(randomized_trajectory[0]), size=int(self.batch_size))
            labelind = np.random.randint(0, len(labels[0]), size=int(self.batch_size))

            randomized_batch = randomized_trajectory[randind]
            labels = labels[labelind]

            predicted = self.regressor(randomized_batch)
            self.regression_optimizer.zero_grad()

            mse_loss = self.regression_criterion(predicted, labels)
            mse_loss.backward()

            self.regression_optimizer.step()

    def _trajectory2tensor(self, trajectory):
        return torch.from_numpy(trajectory).float().to(device)
