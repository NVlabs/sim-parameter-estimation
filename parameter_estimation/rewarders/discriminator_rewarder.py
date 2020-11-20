import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAMLPDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim):
        super(SAMLPDiscriminator, self).__init__()
        
        self.l1 = nn.Linear((state_dim + action_dim + state_dim), 128)
        self.l2 = nn.Linear(128, 128)
        self.logic = nn.Linear(128, 1)
        
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.logic(x)
        return torch.sigmoid(x)


class DiscriminatorRewarder(object):
    def __init__(self, state_dim, action_dim, discriminator_batchsz, reward_scale, discriminator_lr=3e-3, add_pz=True):
        self.discriminator = SAMLPDiscriminator(
            state_dim=state_dim,
            action_dim=action_dim).to(device)

        self.discriminator_criterion = nn.BCELoss()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.reward_scale = reward_scale
        self.batch_size = discriminator_batchsz 
        self.add_pz = add_pz

    def __call__(self, randomized_trajectory, reference_trajectory=None):
        traj_tensor = self._trajectory2tensor(randomized_trajectory).float()

        with torch.no_grad():
            score = (self.discriminator(traj_tensor).cpu().detach().numpy()+1e-8)
            return score.mean(axis=1)

    def calculate_rewards(self, randomized_trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        score, _, _ = self.get_score(randomized_trajectory)
        reward = np.log(score)

        if self.add_pz:
            reward -= np.log(0.5)

        return self.reward_scale * reward

    def get_score(self, trajectory):
        """Discriminator based reward calculation
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        traj_tensor = self._trajectory2tensor(trajectory).float()

        with torch.no_grad():
            score = (self.discriminator(traj_tensor).cpu().detach().numpy()+1e-8)
            return score.mean(), np.median(score), np.sum(score)

    def train_discriminator(self, reference_trajectory, randomized_trajectory, iterations):
        """Trains discriminator to distinguish between reference and randomized state action tuples
        """
        for _ in range(iterations):
            randind = np.random.randint(0, len(randomized_trajectory[0]), size=int(self.batch_size))
            refind = np.random.randint(0, len(reference_trajectory[0]), size=int(self.batch_size))

            randomized_batch = self._trajectory2tensor(randomized_trajectory[randind])
            reference_batch = self._trajectory2tensor(reference_trajectory[refind])

            g_o = self.discriminator(randomized_batch)
            e_o = self.discriminator(reference_batch)

            self.discriminator_optimizer.zero_grad()

            discrim_loss = self.discriminator_criterion(g_o, torch.ones((len(randomized_batch), 1), device=device)) + \
                           self.discriminator_criterion(e_o, torch.zeros((len(reference_batch), 1), device=device))
            discrim_loss.backward()

            self.discriminator_optimizer.step()

    def _trajectory2tensor(self, trajectory):
        return torch.from_numpy(trajectory).float().to(device)
