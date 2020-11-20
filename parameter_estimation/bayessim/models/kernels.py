import torch
import numpy as np

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def set_lengthscale(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class Laplace(torch.nn.Module):
    def __init__(self, sigma=None):
        super(Laplace, self).__init__()
        self.sigma = sigma

    def _torch_sqrt(self, x, eps=1e-6):
        """
        A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
        at 0.
        """
        # Ref: https://github.com/pytorch/pytorch/issues/2421
        return (x + eps).sqrt()

    def set_lengthscale(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        dnorm = self._torch_sqrt(dnorm2)
        K_XY = (-gamma * dnorm).exp()

        return K_XY


class Matern32(torch.nn.Module):
    def __init__(self, sigma=None):
        super(Matern32, self).__init__()
        self.sigma = sigma

    def set_lengthscale(self, sigma):
        self.sigma = sigma

    def _torch_sqrt(self, x, eps=1e-4):
        """
        A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
        at 0.
        """
        # Ref: https://github.com/pytorch/pytorch/issues/2421
        return (x + eps).sqrt()

    def __call__(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        dnorm = self._torch_sqrt(dnorm2)

        sqrt3_d = 3 ** 0.5 * dnorm
        K_XY = (1 + gamma * sqrt3_d) * torch.exp(-gamma * sqrt3_d)
        return K_XY