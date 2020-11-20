import numpy as np  # basic math and random numbers
import torch  # package for building functions with learnable parameters
import torch.nn as nn  # prebuilt functions specific to neural networks
from src.models.random_features_torch import RFF
import src.utils.pdf as pdf
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import itertools
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class ConditionalFlow(nn.Module):
    def __init__(self, hidden_layers=[50, 50], network_type="FCNN",
                 flow="RealNVP", nflows=3, inputd=1, outputd=1, gpu=True):
        super().__init__()
        last_layer_size = None
        self.network = []
        self.n_outputs = outputd
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(1)
            self.device = "cuda:1"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

        for ix, layer_size in enumerate(hidden_layers):
            if ix == 0:
                self.network.append(nn.Linear(inputd, layer_size).to(self.device))
            else:
                self.network.append(nn.Linear(last_layer_size, layer_size).to(self.device))
            self.network.append(nn.Tanh().to(self.device))
            last_layer_size = layer_size




        flow = eval(flow)
        act_norm_flag = True

        # Builds the flow
        flows = [flow(dim=last_layer_size, hidden_dim=50, base_network=FCNN) for _ in range(nflows)]
        if act_norm_flag:
            actnorms = [ActNorm(dim=2) for _ in range(nflows)]
            flows = list(itertools.chain(*zip(actnorms, flows)))
        base_dist = MultivariateNormal(torch.zeros(outputd).type(self.dtype).to(self.device),
                                       torch.eye(outputd).type(self.dtype).to(self.device))
        self.flow_dist = NormalizingFlowModel(base_dist, flows)

    def forward(self, x):
        for ix, layer in enumerate(self.network):
            if ix == 0:
                z_h = layer(x)
            else:
                z_h = layer(z_h)

        self.flow_dist(z_h)

    # Objective function
    def loss_fn(self, x, y):
        z, prior_logprob, log_det = self.flow_dist(x, y)
        loss = -torch.mean(prior_logprob + log_det)
        return loss

    def train(self, x_data, y_data, nepoch=1000, batch_size=100, save=False, verbose=True):

        optimizer = torch.optim.Adam(self.parameters())

        x_variable = torch.from_numpy(x_data).type(self.dtype).to(self.device)
        y_variable = torch.from_numpy(y_data).type(self.dtype).to(self.device)

        batch_size = len(x) if batch_size is None else batch_size

        print("Training mdn network on {} datapoints".format(len(x_data)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x_data), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        lossHistory = []
        for epoch in tqdm(range(nepoch)):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            loss = self.loss_fn(x_batch, y_batch)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                print("Initial Loss is: {}".format(loss.item()))

            elif epoch % 100 == 0 and verbose:
                if epoch != 0:
                    print(" Iteration:", epoch, "Loss", loss.item())

            lossHistory.append(loss.item())
        print("Training Finished, final loss is {}".format(loss.item()))
        return self, lossHistory


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Hardtanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Hardtanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 8, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim)

    def forward(self, x):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def backward(self, z):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det


class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 5, B = 3, hidden_dim = 8, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def backward(self, z):
        log_det = torch.zeros(z.shape[0])
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = self.unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = self.unconstrained_RQS(
            upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    """
    Implementation of rational-quadratic splines in this file is taken from
    https://github.com/bayesiains/nsf.

    Thank you to the authors for providing well-documented source code!
    """
    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1

    #DEFAULT_MIN_BIN_WIDTH = 1e-3
    #DEFAULT_MIN_BIN_HEIGHT = 1e-3
    #DEFAULT_MIN_DERIVATIVE = 1e-3
    def unconstrained_RQS(self, inputs, unnormalized_widths, unnormalized_heights,
                          unnormalized_derivatives, inverse=False,
                          tail_bound=1., min_bin_width=1e-3,
                          min_bin_height=1e-3,
                          min_derivative=1e-3):
        inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
        outside_interval_mask = ~inside_intvl_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0

        outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = self.RQS(
            inputs=inputs[inside_intvl_mask],
            unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
            unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )
        return outputs, logabsdet

    def RQS(self, inputs, unnormalized_widths, unnormalized_heights,
            unnormalized_derivatives, inverse=False, left=0., right=1.,
            bottom=0., top=1., min_bin_width=1e-3,
            min_bin_height=1e-3,
            min_derivative=1e-3):
        if torch.min(inputs) < left or torch.max(inputs) > right:
            raise ValueError("Input outside domain")

        num_bins = unnormalized_widths.shape[-1]

        if min_bin_width * num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if min_bin_height * num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if inverse:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
        input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (((inputs - input_cumheights) * (input_derivatives \
                                                 + input_derivatives_plus_one - 2 * input_delta) \
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives - (inputs - input_cumheights) \
                 * (input_derivatives + input_derivatives_plus_one \
                    - 2 * input_delta))
            c = - input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta \
                          + ((input_derivatives + input_derivatives_plus_one \
                              - 2 * input_delta) * theta_one_minus_theta)
            derivative_numerator = input_delta.pow(2) \
                                   * (input_derivatives_plus_one * root.pow(2) \
                                      + 2 * input_delta * theta_one_minus_theta \
                                      + input_derivatives * (1 - root).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2) \
                                         + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + ((input_derivatives \
                                          + input_derivatives_plus_one - 2 * input_delta) \
                                         * theta_one_minus_theta)
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) \
                                   * (input_derivatives_plus_one * theta.pow(2) \
                                      + 2 * input_delta * theta_one_minus_theta \
                                      + input_derivatives * (1 - theta).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, logabsdet


class ActNorm(nn.Module):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype = torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype = torch.float))

    def forward(self, x):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma)
        return z, log_det

    def backward(self, z):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma)
        return x, log_det


# Defines the model
class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).type(dtype)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).type(dtype)
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.backward(z)
        return x