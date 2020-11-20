import torch
import numpy as np
from datetime import datetime
import src.utils.pdf as pdf
from torch.distributions import Uniform
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import  cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm


class SVGD(object):
    """
    Adapted from: https://github.com/activatedgeek/svgd
    """
    def __init__(self, kernel, surrogate, optimizer_func, n_particles=50,
                 generator=None, p_lower=None, p_upper=None, n_iter=100, gpu=True):
        self.kernel = kernel
        self.surrogate = surrogate
        self.optimizer_func = optimizer_func
        self.optimizer = None
        self.p_lower = np.array(p_lower).reshape(-1,)
        self.p_upper = np.array(p_upper).reshape(-1,)
        self.n_particles = n_particles
        self.generator = generator
        self.data_test = None
        self.scaler = None
        self.n_iter = n_iter
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

    def scale_particles(self, x):
        range_v = torch.tensor(self.p_upper - self.p_lower).type(self.dtype).to(self.device)
        x_normalised = (x - torch.tensor(self.p_lower).type(self.dtype).to(self.device)) / range_v
        #return x_normalised
        return x

    def unscale_particles(self, x):
        range_v = torch.tensor(self.p_upper - self.p_lower).type(self.dtype).to(self.device)
        x_unnormalised = (x * range_v) + torch.tensor(self.p_lower).type(self.dtype).to(self.device)
        #return x_unnormalised
        return x

    def fix_x(self, x):
        for i in range(x.shape[1]):
            mid = self.p_lower[i] + 0.5*(self.p_upper[i] - self.p_lower[i])
            x[x[:, i] < self.p_lower[i], i] = mid
            x[x[:, i] > self.p_upper[i], i] = mid
        return x

    def set_prior_from_generator(self, generator):
        self.p_lower, self.p_upper = generator.get_prior()

    def phi(self, x):
        x = x.detach().requires_grad_(True)
        score_func = torch.autograd.grad(self.surrogate.forward(x).sum(), x)[0]
        k_xx = self.kernel(x, x.detach())
        grad_k = -torch.autograd.grad(k_xx.sum(), x)[0]
        phi = (k_xx.detach().matmul(score_func) + grad_k) / x.size(0)
        return phi

    def step(self, x):
        self.optimizer.zero_grad()
        x.grad = -self.phi(x)
        self.optimizer.step()

    def set_prior(self, p_lower, p_upper):
        self.p_lower = p_lower
        self.p_upper = p_upper

    def objective_func(self, x, loss='L2'):
        #print(x)
        if hasattr(self.generator, 'num_envs'):
            # Parallel simulation
            if isinstance(x, torch.Tensor):
                _, data = self.generator.run_forward_model(x.detach().cpu().numpy(), 1)
            else:
                _, data = self.generator.run_forward_model(x.reshape(-1, self.p_lower.size), 1)
            if self.scaler is not None:
                data = self.scaler.transform(data)
            SS_x = torch.tensor(data).type(self.dtype).to(self.device)
        else:
            # No parallel simulation
            SS_x = torch.zeros([x.shape[0], self.data_test.shape[1]]).type(self.dtype).to(self.device)
            for i in range(x.shape[0]):
                if isinstance(x, torch.Tensor):
                    _, data = self.generator.run_forward_model(x[i].detach().cpu().numpy(), 1)
                else:
                    _, data = self.generator.run_forward_model(x[i], 1)
                if self.scaler is not None:
                    data = self.scaler.transform(data)
                SS_x[i, :] = torch.tensor(data).type(self.dtype).to(self.device)

        if loss == 'L2':
            return torch.norm(SS_x - self.data_test, p=2, dim=1)
        elif loss == 'L1':
            return torch.norm(SS_x - self.data_test, p=1, dim=1)
        elif loss == 'max':
            return torch.max(torch.abs(SS_x - self.data_test), dim=1)[0]
        elif loss == 'cosine':
            return 10./cosine_similarity(SS_x, self.data_test.repeat(x.shape[0], 1), dim=1)

    def log_p(self, x):
        x_unnormalised = self.unscale_particles(x)
        return (-self.objective_func(x_unnormalised) +
                torch.log(self.prior_func(x_unnormalised))).reshape([-1, 1])

    def prior_func(self, x):
        # Sets uniform prior
        prior_x = (1./x.shape[1]**2)*torch.ones(x.shape[0]).type(self.dtype).to(self.device)
        for i in range(x.shape[1]):
            prior_x[x[:, i] < self.p_lower[i]] = 1e-8
            prior_x[x[:, i] > self.p_upper[i]] = 1e-8
        return prior_x

    def predict(self, data_test):
        # Initialisation with a uniform distribution
        x = (Uniform(torch.tensor(self.p_lower),
                     torch.tensor(self.p_upper))
             .sample((self.n_particles,))).type(self.dtype).to(self.device)

        x = self.scale_particles(x)
        self.optimizer = self.optimizer_func(params=[x])
        x_all = [x.clone().cpu().numpy()]
        if self.scaler is not None:
            self.data_test = torch.tensor(self.scaler.transform(data_test)).type(self.dtype).to(self.device)
        else:
            self.data_test = torch.tensor(data_test).type(self.dtype).to(self.device)
        self.surrogate.objective = self.log_p

        for _ in tqdm(range(self.n_iter)):
            self.surrogate.update(x)
            self.step(x)
            self.fix_x(x) # keeps x within limits
            x_all.extend([self.unscale_particles(x).clone().cpu().numpy()])
        return self.unscale_particles(x), x_all

    # Prediction function returning a MoG
    def predict_mog(self, data_test, sigma=0.01):
        # Builds a MoG representation to be compatible with MDN
        # Parameters of the mixture
        ntest, _ = data_test.shape  # test dimensionality and number of queries

        mog = []
        for pt in range(ntest):
            start_time = datetime.now()
            #x_testS = self.scaler.transform(x_test) no scaling implemented
            y_pred, y_all = self.predict(data_test)
            end_time = datetime.now()
            print('\n')
            print("*********************************  Prediction ends  *********************************")
            print('\n')
            print('Duration: {}'.format(end_time - start_time))

            a = [1./self.n_particles for i in range(self.n_particles)]
            if isinstance(y_pred, torch.Tensor):
                ms = [y_pred[i, :].clone().cpu().numpy() for i in range(self.n_particles)]
                p_std = 0.0001 + 0.005 * np.std(y_pred.clone().cpu().numpy(), axis=0)
            else:
                # Assuming output from BlackBoxOptimizer
                ms = [y_pred[i, :] for i in range(self.n_particles)]
                p_std = 0.0001 + 0.001 * np.ones(y_pred.shape[1])

            #Ss = [sigma*np.eye(y_pred[i, :].shape[0]) for i in range(self.n_particles)]
            Ss = [np.diag(p_std) for i in range(self.n_particles)]
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        #self.plot_objective([-2, 0])
        return mog

    def plot_objective(self, bar_range=None):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches((10, 10))
        cmap = cm.cool
        # ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(self.p_lower[0], self.p_upper[0], 0.01)
        Y = np.arange(self.p_lower[1], self.p_upper[1], 0.01)
        X, Y = np.meshgrid(X, Y)
        tmp = np.array([X, Y]).reshape([2, X.size]).transpose()
        Z = self.log_p(torch.tensor(tmp).type(self.dtype).to(self.device))

        if bar_range is not None:
            Z[Z > bar_range[1]] = bar_range[1]
            Z[Z < bar_range[0]] = bar_range[0]
        # Plot the function.
        plt.pcolormesh(X, Y, Z.reshape([X.shape[0], X.shape[1]]), shading='gouraud', cmap=cmap)
        plt.colorbar(spacing='proportional')
        ax.contour(X, Y, Z.reshape(X.shape), alpha=0.8)

        # Customize the z axis.
        # ax.set_zlim(0, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # And a corresponding grid
        ax.grid(which='both')  # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.6)
        ax.grid(which='major', alpha=0.8)
        plt.show()



# Gradient-free SVGD simple update as described in equations 22 and 23
# https://arxiv.org/pdf/1806.02775.pdf
class GFSVGD(SVGD):
    """
    Adapted from: https://github.com/activatedgeek/svgd
    """
    def __init__(self, kernel, surrogate, optimizer_func, n_particles=50,
                 generator=None, p_lower=None, p_upper=None, n_iter=100):
        super().__init__(kernel, surrogate, optimizer_func, n_particles, generator, p_lower, p_upper, n_iter)

    def phi(self, x):
        x = x.detach().requires_grad_(True)
        score_func = torch.tensor(self.log_p(x)).type(self.dtype).to(self.device).exp()
        k_xx = self.kernel(x, x.detach())
        grad_k = torch.autograd.grad(k_xx.matmul(score_func).sum(), x)[0]
        Z = (1 / score_func).sum()
        phi = grad_k.detach() / Z
        return phi

    def phi_surrogate(self, x):
        x = x.detach().requires_grad_(True)
        surr_v = self.surrogate.forward(x)
        score_func = torch.autograd.grad(surr_v.sum(), x)[0]
        k_xx = self.kernel(x, x.detach())
        grad_k = -torch.autograd.grad(k_xx.sum(), x)[0]
        w = surr_v - self.log_p(x)
        w = w - w.logsumexp(dim=0)  # normalisation
        w = w.exp()
        phi = w.repeat(1, x.shape[1]) * (k_xx.detach().matmul(score_func) + grad_k)
        return phi

    def step(self, x):
        self.optimizer.zero_grad()
        if self.surrogate is None:
            # Eq 23
            x.grad = -self.phi(x)
        else:
            # Eq 22
            x.grad = -self.phi_surrogate(x)
        self.optimizer.step()


# Reformatted SVGD code based on the original implementation. Currently not tested with the BayesSim framework.
class SVGDOriginal(SVGD):
    def __init__(self, log_p):
        self.iter = 0
        self.historical_grad_square = 0
        self.log_p = log_p

    def d_log_p(self, x):
        x = x.detach().requires_grad_(True)
        return torch.autograd.grad(self.log_p(x).sum(), x)[0]

    def RBF_kernel(self, X, Y, h=-1):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())
        pairwise_dists = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists.detach().cpu().numpy())
            h = h / np.log(X.shape[0] + 1)

        kernel_Xj_Xi = (- pairwise_dists / h).exp()
        d_kernel_Xi = torch.zeros(X.shape)
        for i in range(X.shape[0]):
            d_kernel_Xi[i] = kernel_Xj_Xi[i].matmul(X[i] - X) * 2 / h
        return kernel_Xj_Xi, d_kernel_Xi

    # Computes a kernel using pairwise relations between the dimensions.
    # pairs_idx is a list with pairs of dimensions. For example for D=3
    # pairs_idx can be [[0,1],[1,2]].
    # If pairs_idx is None, it uses pairs sequentially, [[0,1],[1,2],[2,3],...]
    def pairwise_RBF_kernel(self, X, Y, h=-1, pairs_idx=None):
        if pairs_idx is None:
            pairs_idx = [[i, i + 1] for i in range(X.shape[1] - 1)]

        kernel_Xj_Xi = torch.zeros(X.shape[0], Y.shape[0]).type(self.dtype).to(self.device)
        d_kernel_Xi = torch.zeros(X.shape).type(self.dtype).to(self.device)

        for i in range(len(pairs_idx)):
            k_tmp, dk_tmp = self.RBF_kernel(X[:, pairs_idx[i]].reshape(-1, 1),
                                            Y[:, pairs_idx[i]].reshape(-1, 1), h)
            kernel_Xj_Xi += k_tmp
            d_kernel_Xi[:, i] = dk_tmp[:, 0]
        return kernel_Xj_Xi, d_kernel_Xi


    def step(self, x, stepsize=5e-2, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x is None or self.log_p is None:
            raise ValueError('x or lnprob cannot be None!')

        # adagrad with momentum
        eps_factor = 1e-8

        if debug and (iter + 1) % 1000 == 0:
            print('iter ' + str(self.iter + 1))

        kernel_xj_xi, d_kernel_xi = self.RBF_kernel(x, x, h=bandwidth)
        current_grad = (kernel_xj_xi.matmul(self.d_log_p(x)) + d_kernel_xi) / x.shape[0]
        # print(d_kernel_xi)
        # print(current_grad)
        # print(current_grad-d_kernel_xi)
        if self.iter == 0:
            self.historical_grad_square += current_grad ** 2
        else:
            self.historical_grad_square = alpha * self.historical_grad_square + (1 - alpha) * (current_grad ** 2)
        adj_grad = current_grad / np.sqrt(self.historical_grad_square + eps_factor)

        # print(adj_grad.sum())
        x += stepsize * adj_grad
        self.iter += 1
        return x
