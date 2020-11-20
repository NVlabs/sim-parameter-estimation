import torch
from torch.distributions import MultivariateNormal
import numpy as np
import nlopt
import sys
# sys.path.append('/home/framos/Code/RafaOliveira/ssgp')
# import ssgp


# Simple fast GP
class FGP(object):
    def __init__(self, kernel, log_p=None, gpu=True):
        self.k = kernel
        self.mean = -1.
        self.noise = 1e-2
        self.signal_stdev = 2.

        self.objective = log_p
        self.N = 4
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"
        self.X = torch.tensor([]).type(self.dtype).to(self.device)
        self.y = torch.tensor([]).type(self.dtype).to(self.device)
        self.Xplus = torch.tensor([]).type(self.dtype).to(self.device)
        self.yplus = torch.tensor([]).type(self.dtype).to(self.device)
        self.K_inv = torch.tensor([]).type(self.dtype).to(self.device)

    def update(self, X):
        self.X = X
        dim = self.X.shape[1]

        if self.objective is not None:
            #sampler = MultivariateNormal(self.X, 1e-2*torch.eye(dim).type(self.dtype).to(self.device))
            #self.Xplus = torch.cat((self.X, sampler.sample((self.N,)).reshape([-1, dim])), 0)
            self.Xplus = torch.cat((self.X, self.X - 1e-2*self.X.reshape([-1, dim])), 0)
            self.yplus = self.objective(self.Xplus)
        else:
            self.Xplus = self.X
            self.yplus = self.y

        #Fix for potential nan
        nan_idx = torch.isnan(self.yplus)
        self.Xplus = self.Xplus[~nan_idx[:, 0], :]
        self.yplus = self.yplus[~nan_idx[:, 0]]

        K = (self.signal_stdev**2)*self.k(self.Xplus, self.Xplus)
        K = (K + K.t()).mul(0.5)
        K = K + (self.noise**2) * torch.eye(self.Xplus.shape[0]).type(self.dtype).to(self.device)

        # Compute K's inverse with Cholesky factorisation.
        try:
            self.L = torch.cholesky(K)
            self.L_inv = self.L.inverse()
            self.K_inv = self.L_inv.t().mm(self.L_inv)
        except:
            self.K_inv = torch.inverse(K)
            print("Warning: using pseudoinverse")

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x).type(self.dtype).to(self.device)
        K_s = self.k(x, self.Xplus)
        return K_s.mm(self.K_inv.mm(self.yplus))


# Full GP
class FullGP(object):
    def __init__(self, kernel, log_p=None, gpu=False):
        self.k = kernel
        self.objective = log_p
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

        self.mean = torch.tensor(-1.).type(self.dtype).to(self.device).detach().requires_grad_(False)
        self.noise = torch.tensor(0.1).type(self.dtype).to(self.device).detach().requires_grad_(False)
        self.signal_stdev = torch.tensor(1.).type(self.dtype).to(self.device).detach().requires_grad_(False)
        self.lenghscales = torch.tensor(1.).type(self.dtype).to(self.device).detach().requires_grad_(False)
        self.X = torch.tensor([]).type(self.dtype).to(self.device)
        self.y = torch.tensor([]).type(self.dtype).to(self.device)
        self.K_inv = torch.tensor([]).type(self.dtype).to(self.device)
        self.L = torch.tensor([]).type(self.dtype).to(self.device)
        self.L_inv = torch.tensor([]).type(self.dtype).to(self.device)
        self.max_K_size = int(4000)
        self.n_updates = 0
        self.n_optimize = 300  #optimizes the hyperparameters after this many updates

    def update(self, X):
        if self.objective is not None:
            self.X = torch.cat((self.X, X), 0)
            y = self.objective(X)
            self.y = torch.cat((self.y, y - self.mean), 0)

        size_X = self.X.shape[0]
        if size_X > self.max_K_size:
            self.X = self.X[size_X - self.max_K_size: -1, :]
            self.y = self.y[size_X - self.max_K_size: -1, :]

        K = (self.signal_stdev ** 2) * self.k(self.X, self.X)
        K = (K + K.t()).mul(0.5)
        K = K + (self.noise ** 2) * torch.eye(self.X.shape[0]).type(self.dtype).to(self.device)

        # Compute K's inverse with Cholesky factorisation.
        try:
            self.L = torch.cholesky(K)
            self.L_inv = self.L.inverse()
            self.K_inv = self.L_inv.t().mm(self.L_inv)
        except:
            self.K_inv = torch.inverse(K)
            print("Warning: using pseudoinverse")
        #print("Loglikelihood = ",
        #      self.log_likelihood([self.lenghscales, self.signal_stdev, self.noise, self.mean]))
        self.n_updates += 1
        if self.n_updates%self.n_optimize == 0:
            params = self.optimize(pars=[self.lenghscales,
                                         self.signal_stdev,
                                         self.noise,
                                         self.mean], n_steps=10)
            print(params)
            self.lenghscales = params[0]
            self.k.set_lengthscale(params[0])
            #self.signal_stdev = params[1]
            #self.noise = params[2]
            self.mean = params[3]

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x).type(self.dtype).to(self.device)
        K_s = self.k(x, self.X)
        return self.mean + K_s.mm(self.K_inv.mm(self.y))

    def log_likelihood(self, pars):
        self.k.set_lengthscale(pars[0])
        #K = (pars[1] ** 2) * self.k(self.X, self.X)
        K = (self.signal_stdev ** 2) * self.k(self.X, self.X)
        K = (K + K.t()).mul(0.5)
        K = K + (pars[2]**2) * torch.eye(self.X.shape[0]).type(self.dtype).to(self.device)

        # Compute K's inverse with Cholesky factorisation.
        try:
            L = torch.cholesky(K)
            L_inv = L.inverse()
            K_inv = L_inv.t().mm(L_inv)
        except:
            K = K + 1e-1*torch.eye(self.X.shape[0]).type(self.dtype).to(self.device)
            L = torch.cholesky(K)
            L_inv = L.inverse()
            K_inv = L_inv.t().mm(L_inv)
            #K_inv = torch.inverse(K)

        # Compute the log likelihood.
        y_mean = self.y - pars[3]
        log_likelihood_dims = -0.5 * y_mean.t().mm(K_inv.mm(y_mean)).sum(dim=0)
        log_likelihood_dims -= L.diag().log().sum()
        log_likelihood_dims -= L.shape[0] / 2.0 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(dim=-1)
        print("Negative log likelihood = ", -log_likelihood)
        return -log_likelihood

    def optimize(self, n_steps=10, pars=None):

        # optimizer = torch.optim.LBFGS(params=pars)
        # for i in range(n_steps):
        #     def closure():
        #         optimizer.zero_grad()
        #         loss = self.log_likelihood(pars)
        #         loss.backward()
        #         return loss
        #     optimizer.step(closure)
        optimizer = torch.optim.RMSprop(params=pars)
        for i in range(n_steps):
            optimizer.zero_grad()
            loss = self.log_likelihood(pars)
            loss.backward()
            optimizer.step()
        return pars
        # n_hp = 2
        #
        # hp_lower_bounds = np.asarray([.5, -50.])
        # hp_upper_bounds = np.asarray([10., 50.])
        #
        # nlopt_obj = self.log_likelihood
        # # ssgp.tuning.NLoptTuningObjective(
        # #    self.model,["lengthscale","mean_params"],
        # #    compute_grad=True) # length-scale is mandatory to be included
        #
        # hp_opt = nlopt.opt(nlopt.LD_MMA, n_hp)
        # hp_opt.set_lower_bounds(hp_lower_bounds)
        # hp_opt.set_upper_bounds(hp_upper_bounds)
        # hp_opt.set_min_objective(nlopt_obj)
        # hp_opt.set_maxeval(n_steps)
        #
        # initial_hp = [1., -50.]
        # print("Initial setting:", initial_hp)
        #
        # print("Optimising hyperparmeters...")
        # final_hp = hp_opt.optimize(initial_hp)
        #
        # self.model.set_hyperparameters(**nlopt_obj.hp_map(final_hp))
        # print("GP hyperparameters: {}\n".format(self.model.get_hyperparameters()))


class SSGP(object):
    def __init__(self, kernel, log_p=None, gpu=False):
        self.dim = None
        self.n_freqs = 200
        self.mean = -100
        self.noise = 0.01
        self.signal_stdev = 2.
        self.lengthscale = kernel.sigma
        if type(kernel).__name__ is "RBF":
            self.k = "squared_exponential"
        else:
            ValueError('Not implemented yet')
        self.objective = log_p
        self.model = None
        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

    def update(self, X):
        if self.model is None:
            self.dim = X.shape[1]
            self.model = ssgp.models.ISSGPR(self.n_freqs, self.dim,
                               lengthscale=self.lengthscale,
                               noise_stddev=self.noise,
                               signal_stddev=self.signal_stdev,
                               kernel_type=self.k,
                               mean_function=ssgp.mean_functions.ConstantMean(self.mean))

        y = self.objective(X).reshape([-1, 1])
        for i in range(y.shape[0]):
            self.model.update(X[i].reshape([1, -1]), y[i])

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.model.predict(x)[0]

    def optimise(self, n_steps=10):
        n_hp = 2
        hp_lower_bounds = np.asarray([.5, -50.])
        hp_upper_bounds = np.asarray([10., 50.])

        nlopt_obj = ssgp.tuning.NLoptTuningObjective(
            self.model, ["lengthscale", "mean_params"],
            compute_grad=True)  # length-scale is mandatory to be included

        hp_opt = nlopt.opt(nlopt.LD_MMA, n_hp)
        hp_opt.set_lower_bounds(hp_lower_bounds)
        hp_opt.set_upper_bounds(hp_upper_bounds)
        hp_opt.set_min_objective(nlopt_obj)
        hp_opt.set_maxeval(n_steps)

        initial_hp = [1., -50.]
        print("Initial setting:", initial_hp)

        print("Optimising hyperparmeters...")
        final_hp = hp_opt.optimize(initial_hp)

        self.model.set_hyperparameters(**nlopt_obj.hp_map(final_hp))
        print("GP hyperparameters: {}\n".format(self.model.get_hyperparameters()))