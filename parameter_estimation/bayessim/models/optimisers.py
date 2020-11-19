import numpy as np
import cma
import sys
import torch
# sys.path.append('/home/framos/Code/RafaOliveira/hdbo')
# from hdbo.models.ssgpr import ISSGPR
# from hdbo.af.ucb import UCB
# from hdbo.nloptoptimizer import NLoptOptimizer
# from hdbo.bo import BayesOpt
from src.models.svgd import SVGD
from torch.nn.functional import  cosine_similarity


class BlackBoxOptimizer(SVGD):
    def __init__(self, algorithm="cma",
                 generator=None, p_lower=None, p_upper=None, n_iter=100):
        super().__init__(None, None, None, 1, generator, p_lower, p_upper, n_iter)
        self.algorithm = algorithm
        self.n_particles = 1
        self.gpu = False
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

    def predict(self, data_test):
        if self.scaler is not None:
            self.data_test = torch.tensor(self.scaler.transform(data_test), dtype=torch.float32)
        else:
            self.data_test = torch.tensor(data_test, dtype=torch.float32)

        #x0 = np.random.uniform(self.p_lower, self.p_upper)
        x0 = self.generator.sample_params_from_uniform_prior(1)
        if self.algorithm == "cma":
            x_opt = self.cma(x0)
            x_all = None
        elif self.algorithm == "BayesianOptimization":
            pass
        else:
            x_opt = self.NLOpt(x0, self.algorithm)
            x_all = None
        return x_opt.reshape(1, -1), x_all

    def cma(self, starting_point):
        def parallel_objective(x):
            results = self.objective_func(np.array(x))
            return list(results)

        xcma = cma.fmin(self.objective_func,
                        starting_point, .25,
                        {'bounds':[self.p_lower.reshape(starting_point.size,),
                        self.p_upper.reshape(starting_point.size,)],
                         'maxfevals': self.n_iter},
                        parallel_objective=parallel_objective)
        return xcma[0]

    def BayesianOptimization(self):
        dim = len(env_params)
        n_iter = 50
        n_features = 100
        noise_stddev = 0.01
        lengthscale = 1.
        afopt_algorithm = 'DIRECT'
        afopt_max_iter = 50

        # Defines the GP models
        model = ISSGPR(n_features, dim, noise_stddev)

        bounds = [(self.p_lower[i], self.p_upper[i]) for i in range(self.p_lower.size)]
        bounds_array = np.asarray(bounds)
        starting_point = np.atleast_2d(np.random.uniform(bounds_array[:, 0], bounds_array[:, 1]))
        starting_y = self.objective_func(starting_point)
        model.update(starting_point, starting_y)

        # Sets the acquisition function
        af = UCB(model, uncertainty_factor=0.5, maximize=False)

        acquisition_optimizer = NLoptOptimizer(bounds,
                                               afopt_max_iter,
                                               afopt_algorithm,
                                               maximize=True, xtol_rel=1e-6)

        bopt = BayesOpt(model, af,
                        acquisition_optimizer,
                        bounds_array[:, 0], bounds_array[:, 1])

        # Runs BO
        X, y = bopt.run(self.objective_func, n_iter)
        f_best = np.ones_like(y) * np.asscalar(self.objective_func(X[0]))
        f = np.asarray([np.asscalar(self.objective_func(p)) for p in X])
        f_best[1:] = np.asarray([f[:i].min() for i in np.arange(1, f.size)])

        # Best parameter index
        ind = np.argmin(y)

        return X[ind], X, y

    def NLOpt(self, starting_point, opt_alg):
        self.p_lower = self.p_lower.reshape(-1, )
        self.p_upper = self.p_upper.reshape(-1, )
        bounds = [(self.p_lower[i], self.p_upper[i]) for i in range(self.p_lower.size)]
        nlopt = NLoptOptimizer(bounds,
                               self.n_iter,
                               opt_alg,
                               maximize=False, xtol_rel=1e-3)

        xnlopt, ynlopt = nlopt.optimize(starting_point.reshape(1, -1), self.objective_func)
        return xnlopt

    def objective_func(self, x, loss='L2'):
        return super().objective_func(x.reshape(-1, self.p_lower.size), loss).clone().cpu().numpy()

