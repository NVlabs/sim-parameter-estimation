# from src.utils.param_inference import *
from scipy import stats
from src.utils.generator import Default


class BayesSim(object):
    def __init__(self,
                 model=None,
                 obs=None,
                 generator=Default,
                 n_components=1,
                 seed=None,
                 verbose=True,
                 prior_norm=False,
                 pilot_samples=50,
                 params_dim=None,
                 stats_dim=None):

        self.generator = generator # generates the data
        self.generator.proposal = None
        self.obs = obs # observation
        self.n_components = n_components # number of components for the mdn or number
        # of particles for SVGD
        self.seed = seed
        self.verbose = verbose
        self.round = 0
        self.model = model

        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        # MDN model
        # if model is None:
        #     self.model = MDNN(ncomp=self.n_components, outputd=params_dim,
        #                       inputd=stats_dim, nhidden=2, nunits=[24, 24])


        # if np.any(np.isnan(self.obs)):
        #     raise ValueError("Observed data contains NaNs")

        # parameters for z-transform of params
        if prior_norm:
            # z-transform for params based on prior
            self.params_mean = self.generator.prior.mean
            self.params_std = self.generator.prior.std
        else:
            # parameters are set such that z-transform has no effect
            self.params_mean = np.zeros((params_dim,))
            self.params_std = np.ones((params_dim,))

        # parameters for z-transform for stats
        if pilot_samples is not None and pilot_samples != 0:
            # determine via pilot run
            if seed is not None:  # reseed generator for consistent inits
                self.generator.reseed(self.gen_newseed())
        else:
            # parameters are set such that z-transform has no effect
            self.stats_mean = np.zeros((stats_dim,))
            self.stats_std = np.ones((stats_dim,))

    def gen(self, n_samples, prior_mixin=0, verbose=None, batch_size=1000):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        """
        params, stats = self.generator.gen(n_samples)

        # z-transform params and stats
        # params = (params - self.params_mean) / self.params_std
        # stats = (stats - self.stats_mean) / self.stats_std
        return params, stats

    def run(self, n_train=500, epochs=1000, n_rounds=2, batch_size=1000):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """

        logs = []
        trn_datasets = []

        trn_data = self.gen(n_train)
        _, log = self.model.train(x_data=trn_data[1],
                                  y_data=trn_data[0],
                                  nepoch=epochs,
                                  batch_size=batch_size)

        logs.append({'loss': log})
        trn_datasets.append(trn_data)

        return logs, trn_datasets

    def predict(self, x, threshold=0.005):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x = x.reshape(1, -1)
        if self.generator.proposal is None:
            # no correction necessary
            return self.model.predict_mog(x)[0]  # via super
        else:
            # mog is posterior given proposal prior
            mog = self.model.predict_mog(x)[0]  # via super
            mog.prune_negligible_components(threshold=threshold)

            # compute posterior given prior by analytical division step
            if isinstance(self.generator.prior, dd.Uniform):
                posterior = mog / self.generator.proposal
            elif isinstance(self.generator.prior, dd.Gaussian):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented
            return posterior

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    def pilot_run(self, n_samples):
        """Pilot run in order to find parameters for z-scoring stats
        """
        verbose = '(pilot run) ' if self.verbose else False
        params, stats = self.generator.gen(n_samples)
        self.stats_mean = np.nanmean(stats, axis=0)
        self.stats_std = np.nanstd(stats, axis=0)




