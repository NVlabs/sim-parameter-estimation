import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import elfi
import src.utils.pdf as pdf
import scipy


class ABCRejection(object):
    def __init__(self, generator=None, p_lower=None, p_upper=None, n_particles=100):
        self.p_lower = p_lower
        self.p_upper = p_upper
        self.n_particles = n_particles
        self.generator = generator
        self.model = elfi.new_model()
        self.threshold = 0.1  # %% acceptance threshold

    def identity(self, x):
        return x[0, :].reshape(1, -1)

    def set_prior_from_generator(self, generator):
        self.p_lower, self.p_upper = generator.get_prior()
        #self.prior = [elfi.Prior('uniform', self.p_lower[i], self.p_upper[i]) for i in range(self.p_lower.shape[0])]
        #for i in range(self.p_lower.shape[0]):
        #    self.prior.append(elfi.Prior('uniform', self.p_lower[i], self.p_upper[i], model=self.model, name=str(i)))

    def simulator(self, x, batch_size=1, random_state=None):
        _, data = self.generator.run_forward_model(x.reshape(-1, self.p_lower.size), batch_size)
        return data.reshape(1, -1)

    def predict(self, data_test):
        elfi.new_model("Rejection")
        prior = elfi.Prior(MVUniform, self.p_lower, self.p_upper)
        sim = elfi.Simulator(self.simulator, prior, observed=data_test, name='sim')
        SS = elfi.Summary(self.identity, sim, name='identity')
        d = elfi.Distance('euclidean', SS, name='d')
        rej = elfi.Rejection(d, batch_size=1, seed=42)
        samples = rej.sample(self.n_particles, threshold=self.threshold)
        return samples.samples_array

    # Prediction function returning a MoG
    def predict_mog(self, data_test, sigma=0.01):
        # Builds a MoG representation to be compatible with MDN
        # Parameters of the mixture
        ntest, _ = data_test.shape  # test dimensionality and number of queries

        mog = []
        for pt in range(ntest):
            start_time = datetime.now()
            #x_testS = self.scaler.transform(x_test) no scaling implemented
            y_pred = self.predict(data_test)
            end_time = datetime.now()
            print('\n')
            print("*********************************  Prediction ends  *********************************")
            print('\n')
            print('Duration: {}'.format(end_time - start_time))

            a = [1./self.n_particles for i in range(self.n_particles)]

            # Assuming output from BlackBoxOptimizer
            ms = [y_pred[i, :] for i in range(self.n_particles)]
            p_std = 0.005 * np.ones(y_pred.shape[1])

            Ss = [np.diag(p_std) for i in range(self.n_particles)]
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        #self.plot_objective([-2, 0])
        return mog


class ABCSMC(ABCRejection):
    def __init__(self, generator=None, p_lower=None, p_upper=None, n_particles=100):
        super().__init__(generator, p_lower, p_upper, n_particles)

    def predict(self, data_test):
        elfi.new_model("SMC")
        prior = elfi.Prior(MVUniform, self.p_lower, self.p_upper)
        sim = elfi.Simulator(self.simulator, prior, observed=data_test, name='sim')
        SS = elfi.Summary(self.identity, sim, name='identity')
        d = elfi.Distance('euclidean', SS, name='d')
        smc = elfi.SMC(d, batch_size=1, seed=42)
        samples = smc.sample(self.n_particles, [self.threshold])
        return samples.samples_array


class BOLFI(ABCRejection):
    def __init__(self, generator=None, p_lower=None, p_upper=None, n_particles=100):
        super().__init__(generator, p_lower, p_upper, n_particles)

    def predict(self, data_test):
        elfi.new_model("BOLFI")
        prior = elfi.Prior(MVUniform, self.p_lower, self.p_upper)
        sim = elfi.Simulator(self.simulator, prior, observed=data_test, name='sim')
        SS = elfi.Summary(self.identity, sim, name='identity')
        d = elfi.Distance('euclidean', SS, name='d')
        log_d = elfi.Operation(np.log, d)
        bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20,
                           update_interval=10, acq_noise_var=self.p_lower.size*[0.1],
                           bounds=None, seed=42)
        bolfi.fit(n_evidence=self.n_particles)
        post = bolfi.extract_posterior(-1.)
        samples = post.model.X
        return samples


class MVUniform(elfi.Distribution):
    def rvs(lower, upper, size=1, random_state=None):
        if type(size) is tuple:
            length = size[0]
        else:
            length = size
        u = np.zeros([length, lower.shape[0]])
        for i in range(lower.shape[0]):
            u[:, i] = scipy.stats.uniform.rvs(loc=lower[i], scale=upper[i]-lower[i], size=size, random_state=random_state)
        return u