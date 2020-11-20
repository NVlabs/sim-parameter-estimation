import numpy as np  # basic math and random numbers
import torch  # package for building functions with learnable parameters
import torch.nn as nn  # prebuilt functions specific to neural networks
from parameter_estimation.bayessim.models.random_features_torch import RFF
import parameter_estimation.bayessim.utils.pdf as pdf
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import Independent
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle


class MDNNTorch(nn.Module):
    def __init__(self, hidden_layers=[50, 50], ncomp=10, inputd=1, outputd=1, gpu=True):
        super(MDNNTorch, self).__init__()
        last_layer_size = None
        self.network = []
        self.n_outputs = outputd
        self.n_gaussians = ncomp
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

        self.pi = nn.Linear(last_layer_size, self.n_gaussians).to(self.device)
        self.L = nn.Linear(last_layer_size, int(0.5 * self.n_gaussians * self.n_outputs * (self.n_outputs - 1))).to(self.device)
        self.L_diagonal = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs).to(self.device)
        self.mu = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs).to(self.device)

    def forward(self, x):
        for ix, layer in enumerate(self.network):
            if ix == 0:
                z_h = layer(x)
            else:
                z_h = layer(z_h)

        pi = nn.functional.softmax(self.pi(z_h), -1)
        L_diagonal = torch.exp(self.L_diagonal(z_h)).reshape(-1, self.n_outputs, self.n_gaussians)
        L = self.L(z_h).reshape(-1, int(0.5 * self.n_outputs * (self.n_outputs - 1)), self.n_gaussians)
        mu = self.mu(z_h).reshape(-1, self.n_outputs, self.n_gaussians)
        return pi, mu, L, L_diagonal

    def generate_data(self, n_samples=1000):
        # evenly spaced samples from -10 to 10

        x_test_data = np.linspace(-15, 15, n_samples).reshape(n_samples, 1)
        x_test = torch.from_numpy(x_test_data).to(self.device)

        pi, mu, L, L_d,  = self(x_test)

        pi_data = pi.data.cpu().numpy()
        L_data = L.data.cpu().numpy()
        L_diagonal_data = L_d.data.cpu().numpy()
        mu_data = mu.data.cpu().numpy()

        return x_test_data, pi_data, mu_data, L_data, L_diagonal_data

    def mdn_loss_fn(self, pi, mu, L, L_d, y, epsilon=1e-9):

        result = torch.zeros(y.shape[0], self.n_gaussians).to(self.device)
        tril_idx = np.tril_indices(self.n_outputs, -1)
        diag_idx = np.diag_indices(self.n_outputs)

        for idx in range(self.n_gaussians):
            tmp_mat = torch.zeros(y.shape[0], self.n_outputs, self.n_outputs).to(self.device)
            tmp_mat[:, tril_idx[0], tril_idx[1]] = L[:, :, idx]
            tmp_mat[:, diag_idx[0], diag_idx[1]] = L_d[:, :, idx]
            mvgaussian = MultivariateNormal(loc=mu[:, :, idx], scale_tril=tmp_mat)
            result_per_gaussian = mvgaussian.log_prob(y)
            result[:, idx] = result_per_gaussian + pi[:, idx].log()
        return -torch.mean(torch.logsumexp(result, dim=1))

    def train(self, x_data, y_data, nepoch=1000, batch_size=100, save=False, verbose=True):

        optimizer = torch.optim.Adam(self.parameters())

        x_variable = torch.from_numpy(np.array(x_data)).type(self.dtype).to(self.device)
        y_variable = torch.from_numpy(np.array(y_data)).type(self.dtype).to(self.device)

        batch_size = len(x) if batch_size is None else batch_size

        # print("Training mdn network on {} datapoints".format(len(x_data)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x_data), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        lossHistory = []
        for epoch in range(nepoch):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            pi, mu, L, L_diagonal = self(x_batch)
            loss = self.mdn_loss_fn(pi, mu, L, L_diagonal, y_batch)
            loss.backward()
            optimizer.step()

            # if epoch == 0:
            #     print("Initial Loss is: {}".format(loss.item()))

            # elif epoch % 100 == 0 and verbose:
            #     if epoch != 0:
            #         print(" Iteration:", epoch, "Loss", loss.item())

            lossHistory.append(loss.item())
        # print("Training Finished, final loss is {}".format(loss.item()))
        return self, lossHistory

    def predict_mog(self, x):
        """
        Return the conditional mog at location x.
        :param network: an MDN network
        :param x: single input location
        :return: conditional mog at x
        """

        ntest, dim = x.shape  # test dimensionality and number of queries
        # gather mog parameters
        pi, mu, L, L_d = self(torch.tensor(x).type(self.dtype).to(self.device))
        pi = pi.data.cpu().numpy()
        mean = mu.data.cpu().numpy()

        tril_idx = np.tril_indices(self.n_outputs, -1)
        diag_idx = np.diag_indices(self.n_outputs)
        mog = []
        for pt in range(ntest):
            Ss = []
            ms = []
            for i in range(self.n_gaussians):
                np_L = np.zeros([self.n_outputs, self.n_outputs])
                np_L[tril_idx[0], tril_idx[1]] = L[pt, :, i].data.cpu().numpy()
                np_L[diag_idx[0], diag_idx[1]] = L_d[pt, :, i].data.cpu().numpy()
                Ss.append(np.matmul(np_L, np_L.T))
                ms.append(mean[pt, :, i])
            # return mog
            mog.append(pdf.MoG(a=pi[pt, :], ms=ms, Ss=Ss))
        return mog


class MDRFFTorch(MDNNTorch):
    def __init__(self,  ncomp=10, nfeat=500, inputd=1, outputd=1,
                 cosOnly=False, kernel="RBF", sigma=1, quasiRandom=True, gpu=True):
        super(MDNNTorch, self).__init__()

        self.n_gaussians = ncomp  # number of mixture components
        self.nfeat = nfeat  # number of features
        self.inputd = inputd  # dimensionality of the input
        self.n_outputs = outputd  # dimensionality of the output
        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        self.sigma = sigma * np.ones(self.inputd)
        self.kernel = kernel

        #self.weights = weights  # weight function
        self.scaler = StandardScaler()

        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(1)
            self.device = "cuda:1"
        else:
            self.dtype = torch.float32
            self.device = "cpu"
        self.rff = RFF(self.nfeat, self.inputd, self.sigma,
                       self.cosOnly, self.quasiRandom, self.kernel, self.gpu)

        self.pi = nn.Linear(nfeat, self.n_gaussians).to(self.device)
        self.L = nn.Linear(nfeat, int(0.5 * self.n_gaussians * self.n_outputs * (self.n_outputs - 1))).to(self.device)
        self.L_diagonal = nn.Linear(nfeat, self.n_gaussians * self.n_outputs).to(self.device)
        self.mu = nn.Linear(nfeat, self.n_gaussians * self.n_outputs).to(self.device)

    def forward(self, x):
        z_h = self.rff.toFeatures(x)
        pi = nn.functional.softmax(self.pi(z_h), -1)
        L_diagonal = torch.exp(self.L_diagonal(z_h)).reshape(-1, self.n_outputs, self.n_gaussians)
        L = self.L(z_h).reshape(-1, int(0.5 * self.n_outputs * (self.n_outputs - 1)), self.n_gaussians)
        mu = self.mu(z_h).reshape(-1, self.n_outputs, self.n_gaussians)
        return pi, mu, L, L_diagonal


class MDLSTMTorch(MDNNTorch):
    def __init__(self, ncomp=10, nhidden=2, hidden_layers=[50, 24, 24],
                 inputd=None, outputd=None, nsteps=200, weights=None, gpu=True):
        super(MDNNTorch, self).__init__()

        last_layer_size = None
        self.network = []
        self.inputd = inputd
        self.n_outputs = outputd
        self.n_gaussians = ncomp
        self.nsteps = nsteps
        self.gpu = gpu
        self.hidden_layers = hidden_layers
        self.nhidden = nhidden
        self.weights = weights
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(1)
            self.device = "cuda:1"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

        for ix, layer_size in enumerate(self.hidden_layers):

            if ix == 0:
                self.network.append(nn.LSTM(inputd, layer_size, nhidden,
                                            dropout=0.,
                                            batch_first=True).to(self.device))
                self.network.append(nn.Linear(layer_size*self.nsteps, layer_size).to(self.device))
            else:
                self.network.append(nn.Linear(last_layer_size, layer_size).to(self.device))
            self.network.append(nn.Tanh().to(self.device))
            last_layer_size = layer_size

        self.pi = nn.Linear(last_layer_size, self.n_gaussians).to(self.device)
        self.L = nn.Linear(last_layer_size, int(0.5 * self.n_gaussians * self.n_outputs * (self.n_outputs - 1))).to(self.device)
        self.L_diagonal = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs).to(self.device)
        self.mu = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs).to(self.device)

    def forward(self, x):
        for ix, layer in enumerate(self.network):
            if ix == 0:
                z_h, hidden = layer(x)
                z_h = z_h.reshape([-1, self.nsteps*self.hidden_layers[0]])
            else:
                z_h = layer(z_h)

        pi = nn.functional.softmax(self.pi(z_h), -1)
        L_diagonal = torch.exp(self.L_diagonal(z_h)).reshape(-1, self.n_outputs, self.n_gaussians)
        L = self.L(z_h).reshape(-1, int(0.5 * self.n_outputs * (self.n_outputs - 1)), self.n_gaussians)
        mu = self.mu(z_h).reshape(-1, self.n_outputs, self.n_gaussians)
        return pi, mu, L, L_diagonal

    def train(self, x_data, y_data, nepoch=1000, batch_size=100, save=False, verbose=True):

        optimizer = torch.optim.Adam(self.parameters())
        x_data = np.array(x_data).reshape(-1, self.nsteps, self.inputd)
        y_data = np.array(y_data)

        x_variable = torch.from_numpy(x_data).type(self.dtype).to(self.device)
        y_variable = torch.from_numpy(y_data).type(self.dtype).to(self.device)

        batch_size = len(x) if batch_size is None else batch_size

        # print("Training mdn network on {} datapoints".format(len(x_data)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x_data), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        lossHistory = []
        for epoch in range(nepoch):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            pi, mu, L, L_diagonal = self(x_batch)
            loss = self.mdn_loss_fn(pi, mu, L, L_diagonal, y_batch)
            loss.backward()
            optimizer.step()

            # if epoch == 0:
            #     print("Initial Loss is: {}".format(loss.item()))

            # elif epoch % 100 == 0 and verbose:
            #     if epoch != 0:
            #         print(" Iteration:", epoch, "Loss", loss.item())

            lossHistory.append(loss.item())
        # print("Training Finished, final loss is {}".format(loss.item()))
        return self, lossHistory

    def predict_mog(self, x):
        """
        Return the conditional mog at location x.
        :param network: an MDN network
        :param x: single input location
        :return: conditional mog at x
        """

        x = x.T
        ntest = x.shape[1]
        # Prepares the data for LSTMs
        x_fixed = np.zeros([self.nsteps, self.inputd, ntest])
        for i in range(ntest):
            x_fixed[:, :, i] = x[:, i].reshape([self.nsteps, self.inputd])
        x_fixed = np.swapaxes(x_fixed, 0, 1)
        x_fixed = np.swapaxes(x_fixed, 0, 2)

        # gather mog parameters
        pi, mu, L, L_d = self(torch.tensor(x_fixed).type(self.dtype).to(self.device))
        pi = pi.data.cpu().numpy()
        mean = mu.data.cpu().numpy()

        tril_idx = np.tril_indices(self.n_outputs, -1)
        diag_idx = np.diag_indices(self.n_outputs)
        mog = []
        for pt in range(ntest):
            Ss = []
            ms = []
            for i in range(self.n_gaussians):
                np_L = np.zeros([self.n_outputs, self.n_outputs])
                np_L[tril_idx[0], tril_idx[1]] = L[pt, :, i].data.cpu().numpy()
                np_L[diag_idx[0], diag_idx[1]] = L_d[pt, :, i].data.cpu().numpy()
                Ss.append(np.matmul(np_L, np_L.T))
                ms.append(mean[pt, :, i])
            # return mog
            mog.append(pdf.MoG(a=pi[pt, :], ms=ms, Ss=Ss))
        return mog

    def save(self,
             save_config=True,
             save_model_weights=True,
             model_config_fname=None, weights_name=""):
        if save_config:
            model_config = {"ncomp": self.n_gaussians, "nhidden": self.nhidden, "nunits": self.hidden_layers, "inputd": self.inputd,
                            "outputd": self.n_outputs, "weights": self.weights}
            f = open(model_config_fname, 'wb')
            pickle.dump(model_config, f)
            f.close()

        #if save_model_weights: //Not yet implemented
        #    self.model.save_weights(weights_name)


class RDNN(MDNNTorch):
    """ Implements a restricted density network. This is basically a MDN with constant weight values
    and equal standard deviation for all components. """
    def __init__(self, hidden_layers=[100, 100, 100], ncomp=50, inputd=1, outputd=1, gpu=True):
        super(MDNNTorch, self).__init__()
        last_layer_size = None
        self.network = []
        self.n_outputs = outputd
        self.n_gaussians = ncomp
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

        self.pi = torch.tensor([1./self.n_gaussians]).to(self.device)
        self.sigma = nn.Linear(last_layer_size, self.n_outputs).to(self.device)
        self.mu = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs).to(self.device)

    def forward(self, x):
        for ix, layer in enumerate(self.network):
            if ix == 0:
                z_h = layer(x)
            else:
                z_h = layer(z_h)

        sigma = torch.exp(self.sigma(z_h)).reshape(-1, self.n_outputs)
        mu = self.mu(z_h).reshape(-1, self.n_outputs, self.n_gaussians)
        return sigma, mu

    def generate_data(self, n_samples=1000):
        # evenly spaced samples from -10 to 10

        x_test_data = np.linspace(-15, 15, n_samples).reshape(n_samples, 1)
        x_test = torch.from_numpy(x_test_data).to(self.device)

        sigma, mu = self(x_test)

        pi_data = self.pi.data.cpu().numpy()
        sigma_data = sigma.data.cpu().numpy()
        mu_data = mu.data.cpu().numpy()

        return x_test_data, pi_data, mu_data, sigma_data

    def mdn_loss_fn(self, mu, sigma, y, epsilon=1e-9):
        # Non-vectorised version
        #result = torch.zeros(y.shape[0], self.n_gaussians).to(self.device)
        # for idx in range(self.n_gaussians):
        #     gaussian = Independent(Normal(loc=mu[:, :, idx], scale=sigma), 1)
        #     result_per_gaussian = gaussian.log_prob(y)
        #     result[:, idx] = result_per_gaussian + self.pi.log()
        # return -torch.mean(torch.logsumexp(result, dim=1))
        gaussian = Independent(Normal(loc=mu,
                                      scale=sigma.reshape(-1, self.n_outputs, 1)
                                      .repeat(1, 1, mu.shape[2])), 0)
        result = gaussian.log_prob(y.reshape([-1, mu.shape[1], 1]).repeat(1, 1, self.n_gaussians))
        result = torch.sum(result, dim=1) + self.pi.log()
        return -torch.mean(torch.logsumexp(result, dim=1))

    def train(self, x_data, y_data, nepoch=1000, batch_size=100, save=False, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.001)

        x_variable = torch.from_numpy(x_data).type(self.dtype).to(self.device)
        y_variable = torch.from_numpy(y_data).type(self.dtype).to(self.device)

        batch_size = len(x) if batch_size is None else batch_size

        # print("Training mdn network on {} datapoints".format(len(x_data)))

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(x_data), batch_size)
                yield x_variable[indexes], y_variable[indexes]

        batch_gen_iter = batch_generator()

        lossHistory = []
        for epoch in range(nepoch):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()
            sigma, mu = self(x_batch)
            loss = self.mdn_loss_fn(mu, sigma, y_batch)
            loss.backward()
            optimizer.step()

            # if epoch == 0:
            #     print("Initial Loss is: {}".format(loss.item()))

            # elif epoch % 100 == 0 and verbose:
            #     if epoch != 0:
            #         print(" Iteration:", epoch, "Loss", loss.item())

            lossHistory.append(loss.item())
        # print("Training Finished, final loss is {}".format(loss.item()))
        return self, lossHistory

    def predict_mog(self, x):
        """
        Return the conditional mog at location x.
        :param network: an MDN network
        :param x: single input location
        :return: conditional mog at x
        """

        ntest, dim = x.shape  # test dimensionality and number of queries
        # gather mog parameters
        sigma, mu = self(torch.tensor(x).type(self.dtype).to(self.device))
        pi = self.pi.data.cpu().numpy()
        mean = mu.data.cpu().numpy()

        tril_idx = np.tril_indices(self.n_outputs, -1)
        diag_idx = np.diag_indices(self.n_outputs)
        mog = []
        for pt in range(ntest):
            Ss = []
            ms = []
            for i in range(self.n_gaussians):
                np_L = np.zeros([self.n_outputs, self.n_outputs])
                #np_L[tril_idx[0], tril_idx[1]] = L[pt, :, i].data.cpu().numpy()
                np_L[diag_idx[0], diag_idx[1]] = sigma.data.cpu().numpy()
                Ss.append(np.matmul(np_L, np_L.T))
                ms.append(mean[pt, :, i])
            # return mog
            mog.append(pdf.MoG(a=pi*np.ones(self.n_gaussians), ms=ms, Ss=Ss))
        return mog


class RDRFF(RDNN):
    def __init__(self,  ncomp=50, nfeat=500, inputd=1, outputd=1,
                 cosOnly=False, kernel="RBF", sigma=1, quasiRandom=True, dropout_p=0.25, gpu=True):
        super(MDNNTorch, self).__init__()
        self.n_gaussians = ncomp  # number of mixture components
        self.nfeat = nfeat  # number of features
        self.inputd = inputd  # dimensionality of the input
        self.n_outputs = outputd  # dimensionality of the output
        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        self.sigma = sigma * np.ones(self.inputd)
        self.dropout_p = dropout_p
        self.kernel = kernel
        #self.weights = weights  # weight function
        self.scaler = StandardScaler()

        self.gpu = gpu
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(1)
            self.device = "cuda:1"
        else:
            self.dtype = torch.float32
            self.device = "cpu"

        self.rff = RFF(self.nfeat, self.inputd, self.sigma,
                       self.cosOnly, self.quasiRandom, self.kernel, self.gpu)
        self.drop_layer = nn.Dropout(p=self.dropout_p)
        self.pi = torch.tensor([1. / self.n_gaussians]).to(self.device)
        self.sigma = nn.Linear(nfeat, self.n_outputs).to(self.device)
        self.mu = nn.Linear(nfeat, self.n_gaussians * self.n_outputs).to(self.device)

    def forward(self, x):
        z_h = self.rff.toFeatures(x)
        z_h = self.drop_layer(z_h)
        sigma = torch.exp(self.sigma(z_h)).reshape(-1, self.n_outputs)
        mu = self.mu(z_h).reshape(-1, self.n_outputs, self.n_gaussians)
        return sigma, mu

