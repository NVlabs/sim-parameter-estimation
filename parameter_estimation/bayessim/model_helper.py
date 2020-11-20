import torch

from parameter_estimation.bayessim.models.mdn import MDNN, MDRFF, MDLSTM, MDRFFLSTM
from parameter_estimation.bayessim.models.mdn_torch import MDNNTorch, MDRFFTorch, RDNN, RDRFF, MDLSTMTorch
from parameter_estimation.bayessim.models.abc_elfi import ABCRejection, ABCSMC, BOLFI
from parameter_estimation.bayessim.models.bayes_sim import BayesSim
from parameter_estimation.bayessim.models.kernels import *
from parameter_estimation.bayessim.models.svgd import *
from parameter_estimation.bayessim.models.approximators import *
from parameter_estimation.bayessim.models.optimisers import BlackBoxOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)


def _get_mdn(n_components=2, nhidden=2, nunits=[24,24], output_dim=None, input_dim=None):
    model = MDNN(ncomp=n_components, outputd=output_dim, inputd=input_dim, nhidden=nhidden, nunits=nunits)
    return model


def _get_mdntorch(n_components=2, nunits=[24, 24], output_dim=None, input_dim=None, gpu=False):
    model = MDNNTorch(ncomp=n_components, outputd=output_dim, inputd=input_dim, hidden_layers=nunits, gpu=gpu).to(device)
    model.
    return model


def _get_rdnn(n_components=2, nunits=[50, 50, 50], output_dim=None, input_dim=None, gpu=False):
    model = RDNN(ncomp=n_components, outputd=output_dim, inputd=input_dim, hidden_layers=nunits, gpu=gpu).to(device)
    return model


def _get_rdrff(n_components=50, kernel="RBF", sigma=4., nfeat=154, quasiRandom=True, output_dim=None, input_dim=None):
    model = RDRFF(ncomp=n_components, outputd=output_dim, inputd=input_dim, nfeat=nfeat,
                  sigma=sigma, kernel=kernel, quasiRandom=quasiRandom)
    return model


def _get_mdrff(n_components=2, kernel="RBF", sigma=4., nfeat=154, quasiRandom=True, output_dim=None, input_dim=None):
    model = MDRFF(ncomp=n_components, outputd=output_dim, inputd=input_dim, nfeat=nfeat,
                  sigma=sigma, kernel=kernel, quasiRandom=quasiRandom)
    return model


def _get_mdrfftorch(n_components=2, kernel="RBF", sigma=4., nfeat=154,
                   quasiRandom=True, output_dim=None, input_dim=None, gpu=False):
    model = MDRFFTorch(ncomp=n_components, outputd=output_dim, inputd=input_dim, nfeat=nfeat,
                  sigma=sigma, kernel=kernel, quasiRandom=quasiRandom, gpu=gpu).to(device)
    return model

def _get_optimizer(algorithm="cma", generator=None, n_iter=500):
    model = BlackBoxOptimizer(algorithm=algorithm, n_iter=n_iter, generator=generator)

    model.set_prior_from_generator(generator)
    # Estimates appropriate scale for the SS
    params, stats = generator.gen(50)
    if model.scaler is not None:
        model.scaler.fit(stats)
    return model


def _get_abc(abc="ABCRejection", n_particles=None, generator=None):
    if abc == "ABCRejection":
        model = ABCRejection(n_particles=n_particles, generator=generator)
    elif abc == "ABCSMC":
        model = ABCSMC(n_particles=n_particles, generator=generator)
    elif abc == "BOLFI":
        model = BOLFI(n_particles=n_particles, generator=generator)

    model.set_prior_from_generator(generator)

    return model

def get_bayessim_model(model_name, **kwargs):
    return eval('_get_{}({})'.format(model_name, **kwargs))