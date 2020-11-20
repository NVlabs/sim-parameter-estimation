import torch

from parameter_estimation.bayessim.models.mdn_torch import MDNNTorch, MDRFFTorch, RDNN, RDRFF, MDLSTMTorch

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)


def _get_mdntorch(n_components=2, nunits=[24, 24], output_dim=None, input_dim=None, gpu=False):
    model = MDNNTorch(ncomp=n_components, outputd=output_dim, inputd=input_dim, hidden_layers=nunits, gpu=gpu).to(device)
    return model

def _get_mdrfftorch(n_components=2, kernel="RBF", sigma=4., nfeat=154,
                   quasiRandom=True, output_dim=None, input_dim=None, gpu=False):
    model = MDRFFTorch(ncomp=n_components, outputd=output_dim, inputd=input_dim, nfeat=nfeat,
                  sigma=sigma, kernel=kernel, quasiRandom=quasiRandom, gpu=gpu).to(device)
    return model

def get_bayessim_model(model_name, **kwargs):
    return eval('_get_{}({})'.format(model_name, **kwargs))