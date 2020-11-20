import numpy as np
import numbers
from scipy.special import erfinv
import ghalton

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.array_ops import concat


class RFF:
    """
    Random Fourier Features, Vanilla or quasi-random using TensorFlow
    Note: make sure input space is normalised
    """
    def toFeatures(self, x):
        pass

    def __init__(self, m, d, sigma, cosOnly=False, quasiRandom=False,
                 kernel="RBF"):
        """
        :param m: number of features
        :param d: input dimension
        :param sigma: feature lengthscale (can be scalar of vector of size d)
        :param cosOnly: Using cos-only formulation of RFF (Default=False)
        :param quasiRandom: Using quasi-random sequence to generate RFF
                            (Default=True)
        :param kernel: Type of kernel to approximate: RBF, Laplace/Matern12,
                       Matern32, Matern52 (Default=RBF)
        RFF for RBF kernel.
        """
        self.m = int(m)
        self.nFeatures = self.m
        self.sigma = sigma
        self.d = int(d)
        self.coeff = None
        self.offset = None
        self.a = 1.0

        # Fix sigma
        if isinstance(sigma, numbers.Number):
            self.sigma = np.ones(d) * sigma
        elif isinstance(sigma, list):
            self.sigma = np.array(sigma)

        if kernel == "RBF":
            rffKernel = RFFKernelRBF()
        elif kernel == "Laplace" or kernel == "Matern12":
            rffKernel = RFFKernelMatern12()
        elif kernel == "Matern32":
            rffKernel = RFFKernelMatern32()
        elif kernel == "Matern52":
            rffKernel = RFFKernelMatern52()
        else:
            raise ValueError("Kernel {} is not recognised.".format(kernel))

        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        if self.cosOnly:  # cos only features
            self.coeff = constant_op.constant(
                self._drawCoeff(rffKernel, m), dtype=dtypes.float32)
            self.offset = constant_op.constant(
                2.0 * np.pi * np.random.rand(1, m), dtype=dtypes.float32)
            self.a = np.sqrt(1.0/float(self.m))
            self.toFeatures = self._toCosOnlyFeatures
        else:  # "cossin"
            assert m % 2 == 0 and "RFF: Number of fts must be multiple of 2."
            self.coeff = constant_op.constant(
                self._drawCoeff(rffKernel, int(m//2)), dtype=dtypes.float32)
            self.a = np.sqrt(1.0/float(self.m/2))
            self.toFeatures = self._toCosSinFeatures

    def _drawCoeff(self, rffKernel, m):
        if self.quasiRandom:
            perms = ghalton.EA_PERMS[:self.d]
            sequencer = ghalton.GeneralizedHalton(perms)
            points = np.array(sequencer.get(m+1))[1:]
            freqs = rffKernel.invCDF(points)
            return freqs / self.sigma.reshape(1, len(self.sigma))

        else:
            freqs = rffKernel.sampleFreqs((m, self.d))
            return freqs / self.sigma.reshape(1, len(self.sigma))

    def _toCosOnlyFeatures(self, x):
        #inner = x.dot(self.coeff.T)
        inner = math_ops.matmul(x, self.coeff, transpose_b=True)
        return self.a * math_ops.cos(math_ops.add(inner, self.offset))

    def _toCosSinFeatures(self, x):
        #inner = x.dot(self.coeff.T)
        inner = math_ops.matmul(x, self.coeff, transpose_b=True)
        return self.a * concat([math_ops.cos(inner), math_ops.sin(inner)], axis=1)


class RFFKernel:
    def sampleFreqs(self, shape):
        raise NotImplementedError

    def invCDF(self, x):
        raise NotImplementedError


class RFFKernelRBF(RFFKernel):
    def sampleFreqs(self, shape):
        return np.random.normal(0.0, 1.0, shape)

    def invCDF(self, x):
        return erfinv(2*x-1) * np.sqrt(2)


class RFFKernelMatern12(RFFKernel):
    def sampleFreqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(1/np.random.chisquare(1, shape))

    def invCDF(self, x):
        # This formula comes from the inv cdf of a standard cauchy
        # distribution (see Laplace RFF).
        return np.tan(np.pi*(x-0.5))


class RFFKernelMatern32(RFFKernel):
    def sampleFreqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(3/np.random.chisquare(3, shape))

    def invCDF(self, x):
        # From https://www.researchgate.net/profile/William_Shaw9/publication/247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/Sampling-Students-T-distribution-use-of-the-inverse-cumulative-distribution-function.pdf
        return (2*x - 1) / np.sqrt(2*x*(1-x))


class RFFKernelMatern52(RFFKernel):
    def sampleFreqs(self, shape):
        return np.random.normal(0, 1, shape) * \
                np.sqrt(5/np.random.chisquare(5, shape))

    def invCDF(self, x):
        # From https://www.researchgate.net/profile/William_Shaw9/publication/247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/Sampling-Students-T-distribution-use-of-the-inverse-cumulative-distribution-function.pdf
        alpha = 4*x*(1-x)
        p = 4 * np.cos(np.arccos(np.sqrt(alpha))/3) / np.sqrt(alpha)
        return np.sign(x-0.5)*np.sqrt(p-4)
