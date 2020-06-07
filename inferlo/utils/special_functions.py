import numba
import numpy as np


@numba.jit
def entropy(x):
    """Entropy is defined as -x*log(x) if x>0 and 0 if x=0."""
    return -np.sum(x * np.log(x + (x == 0)))


@numba.jit("f8[:](f8[:])")
def softmax_1d(x: np.ndarray):
    """Softmax for 1d array."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


@numba.njit("f8(f8[:])")
def logsumexp_1d(x: np.ndarray):
    """LogSumExp function on a vector."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))
