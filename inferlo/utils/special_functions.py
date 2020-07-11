# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
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


@numba.njit("i4[:](f8[:],i4)")
def sample_categorical(probs, num_samples):
    """Samples from categorical distribution.

    Equivalent to ``np.random.choice(len(probs), size=num_samples, p=probs)``,
    slightly less efficient, but works with Numba.

    Probabilities must add up to 1, but it won't check it.
    """
    al_size = probs.shape[0]
    cum_probs = np.cumsum(probs)
    rnd_numbers = np.random.random(num_samples)

    ans = np.zeros(num_samples, dtype=np.int32)
    for i in range(1, al_size):
        mask = rnd_numbers >= cum_probs[i - 1]
        ans[mask] = i
    return ans
