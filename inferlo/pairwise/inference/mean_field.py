# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numba
import numpy as np

from inferlo.utils import special_functions
from inferlo.base.inference_result import InferenceResult

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel

# To prevent numba from complaining that we call np.dot on non-continuous
# arrays (which are in fact continuous).

os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = '1'


@numba.jit("f8(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _logpf_lower_bound(mu, field, edges, inter):
    ans = np.sum(field * mu) + special_functions.entropy(mu)
    for i in range(len(edges)):
        v1, v2 = edges[i, ...]
        ans += np.dot(mu[v1, ...], np.dot(inter[i, ...], mu[v2, ...]))
    return ans


@numba.jit("void(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _naive_mean_field_iteration(mu, field, edges, inter):
    f = np.copy(field)
    for i in range(len(edges)):
        v1, v2 = edges[i, :]
        f[v1, :] += np.dot(inter[i, ...], mu[v2, :])
        f[v2, :] += np.dot(mu[v1, :], inter[i, :])
    for i in range(f.shape[0]):
        mu[i, :] = special_functions.softmax_1d(f[i, :])


@numba.jit("void(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:],i8,i8,i8)")
def _infer_mean_field_internal(best_mu,
                               field,
                               edges,
                               inter,
                               iters_wait,
                               max_iter,
                               num_attempts):
    mu = np.zeros_like(best_mu)
    best_bound = -np.inf
    for _ in range(num_attempts):
        mu[:, :] = np.random.random(size=mu.shape)
        mu /= np.sum(mu, axis=1).reshape(-1, 1)
        steps_without_improvement = 0
        for _ in range(max_iter):
            _naive_mean_field_iteration(mu, field, edges, inter)
            new_bound = _logpf_lower_bound(mu, field, edges, inter)
            if new_bound > best_bound:
                best_bound = new_bound
                best_mu[:, :] = mu
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= iters_wait:
                    break


def infer_mean_field(model: PairWiseFiniteModel,
                     iters_wait=10,
                     max_iter=100,
                     num_attempts=1) -> InferenceResult:
    """Inference with Naive Mean Field.

    Uses Naive Mean Field Algorithm. Time complexity is proportional to
    `gr_size * num_iters`, where `gr_size` is number of variables, `num_iters`
    is number of iterations.

    :param model: Potts model for which to perform inference.
    :param iters_wait: Algorithm stops if result doesn't improve for that long.
    :param max_iter: Maximal number of iterations.
    :param num_attempts: How many times try optimization from new random point.
    :return: InferenceResult object. `log_pf` is guaranteed to be lower bound
        for logarithm of the true partition function. `marg_prob` is
        approximation for marginal probabilities.
    """
    field = model.field
    edges = model.get_edges_array()
    interactions = model.get_all_interactions()

    best_mu = np.zeros_like(model.field)
    _infer_mean_field_internal(best_mu, field, edges, interactions,
                               iters_wait, max_iter, num_attempts)
    best_bound = _logpf_lower_bound(best_mu, field, edges, interactions)
    return InferenceResult(best_bound, best_mu)
