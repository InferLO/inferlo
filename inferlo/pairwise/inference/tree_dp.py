# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from scipy.special import logsumexp

from inferlo.base.inference_result import InferenceResult
from inferlo.utils.special_functions import logsumexp_1d

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


@numba.njit("void(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _dfs1(lz, lzc, dfs_edges, dfs_j):
    """DFS to calculate partition functions for subtrees."""
    dfs_edges_count = dfs_edges.shape[0]
    al_size = lz.shape[1]
    for edge_idx in range(dfs_edges_count - 1, -1, -1):
        vx, to = dfs_edges[edge_idx]
        for i in range(al_size):
            lzc[to][i] = logsumexp_1d(lz[to, :] + dfs_j[edge_idx, i, :])
        lz[vx, :] += lzc[to, :]


@numba.njit("void(f8[:,:],f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _dfs2(lz, lzc, lzr, dfs_edges, dfs_j):
    """DFS to calculate reverse partition functions for subtrees."""
    dfs_edges_count = dfs_edges.shape[0]
    al_size = lz.shape[1]
    for edge_idx in range(dfs_edges_count):
        p, v = dfs_edges[edge_idx]
        for j in range(al_size):
            lzr[v][j] = logsumexp_1d(
                lzr[p, :] + dfs_j[edge_idx, :, j] + lz[p, :] - lzc[v, :])


def infer_tree_dp(model: PairWiseFiniteModel,
                  subtree_mp=False) -> InferenceResult:
    """Inference using DP on tree.

    Performs dynamic programming on tree.

    Applicable only if the interaction graph is a tree or a forest. Otherwise
    throws exception.

    :param model: Model for which to perform inference.
    :param subtree_mp: If true, will return marginal probabilities for
        subtrees, i.e. for each node will return probability of it having
        different values if we leave only it and its subtree.
    :return: InferenceResult object.
    """
    assert not model.get_dfs_result().had_cycles, "Graph has cycles."

    dfs_edges = model.get_dfs_result().dfs_edges
    dfs_j = model.get_interactions_for_edges(dfs_edges)

    lz = model.field.astype(dtype=np.float64, copy=True)  # log(z)
    lzc = np.zeros_like(lz)  # log(zc)
    # Log(z_r). z_r  is partition function for all tree except subtree of given
    # vertex, when value of given vertex is fixed.
    lzr = np.zeros((model.gr_size, model.al_size))

    _dfs1(lz, lzc, dfs_edges, dfs_j)
    log_pf = logsumexp(lz[0, :])

    if subtree_mp:
        return InferenceResult(log_pf, lz)

    _dfs2(lz, lzc, lzr, dfs_edges, dfs_j)

    marg_proba = np.exp(lz + lzr - log_pf)
    return InferenceResult(log_pf, marg_proba)
