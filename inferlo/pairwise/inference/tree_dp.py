from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from networkx import is_tree
from networkx.algorithms.traversal import depth_first_search
from scipy.special import logsumexp, softmax

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.utils.special_functions import logsumexp_1d

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


@numba.jit("void(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def dfs1(lz, lzc, dfs_edges, dfs_j):
    """DFS to calculate partition functions for subtrees."""
    al_size = lz.shape[1]
    for vx, to in dfs_edges[::-1]:
        for i in range(al_size):
            lzc[to][i] = logsumexp_1d(lz[to, :] + dfs_j[to, i, :])
        lz[vx, :] += lzc[to, :]


@numba.jit("void(f8[:,:],f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def dfs2(lz, lzc, lzr, dfs_edges, dfs_j):
    """DFS to calculate reverse partition functions for subtrees."""
    al_size = lz.shape[1]
    for p, v in dfs_edges:
        for j in range(al_size):
            lzr[v][j] = logsumexp_1d(
                lzr[p, :] + dfs_j[v, :, j] + lz[p, :] - lzc[v, :])


def infer_tree_dp(model: PairWiseFiniteModel,
                  subtree_mp=False) -> InferenceResult:
    """Performs inference for the Potts Model.

    Performs dynamic programming on tree. Applicable only if the underlying
        graph is a tree.

    :param model: Potts base for which to perform inference.
    :param subtree_mp: If true, will return marginal probabilities for
        subtrees, i.e. for each node will return probability of it having
        different values if we leave only it and its subtree.
    :return: InferenceResult object.
    """
    assert is_tree(model.get_graph()), "Graph is not a tree."

    # Prepare graph for quick DFS.
    dfs_edges = np.array(
        list(depth_first_search.dfs_edges(model.get_graph(), source=0)))
    dfs_j = np.zeros((model.gr_size, model.al_size, model.al_size))
    for vx, to in dfs_edges:
        dfs_j[to, :, :] = model.get_interaction_matrix(vx, to)
    dfs_edges = np.array(dfs_edges)

    lz = np.array(model.field)  # log(z)
    lzc = np.zeros((model.gr_size, model.al_size))  # log(zc)
    # Log(z_r). z_r  is partition function for all tree except subtree of given
    # vertex, when value of given vertex id fixed.
    lzr = np.zeros((model.gr_size, model.al_size))

    dfs1(lz, lzc, dfs_edges, dfs_j)
    log_pf = logsumexp(lz[0, :])

    if subtree_mp:
        return InferenceResult(log_pf, lz)

    dfs2(lz, lzc, lzr, dfs_edges, dfs_j)

    marg_proba = np.exp(lz + lzr - log_pf)
    return InferenceResult(log_pf, marg_proba)
