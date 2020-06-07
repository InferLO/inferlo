from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from scipy.special import logsumexp

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.utils.special_functions import logsumexp_1d

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def _prepare_interactions(dfs_edges: np.ndarray,
                          model: PairWiseFiniteModel) -> np.ndarray:
    """Prepares interactions for quick DFS.

    Returns np.array of shape (gr_size, al_size, al_size), where at index
    to we have interaction matrix for edge (v, to) in DFS traversal. Such edge
    is unique, unless to is root.

    It's possible that some edges in DFS traversal don't actually exist, but
    were added to make the graph connected. In this case e leave zero
    interaction matrix for them, which is equivalent to not having an edge.

    :param dfs_edges: DFS traversal of the graph.
    :param model: Pairwise model.
    """
    dfs_j = np.zeros((model.gr_size, model.al_size, model.al_size))
    for vx, to in dfs_edges:
        if model.has_edge(vx, to):
            dfs_j[to, :, :] = model.get_interaction_matrix(vx, to)
    return dfs_j


@numba.njit("void(f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _dfs1(lz, lzc, dfs_edges, dfs_j):
    """DFS to calculate partition functions for subtrees."""
    al_size = lz.shape[1]
    for vx, to in dfs_edges[::-1]:
        for i in range(al_size):
            lzc[to][i] = logsumexp_1d(lz[to, :] + dfs_j[to, i, :])
        lz[vx, :] += lzc[to, :]


@numba.njit("void(f8[:,:],f8[:,:],f8[:,:],i4[:,:],f8[:,:,:])")
def _dfs2(lz, lzc, lzr, dfs_edges, dfs_j):
    """DFS to calculate reverse partition functions for subtrees."""
    al_size = lz.shape[1]
    for p, v in dfs_edges:
        for j in range(al_size):
            lzr[v][j] = logsumexp_1d(
                lzr[p, :] + dfs_j[v, :, j] + lz[p, :] - lzc[v, :])


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
    dfs_j = _prepare_interactions(dfs_edges, model)

    lz = np.array(model.field)  # log(z)
    lzc = np.zeros((model.gr_size, model.al_size))  # log(zc)
    # Log(z_r). z_r  is partition function for all tree except subtree of given
    # vertex, when value of given vertex id fixed.
    lzr = np.zeros((model.gr_size, model.al_size))

    _dfs1(lz, lzc, dfs_edges, dfs_j)
    log_pf = logsumexp(lz[0, :])

    if subtree_mp:
        return InferenceResult(log_pf, lz)

    _dfs2(lz, lzc, lzr, dfs_edges, dfs_j)

    marg_proba = np.exp(lz + lzr - log_pf)
    return InferenceResult(log_pf, marg_proba)
