# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


@numba.njit("i4[:](f8[:,:],i4[:,:],f8[:,:,:])")
def _max_likelihood_internal(field, dfs_edges, ints):
    """Max likelihood for pairwise model (internal).

    :param field: Field.
    :param dfs_edges: Edges in DFS tree.
    :param ints: Interactions corresponding to DFS edges.
    :return: Most likely state.
    """
    gr_size = field.shape[0]
    al_size = field.shape[1]
    dfs_edges_count = dfs_edges.shape[0]

    # mlz[v][i] - Max log(Z) for the subtree of v if X[v]=i.
    mlz = np.copy(field)

    # Forward DFS: compute max partition function for subtrees.
    for edge_idx in range(dfs_edges_count - 1, -1, -1):
        v, to = dfs_edges[edge_idx]
        for i in range(al_size):
            mlz[v, i] += np.max(ints[edge_idx, i, :] + mlz[to, :])

    # Backward DFS: determine best values at each vertex.
    result = np.zeros(gr_size, dtype=np.int32)
    result[0] = np.argmax(mlz[0, :])
    for edge_idx in range(dfs_edges_count):
        v, to = dfs_edges[edge_idx]
        result[to] = np.argmax(ints[edge_idx, result[v], :] + mlz[to, :])

    return result


def max_likelihood_tree_dp(model: PairWiseFiniteModel):
    """Max Likelihood for the pairwise model.

    Performs dynamic programming on tree.

    Applicable only if the interaction graph is a tree or a forest. Otherwise
    throws exception.

    :param model: Model for which to find most likely state.
    :return: Most likely state. np.array of ints.
    """
    assert not model.get_dfs_result().had_cycles, "Graph has cycles."

    field = model.field.astype(np.float64, copy=False)
    dfs_edges = model.get_dfs_result().dfs_edges
    ints = model.get_interactions_for_edges(dfs_edges)

    return _max_likelihood_internal(field, dfs_edges, ints)
