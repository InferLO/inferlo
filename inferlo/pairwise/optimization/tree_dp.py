from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from networkx import is_tree
from networkx.algorithms.traversal import depth_first_search

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def max_likelihood_tree_dp(model: PairWiseFiniteModel):
    """Max Likelihood for the pairwise model.

    Performs dynamic programming on tree.

    Applicable only if the interaction graph is a tree or a forest. Otherwise
    throws exception.

    :param model: Model for which to find most likely state.
    :return: Most likely state. np.array of ints.
    """
    assert not model.get_dfs_result().had_cycles, "Graph has cycles."

    dfs_edges = model.get_dfs_result().dfs_edges
    dfs_edges_count = dfs_edges.shape[0]
    ints = model.get_interactions_for_edges(dfs_edges)

    # mlz[v][i] - Max log(Z) for the subtree of v if X[v]=i.
    mlz = np.array(model.field)

    for edge_id in range(dfs_edges_count-1, -1, -1):
        v, to = dfs_edges[edge_id]
        for i in range(model.al_size):
            mlz[v, i] += np.max(ints[edge_id, i, :] + mlz[to, :])

    # Back-tracking: determine best values at each vertex.
    result = np.zeros(model.gr_size, dtype=np.int32)
    result[0] = np.argmax(mlz[0, :])
    for edge_id in range(dfs_edges_count):
        v, to = dfs_edges[edge_id]
        result[to] = np.argmax(ints[edge_id, result[v], :] + mlz[to, :])

    return result
