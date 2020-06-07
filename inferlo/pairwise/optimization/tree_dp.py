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
    # assert not model.get_dfs_result().had_cycles, "Graph has cycles."
    model.make_connected()
    graph = model.get_graph()
    assert is_tree(graph), "Graph is not a tree."

    # dfs_edges = model.get_dfs_result().dfs_edges
    dfs_edges = list(depth_first_search.dfs_edges(model.get_graph()))

    # mlz[v][i] - Max log(Z) for the subtree of v if X[v]=i.
    mlz = np.array(model.field)

    for v, to in dfs_edges[::-1]:
        int_mx = model.get_interaction_matrix(v, to)
        for i in range(model.al_size):
            mlz[v, i] += np.max(int_mx[i, :] + mlz[to, :])

    # Back-tracking: determine best values at each vertex.
    result = np.zeros(model.gr_size, dtype=np.int32)
    result[0] = np.argmax(mlz[0, :])

    for v, to in dfs_edges:
        int_mx = model.get_interaction_matrix(v, to)
        result[to] = np.argmax(int_mx[result[v], :] + mlz[to, :])

    return result
