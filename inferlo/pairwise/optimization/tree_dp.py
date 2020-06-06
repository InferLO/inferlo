from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from networkx import is_tree

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
    model.make_connected()
    graph = model.get_graph()
    assert is_tree(graph), "Graph is not a tree."

    # mlz[v][i] - Max log(Z) for the subtree of v if X[v]=i.
    mlz = np.zeros((model.gr_size, model.al_size))

    # DFS to calculate max likelihood for subtrees.
    def dfs1(v, p):
        # v - current vertex.
        # p - its parent in the dfs tree.
        mlz[v, :] = model.field[v, :]

        for to in graph.neighbors(v):
            if to == p:
                continue
            dfs1(to, v)
            int_mx = model.get_interaction_matrix(v, to)
            for i in range(model.al_size):
                mlz[v, i] += np.max(int_mx[i, :] + mlz[to, :])

    dfs1(0, -1)

    # Back-tracking: determine best values at each vertex.
    result = np.zeros(model.gr_size, dtype=np.int32)
    result[0] = np.argmax(mlz[0, :])

    def dfs2(v, p):
        for to in graph.neighbors(v):
            if to == p:
                continue
            int_mx = model.get_interaction_matrix(v, to)
            result[to] = np.argmax(int_mx[result[v], :] + mlz[to, :])
            dfs2(to, v)

    dfs2(0, -1)

    return result
