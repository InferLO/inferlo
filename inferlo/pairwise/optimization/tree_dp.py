from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from networkx import is_tree

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def max_likelihood_tree_dp(model: PairWiseFiniteModel):
    """Max Likelihood for  the Potts Model.

    Performs dynamic programming on tree.
    """
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
        print(v, mlz[v, :])

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
