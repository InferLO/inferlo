from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from networkx import is_tree, depth_first_search
from scipy.special import softmax

from inferlo.pairwise.inference.tree_dp import infer_tree_dp

if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel


def sample_tree_dp(model: PairWiseFiniteModel, num_samples: int):
    graph = model.get_graph()
    assert is_tree(graph), "Graph is not a tree."

    # Allocate array for the answer.
    # First index is variable number, second index is sample number.
    result = np.zeros((model.num_variables, num_samples), dtype=np.int32)

    log_z = infer_tree_dp(model, subtree_mp=True).marg_prob
    assert log_z.shape == (model.gr_size, model.al_size)

    # Sample values at first vertex. They depend on the field only.
    v0_probs = softmax(log_z[0, :])
    result[0, :] = np.random.choice(model.al_size, size=num_samples,
                                    p=v0_probs)

    # Traverse graph in DFS order and for each edges in DFS tree sample values
    # in children node given values in parent node.
    dfs_edges = np.array(
        list(depth_first_search.dfs_edges(model.get_graph(), source=0)))
    for par, ch in dfs_edges:  # parent, child
        # For every parent's value calculate distribution at child given that
        # parent node has that value.
        J = model.get_interaction_matrix(par, ch)
        ch_probs = np.zeros((model.al_size, model.al_size))
        for par_val in range(model.al_size):
            ch_probs[par_val, :] = softmax(log_z[ch, :] + J[par_val, :])

        # Now, sample.
        for sample_id in range(num_samples):
            par_val = result[par, sample_id]
            p = ch_probs[par_val, :]
            result[ch, sample_id] = np.random.choice(model.al_size, p=p)

    return result.T
