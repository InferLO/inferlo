from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import softmax

from inferlo.pairwise.inference.tree_dp import infer_tree_dp

if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel


def sample_tree_dp(model: PairWiseFiniteModel, num_samples: int):
    """Draws iid samples with dynamic programming on tree."""
    assert not model.get_dfs_result().had_cycles, "Graph has cycles."

    log_z = infer_tree_dp(model, subtree_mp=True).marg_prob
    assert log_z.shape == (model.gr_size, model.al_size)

    dfs_edges = model.get_dfs_result().dfs_edges
    ints = model.get_interactions_for_edges(dfs_edges)

    return _sample_internal(log_z, dfs_edges, ints, num_samples)


def _sample_internal(log_z, dfs_edges, ints, num_samples):
    gr_size = log_z.shape[0]
    al_size = log_z.shape[1]
    dfs_edges_count = dfs_edges.shape[0]

    # Allocate array for the answer.
    # First index is variable number, second index is sample number.
    result = np.zeros((gr_size, num_samples), dtype=np.int32)

    # Sample values at first vertex.
    v0_probs = softmax(log_z[0, :])
    result[0, :] = np.random.choice(al_size, size=num_samples, p=v0_probs)

    # Allocate a buffer for samples.
    smpl = np.empty((al_size, num_samples), dtype=np.int32)

    # Traverse graph in DFS order and for each edges in DFS tree sample values
    # in children node given values in parent node.
    for edge_idx in range(dfs_edges_count):
        par, ch = dfs_edges[edge_idx]  # parent, child

        # For every parent's value calculate distribution at child given that
        # parent node has that value. Then generate `num_samples` independent
        # samples from that distribution.
        # We generate more samples then needed.
        for par_val in range(al_size):
            ch_probs = softmax(log_z[ch, :] + ints[edge_idx, par_val, :])
            smpl[par_val, :] = np.random.choice(al_size, size=num_samples,
                                                p=ch_probs)

        # Now, pick appropriate samples.
        for sample_id in range(num_samples):
            par_val = result[par, sample_id]
            result[ch, sample_id] = smpl[par_val, sample_id]

    return result.T
