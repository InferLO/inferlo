# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from inferlo.pairwise.inference.tree_dp import infer_tree_dp
from inferlo.utils.special_functions import softmax_1d, sample_categorical

if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel


def sample_tree_dp(model: PairWiseFiniteModel, num_samples: int):
    """Draws iid samples with dynamic programming on tree."""
    assert not model.get_dfs_result().had_cycles, "Graph has cycles."

    log_z = infer_tree_dp(model, subtree_mp=True).marg_prob
    log_z = log_z.astype(np.float64, copy=False)
    assert log_z.shape == (model.gr_size, model.al_size)

    dfs_edges = model.get_dfs_result().dfs_edges
    ints = model.get_interactions_for_edges(dfs_edges)
    num_samples = numba.types.int32(num_samples)

    return _sample_internal(log_z, dfs_edges, ints, num_samples)


@numba.njit("i4[:,:](f8[:,:],i4[:,:],f8[:,:,:],i4)")
def _sample_internal(log_z, dfs_edges, ints, num_samples):
    gr_size = log_z.shape[0]
    al_size = log_z.shape[1]
    dfs_edges_count = dfs_edges.shape[0]

    # Allocate array for the answer.
    # First index is variable number, second index is sample number.
    result = np.zeros((gr_size, num_samples), dtype=np.int32)

    # Sample values at first vertex.
    v0_probs = softmax_1d(log_z[0, :])
    result[0, :] = sample_categorical(v0_probs, num_samples)

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
            ch_probs = softmax_1d(log_z[ch, :] + ints[edge_idx, par_val, :])
            smpl[par_val, :] = sample_categorical(ch_probs, num_samples)

        # Now, pick appropriate samples.
        for sample_id in range(num_samples):
            par_val = result[par, sample_id]
            result[ch, sample_id] = smpl[par_val, sample_id]

    return result.T
