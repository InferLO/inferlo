# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from inferlo.graphs.path_decomposition import path_decomposition
from inferlo.base.inference_result import InferenceResult
from inferlo.pairwise.utils import (get_marginal_states,
                                    decode_all_states)

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def get_b(model, layer1, layer2):
    """Returns tensor describing interactions between two layers."""
    l1 = len(layer1)
    l2 = len(layer2)
    edges = []  # all edges from layer1 to layer2.
    for i in range(l1):
        v1 = layer1[i]
        for j in range(l2):
            v2 = layer2[j]
            if model.has_edge(v1, v2):
                edges.append((i, j, model.get_interaction_matrix(v1, v2)))

    all_states1 = decode_all_states(l1, model.al_size)
    all_states2 = decode_all_states(l2, model.al_size)
    st_num2 = model.al_size ** l2

    b = np.zeros(model.al_size ** (l1 + l2))

    r = np.array(range(model.al_size ** (l1 + l2)))
    r1 = r // st_num2
    r2 = r % st_num2

    for u, v, j in edges:
        j_flat = j.reshape(-1)
        idx1 = all_states1[r1, u]
        idx2 = all_states2[r2, v]
        b += j_flat[idx1 * model.al_size + idx2]

    return b.reshape(model.al_size ** l1, model.al_size ** l2)


# Path decomposition of Pairwise model.
# Contains layers of graph's path decomposition and also matrices A and B
# describing interactions within and between layers.
PwPathDecomposition = namedtuple('PwPathDecomposition', ['layers', 'a', 'b'])


def prepare_path_dp(model: PairWiseFiniteModel) -> PwPathDecomposition:
    """Prepares model for path decomposition dynamic programming.

    Builds path decomposition and calculates matrices A describing interactions
    with layers and matrices B describing interactions between them.

    :param model: Pairwise model.
    :return: PwPathDecomposition object with path decomposition and calculated
      matrices A and B.
    """
    layers = path_decomposition(model.get_graph())
    sum_layers_size = sum([len(layer) for layer in layers])
    assert sum_layers_size == model.gr_size, "Graph is not connected."
    max_layer_size = max([len(layer) for layer in layers])
    assert model.al_size ** max_layer_size <= 1e5, (
        "Algorithm won't handle this complexity.")

    a = [model.get_subgraph_factor_values(layer) for layer in layers]
    b = [get_b(model, layers[i], layers[i + 1])
         for i in range(len(layers) - 1)]

    return PwPathDecomposition(layers=layers, a=a, b=b)


def infer_path_dp(model: PairWiseFiniteModel) -> InferenceResult:
    """Inference using DP on path decomposition.

    Performs dynamic programming on the path decomposition of the underlying
    graph of the pairwise model.

    Time complexity is O(gr_size * al_size^(2*PW)), where PW is pathwidth of
    the graph.

    :param model: Potts model for which to perform inference.
    :return: ``InferenceResult`` object.
    """
    decomp = prepare_path_dp(model)
    layers_cnt = len(decomp.layers)

    # Forward dynamic programming.
    z = [None] * layers_cnt
    z[0] = decomp.a[0]
    for i in range(1, layers_cnt):
        z[i] = logsumexp(z[i - 1] + decomp.b[i - 1].T, axis=1) + decomp.a[i]

    # Backward dp.
    z_rev = [None] * layers_cnt
    z_rev[-1] = decomp.a[-1]
    for i in range(layers_cnt - 2, -1, -1):
        z_rev[i] = logsumexp(z_rev[i + 1] + decomp.b[i], axis=1) + decomp.a[i]

    # Partition function.
    log_pf = logsumexp(z[-1])

    # Restore marginal probabilities.
    log_marg_pf = np.zeros((model.gr_size, model.al_size))
    for layer_id in range(layers_cnt):
        layer_size = len(decomp.layers[layer_id])
        state_log_pf = z[layer_id] + z_rev[layer_id] - decomp.a[layer_id]

        marg_states = get_marginal_states(layer_size, model.al_size)
        layer_log_marg_pf = logsumexp(state_log_pf[marg_states], axis=2)
        log_marg_pf[decomp.layers[layer_id], :] = layer_log_marg_pf[:, :]

    return InferenceResult(log_pf, np.exp(log_marg_pf - log_pf))
