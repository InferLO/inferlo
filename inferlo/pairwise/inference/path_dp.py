from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from inferlo.graphs.path_decomposition import path_decomposition
from inferlo.pairwise.inference_result import InferenceResult
from inferlo.pairwise.utils import (get_marginal_states,
                                    decode_all_states)

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def get_a(model, layer: np.ndarray):
    """Returns tensor describing interactions within one layer."""
    layer_size = len(layer)
    edges = []
    for i in range(layer_size):
        v1 = layer[i]
        for j in range(i + 1, layer_size):
            v2 = layer[j]
            if model.has_edge(v1, v2):
                edges.append((i, j, model.get_interaction_matrix(v1, v2)))

    all_states = decode_all_states(layer_size, model.al_size)
    a = np.zeros(model.al_size ** layer_size)
    for u in range(layer_size):
        a += model.field[layer[u]][all_states[:, u]]
    for u, v, j in edges:
        a += j[all_states[:, u], all_states[:, v]]
    return a


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


def infer_path_dp(model: PairWiseFiniteModel) -> InferenceResult:
    """Inference using DP on path decomposition.

    Performs dynamic programming on the path decomposition of the underlying
    graph of the pairwise model.

    Time complexity is O(gr_size * al_size^(2*PW)), where PW is pathwidth of
    the graph.

    :param model: Potts base for which to perform inference.
    :return: ``InferenceResult`` object.
    """
    layers = path_decomposition(model.get_graph())
    sum_layers_size = sum([len(layer) for layer in layers])
    assert sum_layers_size == model.gr_size, "Graph is not connected."
    max_layer_size = max([len(layer) for layer in layers])
    assert model.al_size ** max_layer_size <= 1e5, (
        "Algorithm won't handle this complexity.")

    a = [get_a(model, layer) for layer in layers]
    b = [get_b(model, layers[i], layers[i + 1])
         for i in range(len(layers) - 1)]
    layers_cnt = len(layers)

    # Forward dynamic programming.
    z = [None] * layers_cnt
    z[0] = a[0]
    for i in range(1, layers_cnt):
        z[i] = logsumexp(z[i - 1] + b[i - 1].T, axis=1) + a[i]

    # Backward dp.
    z_rev = [None] * layers_cnt
    z_rev[-1] = a[-1]
    for i in range(layers_cnt - 2, -1, -1):
        z_rev[i] = logsumexp(z_rev[i + 1] + b[i], axis=1) + a[i]

    # Partition function.
    log_pf = logsumexp(z[-1])

    # Restore marginal probabilities.
    log_marg_pf = np.zeros((model.gr_size, model.al_size))
    for layer_id in range(layers_cnt):
        layer_size = len(layers[layer_id])
        state_log_pf = z[layer_id] + z_rev[layer_id] - a[layer_id]

        marg_states = get_marginal_states(layer_size, model.al_size)
        layer_log_marg_pf = logsumexp(state_log_pf[marg_states], axis=2)
        log_marg_pf[layers[layer_id], :] = layer_log_marg_pf[:, :]

    return InferenceResult(log_pf, np.exp(log_marg_pf - log_pf))
