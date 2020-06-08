from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from scipy.special import softmax, logsumexp

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.pairwise.utils import decode_state

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


@numba.njit("void(i4[:],i4,i4)")
def _next_state(state, gr_size, al_size):
    """Go to the next state."""
    # States are like numbers in positional system, with least significant
    # "digit" on the right.
    pos = gr_size - 1
    state[pos] += 1
    while state[pos] == al_size and pos >= 0:
        state[pos] = 0
        pos -= 1
        state[pos] += 1


@numba.njit("f8[:](f8[:,:],i4[:,:],f8[:,:,:])")
def _compute_all_probs_internal(field, edges, inter):
    gr_size = field.shape[0]
    al_size = field.shape[1]
    states_count = al_size ** gr_size
    edge_count = edges.shape[0]
    probs = np.zeros(states_count, dtype=np.float64)
    state = np.zeros(gr_size, dtype=np.int32)

    for state_id in range(states_count):
        state_prob = 0
        for u in range(gr_size):
            state_prob += field[u, state[u]]
        for edge_id in range(edge_count):
            u, v = edges[edge_id]
            state_prob += inter[edge_id][state[u]][state[v]]
        probs[state_id] = state_prob
        _next_state(state, gr_size, al_size)
    return np.exp(probs)


def _compute_all_probs(model: PairWiseFiniteModel):
    """For all possible states finds their probabilities (not normed)."""
    assert model.al_size ** model.gr_size <= 2e7, "Too much states."
    return _compute_all_probs_internal(model.field,
                                       model.get_edges_array(),
                                       model.get_all_interactions())


@numba.njit("f8[:,:](f8[:],i4,i4)")
def _compute_marg_probs(all_probs, gr_size, al_size):
    """Computes marginal probabilities given probabilities of all states.

    For every variable and value sums over all states in which this variables
    has this value. Result should be divided by partition function to get
    marginal probabilities.
    """
    marg = np.zeros((gr_size, al_size))
    state = np.zeros(gr_size, dtype=np.int32)
    for state_id in range(len(all_probs)):
        for i in range(gr_size):
            marg[i][state[i]] += all_probs[state_id]
        _next_state(state, gr_size, al_size)
    return marg


def infer_bruteforce(model: PairWiseFiniteModel) -> InferenceResult:
    """Inference by summing up all states.

    Uses definition to perform inference, so complexity is
    ``O(al_size ** gr_size)``.

    Result is exact.

    :param model: Model for which to perform inference.
    :return: ``InferenceResult`` object.
    """
    if len(model.edges) == 0:
        # Fully isolated model.
        log_pf = sum(logsumexp(model.field[i, :])
                     for i in range(model.gr_size))
        mp = softmax(model.field, axis=1)
        return InferenceResult(log_pf, mp)

    all_probs = _compute_all_probs(model)
    pf = np.sum(all_probs)
    marg_probs = _compute_marg_probs(all_probs, model.gr_size, model.al_size)
    return InferenceResult(np.log(pf), marg_probs / pf)


def max_lh_bruteforce(model: PairWiseFiniteModel) -> np.array:
    """Finds max likelihood for pairwise model by checking all states."""
    if len(model.edges) == 0:
        # Fully isolated model.
        return np.argmax(model.field, axis=1)

    state_id = np.argmax(_compute_all_probs(model))
    return decode_state(state_id, model.gr_size, model.al_size)
