from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import softmax, logsumexp

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.pairwise.utils import decode_state

if TYPE_CHECKING:
    from inferlo.base import PottsModel


def calculate_full_probs(model: PottsModel):
    states_count = model.al_size ** model.gr_size
    assert states_count <= 2e7, "Too much states."
    full_probs = np.zeros(states_count)

    edges, interactions = model.get_compact_interactions()
    edge_count = len(model.edges)
    for state_id in range(states_count):
        state = decode_state(state_id, model.gr_size, model.al_size)
        s1 = 0
        for edge_id in range(edge_count):
            u, v = edges[edge_id]
            s1 += interactions[edge_id][state[u]][state[v]]
        s2 = sum(model.field[u, state[u]] for u in range(model.gr_size))
        full_probs[state_id] = s1 + s2
    return np.exp(full_probs)


def infer_bruteforce(model: PottsModel) -> InferenceResult:
    """Performs inference for the Potts Model.

    Uses Potts Model definition to perform inference, so complexity is
        `al_size^gr_size`.

    :param model: Potts base for which to perform inference.
    :return: InferenceResult object.
    """
    if len(model.edges) == 0:
        log_pf = sum(logsumexp(model.field[i, :])
                     for i in range(model.gr_size))
        mp = softmax(model.field, axis=1)
        return InferenceResult(log_pf, mp)

    full_probs = calculate_full_probs(model)
    marg = np.zeros((model.gr_size, model.al_size))
    for state_id in range(len(full_probs)):
        state = decode_state(state_id, model.gr_size, model.al_size)
        for i in range(model.gr_size):
            marg[i][state[i]] += full_probs[state_id]
    pf = np.sum(full_probs)
    return InferenceResult(np.log(pf), marg / pf)


def sample_potts_bruteforce(model: PottsModel):
    return None


def max_likelihood_potts_bruteforce(model: PottsModel):
    full_probs = calculate_full_probs(model)
    state_id = np.argmax(full_probs)
    return decode_state(state_id, model.gr_size, model.al_size)
