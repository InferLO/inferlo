from __future__ import annotations

from typing import TYPE_CHECKING

import networkx
import numba
import numpy as np
import scipy.special

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.utils.special_functions import logsumexp_1d

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


def _precalc(model: PairWiseFiniteModel):
    # Build list of directed edges.
    dir_edges = []
    for t, s in model.edges:
        dir_edges += [(t, s), (s, t)]
    # Enable continuous lookup by the second vertex.
    dir_edges.sort(key=lambda x: x[1])

    # Compact representation of interactions (transposed).
    intrn = np.array(
        [model.get_interaction_matrix(j, i) for (i, j) in dir_edges])

    # For each vertex, contains interval of edge ids for edges
    # starting with this vertex.
    dir_edges = np.array(dir_edges, dtype=np.int32)
    v_to_e = np.zeros((model.gr_size, 2), dtype=np.int32)
    for i in range(1, len(dir_edges)):
        if dir_edges[i - 1][1] != dir_edges[i][1]:
            v_to_e[dir_edges[i - 1][1]][1] = i
            v_to_e[dir_edges[i][1]][0] = i
    v_to_e[-1, -1] = len(dir_edges)

    return dir_edges, v_to_e, intrn


@numba.jit("void(f8[:,:],i4[:,:],i4[:,:],f8[:,:,:],f8[:,:],i8)")
def _message_passing(mu, dir_edges, v_to_e, intrn, field, max_iter):
    """Runs message passing algorihm.

    Terminates if `max_iter` steps were made or if converged.
    Returns the result in `mu`.
    """
    edge_num = len(dir_edges)
    al_size = field.shape[1]
    new_mu = np.zeros_like(mu)
    for _ in range(max_iter):
        # This loop is one iteration of message passing.
        for edge_id in range(edge_num):
            t, s = dir_edges[edge_id]
            for xs in range(al_size):
                terms = intrn[edge_id, xs, :] + field[t, :]
                for next_edge_id in range(v_to_e[t][0], v_to_e[t][1]):
                    # dir_edges[next_edge_id] is (u, t).
                    if dir_edges[next_edge_id, 0] != s:
                        terms += mu[next_edge_id, :]
                new_mu[edge_id, xs] = logsumexp_1d(terms)

        # If converged, return. Both mu and new_mu contain correct result.
        # if np.max(np.abs(new_mu - mu)) < 1e-9:
        #    return
        # If not converged, repeat again with new mu.
        mu[:, :] = new_mu


def infer_message_passing(model: 'PairWiseFiniteModel',
                          max_iter=None) -> InferenceResult:
    """Inference with Message Passing.

    For acyclic graph returns exact partition function and marginal
        probabilities. For graph with loops may return good approximation to
        the true marginal probabilities, but partition function will be a
        useless number.
    This is an iterative algorithm which terminates when it converged or when
        `max_iter` iterations were made.

    :param model: Potts base.
    :param max_iter: How many iterations without convergence should happen for
        algorithm to terminate. Defaults to maximal diameter of connected
        component.
    :return: InferenceResult object.
    """
    model.make_connected()
    if max_iter is None:
        max_iter = networkx.diameter(model.get_graph())

    dir_edges, v_to_e, intrn = _precalc(model)

    mu = np.zeros((len(dir_edges), model.al_size))
    _message_passing(mu, dir_edges, v_to_e, intrn, model.field, max_iter)

    # Restore partition function for fixed values in nodes.
    log_marg_pf = np.array(model.field)
    for edge_id in range(len(dir_edges)):
        log_marg_pf[dir_edges[edge_id][1], :] += mu[edge_id]

    log_pf = scipy.special.logsumexp(log_marg_pf, axis=-1)
    marg_prob = scipy.special.softmax(log_marg_pf, axis=-1)
    marg_prob /= np.sum(marg_prob, axis=-1).reshape(-1, 1)
    return InferenceResult(np.min(log_pf), marg_prob)
