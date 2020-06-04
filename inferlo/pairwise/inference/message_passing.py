from __future__ import annotations

import time
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


@numba.jit("void(f8[:,:],i4[:,:],i4[:,:],f8[:,:,:],f8[:,:])")
def _update(m, dir_edges, v_to_e, intrn, field):
    edge_num = len(dir_edges)
    al_size = field.shape[1]
    new_m = np.zeros_like(m)
    for edge_id in range(edge_num):
        t, s = dir_edges[edge_id]
        for xs in range(al_size):
            terms = intrn[edge_id, xs, :] + field[t, :]
            for next_edge_id in range(v_to_e[t][0], v_to_e[t][1]):
                # dir_edges[next_edge_id] is (u, t).
                if dir_edges[next_edge_id, 0] != s:
                    terms += m[next_edge_id, :]
            new_m[edge_id, xs] = logsumexp_1d(terms)
    m[:, :] = new_m


def infer_message_passing(model: 'PairWiseFiniteModel',
                          iter_num=None) -> InferenceResult:
    """Performs inference for the Potts Model with message passing algorithm.

    For acyclic graph returns exact partition function and marginal
        probabilities. For graph with loops may return good approximation to
        the true marginal probabilities, but partition function will be a
        useless number.

    :param model: Potts base.
    :param max_iter: Number of iterations to perform. If not set, will be set
        to graph's diameter, which is guaranteed to give exact result for tree.
    :return: InferenceResult object.
    """
    t0 = time.time()

    if iter_num is None:
        iter_num = networkx.diameter(model.get_graph())

    dir_edges, v_to_e, intrn = _precalc(model)

    m = np.zeros((len(dir_edges), model.al_size))
    for i in range(iter_num):
        _update(m, dir_edges, v_to_e, intrn, model.field)

    # Restore partition function for fixed values in nodes.
    log_marg_pf = np.array(model.field)
    for edge_id in range(len(dir_edges)):
        log_marg_pf[dir_edges[edge_id][1], :] += m[edge_id]

    log_pf = scipy.special.logsumexp(log_marg_pf, axis=-1)
    marg_prob = scipy.special.softmax(log_marg_pf, axis=-1)
    marg_prob /= np.sum(marg_prob, axis=-1).reshape(-1, 1)
    return InferenceResult(np.min(log_pf), marg_prob)
