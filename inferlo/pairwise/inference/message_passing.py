# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx
import numba
import numpy as np
import scipy.special

from inferlo.base.inference_result import InferenceResult
from inferlo.utils.special_functions import logsumexp_1d

if TYPE_CHECKING:
    from inferlo.pairwise import PairWiseFiniteModel


@numba.njit("i4[:](i4[:,:])")
def _build_edge_lookup(sorted_edges):
    """Builds edge lookup.

    Edges must be sorted by end vertex (ascending).

    Returns array indicating where is the first edge ending on given vertex.
    In other words, all edges ending on vertex are
    ``sorted_edges[edge_lookup[v]:edge_lookup[v+1], :]``.
    """
    gr_size = sorted_edges[-1, 1] + 1
    # For each vertex, contains interval of edge ids for edges
    # starting with this vertex.
    ans = np.zeros(gr_size + 1, dtype=np.int32)
    for i in range(1, len(sorted_edges)):
        if sorted_edges[i - 1][1] != sorted_edges[i][1]:
            ans[sorted_edges[i][1]] = i
    ans[-1] = len(sorted_edges)
    return ans


@numba.njit("f8[:,:](i4[:,:],f8[:,:,:],f8[:,:],i8)")
def _message_passing(dir_edges, intrn, field, max_iter):
    """Runs message passing algorithm.

    Terminates if `max_iter` steps were made or if converged.

    :param dir_edges: Directed edges. Even though graph is undirected, every
      edge (u,v) must appear in this list twice as (u,v) and (v,u).
    :param intrn: Interaction matrices corresponding to dir_edges.
    :param field: Field.
    :param max_iter: Maximal number of iterations. Can perform less if
      converges earlier.
    :return: Logarithm of mu describing marginal probabilities associated
      with directed edges. See formula (2.5) in [1], except here we work with
      logarithms.
    """
    edge_num = len(dir_edges)
    al_size = field.shape[1]

    # Initialize initial guess with zeros. Since those are logarithms, it
    # corresponds to probabilities all proportional to 1, i.e. equal.
    # `lmu` is logarithm of mu, defined in formula (2.5) in [1].
    lmu = np.zeros((edge_num, al_size), dtype=np.float64)
    new_lmu = np.zeros_like(lmu)

    edge_lookup = _build_edge_lookup(dir_edges)

    for _ in range(max_iter):
        # This loop is one iteration of message passing.
        # It implements formulas (2.8a) and (2.8b) from ref. [1].
        for edge_id in range(edge_num):
            t, s = dir_edges[edge_id]
            for xs in range(al_size):
                terms = intrn[edge_id, :, xs] + field[t, :]
                for next_edge_id in range(edge_lookup[t], edge_lookup[t + 1]):
                    # dir_edges[next_edge_id] is (u, t).
                    # We should skip edge (s, t).
                    if dir_edges[next_edge_id, 0] != s:
                        terms += lmu[next_edge_id, :]
                new_lmu[edge_id, xs] = logsumexp_1d(terms)

        # If converged, return. Both mu and new_mu contain correct result.
        if np.max(np.abs(new_lmu - lmu)) < 1e-9:
            break
        # If not converged, repeat again with new mu as old mu.
        lmu[:, :] = new_lmu

    return lmu


def infer_message_passing(model: PairWiseFiniteModel,
                          max_iter=None) -> InferenceResult:
    """Inference with Message Passing.

    For acyclic graph returns exact partition function and marginal
        probabilities. For graph with loops may return good approximation to
        the true marginal probabilities, but partition function will be a
        useless number.
    This is an iterative algorithm which terminates when it converged or when
        `max_iter` iterations were made.

    :param model: Pairwise model for which to perform inference.
    :param max_iter: How many iterations without convergence should happen for
        algorithm to terminate. Defaults to maximal diameter of connected
        component.
    :return: InferenceResult object.

    Reference
        [1] Wainwright, Jordan. Graphical Models, Exponential Families, and
        Variational Inference. 2008. Section 2.5.1 (p. 26).
    """
    if max_iter is None:
        graph = networkx.Graph()
        graph.add_edges_from(model.get_edges_connected())
        max_iter = networkx.diameter(graph)

    # Build list of directed edges.
    edges = model.get_edges_connected()
    dir_edges = np.concatenate([edges, np.flip(edges, axis=1)])

    # Sort edges by end vertex. This ensures that edges ending with the same
    # vertex are sequential, which allows for efficient lookup.
    dir_edges.view('i4,i4').sort(order=['f1'], axis=0)

    # Compact representation of interactions.
    intrn = model.get_interactions_for_edges(dir_edges)

    # Main algorithm.
    lmu = _message_passing(dir_edges, intrn, model.field, max_iter)

    # Restore partition function for fixed values in nodes.
    log_marg_pf = np.array(model.field)
    for edge_id in range(len(dir_edges)):
        log_marg_pf[dir_edges[edge_id][1], :] += lmu[edge_id]

    log_pf = scipy.special.logsumexp(log_marg_pf, axis=-1)
    marg_prob = scipy.special.softmax(log_marg_pf, axis=-1)
    marg_prob /= np.sum(marg_prob, axis=-1).reshape(-1, 1)
    return InferenceResult(np.min(log_pf), marg_prob)
