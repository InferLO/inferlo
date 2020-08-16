# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import networkx as nx
import numpy as np

from inferlo.base import GraphModel
from inferlo.base.factors import DiscreteFactor
from inferlo.base.inference_result import InferenceResult


def infer_generic_message_passing(model: GraphModel,
                                  max_iter=100,
                                  max_product=False):
    """Generic message passing algorithm.

    Also known as "Loopy belief propagation."

    :param model: Model for which to perform inference.
    :param max_iter: Maximal number of iterations.
    :return: Inference result.

    Warning! This does not work correctly on non-trees.

    Reference
        [1] Kschischang, Frey, Loeliger,
        "Factor graphs and the sum-product algorithm",
        IEEE Transactions on Information Theory, 2001.
        https://ieeexplore.ieee.org/document/910572
    """
    # Prepare discrete factors.
    factors = [DiscreteFactor.from_factor(f) for f in model.get_factors()]

    # Build directed factor graph.
    num_variables = model.num_variables
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_variables + len(factors)))
    edges_count = 0
    for factor_id in range(len(factors)):
        factor_vertex_id = num_variables + factor_id
        for var_id in factors[factor_id].var_idx:
            graph.add_edge(var_id, factor_vertex_id, id=edges_count)
            graph.add_edge(factor_vertex_id, var_id, id=edges_count + 1)
            edges_count += 2
    edges = list(graph.edges)

    # Reverse index on edges. Maps edge to its id.
    edges_ri = {edges[i]: i for i in range(len(edges))}

    def get_variable_for_edge(edge_id):
        start_vx, end_vx = edges[edge_id]
        return start_vx if start_vx < num_variables else end_vx

    # Initialize all messages with "unit functions".
    edge_messages = [np.ones(
        model.get_variable(get_variable_for_edge(edge_id)).domain.size()) for
        edge_id in range(len(edges))]

    def tensor_product(x):
        ans = np.array([1.0])
        for a in x:
            ans = np.kron(ans, a)
        return ans

    def incoming_edge_ids(edge_id):
        start_vx, end_vx = edges[edge_id]
        return [edges_ri[(i, start_vx)] for i in graph.neighbors(start_vx) if
                i != end_vx]

    def new_message_vf(edge_id):
        """Calculate new message for variable->factor edge."""
        ans = np.ones_like(edge_messages[edge_id])
        for i in incoming_edge_ids(edge_id):
            ans = ans * edge_messages[i]
        assert len(ans) == len(edge_messages[edge_id])
        return ans

    def new_message_fv(edge_id):
        """Calculate new message for factor->variable edge."""
        start_vx, end_vx = edges[edge_id]
        factor_id = start_vx - num_variables
        final_var_id = end_vx
        final_domain_size = model.get_variable(final_var_id).domain.size()
        var_idx = factors[factor_id].var_idx

        final_var_id_in_var_idx = -1
        for i in range(len(var_idx)):
            if var_idx[i] == final_var_id:
                final_var_id_in_var_idx = i
        assert final_var_id_in_var_idx >= 0

        perm = [final_var_id_in_var_idx] + [j for j in range(len(var_idx)) if
                                            j != final_var_id_in_var_idx]
        factor_values = factors[factor_id].values.transpose(perm)
        factor_values = factor_values.reshape((final_domain_size, -1))
        msg_tp = tensor_product(
            [edge_messages[i] for i in incoming_edge_ids(edge_id)])

        assert factor_values.shape[1] == msg_tp.shape[0]
        ans = factor_values * msg_tp
        if max_product:
            ans = np.max(ans, axis=1)
        else:
            ans = np.sum(ans, axis=1)
        assert len(ans) == len(edge_messages[edge_id])
        return ans

    # Caclulates new message for edge.
    def calculate_new_message(edge_id):
        start_vx, _ = edges[edge_id]
        if start_vx < num_variables:
            return new_message_vf(edge_id)
        else:
            return new_message_fv(edge_id)

    # Main loop.
    for _ in range(max_iter):
        edge_messages = [calculate_new_message(i) for i in range(len(edges))]

    def get_result_for_variable(var_id):
        ans = 1
        for prev_vx in graph.neighbors(var_id):
            ans *= edge_messages[edges_ri[(prev_vx, var_id)]]
        return ans

    ans = np.zeros((model.num_variables, model.get_max_domain_size()))
    for i in range(model.num_variables):
        ans[i, 0:model[i].domain.size()] = get_result_for_variable(i)

    if max_product:
        return np.argmax(ans, axis=1)

    pf = np.sum(ans, axis=1)[0]
    return InferenceResult(log_pf=np.log(pf), marg_prob=ans / pf)
    # some comment.
