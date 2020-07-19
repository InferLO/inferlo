# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo.base import GraphModel
from inferlo.base.factors import DiscreteFactor
from inferlo.pairwise.inference_result import InferenceResult


def infer_generic_message_passing(model: GraphModel):
    """Generic message passing algorithm.

    Also known as "Loopy belief propagation."

    :param model: Model for which to perform inference.
    :return: Inference result.

    Reference
        [1] Kschischang, Frey, Loeliger,
        "Factor graphs and the sum-product algorithm",
        IEEE Transactions on Information Theory, 2001.
        https://ieeexplore.ieee.org/document/910572
    """
    # Build factor graph.
    # Edge in factor graph is pair (variable, factor), and it's split in two.
    nodes = []
    for i in range(model.num_variables):
        nodes.append({"type": "VARIABLE", "idx": i})
    factors = list(model.get_factors())

    edges = []
    edges_ri = {}
    # TODO: tidy up.
    neighbors = [[] for _ in range(1000)]  # "incoming neighbors"

    def add_edge(start_vx, end_vx):
        edges_ri[(start_vx, end_vx)] = len(edges)
        edges.append((start_vx, end_vx))
        neighbors[end_vx].append(start_vx)

    for factor in factors:
        node_id = len(nodes)
        f = DiscreteFactor.from_factor(factor)
        nodes.append(
            {"type": "FACTOR", "values": f.values, "var_idx": f.var_idx})
        for variable_id in factor.var_idx:
            add_edge(variable_id, node_id)
            add_edge(node_id, variable_id)
    neighbors = neighbors[0:len(nodes)]
    for i in range(len(edges)):
        assert edges_ri[edges[i]] == i

    print("nodes", nodes)
    print("edges", edges)
    print("neighbors", neighbors)

    # Initialize all messages with "unit functions".
    edge_messages = []
    for edge_id in range(len(edges)):
        start_vx, end_vx = edges[edge_id]
        if nodes[start_vx]["type"] == "VARIABLE":
            var_id = nodes[start_vx]["idx"]
        else:
            var_id = nodes[end_vx]["idx"]
        var_size = model.get_variable(var_id).domain.size()
        edge_messages.append(np.ones(var_size))

    def tensor_product(x):
        ans = np.array([1.0])
        for a in x:
            ans = np.kron(ans, a)
        return ans

    # Caclulates new message for edge.
    def calculate_new_message(edge_id):
        start_vx, end_vx = edges[edge_id]
        if nodes[start_vx]["type"] == "VARIABLE":
            assert nodes[end_vx]["type"] == "FACTOR"
            ans = np.ones_like(edge_messages[edge_id])
            for prev_vx in neighbors[start_vx]:
                if prev_vx == end_vx:
                    continue
                ans = ans * edge_messages[edges_ri[(prev_vx, start_vx)]]
            assert len(ans) == len(edge_messages[edge_id])
            return ans
        else:
            assert nodes[start_vx]["type"] == "FACTOR"
            assert nodes[end_vx]["type"] == "VARIABLE"
            assert nodes[end_vx][
                       "idx"] == end_vx  # answer is a function of variabe end_vx.
            factor_id = start_vx - len(nodes)
            final_var_id = end_vx
            final_domain_size = model.get_variable(final_var_id).domain.size()
            var_idx = factors[factor_id].var_idx

            final_var_id_in_var_idx = -1
            for i in range(len(var_idx)):
                if var_idx[i] == final_var_id:
                    final_var_id_in_var_idx = i
            assert final_var_id_in_var_idx >= 0

            perm = [final_var_id_in_var_idx] + [i for i in range(len(var_idx))
                                                if i != final_var_id_in_var_idx]
            factor_values = nodes[start_vx]["values"].transpose(perm)
            factor_values = factor_values.reshape((final_domain_size, -1))
            other_vars_id = [i for i in neighbors[start_vx] if i != end_vx]
            things_going_in_msg_tp = [
                edge_messages[edges_ri[(prev_vx, start_vx)]] for prev_vx in
                other_vars_id]
            msg_tp = tensor_product(things_going_in_msg_tp)

            assert factor_values.shape[1] == msg_tp.shape[0]
            ans = np.einsum('ij,j->i', factor_values, msg_tp)
            assert len(ans) == len(edge_messages[edge_id])
            return ans

    for _ in range(100):
        # TODO: number of iteration must be passed in, and do convergence test.
        edge_messages = [calculate_new_message(i) for i in range(len(edges))]

    def get_result_for_variable(var_id):
        ans = 1
        for prev_vx in neighbors[var_id]:
            ans *= edge_messages[edges_ri[(prev_vx, var_id)]]
        return ans

    ans = np.zeros((model.num_variables, model.get_max_domain_size()))
    for i in range(model.num_variables):
        ans[i, 0:model[i].domain.size()] = get_result_for_variable(i)

    pf = np.sum(ans, axis=1)[0]
    marg_probs = ans / pf
    return InferenceResult(log_pf=np.log(pf), marg_prob=ans / pf)
