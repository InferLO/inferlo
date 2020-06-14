from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in

from inferlo.pairwise.inference_result import InferenceResult
from inferlo.pairwise.utils import decode_state, get_marginal_states

if TYPE_CHECKING:
    from inferlo.pairwise.pwf_model import PairWiseFiniteModel


def build_kron_delta(var_num, al_size, p1, p2):
    """Builds boolean factor requiring that v1[p1]=v2[p2]. """
    ans_size = al_size ** var_num
    ans = np.zeros((ans_size, ans_size), dtype=np.bool)
    for i in range(ans_size):
        val1 = (i // (al_size ** (var_num - p1 - 1))) % al_size
        for j in range(ans_size):
            val2 = (j // (al_size ** (var_num - p2 - 1))) % al_size
            ans[i, j] = (val1 == val2)
    return ans


def build_multi_delta(var_num, al_size, nodes1, nodes2):
    """Build boolean matrix describing Kroneker delta symbol.

    Result[i][j]==True iff state of first supervariable encoded by i and state
    of second supervariable encoded by j are consistent. If the same variable
    appears both in nodes1 and nodes2, then "consistent"  means this variable
    takes the same value in both  supervariables.

    :param: var_num - Number of variables in supervariable. It's possible that
      length of nodes1 or nodes2 less than this value, then we have to do
      padding.
    :al_size: Alphabet size of original model.
    :nodes1: Indices of variables in first supervariable.
    :nodes2: Indices of variables in second supervariable.
    :return: Square boolean matrix of size al_size ** var_num.
    """
    # Padding to account for non-full nodes.
    pad1 = var_num - len(nodes1)
    pad2 = var_num - len(nodes2)
    ans_size = al_size ** var_num
    ans = np.ones((ans_size, ans_size), dtype=np.bool)
    nodes2 = nodes2.tolist()
    for i in range(len(nodes1)):
        for j in range(len(nodes2)):
            if nodes1[i] == nodes2[j]:
                ans *= build_kron_delta(var_num, al_size, i + pad1, j + pad2)
    return ans


@dataclass
class JunctionizedModel:
    """Junction tree decomposition of pairwise model.

    Contains list of new nodes, represented as lists of old nodes. Also
    contains model over junction tree, which is equivalent to the original
    model.

    It's guaranteed that interaction graph of new model doesn't have any
    cycles. This object can be used to restore answers to problems on original
    model if we know answers to new model.
    """
    new_model: PairWiseFiniteModel
    junction_nodes: list
    orig_gr_size: int
    orig_al_size: int

    def restore_original_state(self, new_state):
        """Converts new model state to original model state."""
        old_state = -1 * np.ones(self.orig_gr_size, dtype=np.int32)
        for i in range(len(self.junction_nodes)):
            decoded_state = decode_state(new_state[i],
                                         len(self.junction_nodes[i]),
                                         self.orig_al_size)
            old_nodes = self.junction_nodes[i]
            for i in range(len(old_nodes)):
                if old_state[old_nodes[i]] == -1:
                    old_state[old_nodes[i]] = decoded_state[i]
                else:
                    assert old_state[old_nodes[i]] == decoded_state[i]
        return old_state

    def restore_original_inference_result(
            self,
            new_result: InferenceResult) -> InferenceResult:
        """Converts new model inference result to orig. model inf. result."""
        new_marg_prob = new_result.marg_prob
        old_marg_prob = np.zeros((self.orig_gr_size, self.orig_al_size))
        for i in range(len(self.junction_nodes)):
            marg_states = get_marginal_states(len(self.junction_nodes[i]),
                                              self.orig_al_size)
            for j in range(len(self.junction_nodes[i])):
                orig_v = self.junction_nodes[i][j]
                for val in range(self.orig_al_size):
                    sm = 0
                    for k in marg_states[j, val]:
                        sm += new_marg_prob[i, k]
                    old_marg_prob[orig_v, val] = sm

        for v in range(self.orig_gr_size):
            assert np.allclose(np.sum(old_marg_prob[v, :]), 1)
        return InferenceResult(log_pf=new_result.log_pf,
                               marg_prob=old_marg_prob)


def to_junction_tree_model(model) -> JunctionizedModel:
    """Builds equivalent model on a junction tree.

    First, builds a junction tree using algorithm from NetworkX which uses
    Minimum Fill-in heuristic.

    Then, builds a new model in which variables correspond to nodes in junction
    tree - we will call them "supervariables". Values of new supervariables are
    encoded values of original variables. New alphabet size is original
    alphabet size to the power of maximaljunction size. If some supervariables
    have less variables than others, we just don't use all available for
    encoding "address space". We mark those impossible values as having
    probability 0 (i.e log probability -inf).

    Fields in new model are calculated by multiplying all field and
    interaction factors on variables in the same supervariable. While doing
    this, we make sure that every factor is counted only once. If some factor
    was accounted for in one supervariable field, it won't be accounted for
    again in other supervariables.

    Interaction factors in new model contain consistency requirement. If
    a variable of original model appears in multiple supervariables, we allow
    only those states where it takes the same value in all supervariables. We
    achieve that by using interaction factors which are equal to 1 if values
    of the same original variable in different supervariables are equal, and
    0 if they are not equal. We actually use values 0 and -inf, because we
    work with logarithms.

    See https://en.wikipedia.org/wiki/Tree_decomposition.

    :param model: original model.
    :return: JunctionizedModel object, which contains junction tree and the
      new model, which is equivalent to original model, but whose graph is a
      tree.
    """
    # Build junction tree.
    tree_width, junc_tree = treewidth_min_fill_in(model.get_graph())
    jt_nodes = list(junc_tree.nodes())
    sv_size = tree_width + 1  # Supervariable size.
    new_gr_size = len(jt_nodes)  # New graph size.
    new_al_size = model.al_size ** sv_size  # New alphabet size.
    assert new_al_size <= 1e6, "New domain size is too large."

    # Build edge list in terms of indices in new graph.
    nodes_lookup = {jt_nodes[i]: i for i in range(len(jt_nodes))}
    new_edges = np.array(
        [[nodes_lookup[u], nodes_lookup[v]] for u, v in junc_tree.edges()])

    # Convert node lists to numpy arrays.
    jt_nodes = [np.fromiter(node, dtype=np.int32) for node in jt_nodes]

    # Calculate fields which describe interaction beteen supervariables.
    # If supervariable has less than ``sv_size`` variables, pad with -inf.
    # Then, when decoding, we will just throw away values from the left.
    # We should account for each factor of the old graph in exactly one factor
    # in the new graph. So, for field and interaction factors of the old graph
    # we keep track of whether we already took them, and don't take them for
    # the second time.
    new_field = np.ones((new_gr_size, new_al_size), dtype=np.float64) * -np.inf
    used_node_fields = set()
    for new_node_id in range(new_gr_size):
        old_nodes = jt_nodes[new_node_id]
        node_field = model.get_subgraph_factor_values(
            old_nodes, vars_skip=used_node_fields)
        new_field[new_node_id, 0:len(node_field)] = node_field
        used_node_fields.update(old_nodes)

    # Now, for every edge in new graph - add interaction factor requiring that
    # the same variable appearing in two supervariables always has the same
    # values.
    # We achieve this by using Kroenker delta function.
    # As we working with logarithms, we populate -inf for impossible states,
    # and 0 for possible states.
    new_interactions = np.zeros((len(new_edges), new_al_size, new_al_size))
    for edge_id in range(len(new_edges)):
        u, v = new_edges[edge_id]
        allowed = build_multi_delta(sv_size, model.al_size, jt_nodes[u],
                                    jt_nodes[v])
        new_interactions[edge_id, np.logical_not(allowed)] = -np.inf

    from inferlo.pairwise.pwf_model import PairWiseFiniteModel
    new_model = PairWiseFiniteModel.create(new_field, new_edges,
                                           new_interactions)
    return JunctionizedModel(new_model, jt_nodes, model.gr_size, model.al_size)


def infer_junction_tree(model):
    """Performs inference using Junction Tree decomposition.

    Decomposes graph into junction tree, builds equivalent model on that tree,
    performs inference for the new model and restores answer for the original
    model.

    :param model: Model, for which to perform inference.
    :param result: InferenceResult object.
    """
    junct_model = to_junction_tree_model(model)
    result = junct_model.new_model.infer(algorithm='tree_dp')
    return junct_model.restore_original_inference_result(result)


def max_likelihood_junction_tree(model):
    """Finds most probable state using Junction Tree decomposition.

    Decomposes graph into junction tree, builds equivalent model on that tree,
    finds the most probable state for the new model and restores answer for the
    original model.

    :param model: Model, for which to find the most probable state.
    :param result: The most probable state. Integer np.array of length
      ``model.gr_size``.
    """
    junct_model = to_junction_tree_model(model)
    state = junct_model.new_model.max_likelihood(algorithm='tree_dp')
    return junct_model.restore_original_state(state)
