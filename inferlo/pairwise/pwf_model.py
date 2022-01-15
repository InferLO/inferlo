# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Set, Union, List

import numpy as np
import networkx as nx
from networkx import Graph

from inferlo.base.domain import DiscreteDomain
from inferlo.base.factors import DiscreteFactor
from inferlo.base.graph_model import GraphModel
from .bruteforce import (
    infer_bruteforce,
    max_lh_bruteforce,
    sample_bruteforce, TooMuchStatesError)
from .inference.mean_field import infer_mean_field
from .inference.message_passing import infer_message_passing
from .inference.path_dp import infer_path_dp
from .inference.tree_dp import infer_tree_dp
from .junction_tree import infer_junction_tree, max_likelihood_junction_tree, \
    sample_junction_tree
from .optimization.path_dp import max_lh_path_dp
from .optimization.tree_dp import max_likelihood_tree_dp
from .sampling.tree_dp import sample_tree_dp
from .utils import decode_state, encode_state, decode_all_states
from ..generic.inference import belief_propagation
from ..graphs import fast_dfs
from ..graphs.fast_dfs import FastDfsResult

if TYPE_CHECKING:
    from inferlo.base.factors import Factor
    from inferlo.base.inference_result import InferenceResult


class PairWiseFiniteModel(GraphModel):
    """Pairwise finite graphical model.

    Represents a graphical model in which all variables have the same discrete
    domain, all factor depends on at most two variables.

    Model is represented by field F and interactions J. Probability of
    configuration ``X`` is proportional to
    ``exp(sum F[i][X_i] + 0.5*sum J[i][j][X[i]][X[j]])``.

    Field is stored explicitly as a matrix of shape ``(gr_size, al_size)``.

    Interactions are stored only for those pairs of variables for which they
    are non-zero. So, interactions are represented by undirected graph, where
    for each edge (i,j) we store matrix `J[i,j]`, which has shape
    ``(al_size, al_size)``.

    Names
        "Field" is called like that because in physical models (such as
        Ising model) these values correspond to local magnetic fields. They
        are also known as biases. "Interactions" are called like that because
        in physical models they correspond to strength of spin-spin
        interactions. The fact that all these terms enter the probability
        density function inside the exponent also refers to physical models,
        because fields and interactions are terms in energy and according to
        Bolzmann distribution probability of the state with energy E is
        proportional to ``exp(-E/(kT))``.
    """

    def __init__(self, size, al_size):
        """Initializes PairWiseFiniteModel.

        :param num_variables: Number of variables.
        :param al_size: Size of the alphabet (domain).

        Domain will consist of integers in range 0, 1, ... al_size - 1.
        """
        super().__init__(size, DiscreteDomain.range(al_size))

        self.gr_size = size
        self.al_size = al_size

        self.field = np.zeros((self.gr_size, self.al_size), dtype=np.float64)

        self.edges = []
        self._edges_interactions = []
        # Maps  (u,v) and (v,u) to index of one of them in self.edges.
        self._edge_ids = dict()

        # Cached properties that are invalidated when graph changes.
        self._graph = None
        self._edges_array = None
        self._dfs_result = None

    def set_field(self, field: np.ndarray):
        """Sets values of field (biases) in all vertices."""
        assert field.shape == (self.gr_size, self.al_size)
        self.field = np.array(field, dtype=np.float64)

    def add_interaction(self, u, v, interaction):
        """Adds factor corresponding to interaction between nodes u and v.

        Factor is f(x) = exp(interaction[x[u], x[v]]).
        If there already is interaction between these edges, this interaction
        will be added to it (old interaction isn't discarded).
        """
        if (u, v) in self._edge_ids:
            edge_id = self._edge_ids[(u, v)]
            if self.edges[edge_id] == (v, u):
                interaction = interaction.T
            self._edges_interactions[edge_id] += interaction
        else:
            self._on_graph_changed()
            self.edges.append((u, v))
            self._edges_interactions.append(np.array(interaction,
                                                     dtype=np.float64))
            self._edge_ids[(u, v)] = len(self.edges) - 1
            self._edge_ids[(v, u)] = len(self.edges) - 1

    def get_interaction_matrix(self, u, v):
        """Returns interaction matrix between nodes u and v.

        Returns np.array of shape (al_size, al_size).
        If there is no interaction between these nodes, raises KeyError.
        """
        edge_id = self._edge_ids[(u, v)]
        if self.edges[edge_id] == (u, v):
            return self._edges_interactions[edge_id]
        else:
            return self._edges_interactions[edge_id].T

    def get_interactions_for_edges(self, edges) -> np.ndarray:
        """Returns interaction for given edges.

        If some edges don't exist, interaction matrix for them will be a zero
        matrix.

        :param edges: Edge list. np.array of shape ``(x, 2)``.
        :return: np.array of shape (x, al_size, al_size).
        """
        edges_num = edges.shape[0]
        assert edges.shape == (edges_num, 2)
        result = np.zeros((edges_num, self.al_size, self.al_size),
                          dtype=np.float64)

        for i in range(edges_num):
            u, v = edges[i]
            if self.has_edge(u, v):
                result[i, :, :] = self.get_interaction_matrix(u, v)

        return result

    def has_edge(self, u, v) -> bool:
        """Whether there is edge between vertices u and v."""
        return (u, v) in self._edge_ids

    def get_graph(self):
        """Returns interaction graph."""
        if self._graph is None:
            self._graph = Graph()
            self._graph.add_nodes_from(range(self.gr_size))
            for u, v in self.edges:
                self._graph.add_edge(u, v)
        return self._graph

    def get_dfs_result(self) -> FastDfsResult:
        """Performs DFS for interaction graph."""
        if self._dfs_result is None:
            self._dfs_result = fast_dfs(self.gr_size, self.get_edges_array())
        return self._dfs_result

    def is_graph_acyclic(self):
        """Whether interaction graph is acyclic."""
        return not self.get_dfs_result().had_cycles

    def get_edges_array(self) -> np.ndarray:
        """Returns edge list as np.array."""
        if self._edges_array is None:
            if len(self.edges) == 0:
                self._edges_array = np.empty((0, 2), dtype=np.int32)
            else:
                self._edges_array = np.array(self.edges, dtype=np.int32)
        return self._edges_array

    def get_edges_connected(self) -> np.ndarray:
        """Returns edges, ensuring that graph is connected.

        If graph is already connected, equivalent to ``get_edges_array``.
        If graph is not connected, adds minimal amount of edges to make it
        connected.

        This is needed for algorithms which require connected graph to work
        correctly.
        """
        if not self.get_dfs_result().was_disconnected:
            return self.get_edges_array()

        additional_edges = [(u, v) for u, v in self.get_dfs_result().dfs_edges
                            if not self.has_edge(u, v)]
        return np.concatenate([self.get_edges_array(), additional_edges])

    def _on_graph_changed(self):
        """Invalidates cached graphs."""
        self._graph = None
        self._edges_array = None
        self._dfs_result = None

    def get_all_interactions(self) -> np.ndarray:
        """Returns all interaction matrices in compact form.

        :return: np.array of shape ``(edge_num, al_size, al_size)`` with
          interaction matrix for every edge. Matrices correspond to edges in
          the same order as returned by get_edges.array.
        """
        if len(self.edges) == 0:
            shape = (0, self.al_size, self.al_size)
            return np.empty(shape, dtype=np.float64)
        return np.array(self._edges_interactions, dtype=np.float64)

    def add_factor(self, factor: Factor):
        """Adds a factor."""
        if isinstance(factor, DiscreteFactor):
            self._add_discrete_factor(factor)
        elif factor.is_discrete():
            self._add_discrete_factor(DiscreteFactor.from_factor(factor))
        else:
            raise ValueError("Can't add non-discrete factor.")

    def _add_discrete_factor(self, factor: DiscreteFactor):
        assert factor.model == self
        with np.errstate(divide='ignore'):
            log_factor = np.log(factor.values)
        if len(factor.var_idx) > 2:
            raise ValueError("Can't add factor with more than 2 variables.")
        if len(factor.var_idx) == 1:
            assert factor.values.shape == (self.al_size,)
            self.field[factor.var_idx[0], :] += log_factor
        elif len(factor.var_idx) == 2:
            v1, v2 = factor.var_idx
            self.add_interaction(v1, v2, log_factor)

    def get_factors(self) -> Iterable[Factor]:
        """Generates explicit list of factors."""
        for i in range(self.gr_size):
            if np.linalg.norm(self.field[i, :]) > 1e-9:
                yield DiscreteFactor(self, [i], np.exp(self.field[i, :]))
        for u, v in self.edges:
            factor = DiscreteFactor(self, [u, v],
                                    np.exp(self.get_interaction_matrix(u, v)))
            if self.num_variables < 10:
                factor.name = 'J%d%d' % (u, v)
            else:
                factor.name = 'J_%d_%d' % (u, v)
            yield factor

    def infer(self, algorithm='auto', **kwargs) -> InferenceResult:
        """Performs inference.

        Available algorithms
            * ``auto`` - Automatic.
            * ``bruteforce`` - Brute force (by definition). Exact
            * ``mean_field`` - Naive Mean Field. Approximate.
            * ``message_passing`` - Message passing. Approximate, exact only
              for trees.
            * ``path_dp`` - Dynamic programming on path decomposition. Exact.
              Effective on graphs of small pathwidth.
            * ``tree_dp`` - Dynamic programming on tree. Exact. Works only on
              trees.
            * ``junction_tree`` - DP on junction tree. Exact. Effective on
              graphs of small treewidth.

        :param algorithm: Which algorithm to use. String.
        :return: `InferenceResult` object, which contains logarithm of
          partition function and matrix of marginal probabilities.
        """
        if algorithm == 'auto':
            if self.is_graph_acyclic():
                return infer_tree_dp(self)
            try:
                return infer_junction_tree(self)
            except TooMuchStatesError:
                return belief_propagation(self)
        elif algorithm == 'bruteforce':
            return infer_bruteforce(self)
        elif algorithm == 'mean_field':
            return infer_mean_field(self, **kwargs)
        elif algorithm == 'message_passing':
            return infer_message_passing(self, **kwargs)
        elif algorithm == 'path_dp':
            return infer_path_dp(self)
        elif algorithm == 'tree_dp':
            return infer_tree_dp(self)
        elif algorithm == 'junction_tree':
            return infer_junction_tree(self, **kwargs)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        """Finds the most probable state.

        Available algorithms
            * ``auto`` - Automatic.
            * ``bruteforce`` - Brute force (by definition).
            * ``path_dp`` - Dynamic programming on path decomposition. Exact.
              Effective on graphs of small pathwidth.
            * ``tree_dp`` - Dynamic programming on tree. Exact. Works only on
              trees.
            * ``junction_tree`` - DP on junction tree. Exact. Effective on
              graphs of small treewidth.

        :param algorithm: Which algorithm to use. String.
        :return: The most probable state as numpy int array.
        """
        if algorithm == 'auto':
            if self.is_graph_acyclic():
                return max_likelihood_tree_dp(self)
            else:
                try:
                    return max_lh_bruteforce(self)
                except TooMuchStatesError:
                    return max_likelihood_junction_tree(self)
        elif algorithm == 'bruteforce':
            return max_lh_bruteforce(self)
        elif algorithm == 'tree_dp':
            return max_likelihood_tree_dp(self)
        elif algorithm == 'path_dp':
            return max_lh_path_dp(self)
        elif algorithm == 'junction_tree':
            return max_likelihood_junction_tree(self)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def sample(self, num_samples: int = 1, algorithm='auto',
               **kwargs) -> np.ndarray:
        """Draws i.i.d. samples from the distribution.

        Available algorithms
            * ``auto`` - Automatic.
            * ``bruteforce`` - Sampling from explicitly calculated
              probabilities for each state.
            * ``tree_dp`` - Dynamic programming on tree. Works only on trees.
            * ``junction_tree`` - DP on junction tree.

        :param num_samples: How many samples to generate.
        :param algorithm: Which algorithm to use.
        :return: ``np.array`` of type ``np.int32`` and shape
          ``(num_samples, gr_size)``. Every row is an independent sample.
        """
        if algorithm == 'auto':
            if self.is_graph_acyclic():
                return sample_tree_dp(self, num_samples=num_samples)
            else:
                try:
                    return sample_bruteforce(self, num_samples=num_samples)
                except TooMuchStatesError:
                    return sample_junction_tree(self, num_samples=num_samples)
        elif algorithm == 'bruteforce':
            return sample_bruteforce(self, num_samples=num_samples)
        elif algorithm == 'tree_dp':
            return sample_tree_dp(self, num_samples=num_samples)
        elif algorithm == 'junction_tree':
            return sample_junction_tree(self, num_samples=num_samples)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def encode_state(self, state):
        """Returns state represented by its integer id."""
        return encode_state(state, self.gr_size, self.al_size)

    def decode_state(self, state):
        """Returns id of given state.

        State id is integer between `0` and `al_size**gr_size-1`.
        """
        return decode_state(state, self.gr_size, self.al_size)

    @staticmethod
    def create(field: np.ndarray,
               edges: Union[np.ndarray, List],
               interactions: np.ndarray):
        """Creates PairwiseFiniteModel from compact representation.

        Infers number of variables and size of alphabet from shape of
        ``field``.

        :param field: Values of the field. ``np.array`` of shape
          ``(gr_size, al_size)``.
        :param edges: List of edges with interactions. ``np.array`` of integer
          dtype and shape ``(edge_num, 2)``. Edges can't repeat. If there is
          edge (u,v), you can't have edge (v,u).
        :param interactions: ``np.array`` of shape
          ``(edge_num, al_size, al_size)``, or Iterable which can be converted
          to such an array. ``interactons[i,:,:]`` is a matrix  decribing
          interactions between variables ``edges[i, 0]`` and ``edges[i, `]``.
        """
        size, al_size = field.shape
        model = PairWiseFiniteModel(size, al_size)
        model.set_field(field)
        idx = 0
        assert len(edges) == len(interactions)
        for v1, v2 in edges:
            model.add_interaction(v1, v2, interactions[idx])
            idx += 1
        return model

    def draw_pairwise_graph(self, ax):
        """Draws pairwise graph."""
        graph = self.get_graph()
        pos = nx.kamada_kawai_layout(graph)
        node_labels = {i: self[i].name for i in range(self.num_variables)}
        nx.draw_networkx(graph, pos, ax,
                         labels=node_labels,
                         edge_color='green',
                         node_color='#ffaaaa')
        edge_labels = {(u, v): "J_%d_%d" % (u, v) for u, v in self.edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    def get_subgraph_factor_values(self,
                                   vars_idx: np.ndarray,
                                   vars_skip: Set = frozenset()) -> np.ndarray:
        """Calculates factor values for subgraph.

        Consider model on subgraph containing only variables with indices
        ``vars``. That is, containing only factors which depend only on
        variables from ``vars``. For every possible combination of those
        variable values, calculate product of all factors in the new model -
        that's what this function returns.

        This can also be described as "interactions within subgraph". Or if we
        condense all variables in ``vars`` in single "supervariable", this
        function returns field for the new supervariable.

        :param vars_idx: Indices of variables in subgraph.
        :param vars_skip_factors: Set. Indices of variables, which should be
          skipped for factor calculation. Field factors for these variables
          won't be included in the result. Interaction factors oth arguments
          of which are in ``vars_skip_factors``, won't be included in the
          result. However, interaction factors where only one variable appears
          in ``vars_skip_factors``, will be included in result. This parameter
          is useful when building junction tree, to avoid double-counting
          factors.
        :return: ``np.array`` of length ``al_size ** len(vars)``. Each value
          is logarithm of product of all relevant factors for certain variable
          values. Correspondence between indices in this array and states
          is consistent with ``decode_state``.
        """
        vars_num = len(vars_idx)
        edges = []
        for i in range(vars_num):
            v1 = vars_idx[i]
            for j in range(i + 1, vars_num):
                v2 = vars_idx[j]
                should_skip = v1 in vars_skip and v2 in vars_skip
                if not should_skip and self.has_edge(v1, v2):
                    edges.append(
                        (i, j, self.get_interaction_matrix(v1, v2)))

        all_states = decode_all_states(vars_num, self.al_size)
        a = np.zeros(self.al_size ** vars_num)
        for u in range(vars_num):
            if vars_idx[u] in vars_skip:
                continue
            a += self.field[vars_idx[u]][all_states[:, u]]
        for u, v, j in edges:
            a += j[all_states[:, u], all_states[:, v]]
        return a

    @staticmethod
    def from_model(original_model: GraphModel) -> PairWiseFiniteModel:
        """Constructs Pairwise Finite model which is equivalent to given model.

        All variables must be discrete. All factors must depend on at most 2
        variables.

        New model will have the same number of variables and factors. If
        variables in original model have different domain sizes, in new model
        they will be extended to have the same domain size.
        """
        al_size = max(v.domain.size() for v in original_model.get_variables())
        old_factors = list(original_model.get_factors())

        def pad_tensor(t):
            padding = [[0, al_size - dim] for dim in t.shape]
            return np.pad(t, padding)

        # Validate model.
        if al_size > 1000:
            raise ValueError("Not all variables are discrete.")
        if max(len(f.var_idx) for f in old_factors) > 2:
            raise ValueError("Model is not pairwise.")

        new_model = PairWiseFiniteModel(original_model.num_variables, al_size)
        for old_factor in old_factors:
            values = DiscreteFactor.from_factor(old_factor).values
            values = pad_tensor(values)
            new_factor = DiscreteFactor(new_model, old_factor.var_idx, values)
            new_model.add_factor(new_factor)

        return new_model
