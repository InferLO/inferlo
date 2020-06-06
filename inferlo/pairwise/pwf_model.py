from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
from networkx import Graph, is_tree, nx

from inferlo.base.domain import DiscreteDomain
from inferlo.base.factors import DiscreteFactor
from inferlo.base.graph_model import GraphModel
from .bruteforce import (infer_bruteforce, max_lh_bruteforce)
from .inference.mean_field import infer_mean_field
from .inference.message_passing import infer_message_passing
from .inference.path_dp import infer_path_dp
from .inference.tree_dp import infer_tree_dp
from .optimization.tree_dp import max_likelihood_tree_dp
from .sampling.tree_dp import sample_tree_dp
from .utils import decode_state, encode_state

if TYPE_CHECKING:
    from inferlo.base.factors import Factor
    from inferlo.pairwise.inference_result import InferenceResult


class PairWiseFiniteModel(GraphModel):
    """Pairwise finite graphical model

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
        proportional to $e^{-\beta E}.
    """

    def __init__(self, size, al_size):
        """Creates PairWiseFiniteModel.

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
        self._edge_ids = dict()  # Maps vertex pair to edge id.

        self._graph = None

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
            self._graph = None
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

    def make_connected(self):
        """Makes graph connected without changing the distribution.

        Adds minimal amount of edges with zero interactions. If graph was a
        forest, it becomes a tree. If graph as connected, does nothing.
        """
        # TODO: ideally this shouldn't mutate the model.
        con_comps = list(nx.connected_components(self.get_graph()))
        if len(con_comps) > 1:
            zeros = np.zeros((self.al_size, self.al_size))
            # Node to which attach other components.
            v0 = list(con_comps[0])[0]
            for cc in con_comps[1:]:
                self.add_interaction(v0, list(cc)[0], zeros)
            self._graph = None

    def get_compact_interactions(self):
        """Returns interactions in compact form.

        Returns two np.arrays. First array has shape (edges_num, 2) and is a
        list of edges in interaction graph. No edge appears twice. Second array
        has shape (edge_id, al_size, al_size), and for every edge in the first
        array contains interaction matrix for that edge.
        """
        edges = np.array(self.edges, dtype=np.int32)
        inter = np.array(self._edges_interactions, dtype=np.float64)
        if len(self.edges) == 0:
            edges = np.empty((0, 2), dtype=np.int32)
            inter = np.empty(
                (0, self.al_size, self.al_size,), dtype=np.float64)
        return edges, inter

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
        if len(factor.var_idx) > 2:
            raise ValueError("Can't add factor with more than 2 variables.")
        if len(factor.var_idx) == 1:
            assert factor.values.shape == (self.al_size,)
            self.field[factor.var_idx[0], :] += np.log(factor.values)
        elif len(factor.var_idx) == 2:
            v1, v2 = factor.var_idx
            self.add_interaction(v1, v2, np.log(factor.values))

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
            * ``path_dp`` - Dynamic programming on path decomposition.
              Approximate. Effective on graphs of small pathwidth.
            * ``tree_dp`` - Dynamic programming on tree. Exact. Works only on
              trees.

        :param algorithm: Which algorithm to use. String.
        :return: `InferenceResult` object, which contains logarithm of
          partition function and matrix of marginal probabilities.
        """
        if algorithm == 'auto':
            if is_tree(self.get_graph()):
                return infer_tree_dp(self)
            else:
                return infer_message_passing(self)
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
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        """Finds the most probable state.

        Available algorithms
            * ``auto`` - Automatic.
            * ``bruteforce`` - Brute force (by definition).
            * ``tree_dp`` - Dynamic programming on tree. Exact. Works only on
              trees.

        :param algorithm: Which algorithm to use. String.
        :return: The most probable state as numpy int array.
        """
        if algorithm == 'auto':
            if is_tree(self.get_graph()):
                return max_likelihood_tree_dp(self)
            else:
                return max_lh_bruteforce(self)
        elif algorithm == 'bruteforce':
            return max_lh_bruteforce(self)
        elif algorithm == 'tree_dp':
            return max_likelihood_tree_dp(self)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def sample(self, num_samples: int = 1, algorithm='auto',
               **kwargs) -> np.ndarray:
        """Draws i.i.d. samples from the distribution.

        Returns ``np.array`` of type ``np.int32`` shape
        ``(num_samples, gr_size)``. Every row is an independent sample.

        Available algorithms
            * ``auto`` - Automatic.
            * ``tree_dp`` - Dynamic programming on tree. Works only on trees.

        :param num_samples: How many samples to generate.
        :param algorithm: Which algorithm to use.
        """
        if algorithm == 'auto':
            if is_tree(self.get_graph()):
                return sample_tree_dp(self, num_samples=num_samples)
            else:
                raise NotImplementedError("Can handle only trees so far.")
        elif algorithm == 'tree_dp':
            return sample_tree_dp(self, num_samples=num_samples)
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
               edges: Iterable,
               interactions: Iterable):
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
        for v1, v2 in edges:
            model.add_interaction(v1, v2, interactions[idx])
            idx += 1
        return model

    def draw_pairwise_graph(self, ax):
        """Draws pairwise graph."""
        graph = self.get_graph()
        pos = nx.spring_layout(graph)
        node_labels = {i: self[i].name for i in range(self.num_variables)}
        nx.draw_networkx(graph, pos, ax,
                         labels=node_labels,
                         edge_color='green',
                         node_color='#ffaaaa')
        edge_labels = {(u, v): "J_%d_%d" % (u, v) for u, v in self.edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
