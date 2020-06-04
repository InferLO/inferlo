from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
from networkx import Graph, is_tree, nx

from inferlo.base.domain import DiscreteDomain
from inferlo.base.factors import DiscreteFactor
from inferlo.base.graph_model import GraphModel
from .bruteforce import (infer_bruteforce, max_likelihood_potts_bruteforce)
from .inference.mean_field import infer_mean_field
from .inference.message_passing import infer_message_passing
from .inference.path_dp import infer_path_dp
from .inference.tree_dp import infer_tree_dp
from .optimization.tree_dp import max_likelihood_tree_dp
from .sampling.tree_dp import sample_tree_dp
from .utils import decode_state, encode_state

if TYPE_CHECKING:
    from inferlo.base.factors import Factor


class PairWiseFiniteModel(GraphModel):
    """Class describing Potts Model in the most general sense.

    Potts Model is a probabilistic base, in which every variable can take
        values in the same finite set (alphabet), and probability of every
        configuration is proportional to
        exp(sum F[i][X_i] + 0.5*sum J[i][j][X[i]][X[j]]), where M is field,
        J is interaction tensor.
    Potts base has underlying undirected graph, in which there is edge
        (i, j) iff `J[i,j,:,:] != 0`.
    """

    def __init__(self, size, alphabet_size):
        """Creates PottsModel instance.

        :param num_variables: Number of variables.
        :param alphabet_size: Size of the alphabet.
        """
        super().__init__(size, DiscreteDomain(list(range(alphabet_size))))

        self.gr_size = size
        self.al_size = alphabet_size

        self.field = np.zeros((self.gr_size, self.al_size))

        self.edges = []
        self._edges_interactions = []
        self._edge_ids = dict()  # Maps vertex pair to edge id.

        self._graph = None

    def set_field(self, field: np.ndarray):
        assert field.shape == (self.gr_size, self.al_size)
        self.field = field

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
        return (u, v) in self._edge_ids

    def get_graph(self):
        if self._graph is None:
            self._graph = Graph()
            self._graph.add_nodes_from(range(self.gr_size))
            for u, v in self.edges:
                self._graph.add_edge(u, v)
        return self._graph

    def get_compact_interactions(self):
        edges = np.array(self.edges, dtype=np.int32)
        int = np.array(self._edges_interactions, dtype=np.float64)
        if len(self.edges) == 0:
            edges = np.empty((0, 2), dtype=np.int32)
            int = np.empty((0, self.al_size, self.al_size,), dtype=np.float64)
        return edges, int

    def add_factor(self, factor: Factor):
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

    def infer(self, algorithm='auto', **kwargs):
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
        elif algorithm == 'tree_dp':
            return infer_tree_dp(self)
        elif algorithm == 'path_dp':
            return infer_path_dp(self)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        if algorithm == 'auto':
            if is_tree(self.get_graph()):
                return max_likelihood_tree_dp(self)
            else:
                return max_likelihood_potts_bruteforce(self)
        elif algorithm == 'bruteforce':
            return max_likelihood_potts_bruteforce(self)
        elif algorithm == 'tree_dp':
            return max_likelihood_tree_dp(self)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def sample(self, num_samples: int = 1, algorithm='auto',
               **kwargs) -> np.ndarray:
        if algorithm == 'auto':
            if is_tree(self.get_graph()):
                return sample_tree_dp(self, num_samples=num_samples)
            else:
                raise NotImplemented("Can handle only trees so far.")
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
        """Creates a Potts base represented by field and interactions."""
        size, al_size = field.shape
        model = PairWiseFiniteModel(size, al_size)
        model.set_field(field)
        idx = 0
        for v1, v2 in edges:
            model.add_interaction(v1, v2, interactions[idx])
            idx += 1
        return model

    def draw_pairwise_graph(self, ax):
        graph = self.get_graph()
        pos = nx.spring_layout(graph)
        node_labels = {i: self[i].name for i in range(self.num_variables)}
        nx.draw_networkx(graph, pos, ax,
                         labels=node_labels,
                         edge_color='green',
                         node_color='#ffaaaa')
        edge_labels = {(u, v): "J_%d_%d" % (u, v) for u, v in self.edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
