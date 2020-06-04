from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterable, Tuple, Dict

import networkx as nx
import numpy as np

from inferlo.base.variable import Variable

if TYPE_CHECKING:
    from inferlo.base import Domain, Factor


class GraphModel(abc.ABC):
    """Abstract class representing any graphical model."""

    def __init__(self, num_variables: int, domain: Domain):
        self.num_variables = num_variables
        self._default_domain = domain
        self._vars = dict()

    def get_variable(self, idx: int):
        if not 0 <= idx < self.num_variables:
            raise IndexError(
                "index %d is out of bounds for random vector of size %d" % (
                    idx, self.num_variables))
        if idx not in self._vars:
            v = Variable(self, idx, self._default_domain)
            self._vars[idx] = v
        return self._vars[idx]

    def get_variables(self) -> Iterable[Variable]:
        return [self.get_variable(i) for i in range(self.num_variables)]

    def __getitem__(self, idx: int) -> Variable:
        return self.get_variable(idx)

    @abc.abstractmethod
    def add_factor(self, factor: Factor):
        pass

    def __imul__(self, other: Factor):
        self.add_factor(other)
        return self

    def __len__(self):
        return self.num_variables

    @abc.abstractmethod
    def infer(self, algorithm='auto', **kwargs):
        pass

    @abc.abstractmethod
    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        pass

    def sample(self, num_samples: int, algorithm='auto',
               **kwargs) -> np.ndarray:
        pass

    def fit(self, data: np.ndarray, algorithm='auto', **kwargs):
        pass

    @abc.abstractmethod
    def get_factors(self) -> Iterable[Factor]:
        pass

    def get_factor_graph(self) -> Tuple[nx.Graph, Dict[int, str]]:
        """Builds factor graph for the model

        Factor graph is a bipartite graph with variables in one part and
        factors in other graph. Edge denotes that factor depends on variable.
        """
        factors = list(self.get_factors())
        var_labels = [v.name for v in self.get_variables()]
        fact_labels = [f.get_name() for f in factors]
        labels = var_labels + fact_labels
        labels = {i: labels[i] for i in range(len(labels))}

        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_variables), bipartite=0)
        graph.add_nodes_from(
            range(self.num_variables, self.num_variables + len(factors)),
            bipartite=1)
        for factor_id in range(len(factors)):
            for var_id in factors[factor_id].var_idx:
                graph.add_edge(var_id, self.num_variables + factor_id)
        return graph, labels

    def draw_factor_graph(self, ax):
        graph, labels = self.get_factor_graph()
        top = nx.bipartite.sets(graph)[0]
        vc = self.num_variables
        fc = len(nx.bipartite.sets(graph)[1])
        pos = nx.bipartite_layout(graph, top)
        nx.draw_networkx(graph, pos, ax, labels=labels, node_shape='o',
                         nodelist=list(range(vc)),
                         node_color='#ffaaaa')
        # Draw factors in another color.
        nx.draw_networkx(graph, pos, ax, labels=labels,
                         nodelist=list(range(vc, vc + fc)),
                         node_shape='s',
                         edgelist=[],
                         node_color='lightgreen')

    def evaluate(self, x: np.ndarray) -> float:
        """Returns value of non-normalized pdf in point.

        In other words, just substitutes values into factors and multiplies
        them.
        """
        assert x.shape == (self.num_variables,)
        result = 1.0
        for factor in self.get_factors():
            result *= factor.value(x[factor.var_idx])
        return result
