# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import abc
import itertools
from typing import TYPE_CHECKING, Iterable, Tuple, Dict, List

import networkx as nx
import numpy as np

from inferlo.base.factors import FunctionFactor
from inferlo.base.variable import Variable

if TYPE_CHECKING:
    from inferlo.base import Domain, Factor


class GraphModel(abc.ABC):
    """Abstract class representing any graphical model."""

    def __init__(self, num_variables: int, domain: Domain):
        """
        :param num_variables: Number of variables in the model.
        :param domain: Default domain of each variable.
        """
        self.num_variables = num_variables
        self._default_domain = domain
        self._vars = dict()

    def get_variable(self, idx: int) -> Variable:
        """Returns variable by its index."""
        if not 0 <= idx < self.num_variables:
            raise IndexError(
                "index %d is out of bounds for random vector of size %d" % (
                    idx, self.num_variables))
        if idx not in self._vars:
            v = Variable(self, idx, self._default_domain)
            self._vars[idx] = v
        return self._vars[idx]

    def get_variables(self) -> List[Variable]:
        """Returns all variables."""
        return [self.get_variable(i) for i in range(self.num_variables)]

    def __getitem__(self, idx: int) -> Variable:
        return self.get_variable(idx)

    @abc.abstractmethod
    def add_factor(self, factor: Factor):
        """Adds a factor to the model."""

    def __imul__(self, other: Factor):
        self.add_factor(other)
        return self

    def __len__(self):
        return self.num_variables

    @abc.abstractmethod
    def infer(self, algorithm='auto', **kwargs):
        """Performs inference."""

    @abc.abstractmethod
    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        """Finds the most probable state."""

    def sample(self, num_samples: int, algorithm='auto',
               **kwargs) -> np.ndarray:
        """Generates samples."""

    @abc.abstractmethod
    def get_factors(self) -> Iterable[Factor]:
        """Returns all factors."""

    def get_symbolic_variables(self) -> List[FunctionFactor]:
        """Prepares variables for usage in expressions.

        Returns lists of trivial ``FunctionFactor`` s, each of them
        representing a factor on one variable with identity function.
        They can  be used in mathematical expressions, which will result in
        another ``FunctionFactor``.
        """
        return [FunctionFactor(self, [i], lambda x: x[0]) for i in
                range(self.num_variables)]

    def get_factor_graph(self) -> Tuple[nx.Graph, Dict[int, str]]:
        """Builds factor graph for the model.

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
        """Draws the factor graph."""
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
        x = np.array(x)
        assert x.shape == (self.num_variables,)
        result = 1.0
        for factor in self.get_factors():
            result *= factor.value(x[factor.var_idx])
        return result

    def part_func_bruteforce(model):
        """Evaluates partition function in very inefficient way."""
        part_func = 0
        for x in itertools.product(
                *(v.domain.values for v in model.get_variables())):
            part_func += model.evaluate(np.array(x))
        return part_func

    def max_likelihood_bruteforce(model):
        """Evaluates most likely state in a  very inefficient way."""
        best_state = None
        best_prob = 0.0
        for x in itertools.product(
                *(v.domain.values for v in model.get_variables())):
            prob = model.evaluate(np.array(x))
            if prob >= best_prob:
                best_state = x
                best_prob = prob
        return best_state

    def get_max_domain_size(self):
        """Returns the biggest domain size over all variables."""
        return max([var.domain.size() for var in self.get_variables()])
