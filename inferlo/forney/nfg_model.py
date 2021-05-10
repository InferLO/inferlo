# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import networkx as nx
import numpy as np

from inferlo.base import GraphModel, DiscreteFactor, FunctionFactor
from inferlo.forney.edge_elimination import infer_edge_elimination

if TYPE_CHECKING:
    from inferlo import Domain, Factor


class NormalFactorGraphModel(GraphModel):
    """Normal Factor Graph model.

    Also known as Forney-style or Edge-Variable graphical model.

    Variables correspond to edges of a graph, factors correspond to vertices of
    a graph.

    Every variable appears in exactly two factors.
    """

    def __init__(self, num_variables: int, domain: Domain):
        """Initializes Normal Factor Graph model.

        :param num_variables: Number of variables in the model.
        :param domain: Default domain of each variable.
        """
        super().__init__(num_variables, domain)

        # self.edges[i] -- two indices of factors, depending on i-th variable.
        self.edges = [[] for _ in range(num_variables)]

        self.factors = []
        self.built = False

    def add_factor(self, factor: Factor):
        """Add a factor."""
        assert not self.built, "Model is built."
        assert factor.model == self
        factor_idx = len(self.factors)
        self.factors.append(factor)

        for var_id in factor.var_idx:
            assert len(self.edges[var_id]) < 2, (
                "One variable can't belong to more than 2 factors.")

        for var_id in factor.var_idx:
            self.edges[var_id].append(factor_idx)

    def build(self):
        """Validates the model and makes it immutable."""
        for i in range(self.num_variables):
            assert len(self.edges[i]) == 2, (
                "Can't build Forney-style model. Variable %d appears in "
                "%d factors, but must appear in exactly 2 factors." % (
                    i, len(self.edges[i])))

        self.built = True

    def infer(self, algorithm='auto', **kwargs):
        """Calculates partition function.

        Available algorithms
            * ``auto`` - Automatic.
            * ``edge_elimination`` - Edge elimination.

        :param algorithm: Which algorithm to use. String.
        :return: Partition function.
        """
        if algorithm == 'auto':
            return infer_edge_elimination(self)
        elif algorithm == 'edge_elimination':
            return infer_edge_elimination(self)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)

    def max_likelihood(self, algorithm, **kwargs) -> np.ndarray:
        raise NotImplemented

    def get_factors(self) -> Iterable[Factor]:
        return self.factors

    @staticmethod
    def from_model(original_model: GraphModel):
        """Constructs NFG model which is equivalent to given model.

        If necessary, clones variables and adds factors.

        Guarantees that first N variables in new model will correspond to
        variables in original model, where N is number of variables in original
        model.
        """
        old_factors = list(original_model.get_factors())

        # For every variable, find all factors in which it appears.
        var_to_factors = [[] for _ in range(original_model.num_variables)]
        for factor_id in range(len(old_factors)):
            for var_id in old_factors[factor_id].var_idx:
                var_to_factors[var_id].append(factor_id)

        # Calculate in advance how many variables will the model have.
        # For every factor of size k >= 2 we will add (k-1) new variables.
        new_vars_count = 0
        for i in range(original_model.num_variables):
            factors_num = len(var_to_factors[i])
            if factors_num <= 2:
                new_vars_count += 1
            else:
                new_vars_count += factors_num

        # Create new model.
        new_model = NormalFactorGraphModel(new_vars_count,
                                           original_model._default_domain)

        # Clone factors, change reference to the model.
        new_factors = [f.clone(new_model) for f in old_factors]

        # Helper function to create trivial constant factor.
        def create_constant_factor(var_id):
            const_factor = FunctionFactor(new_model, [var_id], lambda x: 1)
            const_factor.name = '1'
            if original_model[var_id].domain.is_discrete():
                return DiscreteFactor.from_factor(const_factor)
            else:
                return const_factor

        # Helper function to create Kroeneker delta factor.
        def create_delta_factor(var_ids):
            delta_factor = FunctionFactor(new_model, var_ids,
                                          lambda x: x[1:] == x[:-1])
            delta_factor.name = '='
            if original_model[var_ids[0]].domain.is_discrete():
                return DiscreteFactor.from_factor(delta_factor)
            else:
                return delta_factor

        # Counter of used variables.
        used_variables_count = original_model.num_variables

        for i in range(original_model.num_variables):
            k = len(var_to_factors[i])

            # For every variable, which appears only in one factor - copy this
            # variable (with the same index) and add a trivial factor
            # (constant 1) on another end of the edge.
            if len(var_to_factors[i]) == 1:
                new_model[i].domain = original_model[i].domain
                new_model[i].name = original_model[i].name
                new_factors.append(create_constant_factor(i))

            # For every variable, which appears in exactly two factors - copy
            # this variable and don't add new factors.
            elif len(var_to_factors[i]) == 2:
                new_model[i].domain = original_model[i].domain
                new_model[i].name = original_model[i].name

            # For ever variable appearing in k > 2 factors - create k copies of
            # it, remap factors such that every factor depends on a unique copy
            # of it and add one new delta factor, which requires all these
            # variables to be equal.
            else:
                var_idx = [i]
                for _ in range(k - 1):
                    var_idx.append(used_variables_count)
                    used_variables_count += 1

                # Copy domains.
                for j in var_idx:
                    new_model[j].domain = original_model[i].domain

                for j in range(k):
                    new_name = '%s_%d' % (original_model[i].name, j)
                    new_model[var_idx[j]].name = new_name

                # Add delta factor.
                new_factors.append(create_delta_factor(var_idx))

                # Remap indices.
                # Can skip first factor, which still depends on old variable.
                for j in range(1, k):
                    new_var_id = var_idx[j]
                    factor_id = var_to_factors[i][j]
                    pos = new_factors[factor_id].var_idx.index(i)
                    new_factors[factor_id].var_idx[pos] = new_var_id

        # Now we can add all new factors to the model.
        for factor in new_factors:
            new_model.add_factor(factor)
        new_model.build()
        return new_model

    def get_edge_variable_graph(self) -> nx.Graph:
        """Returns edge-variable graph."""
        self.check_built()
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.factors)))
        graph.add_edges_from(self.edges)
        return graph

    def draw_edge_variable_graph(self, ax):
        """Draws edge-variable graph."""
        graph = self.get_edge_variable_graph()
        pos = nx.kamada_kawai_layout(graph)
        node_labels = {i: self.factors[i].get_name() for i in
                       range(len(self.factors))}
        edge_labels = {(*self.edges[i],): self[i].name for i in
                       range(self.num_variables)}
        nx.draw_networkx(graph, pos, ax,
                         labels=node_labels,
                         node_shape='s',
                         node_color='lightgreen',
                         edge_color='red')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    def check_built(self):
        """Checks that model is built."""
        assert self.built, "Model is not built, call build()."
