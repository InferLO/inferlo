# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

from inferlo.base.domain import DiscreteDomain
from inferlo.base.graph_model import GraphModel
from inferlo.base.variable import Variable

if TYPE_CHECKING:
    from inferlo.base.factors.factor import Factor
    from inferlo.base.factors.discrete_factor import DiscreteFactor


class DiscreteModel(GraphModel):
    """Discrete Graphical model in the most general form.

    Explicitly specified by lists of variables and factors. All variables are discrete.
    """

    def __init__(self, variables: List[Variable]):
        super().__init__()
        self.variables = variables
        self.factors = []  # type: List[DiscreteFactor]

    @staticmethod
    def create(num_variables: int, domain_size: int):
        domain = DiscreteDomain.range(domain_size)
        variables = [Variable(idx, domain) for idx in range(num_variables)]
        return DiscreteModel(variables)

    def add_factor(self, factor: Factor):
        """Adds factor."""
        assert factor.model == self
        self.factors.append(factor)

    def get_factors(self) -> Iterable[Factor]:
        """Returns all factors."""
        return self.factors

    @staticmethod
    def from_model(model: GraphModel):
        """Creates copy of a given model."""
        new_model = DiscreteModel(model.variables)
        for old_factor in model.get_factors():
            new_model.factors.append(old_factor.clone(new_model))
        return new_model

    def copy(self):
        """Makes a copy of itself."""
        return DiscreteModel.from_model(self)
