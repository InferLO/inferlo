# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

import numpy as np

from inferlo.base.domain import RealDomain
from inferlo.base.graph_model import GraphModel

if TYPE_CHECKING:
    from inferlo.base.factors.factor import Factor


class GenericGraphModel(GraphModel):
    """Graphical model in the most general form.

    Explicitly specified by list of factors.
    """

    def __init__(self, num_variables: int, domain=None):
        if domain is None:
            domain = RealDomain()
        super().__init__(num_variables, domain)
        self.factors = []  # type: List[Factor]

    def add_factor(self, factor: Factor):
        """Adds factor."""
        assert factor.model == self
        self.factors.append(factor)

    def get_factors(self) -> Iterable[Factor]:
        """Returns all factors."""
        return self.factors

    def infer(self, algorithm='auto', **kwargs):
        """Performs inference."""
        raise NotImplemented

    def max_likelihood(self, algorithm='auto', **kwargs) -> np.ndarray:
        """Finds most probable state."""
        raise NotImplemented

    @staticmethod
    def from_model(model: GraphModel):
        """Creates copy of a given model."""
        n = model.num_variables
        new_model = GenericGraphModel(n)
        for i in range(n):
            new_model.get_variable(i).domain = model.get_variable(i).domain
        for old_factor in model.get_factors():
            new_model.factors.append(old_factor.clone(new_model))
        return new_model

    def copy(self):
        """Makes a copy of itself."""
        return GenericGraphModel.from_model(self)
