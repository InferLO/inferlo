from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from inferlo.base.domain import RealDomain
from inferlo.base.graph_model import GraphModel

if TYPE_CHECKING:
    from inferlo.base.factors.factor import Factor


class GenericGraphModel(GraphModel):
    """Graphical base in the most general form.

    Explicitly specified by factors.
    """

    def __init__(self, num_variables: int, domain=None):
        if domain is None:
            domain = RealDomain()
        super().__init__(num_variables, domain)
        self.factors = []

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
