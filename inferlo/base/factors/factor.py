from __future__ import annotations

import abc
import copy
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from inferlo.base import GraphModel


class Factor(abc.ABC):
    """Abstract factor - function of one or more random variables."""

    def __init__(self, model: GraphModel, var_idx: List[int]):
        """
        :param model: Graphical base this factor belongs to.
        :param var_idx: Indices of random variables in the base, on which this
            factor depends.
        """
        self.model = model
        assert len(set(var_idx)) == len(var_idx)  # No duplicate indices.
        self.var_idx = var_idx

    @abc.abstractmethod
    def value(self, values: List[float]):
        pass

    def is_discrete(self):
        return all([self.model[i].domain.is_discrete() for i in
                    self.var_idx])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '%s(%s)' % (self.get_name(), ','.join(
            self.model[i].name for i in self.var_idx))

    def get_name(self):
        return self.name if hasattr(self, 'name') else 'f'

    def clone(self, new_model: GraphModel):
        """Creates the same factor, but pointing to new model."""
        new_factor = copy.copy(self)
        new_factor.model = new_model
        new_factor.var_idx = copy.copy(self.var_idx)
        return new_factor
