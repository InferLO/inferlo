# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import copy
from typing import List, TYPE_CHECKING

import numpy as np

from inferlo.base.factors.factor import Factor
from inferlo.utils.special_functions import logsumexp

if TYPE_CHECKING:
    from inferlo.base import GraphModel


class OldDiscreteFactor(Factor):
    """A factor of several discrete variables."""

    # TODO: make immutable.

    def __init__(self,
                 model: GraphModel,
                 var_idx: List[int],
                 log_values: np.ndarray):
        """
        :param model: Model to which factor belongs (needed to resolve variable indices).
        :param var_idx: Indices of variables on which this factor depends.
        :param log_values: Logarithms of values of the factor.
        """
        super().__init__(model, var_idx)

        expected_shape = [self.model[i].domain.size() for i in self.var_idx]
        assert list(log_values.shape) == expected_shape, (
                "Got values of shape %s, but variables imply %s." %
                (log_values.shape, expected_shape))
        self.log_values = log_values

    @property
    def values(self) -> np.array:
        """Values of this factor for all possible combinations of variables."""
        return np.exp(self.log_values)

    def evaluate(self, x: List[float]):
        """Returns value of the factor if variables take given values."""
        assert len(x) == len(self.var_idx)
        idx = tuple(self.model[self.var_idx[i]].domain.get_value_index(x[i])
                    for i in range(len(x)))
        return np.exp(self.log_values[idx])

    @staticmethod
    def from_factor(factor: Factor) -> OldDiscreteFactor:
        """Converts arbitrary factor to DiscreteFactor."""
        if factor.__class__.__name__ == 'DiscreteFactor':
            return factor

        assert factor.is_discrete()
        # Dimensions of the tensor holding all the values.
        vals = [factor.model[i].domain.values for i in factor.var_idx]
        rank = len(vals)
        dims = [len(v) for v in vals]
        values_count = int(np.prod(dims))
        flat_values = np.zeros(values_count)
        prod = [np.prod(dims[i + 1:]) for i in range(rank)]
        prod = np.array(prod, dtype=np.int32)
        for flat_idx in range(values_count):
            args = []
            for i in range(rank):
                args.append(vals[i][(flat_idx // prod[i]) % dims[i]])
            flat_values[flat_idx] = factor.evaluate(args)
        new_factor = OldDiscreteFactor.from_values(factor.model, factor.var_idx,
                                                   flat_values.reshape(dims))
        new_factor.name = factor.get_name()
        return new_factor

    @staticmethod
    def from_values(model: GraphModel,
                    var_idx: List[int],
                    values: np.ndarray):
        assert np.min(values) >= 0, "Factor values must be non-negative."
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.log(values)
        return OldDiscreteFactor(model, var_idx, log_values)

    @staticmethod
    def from_flat_values(model: GraphModel,
                         var_idx: List[int],
                         values_flat: np.ndarray):
        """Creates factor specified by list of values.

        :param model: GM to which this factor belongs.
        :param var_idx: Indices of variables.
        :param values_flat: 1D list of values. Last variable is "least significant".
        """
        shape = [model[i].domain.size() for i in var_idx]
        values = np.array(values_flat).reshape(shape)
        return OldDiscreteFactor.from_values(model, var_idx, values)

    def copy(self):
        return OldDiscreteFactor(self.model, self.var_idx, self.log_values)

    def clone(self, new_model: GraphModel):
        return OldDiscreteFactor(new_model, copy.copy(self.var_idx), self.log_values)

    def restrict(self, variable_id, fixed_value) -> OldDiscreteFactor:
        """Fixes value of one variable.

        Returns new factor which is equivalent to original factor, but with
        value of one variable fixed.
        """
        assert variable_id in self.var_idx
        if len(self.var_idx) == 1:
            # TODO: remove this hack - this case should return scalar factor.
            assert self.var_idx[0] == variable_id
            new_values = np.ones_like(self.log_values) * -np.inf
            new_values[fixed_value] = self.log_values[fixed_value]
            return OldDiscreteFactor(self.model, [variable_id], new_values)
        new_var_idx = [i for i in self.var_idx if i != variable_id]
        idx = tuple(fixed_value if i == variable_id else slice(None) for i in
                    self.var_idx)
        new_log_values = self.log_values[idx]
        return OldDiscreteFactor(self.model, new_var_idx, new_log_values)

    def marginalize(self, var_idx: List[int] = None, operator="sum", **kwargs) -> OldDiscreteFactor:
        """Marginalizes factor over given variables.

        :param var_idx: Indices of variables, over which to marginalize.
        :param operator: sum, weighted_sum, min or max.
        """
        vars_remove = set(var_idx) if var_idx is not None else set(self.var_idx)
        new_vars_idx = []
        axis_drop = []
        for i, var_id in enumerate(self.var_idx):
            if var_id not in vars_remove:
                new_vars_idx.append(var_id)
            else:
                axis_drop.append(i)
        axis_drop = tuple(axis_drop)

        if operator == "sum":
            new_log_values = logsumexp(self.log_values, axis=axis_drop)
        elif operator == "weighted_sum":
            w = kwargs["weight"]
            if w != 0:
                new_log_values = logsumexp(self.log_values / w, axis=axis_drop) * w
            else:
                new_log_values = np.amax(self.log_values, axis=axis_drop)
        elif operator == "max":
            new_log_values = np.amax(self.log_values, axis=axis_drop)
        elif operator == "min":
            new_log_values = np.amin(self.log_values, axis=axis_drop)
        else:
            raise ValueError("Unsupported operator.")

        return OldDiscreteFactor(self.model, new_vars_idx, new_log_values)
