# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import math
from typing import List, Callable, Any, TYPE_CHECKING

from inferlo.base.factors.factor import Factor

if TYPE_CHECKING:
    from inferlo.base.graph_model import GraphModel


class FunctionFactor(Factor):
    """A factor given explicitly by function."""

    def __init__(self, model: GraphModel,
                 var_idx: List[int],
                 func: Callable[[List[float]], float]):
        """Create function factor.

        :param model: Graphical model this factor belongs to.
        :param var_idx: Indices of variables in the model, on which this
            factor depends.
        :param func: Function of this factor (as Python callable).
        """
        super().__init__(model, var_idx)
        self.func = func

    def value(self, values: List[float]):
        return self.func(values)

    # Numeric operations on factors.
    @staticmethod
    def combine_factors(factor1: FunctionFactor, factor2: FunctionFactor,
                        func: Callable[
                            [float, float], float]) -> FunctionFactor:
        """Returns a factor which is a function of other 2 factors."""
        assert factor1.model == factor2.model

        # List of variable indices in the new factor.
        new_idx = list(set(factor1.var_idx + factor2.var_idx))
        # Maps position in new factor to variable index.
        new_idx_rev = {new_idx[i]: i for i in range(len(new_idx))}
        # Maps position in args list of first factor to position of the same
        # variable in the new factor.
        idx1 = [new_idx_rev[i] for i in factor1.var_idx]
        # Maps position in args list of second factor to position of the same
        # variable in the new factor.
        idx2 = [new_idx_rev[i] for i in factor2.var_idx]

        def new_func(all_args: List[float]) -> float:
            first_factor_args = [all_args[i] for i in idx1]
            second_factor_args = [all_args[i] for i in idx2]
            return func(factor1.func(first_factor_args),
                        factor2.func(second_factor_args))

        return FunctionFactor(factor1.model, new_idx, new_func)

    def apply_function(self, func: Callable[[float], float]):
        """Returns factor func(g(x)), where g(x) is given factor."""
        return FunctionFactor(self.model, self.var_idx,
                              lambda x: func(self.func(x)))

    def combine_with(self, other: Any, func: Callable[[float, float], float]):
        """Returns factor func(g(x), other), where g(x) is given factor.

        `other` may be a number, variable or another factor.
        """
        if isinstance(other, (int, float)):
            return self.apply_function(lambda x: func(x, other))
        elif other.__class__.__name__ == 'FunctionFactor':
            return FunctionFactor.combine_factors(self, other, func)
        else:
            raise TypeError(
                'Cannot combine FunctionFactor with %s' % type(other))

    def __add__(self, other: Any):
        return self.combine_with(other, lambda x, y: x + y)

    def __radd__(self, other: Any):
        return self.combine_with(other, lambda x, y: x + y)

    def __sub__(self, other: Any):
        return self.combine_with(other, lambda x, y: x - y)

    def __rsub__(self, other: Any):
        return self.combine_with(other, lambda x, y: y - x)

    def __mul__(self, other: Any):
        return self.combine_with(other, lambda x, y: x * y)

    def __rmul__(self, other: Any):
        return self.combine_with(other, lambda x, y: x * y)

    def __truediv__(self, other: Any):
        return self.combine_with(other, lambda x, y: x / y)

    def __rtruediv__(self, other: Any):
        return self.combine_with(other, lambda x, y: y / x)

    def __pow__(self, other):
        return self.combine_with(other, lambda x, y: x ** y)

    def __rpow__(self, other):
        return self.combine_with(other, lambda x, y: y ** x)

    def __neg__(self):
        return self.apply_function(lambda x: -x)

    def __abs__(self):
        return self.apply_function(lambda x: abs(x))

    def exp(self):
        """Exponent of this factor."""
        return self.apply_function(math.exp)

    def sin(self):
        """Sine of this factor."""
        return self.apply_function(math.sin)

    def cos(self):
        """Sine of this factor."""
        return self.apply_function(math.cos)
