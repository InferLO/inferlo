from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

from inferlo.base.factors.factor import Factor

if TYPE_CHECKING:
    from inferlo.base import GraphModel

# TODO: make immutable.


class DiscreteFactor(Factor):
    """A factor of several discrete variables."""

    def __init__(self, model: GraphModel, var_idx: List[int],
                 values: np.ndarray):
        super().__init__(model, var_idx)

        values = np.array(values)
        expected_shape = [self.model[i].domain.size() for i in
                          self.var_idx]
        assert list(values.shape) == expected_shape
        assert np.min(values) >= 0, "Factors should be non-negative."
        self.values = values

    def value(self, x: List[float]):
        assert len(x) == len(self.var_idx)
        ans = self.values
        for i in range(len(x)):
            ans = ans[self.model[self.var_idx[i]].domain.get_value_index(x[i])]
        return ans

    @staticmethod
    def from_factor(factor: Factor) -> DiscreteFactor:
        """Converts arbitrary factor to DiscreteFactor.

        Returns `None` if some variables of the factor are not discrete.
        """
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
            flat_values[flat_idx] = factor.value(args)
        new_factor = DiscreteFactor(factor.model, factor.var_idx,
                                    flat_values.reshape(dims))
        new_factor.name = factor.get_name()
        return new_factor
