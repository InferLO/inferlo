# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import abc

import numpy as np


class Domain(abc.ABC):
    """Describes set of values which a variable can take."""

    def size(self):
        """Cardinality of the domain."""
        return np.inf

    def is_discrete(self):
        """Whether the domain is discrete."""
        return False


class DiscreteDomain(Domain):
    """Domain consisting of finite set of numbers."""

    def __init__(self, values):
        self.values = np.array(values)
        self._rev_idx = {values[i]: i for i in range(len(values))}

    def size(self):
        """Cardinality of the domain."""
        return len(self.values)

    def get_value_index(self, val):
        """Gets number of given value in list of al values in domain."""
        if val in self._rev_idx:
            return self._rev_idx[val]
        raise ValueError("Value %f is not in domain." % val)

    def is_discrete(self):
        """Whether the domain is discrete."""
        return True

    @staticmethod
    def binary():
        """Creates domain with two values 0 and 1."""
        return DiscreteDomain([0, 1])

    @staticmethod
    def range(n: int):
        """Creates domain with integer values from 0 to n-1."""
        return DiscreteDomain(range(n))

    def __repr__(self):
        return 'DiscreteDomain([%s])' % ','.join([str(v) for v in self.values])


class RealDomain(Domain):
    """Domain consisting of all real values."""

    def __repr__(self):
        return 'RealDomain'
