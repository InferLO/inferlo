import abc

import numpy as np


class Domain(abc.ABC):
    """Describes set of values which a variable can take."""

    def size(self):
        return np.inf

    def is_discrete(self):
        return False


class DiscreteDomain(Domain):
    """Domain consisting of finite set of numbers."""

    def __init__(self, values):
        self.values = np.array(values)
        self._rev_idx = {values[i]: i for i in range(len(values))}

    def size(self):
        return len(self.values)

    def get_value_index(self, val):
        if val in self._rev_idx:
            return self._rev_idx[val]
        raise ValueError("Value %f is not in domain." % val)

    def is_discrete(self):
        return True

    @staticmethod
    def binary():
        return DiscreteDomain([0, 1])


class RealDomain(Domain):
    """All real values."""
