# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import numpy as np

from inferlo import GenericGraphModel, DiscreteDomain, DiscreteFactor


class UaiReader():
    """Reads files in UAI format."""

    def __init__(self):
        self.pos = 0
        self.tokens = []

    def _read_file(self, path):
        with open(path, 'r') as file:
            self.tokens = ''.join(file.readlines()).split()
        self.pos = 0

    def _next_token(self):
        self.pos += 1
        return self.tokens[self.pos - 1]

    def _next_int(self):
        return int(self._next_token())

    def read_model(self, path) -> GenericGraphModel:
        """Reads Graphical model form a file in UAI format.

        Format description:
        http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html

        :param path: Local path to text file with GM in UAI format.
        :return: Graphical model read from the file.
        """
        # Read model type.
        self._read_file(path)
        model_type = self._next_token()
        assert model_type == 'MARKOV', 'Unsupported model type: %s' % model_type

        # Read variables' cardinalities and initialize the model.
        num_vars = int(self._next_int())
        model = GenericGraphModel(num_vars)
        for i in range(num_vars):
            model[i].domain = DiscreteDomain.range(int(self._next_token()))

        # Read which variables are in which factors.
        num_factors = self._next_int()
        var_idx_list = []
        for factor_id in range(num_factors):
            var_count = self._next_int()
            var_idx = [self._next_int() for _ in range(var_count)]
            var_idx_list.append(var_idx)

        # Read factors' values and add factors to the model.
        for factor_id in range(num_factors):
            vals_count = self._next_int()
            vals = np.array(
                [np.float64(self._next_token()) for _ in range(vals_count)])
            factor = DiscreteFactor.from_flat_values(model,
                                                     var_idx_list[factor_id],
                                                     vals)
            model.add_factor(factor)

        return model

    def read_marginals(self, path) -> np.array:
        """Reads true marginals form a file in UAI format.

        Format description:
        http://www.hlt.utdallas.edu/~vgogate/uai14-competition/resformat.html

        :param path: Local path to text file with true marginals in UAI format.
        :return: 2D numpy array. First dimension is number of variables,
          second is maximal domain size of a variable. ``result[i][j]`` is a
          probability that ``i``-th variable takes ``j``-th value (or 0 if this
          variable's domain size is less than ``j+1``).
        """
        self._read_file(path)
        assert self._next_token() == "MAR"

        vars_num = self._next_int()
        marginals = []
        for i in range(vars_num):
            domain_size = self._next_int()
            marginals.append(
                [np.float64(self._next_token()) for _ in range(domain_size)])
        max_domain_size = max([len(x) for x in marginals])
        ans = np.zeros((vars_num, max_domain_size))
        for i in range(vars_num):
            for j in range(len(marginals[i])):
                ans[i, j] = marginals[i][j]
        return ans
