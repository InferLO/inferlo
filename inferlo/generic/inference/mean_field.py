# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import warnings
from copy import copy

import numpy as np

from .factor import Factor, product_over_, entropy
from .graphical_model import GraphicalModel


def _default_message_name(prefix="_M"):
    _default_message_name.cnt += 1
    return prefix + str(_default_message_name.cnt)


_default_message_name.cnt = 0


class MeanField:
    """Mean Field algorithm."""

    def __init__(self, model: GraphicalModel, **kwargs):
        self.model = model.copy()

        mean_field_init_method = kwargs.get("mean_field_init_method")
        if mean_field_init_method == "random":
            init_np_func = np.ones
        elif mean_field_init_method == "uniform":
            init_np_func = np.random.random
        else:
            init_np_func = np.ones

        self.mean_fields = {}
        for var in self.model.variables:
            self.mean_fields[var] = Factor.initialize_with_(
                _default_message_name(),
                [var],
                init_np_func,
                self.model.get_cardinality_for_(var))
            self.mean_fields[var].normalize()

    def run(self, max_iter=1000, converge_thr=1e-2):
        """Runs the algorithm, returns log(Z)."""
        converged = False
        for _ in range(max_iter):
            old_mean_field = {
                var: copy(
                    self.mean_fields[var]) for var in self.model.variables}
            self._update_mean_fields()
            if self._is_converged(
                    self.mean_fields,
                    old_mean_field,
                    converge_thr):
                converged = True
                break

        if not converged:
            warnings.warn(
                "Mean field did not converge after %d iterations." % max_iter)

        return self._get_log_z()

    def _get_log_z(self):
        logZ = 0
        for var in self.model.variables:
            logZ += entropy(self.mean_fields[var])

        for fac in self.model.factors:
            m = product_over_(*[self.mean_fields[var]
                                for var in fac.variables])
            index_to_keep = m.values != 0
            logZ += np.sum(m.values[index_to_keep] *
                           fac.log_values[index_to_keep])

        return logZ

    def _update_mean_fields(self):
        variable_order = np.random.permutation(self.model.variables)
        for var in variable_order:
            next_mean_field = Factor.full_like_(self.mean_fields[var], 0.0)
            for fac in self.model.get_adj_factors(var):
                tmp = Factor(
                    name="tmp",
                    variables=[var],
                    values=np.ones(self.model.get_cardinality_for_(var)),
                )
                tmp = product_over_(
                    tmp, *[self.mean_fields[var1] for var1 in fac.variables if
                           var1 != var]
                )
                tmp.transpose_by_(fac.variables)
                tmp.log_values = fac.log_values * tmp.values
                next_mean_field = next_mean_field + tmp.marginalize_except_(
                    [var], inplace=False)

            self.mean_fields[var] = next_mean_field.exp(
                inplace=False).normalize(inplace=False)
            self.mean_fields[var].log_values = np.nan_to_num(
                self.mean_fields[var].log_values)

    def _is_converged(self, mean_field, old_mean_field, converge_thr):
        for var in self.model.variables:
            if np.sum(
                    np.abs(
                        mean_field[var].values -
                        old_mean_field[var].values)) > converge_thr:
                return False
        return True
