# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import random
from copy import copy
from typing import List

import numpy as np

from .factor import Factor, product_over_, logsumexp
from .graphical_model import GraphicalModel
from ... import InferenceResult
from ...base.inference_result import marg_probs_to_array


class BucketElimination:
    """Bucket elimination algorithm."""

    def __init__(self, model: GraphicalModel):
        self.model = model.copy()

    def run(self, elimination_order_method="random", **kwargs) -> float:
        """Runs the algorithm, returns log(Z)."""
        if elimination_order_method == "random":
            elimination_order = copy(self.model.variables)
            random.shuffle(elimination_order)
        elif elimination_order_method == "not_random":
            elimination_order = copy(self.model.variables)
        elif elimination_order_method == "given":
            elimination_order = kwargs["elimination_order"]

        eliminated_model = self._eliminate_variables(self.model,
                                                     elimination_order)
        Z = Factor.scalar(1.0)
        for fac in eliminated_model.factors:
            Z = Z * fac
        return Z.log_values

    def run_bt(self) -> InferenceResult:
        """Runs the algorithm with binary tree order (TODO: explain)."""
        self._final_factors = dict()
        self._run_bt_rec(self.model, self.model.variables)

        log_z = 0
        marg_probs = []
        for variable in self.model.variables:
            log_values = self._final_factors[variable].log_values
            log_z = logsumexp(log_values)
            marg_probs.append(np.exp(log_values - log_z))
        return InferenceResult(log_pf=log_z,
                               marg_prob=marg_probs_to_array(marg_probs))

    def _run_bt_rec(self,
                    model: GraphicalModel,
                    remaining_variables: List[str]):
        """Recursive step of BE with Binary Tree order."""
        assert len(remaining_variables) >= 1
        if len(remaining_variables) == 1:
            final_factor = Factor.scalar(1.0)
            for fac in model.factors:
                final_factor = final_factor * fac
            self._final_factors[remaining_variables[0]] = final_factor
        else:
            n = len(remaining_variables) // 2
            vars1 = remaining_variables[:n]
            vars2 = remaining_variables[n:]
            model1 = self._eliminate_variables(model, vars1)
            self._run_bt_rec(model1, vars2)
            model2 = self._eliminate_variables(model, vars2)
            self._run_bt_rec(model2, vars1)

    def _eliminate_variables(self,
                             model: GraphicalModel,
                             elimination_order: List[str]):
        model = model.copy()
        for var in elimination_order:
            model.contract_variable(var)
        return model

    def get_marginal_factor(self,
                            elimination_order_method="random",
                            **kwargs) -> Factor:
        """Returns marginal factor."""
        if elimination_order_method == "random":
            elimination_order = copy(self.model.variables)
            random.shuffle(elimination_order)
        elif elimination_order_method == "not_random":
            elimination_order = copy(self.model.variables)
        elif elimination_order_method == "given":
            elimination_order = kwargs["elimination_order"]

        if "exception_variables" in kwargs:
            exception_variables = kwargs["exception_variables"]
        else:
            exception_variables = []

        eliminated_model = self.model.copy()
        max_ibound = 0
        for var in elimination_order:
            if var not in exception_variables:
                max_ibound = max(max_ibound, get_bucket_size(
                    [fac for fac in eliminated_model.get_adj_factors(var)]), )
                eliminated_model.contract_variable(var)

        final_factor = product_over_(*eliminated_model.factors)
        if "exception_variables" in kwargs:
            final_factor.transpose_by_(exception_variables)

        return final_factor


def get_bucket_size(bucket: List[Factor]):
    """Counts variables referenced by factors in a bucket."""
    s = set()
    for fac in bucket:
        s = s.union(set(fac.variables))
    return len(s)
