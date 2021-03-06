# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from typing import Callable, List

import numpy as np

from .factor import logsumexp
from .graphical_model import GraphicalModel
from ... import InferenceResult
from ...base.inference_result import marg_probs_to_array


class BinaryTreeElimination:
    """Meta-algorithm for efficient computing of marginal probabilities using an
     elimination-based algorithm.
     """

    def __init__(
            self,
            eliminator: Callable[[GraphicalModel, List[str]], GraphicalModel]):
        self.eliminator = eliminator

    def run(self, model: GraphicalModel) -> InferenceResult:
        """Finds all marginal probabilities by running Bucket Elimination on
        Binary Tree.
        """
        self._final_factors = dict()
        self._run_recursive(model, model.variables)

        log_z = 0
        marg_probs = []
        for variable in model.variables:
            log_values = self._final_factors[variable].log_values
            log_z = logsumexp(log_values)
            marg_probs.append(np.exp(log_values - log_z))
        return InferenceResult(log_pf=log_z,
                               marg_prob=marg_probs_to_array(marg_probs))

    def _run_recursive(self,
                       model: GraphicalModel,
                       remaining_variables: List[str]):
        """Recursive step of BE on Binary Tree."""
        assert len(remaining_variables) >= 1
        if len(remaining_variables) == 1:
            final_factor = model.factors[0]
            for fac in model.factors[1:]:
                final_factor = final_factor * fac
            self._final_factors[remaining_variables[0]] = final_factor
        else:
            n = len(remaining_variables) // 2
            vars1 = remaining_variables[:n]
            vars2 = remaining_variables[n:]
            model1 = self.eliminator(model, vars1)
            self._run_recursive(model1, vars2)
            model2 = self.eliminator(model, vars2)
            self._run_recursive(model2, vars1)
