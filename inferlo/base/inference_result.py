# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class InferenceResult:
    """Inference result for discrete graphical model.

    Inference result consists of log Z (logarithm of partition function) and
    marginal probabilities.

    Marginal probabilities specify for each variable and for each value what is
    the probability of this variable to assume given value when other variables
    can take any value.
    """
    log_pf: float
    marg_prob: np.array


def marg_probs_to_array(marg_probs: List[np.array]) -> np.array:
    """ Collects marginal probabilities to 2D np.array.

    If variables have different domains, pads absent values with zeroes.
    """
    num_vars = len(marg_probs)
    max_domain_size = max(len(mp) for mp in marg_probs)
    result = np.zeros((num_vars, max_domain_size), dtype=float)
    for i in range(num_vars):
        result[i, :len(marg_probs[i])] = marg_probs[i]
    return result
