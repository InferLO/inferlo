# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from dataclasses import dataclass

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
