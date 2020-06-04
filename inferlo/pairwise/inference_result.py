from __future__ import annotations

import numpy as np


class InferenceResult:
    """Inference result for Potts Model.

    Inference result consists of log Z (logarithm of partition function) and
        marginal probabilities.
    Marginal probabilities specify for each vertex and for each value what is
        the probability of this vertex to assume the given value.
    """

    def __init__(self, log_pf, marg_prob):
        self.log_pf = log_pf
        self.marg_prob = marg_prob

    def __repr__(self):
        return str({"log_pf": self.log_pf,
                    "marg_prob": self.marg_prob})
