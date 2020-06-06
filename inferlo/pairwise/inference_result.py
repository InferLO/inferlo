from __future__ import annotations


class InferenceResult:
    """Inference result for PairwiseFiniteModel.

    Inference result consists of log Z (logarithm of partition function) and
    marginal probabilities.

    Marginal probabilities specify for each variable and for each value what is
    the probability of this variable to assume given value when other variables
    can take any value.
    """

    def __init__(self, log_pf, marg_prob):
        self.log_pf = log_pf
        self.marg_prob = marg_prob

    def __repr__(self):
        return str({"log_pf": self.log_pf,
                    "marg_prob": self.marg_prob})
