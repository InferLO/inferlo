import numpy as np

from inferlo.pairwise import InferenceResult


def assert_results_close(res1: InferenceResult,
                         res2: InferenceResult,
                         log_pf_tol=1e-9,
                         mp_mse_tol=1e-9):
    """
    :param log_pf_tol: Absolute tolerance for log partition function.
    :param mp_mse_tol: Absolute tolerance for mean square error of marginal
        probabilities.
    """
    log_pf_diff = np.abs(res1.log_pf - res2.log_pf)
    assert log_pf_diff < log_pf_tol
    mp_mse = np.mean(np.square(res1.marg_prob - res2.marg_prob))
    assert mp_mse < mp_mse_tol
