# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo.base.inference_result import InferenceResult


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


def check_samples(*, samples: np.ndarray,
                  true_marg_probs: np.ndarray,
                  tol: float):
    """
    :param samples: Generated samples.
    :param true_marg_probs: True marginal probabilitites.
    :param tol: tolerance.
    """
    # Calculate empirical marginal probabilities.
    num_samples, gr_size = samples.shape
    emp_mp = np.zeros_like(true_marg_probs)
    for sample in samples:
        for i in range(gr_size):
            emp_mp[i, sample[i]] += 1
    emp_mp /= num_samples

    assert np.mean(np.square(emp_mp - true_marg_probs)) < tol
