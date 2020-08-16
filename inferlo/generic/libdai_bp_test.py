# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np
import pytest

from inferlo.generic.libdai_bp import BP
from inferlo.interop import LibDaiInterop
from inferlo.testing import (assert_results_close, tree_potts_model,
                             grid_potts_model, random_generic_model)

libdai = LibDaiInterop()


# This test verifies that InferLO's algorithm is numerically equivalent to
# libDAI's BP algorithm.
@pytest.mark.skipif(not libdai.is_libdai_ready(),
                    reason="libDAI is not installed on the system.")
def test_libdai_bp_regression():
    model = random_generic_model(
        num_variables=20,
        num_factors=20,
        max_domain_size=4,
        max_factor_size=3)
    default_opts = {
        'tol': 1e-9,
        'logdomain': 0,
        'updates': 'SEQFIX',
        'verbose': 0}
    configs = [
        {**default_opts},
        {**default_opts, 'logdomain': 1},
        {**default_opts, 'updates': 'SEQMAX'},
        {**default_opts, 'updates': 'PARALL'},
        {**default_opts, 'updates': 'SEQRND'},
        {**default_opts, 'maxiter': 0},
        {**default_opts, 'maxiter': 1},
        {**default_opts, 'maxiter': 5, 'logdomain': 1},
        {**default_opts, 'maxiter': 50},
        {**default_opts, 'damping': 0.1, 'logdomain': 0, 'maxiter': 5},
        {**default_opts, 'damping': 0.2, 'logdomain': 1, 'maxiter': 10},
    ]

    for options in configs:
        result_libdai = libdai.infer(model, algorithm='BP', options=options)
        result_bp = BP.infer(model, options)
        assert_results_close(result_libdai, result_bp)


def test_tree():
    model = tree_potts_model(gr_size=100, al_size=3)
    true_result = model.infer(algorithm='tree_dp')
    result_bp = BP.infer(model)
    assert_results_close(true_result, result_bp, log_pf_tol=2e-9)


def test_grid():
    model = grid_potts_model(10, 5, al_size=2)
    true_result = model.infer(algorithm='path_dp')
    result_bp = BP.infer(model)
    assert_results_close(
        true_result,
        result_bp,
        log_pf_tol=0.1,
        mp_mse_tol=2e-3)


def test_small_model():
    # Generate small random model and compare Z with bruteforce result.
    model = random_generic_model(
        num_variables=10,
        num_factors=4,
        max_domain_size=2,
        max_factor_size=3)
    true_log_z = np.log(model.part_func_bruteforce())
    bp_log_z = BP.infer(model).log_pf
    assert np.allclose(true_log_z, bp_log_z)
