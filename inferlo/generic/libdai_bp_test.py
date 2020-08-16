# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import warnings

import numpy as np

from inferlo import GenericGraphModel, DiscreteDomain, DiscreteFactor
from inferlo.generic.libdai_bp import BP
from inferlo.interop import LibDaiInterop
from inferlo.pairwise.testing import (assert_results_close, tree_potts_model,
                                      grid_potts_model)


def _random_generic_model(num_variables=10,
                          num_factors=10,
                          max_domain_size=3,
                          max_factor_size=3):
    model = GenericGraphModel(num_variables=num_variables)
    for var_id in range(num_variables):
        domain_size = 2 + np.random.randint(max_domain_size - 1)
        model.get_variable(var_id).domain = DiscreteDomain.range(domain_size)
    for _ in range(num_factors):
        factor_size = 1 + np.random.randint(max_factor_size)
        var_idx = np.random.choice(
            num_variables,
            size=factor_size,
            replace=False)
        values_shape = [model.get_variable(i).domain.size() for i in var_idx]
        values = np.random.random(size=values_shape)
        factor = DiscreteFactor(model, var_idx, values)
        model.add_factor(factor)
    return model


# This test verifies that InferLO's algorithm is numerically equivalent to
# libDAI's BP algorithm.
def test_libdai_bp_regression():
    model = _random_generic_model(
        num_variables=20,
        num_factors=20,
        max_domain_size=4,
        max_factor_size=3)
    libdai = LibDaiInterop()
    if not libdai.is_libdai_ready:
        warnings.warn(
            "LibDAI not installed - won't run test_libdai_bp_regression.")
        return
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
