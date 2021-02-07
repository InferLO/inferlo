# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo.pairwise.pwf_model import PairWiseFiniteModel
from inferlo.testing import (assert_results_close, line_potts_model,
                             tree_potts_model, grid_potts_model,
                             clique_potts_model)


def test_isolated_exact():
    np.random.seed(0)
    gr_size = 1000
    al_size = 5
    model = PairWiseFiniteModel(gr_size, al_size)
    model.set_field(np.random.random((gr_size, al_size)))
    gt = model.infer(algorithm='bruteforce')

    result = model.infer(algorithm='mean_field')

    assert_results_close(result, gt)


def test_cycle3():
    np.random.seed(0)
    model = PairWiseFiniteModel(3, 3)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        model.add_interaction(i, j, np.random.random(size=(3, 3)))
    gt = model.infer(algorithm='bruteforce')

    result = model.infer(algorithm='mean_field')

    assert_results_close(result, gt, log_pf_tol=0.1, mp_mse_tol=1e-4)


def test_line():
    model = line_potts_model(gr_size=40, al_size=3, seed=0)
    gt = model.infer(algorithm='path_dp')

    result = model.infer(algorithm='mean_field')

    assert_results_close(result, gt, log_pf_tol=3.0, mp_mse_tol=1e-2)


def test_tree():
    model = tree_potts_model(gr_size=50, al_size=3, seed=0)
    gt = model.infer(algorithm='tree_dp')

    result = model.infer(algorithm='mean_field')

    assert_results_close(result, gt, log_pf_tol=10.0, mp_mse_tol=0.03)


def test_grid():
    model = grid_potts_model(4, 5, al_size=2, seed=0)
    gt = model.infer(algorithm='path_dp')

    result = model.infer(algorithm='mean_field', max_iter=1000)

    assert_results_close(result, gt, log_pf_tol=5.0, mp_mse_tol=0.05)


def test_clique():
    model = clique_potts_model(gr_size=5, al_size=3, seed=0)
    gt = model.infer(algorithm='bruteforce')

    result = model.infer(algorithm='mean_field')

    assert_results_close(result, gt, log_pf_tol=1.0, mp_mse_tol=1e-4)
