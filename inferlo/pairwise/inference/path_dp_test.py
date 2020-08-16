# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo.pairwise.bruteforce import infer_bruteforce
from inferlo.testing import (assert_results_close, grid_potts_model,
                             line_potts_model, clique_potts_model)


def test_grid_3x3():
    model = grid_potts_model(3, 3, al_size=2, seed=123)
    gt = model.infer(algorithm='bruteforce')
    result = model.infer(algorithm='path_dp')
    assert_results_close(result, gt)


def test_grid_4x2():
    model = grid_potts_model(4, 2, al_size=3, seed=123)
    gt = model.infer(algorithm='bruteforce')
    result = model.infer(algorithm='path_dp')
    assert_results_close(result, gt)


def test_long_line():
    gr_size = 1000
    al_size = 5
    j = np.ones((al_size, al_size)) + np.eye(al_size)
    model = line_potts_model(gr_size=gr_size, al_size=al_size,
                             seed=111, same_j=j, zero_field=True)
    result = model.infer(algorithm='path_dp')
    assert np.allclose(result.marg_prob,
                       np.ones((gr_size, al_size)) / al_size)


def test_long_line_compare_with_tree():
    # Test to ensure Tree DP and Path DP are consistent.
    gr_size = 1000
    al_size = 3
    model = line_potts_model(gr_size=gr_size, al_size=al_size, seed=111)
    result1 = model.infer(algorithm='path_dp')
    result2 = model.infer(algorithm='tree_dp')
    assert_results_close(result1, result2)


def test_clique():
    model = clique_potts_model(gr_size=5, al_size=3, seed=0)
    gt = infer_bruteforce(model)

    result = model.infer(algorithm='path_dp')

    assert_results_close(result, gt)
