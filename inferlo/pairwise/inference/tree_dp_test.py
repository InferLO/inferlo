# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.testing import (tree_potts_model,
                             line_potts_model, assert_results_close)


def test_vert15_alph2():
    model = tree_potts_model(gr_size=10, al_size=2, seed=123)
    ground_truth = model.infer(algorithm='bruteforce')
    result = model.infer(algorithm='tree_dp')
    assert_results_close(result, ground_truth)


def test_vert5_alph3():
    model = tree_potts_model(gr_size=5, al_size=3, seed=123)
    ground_truth = model.infer(algorithm='bruteforce')
    result = model.infer(algorithm='tree_dp')
    assert_results_close(result, ground_truth)


def test_long_line():
    gr_size = 1000
    al_size = 5
    j = np.ones((al_size, al_size)) + np.eye(al_size)
    model = line_potts_model(gr_size=gr_size, al_size=al_size,
                             seed=111, same_j=j, zero_field=True)
    result = model.infer(algorithm='tree_dp')
    assert np.allclose(result.marg_prob,
                       np.ones((gr_size, al_size)) / al_size)


def test_big_tree():
    gr_size = 1000
    al_size = 5
    j = np.ones((al_size, al_size)) + np.eye(al_size)
    model = tree_potts_model(gr_size=gr_size, al_size=al_size,
                             seed=111, same_j=j, zero_field=True)
    result = model.infer(algorithm='tree_dp')
    assert np.allclose(result.marg_prob,
                       np.ones((gr_size, al_size)) / al_size)


def test_fully_isolated():
    model = PairWiseFiniteModel(10, 2)
    model.set_field(np.random.random(size=(10, 2)))
    ground_truth = model.infer(algorithm='bruteforce')
    result = model.infer(algorithm='tree_dp')
    assert_results_close(result, ground_truth)
