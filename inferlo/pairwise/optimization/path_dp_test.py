# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.pairwise.optimization.path_dp import max_lh_path_dp
from inferlo.testing import grid_potts_model, tree_potts_model, \
    line_potts_model


def test_grid_4x4x2():
    model = grid_potts_model(4, 4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)


def test_grid_3x3x4():
    model = grid_potts_model(3, 3, al_size=4, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)


def test_grid_2x2x10():
    model = grid_potts_model(2, 2, al_size=10, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)


def test_line_1000x10():
    model = line_potts_model(gr_size=1000, al_size=10, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)


def test_tree_50x2():
    model = tree_potts_model(gr_size=50, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)


def test_disconnected():
    model = PairWiseFiniteModel(size=4, al_size=5)
    model.add_interaction(0, 1, np.random.random(size=(5, 5)))
    model.add_interaction(2, 3, np.random.random(size=(5, 5)))
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh = max_lh_path_dp(model)
    assert np.allclose(max_lh, max_lh_gt)
