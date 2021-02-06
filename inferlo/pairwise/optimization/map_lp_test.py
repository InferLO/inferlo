# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.pairwise.optimization.map_lp import map_lp
from inferlo.testing import grid_potts_model, tree_potts_model, \
    line_potts_model


def test_grid_4x4x2():
    model = grid_potts_model(4, 4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert (round(max_lh_ub, 3) >= round(np.log(model.evaluate(max_lh_gt)), 3)
            >= round(max_lh_lb, 3))


def test_grid_3x3x4():
    model = grid_potts_model(3, 3, al_size=4, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert (round(max_lh_ub, 3) >= round(np.log(model.evaluate(max_lh_gt)), 3)
                                >= round(max_lh_lb, 3))


def test_grid_2x2x10():
    model = grid_potts_model(2, 2, al_size=10, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert (round(max_lh_ub, 3) >= round(np.log(model.evaluate(max_lh_gt)), 3)
                                >= round(max_lh_lb, 3))


def test_line_3x3():
    model = line_potts_model(gr_size=3, al_size=3, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)), max_lh_lb)


def test_line_20x10():
    model = line_potts_model(gr_size=20, al_size=10, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)), max_lh_lb)


def test_tree_50x2():
    model = tree_potts_model(gr_size=50, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)), max_lh_lb)


def test_disconnected():
    model = PairWiseFiniteModel(size=4, al_size=5)
    model.add_interaction(0, 1, np.random.random(size=(5, 5)))
    model.add_interaction(2, 3, np.random.random(size=(5, 5)))
    max_lh_gt = model.max_likelihood(algorithm='path_dp')
    lp_res = map_lp(model)
    max_lh_ub = lp_res.upper_bound
    max_lh_lb = lp_res.lower_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)), max_lh_lb)
