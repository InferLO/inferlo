import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.pairwise.optimization.convex_bounds import LPrelaxation
from inferlo.pairwise.testing import grid_potts_model, tree_potts_model, \
    line_potts_model


def test_grid_4x4x2():
    model = grid_potts_model(4, 4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh_gt_value = np.log(model.evaluate(max_lh_gt))
    max_lh_value_relax = LPrelaxation(model)
    assert (max_lh_gt_value <= max_lh_value_relax)


def test_grid_3x3x4():
    model = grid_potts_model(3, 3, al_size=4, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh_gt_value = np.log(model.evaluate(max_lh_gt))
    max_lh_value_relax = LPrelaxation(model)
    assert (max_lh_gt_value <= max_lh_value_relax)


def test_grid_2x2x10():
    model = grid_potts_model(2, 2, al_size=10, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh_gt_value = np.log(model.evaluate(max_lh_gt))
    max_lh_value_relax = LPrelaxation(model)
    assert (max_lh_gt_value <= max_lh_value_relax)


def test_disconnected():
    model = PairWiseFiniteModel(size=4, al_size=5)
    model.add_interaction(0, 1, np.random.random(size=(5, 5)))
    model.add_interaction(2, 3, np.random.random(size=(5, 5)))
    max_lh_gt = model.max_likelihood(algorithm='bruteforce')
    max_lh_gt_value = np.log(model.evaluate(max_lh_gt))
    max_lh_value_relax = LPrelaxation(model)
    assert (max_lh_gt_value <= max_lh_value_relax)