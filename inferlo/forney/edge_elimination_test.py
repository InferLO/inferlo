# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

import inferlo
from inferlo import PairWiseFiniteModel
from inferlo.forney.edge_elimination import (convolve_factor,
                                             convolve_two_factors,
                                             infer_edge_elimination)
from inferlo.testing import tree_potts_model, grid_potts_model


def test_convolve_factor():
    model = inferlo.GenericGraphModel(4, inferlo.DiscreteDomain.binary())
    model[2].domain = inferlo.DiscreteDomain.range(3)
    factor1 = inferlo.FunctionFactor(model, [0, 1, 2],
                                     lambda x: x[0] + 10 * x[1] + 100 * x[2])
    factor1 = inferlo.DiscreteFactor.from_factor(factor1)

    factor2 = convolve_factor(factor1, 1)

    assert factor2.model == model
    assert factor2.var_idx == [0, 2]
    assert np.allclose(factor2.values,
                       np.array([[10, 210, 410], [12, 212, 412]]))


def test_convolve_two_factors():
    model = inferlo.GenericGraphModel(3)
    model[0].domain = inferlo.DiscreteDomain.range(4)
    model[1].domain = inferlo.DiscreteDomain.range(5)
    model[2].domain = inferlo.DiscreteDomain.range(6)
    v1 = np.random.random(size=(4, 5))
    v2 = np.random.random(size=(5, 6))
    factor1 = inferlo.DiscreteFactor(model, [0, 1], v1)
    factor2 = inferlo.DiscreteFactor(model, [1, 2], v2)

    factor3 = convolve_two_factors(factor1, factor2, 1)
    assert factor3.model == model
    assert factor3.var_idx == [0, 2]
    assert np.allclose(factor3.values, v1 @ v2)


def test_convolve_two_factors_leaf():
    model = inferlo.GenericGraphModel(2, inferlo.DiscreteDomain.binary())
    factor1 = inferlo.DiscreteFactor(model, [0], np.array([1, 2]))
    factor2 = inferlo.DiscreteFactor(model, [0, 1], np.array([[3, 4], [5, 6]]))

    factor3 = convolve_two_factors(factor1, factor2, 0)

    assert factor3.model == model
    assert factor3.var_idx == [1]
    assert np.allclose(factor3.values, np.array([13, 16]))


def test_infer_small_line():
    np.random.seed(10)
    domain = inferlo.DiscreteDomain.range(5)
    model = inferlo.NormalFactorGraphModel(3, domain)
    f0 = np.random.random(size=(5,))
    f01 = np.random.random(size=(5, 5))
    f12 = np.random.random(size=(5, 5))
    f2 = np.random.random(size=(5,))
    model *= inferlo.DiscreteFactor(model, [0], f0)
    model *= inferlo.DiscreteFactor(model, [0, 1], f01)
    model *= inferlo.DiscreteFactor(model, [1, 2], f12)
    model *= inferlo.DiscreteFactor(model, [2], f2)
    model.build()

    z = infer_edge_elimination(model)

    assert np.allclose(z, f0 @ f01 @ f12 @ f2)


def test_ifer_clique_3vars():
    al_size = 5
    np.random.seed(10)
    domain = inferlo.DiscreteDomain.range(al_size)
    model = inferlo.NormalFactorGraphModel(3, domain)
    f01 = np.random.random(size=(al_size, al_size))
    f12 = np.random.random(size=(al_size, al_size))
    f20 = np.random.random(size=(al_size, al_size))
    model *= inferlo.DiscreteFactor(model, [0, 1], f01)
    model *= inferlo.DiscreteFactor(model, [1, 2], f12)
    model *= inferlo.DiscreteFactor(model, [2, 0], f20)
    model.build()

    z = infer_edge_elimination(model)

    assert np.allclose(z, model.part_func_bruteforce())


def test_infer_compare_with_pairwise_tree():
    pw_model = tree_potts_model(gr_size=50, al_size=5, seed=0)
    true_pf = np.exp(pw_model.infer(algorithm='tree_dp').log_pf)
    nfg_model = inferlo.NormalFactorGraphModel.from_model(pw_model)

    pf = infer_edge_elimination(nfg_model)

    assert np.allclose(true_pf, pf)


def test_infer_compare_with_pairwise_grid_5x5():
    pw_model = grid_potts_model(5, 5, al_size=2, seed=0)
    true_pf = np.exp(pw_model.infer(algorithm='path_dp').log_pf)
    nfg_model = inferlo.NormalFactorGraphModel.from_model(pw_model)

    pf = infer_edge_elimination(nfg_model)

    assert np.allclose(true_pf, pf)


def test_infer_disconnected():
    pw_model = PairWiseFiniteModel(5, al_size=2)
    pw_model.set_field(np.random.random(size=(5, 2)))
    pw_model.add_interaction(2, 3, np.random.random(size=(2, 2)))
    true_pf = np.exp(pw_model.infer(algorithm='bruteforce').log_pf)
    nfg_model = inferlo.NormalFactorGraphModel.from_model(pw_model)

    pf = infer_edge_elimination(nfg_model)

    assert np.allclose(true_pf, pf)
