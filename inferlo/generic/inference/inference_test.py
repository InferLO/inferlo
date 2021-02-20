# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np

from inferlo import GenericGraphModel
from inferlo.generic import inference as inf
from inferlo.testing import grid_potts_model, tree_potts_model, \
    clique_potts_model, assert_results_close


def test_all_tree_30():
    model = tree_potts_model(30, al_size=4, seed=101)
    true_log_pf = model.infer(algorithm="tree_dp").log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=40.0)
    assert np.allclose(inf.belief_propagation(model).log_pf, true_log_pf,
                       atol=1e-5)
    assert np.allclose(inf.iterative_join_graph_propagation(model),
                       true_log_pf, atol=1e-5)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf,
                       atol=1e-5)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf, atol=1e-5)
    assert np.allclose(inf.weighted_mini_bucket_elimination(model),
                       true_log_pf, atol=0.5)
    assert np.allclose(inf.bucket_renormalization(model), true_log_pf,
                       atol=1e-5)


def test_all_grid_6x6():
    model = grid_potts_model(6, 6, al_size=2, seed=101)
    true_inference = model.infer(algorithm="junction_tree")
    true_log_pf = true_inference.log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=20.0)
    assert_results_close(inf.belief_propagation(model),
                         true_inference,
                         log_pf_tol=0.1,
                         mp_mse_tol=1e-3)
    assert np.allclose(inf.iterative_join_graph_propagation(model),
                       true_log_pf, atol=1e-5)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf,
                       atol=10.0)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf, atol=1e-5)
    assert np.allclose(inf.weighted_mini_bucket_elimination(model),
                       true_log_pf,
                       atol=10.0)
    assert np.allclose(inf.bucket_renormalization(model),
                       true_log_pf,
                       atol=1.0)


def test_all_clique_6():
    model = clique_potts_model(gr_size=6, al_size=3, seed=101)
    true_log_pf = model.infer(algorithm="junction_tree").log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=10.0)
    assert np.allclose(inf.belief_propagation(model).log_pf, true_log_pf,
                       atol=0.1)
    assert np.allclose(inf.iterative_join_graph_propagation(model),
                       true_log_pf,
                       atol=1e-4)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf,
                       atol=1e-5)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf, atol=1e-5)
    assert np.allclose(inf.weighted_mini_bucket_elimination(model),
                       true_log_pf, atol=1e-5)
    assert np.allclose(inf.bucket_renormalization(model), true_log_pf,
                       atol=1e-5)


def test_wmbe_grid_9x9():
    model = grid_potts_model(9, 9, al_size=2, seed=0)
    true_log_pf = model.infer(algorithm="path_dp").log_pf
    assert np.allclose(
        inf.weighted_mini_bucket_elimination(model),
        true_log_pf,
        atol=4.0)


def test_bucket_renormalization_grid_9x9():
    model = grid_potts_model(9, 9, al_size=2, seed=0)
    true_log_pf = model.infer(algorithm="path_dp").log_pf
    assert np.allclose(
        inf.bucket_renormalization(model),
        true_log_pf,
        atol=4.0)


def test_get_marginals():
    model = grid_potts_model(4, 3, al_size=4, seed=0)
    true_result = model.infer(algorithm='path_dp')
    model = GenericGraphModel.from_model(model)
    be_result_1 = inf.get_marginals(model, inf.bucket_elimination,
                                    skip_last=False)
    be_result_2 = inf.get_marginals(model, inf.bucket_elimination,
                                    skip_last=True)
    assert_results_close(true_result, be_result_1)
    assert_results_close(true_result, be_result_2)


def test_bucket_elimination_bt():
    model = grid_potts_model(4, 3, al_size=4, seed=1)
    true_result = model.infer(algorithm='path_dp')
    model = GenericGraphModel.from_model(model)
    be_result = inf.bucket_elimination_bt(model)
    assert_results_close(true_result, be_result)


def test_mini_bucket_elimination_bt():
    model = grid_potts_model(5, 5, al_size=3, seed=1)
    true_result = model.infer(algorithm='path_dp')
    model = GenericGraphModel.from_model(model)
    be_result = inf.mini_bucket_elimination_bt(model, ibound=10)
    assert_results_close(true_result, be_result, log_pf_tol=1e-5, mp_mse_tol=1e-4)


def test_mini_bucket_renormalization_bt():
    model = grid_potts_model(5, 5, al_size=3, seed=1)
    true_result = model.infer(algorithm='path_dp')
    model = GenericGraphModel.from_model(model)
    be_result = inf.mini_bucket_renormalization_bt(model, ibound=7)
    assert_results_close(true_result, be_result, log_pf_tol=1e-5, mp_mse_tol=1e-4)
