# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np

from inferlo import GenericGraphModel, DiscreteDomain, DiscreteFactor
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


# This test only checks that get_marginals runs without failures with various underlying algos.
def test_get_marginals_with_different_algos():
    model = GenericGraphModel(5)
    for i in range(5):
        model[i].domain = DiscreteDomain.binary()
    model.add_factor(DiscreteFactor(model, [0], [0.05, 0.95]))
    model.add_factor(DiscreteFactor(model, [1], [0.01, 0.99]))
    model.add_factor(DiscreteFactor(model, [0, 1, 2],
                                    [[[0.999, 0.001], [0.9, 0.1]], [[0.95, 0.05], [0.1, 0.9]]]))
    model.add_factor(DiscreteFactor(model, [0, 1, 3],
                                    [[[0.999, 0.001], [0.8, 0.2]], [[0.95, 0.05], [0.05, 0.95]]]))
    model.add_factor(DiscreteFactor(model, [0, 1, 4],
                                    [[[0.999, 0.001], [0.7, 0.3]], [[0.95, 0.05], [0.025, 0.975]]]))
    algos = [
        lambda model: inf.belief_propagation(model).log_pf,
        inf.bucket_elimination,
        inf.mini_bucket_elimination,
        inf.weighted_mini_bucket_elimination,
        inf.bucket_renormalization,
        inf.iterative_join_graph_propagation,
        inf.mean_field
    ]
    for algo in algos:
        result = inf.get_marginals(model, algo, var_ids=[2, 3, 4])
        assert result.marg_prob.shape == (3, 2)


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
