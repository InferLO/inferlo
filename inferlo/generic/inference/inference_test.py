# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.

import numpy as np

from inferlo.generic import inference as inf
from inferlo.testing import grid_potts_model, tree_potts_model, \
    clique_potts_model


def test_all_tree_30():
    model = tree_potts_model(30, al_size=4, seed=101)
    true_log_pf = model.infer(algorithm="tree_dp").log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=40.0)
    assert np.allclose(inf.belief_propagation(model), true_log_pf, atol=1e-5)
    assert np.allclose(
        inf.iterative_join_graph_propagation(model),
        true_log_pf)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf)
    assert np.allclose(
        inf.weighted_mini_bucket_elimination(model),
        true_log_pf)
    assert np.allclose(inf.bucket_renormalization(model), true_log_pf)


def test_all_grid_6x6():
    model = grid_potts_model(6, 6, al_size=2, seed=101)
    true_log_pf = model.infer(algorithm="junction_tree").log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=20.0)
    assert np.allclose(inf.belief_propagation(model), true_log_pf, atol=0.1)
    assert np.allclose(
        inf.iterative_join_graph_propagation(model),
        true_log_pf)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf,
                       atol=10.0)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf)
    assert np.allclose(
        inf.weighted_mini_bucket_elimination(model),
        true_log_pf,
        atol=10.0)
    assert np.allclose(
        inf.bucket_renormalization(model),
        true_log_pf,
        atol=0.1)


def test_all_clique_6():
    model = clique_potts_model(gr_size=6, al_size=3, seed=101)
    true_log_pf = model.infer(algorithm="junction_tree").log_pf
    assert np.allclose(inf.mean_field(model), true_log_pf, atol=10.0)
    assert np.allclose(inf.belief_propagation(model), true_log_pf, atol=0.1)
    assert np.allclose(
        inf.iterative_join_graph_propagation(model),
        true_log_pf,
        atol=1e-4)
    assert np.allclose(inf.mini_bucket_elimination(model), true_log_pf)
    assert np.allclose(inf.bucket_elimination(model), true_log_pf)
    assert np.allclose(
        inf.weighted_mini_bucket_elimination(model),
        true_log_pf)
    assert np.allclose(inf.bucket_renormalization(model), true_log_pf)


def test_wmbe_grid_9x9():
    model = grid_potts_model(9, 9, al_size=2, seed=0)
    true_log_pf = model.infer(algorithm="path_dp").log_pf
    assert np.allclose(
        inf.bucket_renormalization(model),
        true_log_pf,
        atol=1.0)


def test_bucket_renormalization_grid_9x9():
    model = grid_potts_model(9, 9, al_size=2, seed=0)
    true_log_pf = model.infer(algorithm="path_dp").log_pf
    assert np.allclose(
        inf.weighted_mini_bucket_elimination(model),
        true_log_pf,
        atol=2.0)