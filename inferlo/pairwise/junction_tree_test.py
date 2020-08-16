# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo.pairwise.junction_tree import (max_likelihood_junction_tree,
                                            infer_junction_tree)
from inferlo.testing import (clique_potts_model, tree_potts_model,
                             grid_potts_model, assert_results_close)
from inferlo.testing.model_generators import cross_potts_model
from inferlo.testing.test_utils import check_samples


def test_inference_clique_10x2():
    model = clique_potts_model(gr_size=10, al_size=2, seed=0)
    ground_truth = model.infer(algorithm='bruteforce')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_tree_100x5():
    model = tree_potts_model(gr_size=100, al_size=5, seed=0)
    ground_truth = model.infer(algorithm='tree_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_grid_4x50x2():
    model = grid_potts_model(4, 50, al_size=2, seed=0)
    ground_truth = model.infer(algorithm='path_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_cross_50x2x2():
    model = cross_potts_model(length=50, width=2, al_size=2)
    ground_truth = model.infer(algorithm='path_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_max_likelihood_clique_10x2():
    model = clique_potts_model(gr_size=10, al_size=2, seed=0)
    true_ml = model.max_likelihood(algorithm='bruteforce')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_tree_100x5():
    model = tree_potts_model(gr_size=100, al_size=5, seed=0)
    true_ml = model.max_likelihood(algorithm='tree_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_grid_4x50x2():
    model = grid_potts_model(4, 50, al_size=2, seed=0)
    true_ml = model.max_likelihood(algorithm='path_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_cross_50x2x2():
    model = cross_potts_model(length=50, width=2, al_size=2)
    true_ml = model.max_likelihood(algorithm='path_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_sample_grid2x20x2():
    model = grid_potts_model(2, 20, al_size=2, seed=0)
    true_marg_probs = model.infer(algorithm='path_dp').marg_prob
    samples = model.sample(num_samples=1000)
    check_samples(samples=samples, true_marg_probs=true_marg_probs, tol=5e-4)
