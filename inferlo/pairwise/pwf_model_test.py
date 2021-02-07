# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import GenericGraphModel, DiscreteDomain
from inferlo.base import DiscreteFactor
from inferlo.pairwise import PairWiseFiniteModel
from inferlo.base.inference_result import InferenceResult
from inferlo.testing import assert_results_close


def test_build_from_factors():
    model = PairWiseFiniteModel(5, 2)
    x = model.get_symbolic_variables()

    # Field.
    model *= np.exp(5 * x[1])
    model *= np.exp(10 * x[2])
    model *= DiscreteFactor(model, [3], np.exp([7, 8]))

    # Interactions.
    model *= np.exp(2 * x[0] * x[1])
    model *= (1 + x[2] + x[3])
    model *= np.exp(10 * x[0] * x[1])  # Should accumulate.
    model *= DiscreteFactor(model, [0, 4], np.exp([[1, 2], [3, 4]]))

    assert np.allclose(model.field,
                       np.array([[0, 0], [0, 5], [0, 10], [7, 8], [0, 0]]))
    assert model.edges == [(0, 1), (2, 3), (0, 4)]
    assert np.allclose(model.get_interaction_matrix(0, 1),
                       [[0, 0], [0, 12]])
    assert np.allclose(model.get_interaction_matrix(2, 3),
                       np.log([[1, 2], [2, 3]]))
    assert np.allclose(model.get_interaction_matrix(0, 4),
                       [[1, 2], [3, 4]])


def test_build_from_interactions():
    model = PairWiseFiniteModel(10, 5)
    j1 = np.random.random(size=(5, 5))
    j2 = np.random.random(size=(5, 5))
    model.add_interaction(0, 1, j1)
    model.add_interaction(1, 0, j1)
    model.add_interaction(1, 2, j2)

    assert np.allclose(model.field, np.zeros((10, 5)))
    assert model.edges == [(0, 1), (1, 2)]
    assert np.allclose(model.get_interaction_matrix(0, 1), j1 + j1.T)
    assert np.allclose(model.get_interaction_matrix(2, 1), j2.T)


def test_inference_all_methods():
    # Sanity check that all algorithms work on very simple model.
    all_methods = ['auto', 'bruteforce', 'mean_field', 'message_passing',
                   'tree_dp', 'path_dp', 'junction_tree']
    model = PairWiseFiniteModel(2, 2)
    model.add_interaction(0, 1, np.array([[0, 0], [0, 1]]))
    m = np.array([2, 1 + np.exp(1)])
    expected_result = InferenceResult(np.log(3 + np.exp(1)),
                                      np.array([m, m]) / np.sum(m))

    for method in all_methods:
        result = model.infer(algorithm=method)
        assert_results_close(result, expected_result,
                             log_pf_tol=1.0,
                             mp_mse_tol=0.1)


def test_max_likelihood_all_methods():
    # Sanity check that all algorithms work on very simple model.
    all_methods = ['auto', 'bruteforce', 'tree_dp', 'path_dp', 'junction_tree']
    model = PairWiseFiniteModel(2, 2)
    model.add_interaction(0, 1, np.array([[0, 0], [0, 1]]))
    expected_result = np.array([1, 1])

    for method in all_methods:
        result = model.max_likelihood(algorithm=method)
        assert np.allclose(result, expected_result)


def test_sample_all_methods():
    # Sanity check that all algorithms work on very simple model.
    all_methods = ['auto', 'bruteforce', 'tree_dp', 'junction_tree']
    model = PairWiseFiniteModel(2, 2)
    model.set_field(np.array([[100, 0], [100, 0]]))
    num_samples = 10
    expected_result = np.zeros((num_samples, 2))

    for method in all_methods:
        result = model.sample(algorithm=method, num_samples=num_samples)
        assert result.shape == expected_result.shape
        assert np.allclose(result, expected_result)


def test_get_dfs_result():
    model = PairWiseFiniteModel(4, 2)
    j = np.ones((2, 2))
    model.add_interaction(2, 3, j)
    model.add_interaction(2, 1, j)
    model.get_dfs_result()  # To test cache invalidation.
    model.add_interaction(1, 0, j)

    dfs_edges = model.get_dfs_result().dfs_edges

    assert np.allclose(dfs_edges, np.array([[0, 1], [1, 2], [2, 3]]))


def test_decode_state():
    model = PairWiseFiniteModel(5, 2)
    assert model.decode_state(11) == [1, 1, 0, 1, 0]

    model = PairWiseFiniteModel(5, 10)
    assert model.decode_state(12345) == [5, 4, 3, 2, 1]


def test_encode_state():
    model = PairWiseFiniteModel(5, 2)
    assert model.encode_state([1, 1, 0, 1, 0]) == 11

    model = PairWiseFiniteModel(5, 10)
    assert model.encode_state([5, 4, 3, 2, 1]) == 12345


def test_from_model():
    model1 = GenericGraphModel(3)
    model1[0].domain = DiscreteDomain.binary()
    model1[2].domain = DiscreteDomain([0, 10])
    model1[1].domain = DiscreteDomain.range(3)
    x0, x1, x2 = model1.get_symbolic_variables()
    model1.add_factor(np.exp(x0 + x1))
    model1.add_factor(np.exp(x1 + x2))
    model2 = PairWiseFiniteModel.from_model(model1)

    assert np.allclose(model2.field, np.zeros((3, 3)))
    assert model2.edges == [(0, 1), (1, 2)]
    assert np.allclose(model2.get_all_interactions(), np.array([
        [[0, 1, 2], [1, 2, 3], [-np.inf, -np.inf, -np.inf]],
        [[0, 10, -np.inf], [1, 11, -np.inf], [2, 12, -np.inf]]
    ]))
