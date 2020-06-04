import numpy as np

from inferlo.base import FunctionFactor, DiscreteFactor
from inferlo.pairwise import PairWiseFiniteModel, InferenceResult
from inferlo.pairwise.testing import assert_results_close


def test_build_from_factors():
    model = PairWiseFiniteModel(5, 2)
    x = FunctionFactor.prepare_variables(model)

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
    all_methods = ['auto', 'bruteforce', 'mean_field', 'message_passing',
                   'tree_dp', 'path_dp']
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
    all_methods = ['auto', 'bruteforce', 'tree_dp']
    model = PairWiseFiniteModel(2, 2)
    model.add_interaction(0, 1, np.array([[0, 0], [0, 1]]))
    expected_result = np.array([1, 1])

    for method in all_methods:
        result = model.max_likelihood(algorithm=method)
        assert np.allclose(result, expected_result)
