import numpy as np

from inferlo import PairWiseFiniteModel, NormalFactorGraphModel


def test_create_from_model():
    field = np.zeros((4, 2))
    edges = [[0, 1], [0, 2], [0, 3]]
    j1 = np.array([[0, 0], [0, 1]])
    interactions = [j1, j1, j1]
    model1 = PairWiseFiniteModel.create(field, edges, interactions)
    expected_edges = [[0, 3], [0, 0], [1, 1], [2, 2], [1, 3], [2, 3]]
    kron_delta = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    expected_factor_tables = [np.exp(j1)] * 3 + [kron_delta]

    model2 = NormalFactorGraphModel.from_model(model1)

    assert model2.num_variables == 6
    assert len(model2.factors) == 4
    assert model2.edges == expected_edges
    for i in range(4):
        assert np.allclose(model2.factors[i].values, expected_factor_tables[i])
