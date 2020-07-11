# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import (PairWiseFiniteModel, NormalFactorGraphModel,
                     DiscreteDomain, DiscreteFactor)


def test_create_from_model():
    field = np.zeros((4, 2))
    edges = [[0, 1], [0, 2], [0, 3]]
    j1 = np.array([[0, 0], [0, 1]])
    interactions = [j1, j1, j1]
    model1 = PairWiseFiniteModel.create(field, edges, interactions)
    expected_edges = [[0, 3], [0, 4], [1, 5], [2, 6], [1, 3], [2, 3]]
    kron_delta = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    unit_factor = np.array([1, 1])
    expected_factor_tables = [
        np.exp(j1)] * 3 + [kron_delta] + [
        unit_factor] * 3

    model2 = NormalFactorGraphModel.from_model(model1)

    assert model2.num_variables == 6
    assert len(model2.factors) == 7
    assert model2.edges == expected_edges
    for i in range(7):
        assert np.allclose(model2.factors[i].values, expected_factor_tables[i])


def test_inference_all_methods():
    # Sanity check that all algorithms work on very simple model.
    all_methods = ['auto', 'edge_elimination']

    model = NormalFactorGraphModel(2, DiscreteDomain.range(2))
    model *= DiscreteFactor(model, [0], np.array([0, 1]))
    model *= DiscreteFactor(model, [0, 1], np.array([[2, 3], [4, 5]]))
    model *= DiscreteFactor(model, [1], np.array([6, 7]))
    model.build()

    for method in all_methods:
        assert np.allclose(model.infer(method=method), 59)
