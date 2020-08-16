# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.testing import tree_potts_model


def test_tree_12x2():
    for seed in range(5):
        model = tree_potts_model(gr_size=12, al_size=2, seed=seed)
        truth = model.max_likelihood(algorithm='bruteforce')
        result = model.max_likelihood(algorithm='tree_dp')
        assert np.allclose(truth, result)


def test_tree_8x3():
    for seed in range(5):
        model = tree_potts_model(gr_size=8, al_size=3, seed=seed)
        truth = model.max_likelihood(algorithm='bruteforce')
        result = model.max_likelihood(algorithm='tree_dp')
        assert np.allclose(truth, result)


def test_tree_6x4():
    for seed in range(5):
        model = tree_potts_model(gr_size=6, al_size=4, seed=seed)
        truth = model.max_likelihood(algorithm='bruteforce')
        result = model.max_likelihood(algorithm='tree_dp')
        assert np.allclose(truth, result)


def test_tree_5x3():
    model = tree_potts_model(gr_size=5, al_size=3, seed=0)
    truth = model.max_likelihood(algorithm='bruteforce')
    result = model.max_likelihood(algorithm='tree_dp')
    assert np.allclose(truth, result)


def test_pairs():
    n = 5
    j = np.array([[0, 0, 10], [0, 0, 0], [0, 0, 0]])
    model = PairWiseFiniteModel(2 * n, 3)
    for i in range(n):
        model.add_interaction(2 * i, 2 * i + 1, j)
    expected = np.array([0, 2] * n)

    result = model.max_likelihood(algorithm='tree_dp')

    assert np.allclose(expected, result)


def test_fully_isolated():
    model = PairWiseFiniteModel(10, 2)
    model.set_field(np.random.random(size=(10, 2)))
    ground_truth = model.max_likelihood(algorithm='bruteforce')
    result = model.max_likelihood(algorithm='tree_dp')
    assert np.allclose(ground_truth, result)
