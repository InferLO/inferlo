# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np
from scipy.special import softmax

from inferlo.pairwise.bruteforce import infer_bruteforce, sample_bruteforce
from inferlo.pairwise.pwf_model import PairWiseFiniteModel


def _stochastic_vector(n):
    x = np.random.random(size=n)
    return x / np.sum(x)


def test_infer_1_variable():
    al_size = 10
    probs = _stochastic_vector(al_size)
    model = PairWiseFiniteModel(1, al_size)
    model.set_field(np.log(probs).reshape(1, al_size))

    result = infer_bruteforce(model)

    assert np.allclose(result.log_pf, 0)
    assert np.allclose(result.marg_prob, probs)


def test_infer_ising_2_variables():
    model = PairWiseFiniteModel(2, 2)
    j = 5 * np.random.random()
    model.add_interaction(0, 1, np.array([[j, -j], [-j, j]]))

    result = model.infer(algorithm='bruteforce')

    assert np.allclose(result.log_pf, np.log(4 * np.cosh(j)))
    assert np.allclose(result.marg_prob, 0.5 * np.ones((2, 2)))


def test_infer_isolated():
    gr_size = 1000
    model = PairWiseFiniteModel(gr_size, 2)
    model.set_field(np.array([[0, 1]] * gr_size))
    res = model.infer(algorithm='bruteforce')
    assert np.allclose(res.log_pf,
                       gr_size * np.log(1 + np.exp(1)))
    assert np.allclose(res.marg_prob - softmax([0, 1]), 0)


def test_max_likelihood():
    model = PairWiseFiniteModel(3, 2)
    model.set_field(np.array([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]))
    model.add_interaction(0, 1, np.array([[0, 10], [10, 0]]))
    model.add_interaction(1, 2, np.array([[0, 10], [10, 0]]))

    state = model.max_likelihood(algorithm='bruteforce')

    assert np.allclose(state, np.array([1, 0, 1]))


def test_max_likelihood_isolated():
    gr_size = 1000
    model = PairWiseFiniteModel(gr_size, 2)
    model.set_field(np.array([[0, 1]] * gr_size))
    result = model.max_likelihood(algorithm='bruteforce')
    assert np.allclose(result, np.ones(gr_size))


def test_sample_bruteforce():
    gr_size, num_samples = 3, 50
    model = PairWiseFiniteModel(gr_size, 2)
    model.set_field(np.array([[0, 20]] * gr_size))
    samples = sample_bruteforce(model, num_samples=num_samples)
    assert np.allclose(samples, np.ones((num_samples, gr_size)))
