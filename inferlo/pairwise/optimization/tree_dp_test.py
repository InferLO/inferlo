import numpy as np

from inferlo.pairwise.testing import tree_potts_model


def test_tree_15x2():
    model = tree_potts_model(gr_size=15, al_size=2, seed=0)
    truth = model.max_likelihood(algorithm='bruteforce')
    result = model.max_likelihood(algorithm='tree_dp')
    assert np.allclose(truth, result)


def test_tree_10x3():
    model = tree_potts_model(gr_size=10, al_size=3, seed=0)
    truth = model.max_likelihood(algorithm='bruteforce')
    result = model.max_likelihood(algorithm='tree_dp')
    assert np.allclose(truth, result)
