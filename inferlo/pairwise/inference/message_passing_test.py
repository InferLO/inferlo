import numpy as np

from inferlo.pairwise.testing import (tree_potts_model, assert_results_close,
                                      grid_potts_model)


def test_tree():
    model = tree_potts_model(gr_size=50, al_size=3, seed=0)
    assert_results_close(model.infer(algorithm='message_passing'),
                         model.infer(algorithm='tree_dp'))


def test_grid_approx():
    model = grid_potts_model(5, 15, al_size=2, seed=0)
    mp_true = model.infer(algorithm='path_dp').marg_prob
    mp = model.infer(algorithm='message_passing').marg_prob

    assert np.mean(np.square(mp_true - mp)) < 1e-4
