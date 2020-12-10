import numpy as np

from inferlo.pairwise.optimization.convex_hierarchies \
    import sherali_adams, lasserre
from inferlo.testing import line_potts_model


def test_line_potts_4x3_sherali_adams():
    """
    Sherali-Adams is exact on line graph.
    """
    model = line_potts_model(gr_size=4, al_size=3, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    sa_res = sherali_adams(model, level=3)
    max_lh_ub = sa_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))


def test_line_potts_4x2_lasserre():
    """
    Second step of Lasserre hierarchy is exact on line graph.
    """
    model = line_potts_model(gr_size=4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lasserre_res = lasserre(model, level=2)
    max_lh_ub = lasserre_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))
