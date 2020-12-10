import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.pairwise.optimization.convex_hierarchies import sherali_adams, lasserre
from inferlo.testing import grid_potts_model, tree_potts_model, \
    line_potts_model

def line_potts_4x3_sherali_adams():
    model = line_potts_model(gr_size=4, al_size=3, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    sa_res = sherali_adams(model, level=3)
    max_lh_ub = sa_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))

def line_potts_4x3_lasserre():
    model = line_potts_model(gr_size=4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lasserre_res = lasserre(model, level=2)
    max_lh_ub = lasserre_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))
