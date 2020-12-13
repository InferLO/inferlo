import sys
import pytest

import numpy as np

from inferlo.pairwise.optimization.convex_hierarchies \
    import sherali_adams, lasserre, minimal_cycle
from inferlo.testing import line_potts_model, grid_potts_model


def test_line_potts_4x3_sherali_adams():
    """
    Sherali-Adams is exact on line graph.
    """
    model = line_potts_model(gr_size=4, al_size=3, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    sa_res = sherali_adams(model, level=3)
    max_lh_ub = sa_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))


@pytest.mark.skipif(sys.platform == "win32",
                    reason="SCS is unstable")
def test_line_potts_4x2_lasserre():
    """
    Second step of Lasserre hierarchy is exact on line graph.

    This test is skipped for Windows since the performance of SCS
    solver was unstable.
    """
    model = line_potts_model(gr_size=4, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='tree_dp')
    lasserre_res = lasserre(model, level=2)
    max_lh_ub = lasserre_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))


def test_cycle_relaxation_2x2_cycle():
    """
    Sherali-Adams is exact on line graph.
    """
    model = grid_potts_model(2, 2, al_size=2, seed=0)
    max_lh_gt = model.max_likelihood(algorithm='path_dp')
    sa_res = minimal_cycle(model)
    max_lh_ub = sa_res.upper_bound
    assert np.allclose(max_lh_ub, np.log(model.evaluate(max_lh_gt)))
