# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np

from inferlo.testing import grid_potts_model
from inferlo.forney.optimization.nfg_map_lp import map_lp
from inferlo.forney.nfg_model import NormalFactorGraphModel


def test_grid_4x4x2():
    model = grid_potts_model(4, 4, al_size=2, seed=0)
    nfg = NormalFactorGraphModel.from_model(model)
    res = map_lp(nfg)
    ub = res.upper_bound
    lb = res.lower_bound
    assert ((ub >= lb) or (np.allclose(ub, lb)))


def test_line():
    model = grid_potts_model(1, 4, al_size=3, seed=0)
    nfg = NormalFactorGraphModel.from_model(model)
    res = map_lp(nfg)
    ub = res.upper_bound
    lb = res.lower_bound
    assert np.allclose(ub, lb, atol=2e-3)
