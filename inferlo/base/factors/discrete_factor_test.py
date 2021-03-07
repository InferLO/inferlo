# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import FunctionFactor, OldDiscreteFactor, RealDomain, GraphModel
from inferlo.base import DiscreteDomain


def test_convert_from_function():
    model = GraphModel.create(5, RealDomain())
    model[0].domain = DiscreteDomain([-1, 1])
    model[1].domain = DiscreteDomain([0, np.pi])
    model[3].domain = DiscreteDomain([0, 1, 2])

    factor1 = FunctionFactor(model, [0, 1, 3],
                             lambda x: (2 + x[0]) * (1 + np.sin(x[1])) * x[2])
    factor2 = OldDiscreteFactor.from_factor(factor1)

    assert factor2.model == model
    assert factor2.var_idx == [0, 1, 3]
    assert np.allclose(factor2.values, [[[0, 1, 2], [0, 1, 2]],
                                        [[0, 3, 6], [0, 3, 6]]])
