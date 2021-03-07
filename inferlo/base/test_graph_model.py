# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import OldDiscreteFactor, FunctionFactor, GraphModel, RealDomain
from inferlo.base import DiscreteDomain


def test_evaluate():
    model = GraphModel.create(3, RealDomain())
    model[0].domain = DiscreteDomain.binary()
    model[1].domain = DiscreteDomain.binary()

    model.add_factor(OldDiscreteFactor.from_flat_values(model, [0, 1], np.array([1, 2, 3, 4])))
    model.add_factor(FunctionFactor(model, [1, 2], lambda x: x[0] + x[1] ** 2))

    assert model.evaluate(np.array([0, 0, -5])) == 1 * 25
    assert model.evaluate(np.array([1, 1, 4])) == 4 * 17
