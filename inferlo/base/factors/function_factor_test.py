# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

from inferlo import FunctionFactor, RealDomain, GraphModel


def test_value():
    model = GraphModel.create(2, RealDomain())
    factor = FunctionFactor(model, [0, 1], lambda x: x[0] + x[1])
    assert factor.evaluate([10, 20]) == 30


def test_symbolic_factor_creation():
    model = GraphModel.create(10, RealDomain())
    x = model.get_symbolic_variables()

    def f(x, y, z): return (z + y) / 2 + 17 * (x * x * y - z * y) + 36 * x
    factor = f(x[0], x[1], x[2])

    assert np.allclose(factor.evaluate([10, 20, 30]), f(10, 20, 30))
