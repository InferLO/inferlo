import numpy as np

from inferlo import GenericGraphModel, FunctionFactor


def test_symbolic_factor_creation():
    model = GenericGraphModel(10)
    x = FunctionFactor.prepare_variables(model)

    def f(x, y, z): return (z + y) / 2 + 17 * (x * x * y - z * y) + 36 * x
    factor = f(x[0], x[1], x[2])

    assert np.allclose(factor.value([10, 20, 30]), f(10, 20, 30))
