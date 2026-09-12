# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np
import pytest

from .factor import Factor


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("log_values", [
    [0.0, -1000.0, -np.inf],
    [-np.inf, -np.inf, -np.inf],
])
def test_add_zero_and_small_values(log_values, inplace):
    """Addition preserves tiny values and handles zero plus zero."""
    original = np.array(log_values)
    left = Factor("left", ["x"], log_values=original.copy())
    right = Factor("right", ["x"], log_values=original.copy())

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = left.add(right, inplace=inplace)

    if inplace:
        assert result is None
        result = left
    else:
        np.testing.assert_array_equal(left.log_values, original)
    np.testing.assert_allclose(result.log_values, original + np.log(2.0))
    np.testing.assert_array_equal(right.log_values, original)


def test_add_transposed_factor():
    left = Factor("left", ["x", "y"], values=np.array([[0., 1.], [2., 3.]]))
    right = Factor("right", ["y", "x"], values=np.array([[0., 4.], [5., 6.]]))

    with np.errstate(divide="raise", invalid="raise"):
        result = left + right

    assert result.variables == ["x", "y"]
    np.testing.assert_allclose(result.values, [[0., 6.], [6., 9.]])
