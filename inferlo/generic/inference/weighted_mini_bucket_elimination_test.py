# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np
import pytest

from .factor import Factor
from .graphical_model import GraphicalModel
from .weighted_mini_bucket_elimination import WeightedMiniBucketElimination


@pytest.mark.parametrize("second_values, converged", [
    ([0.25, 0.75, 0.0], True),
    ([0.75, 0.25, 0.0], False),
    ([0.0, 1.0, 0.0], False),
])
def test_reparameterization_with_zero_probabilities(second_values, converged):
    """Shared zero states must not mask disagreement on supported states."""
    model = GraphicalModel(["x"], [
        Factor("original", ["x"], values=np.array([0.25, 0.75, 0.0]))])
    replicas = GraphicalModel(["x0", "x1"], [
        Factor("first", ["x0"], values=np.array([0.25, 0.75, 0.0])),
        Factor("second", ["x1"], values=np.array(second_values)),
    ])
    algorithm = WeightedMiniBucketElimination(
        model, elimination_order=["x"], renormalized_model=replicas,
        renormalized_elimination_order=["x0", "x1"],
        variables_replicated_from_={"x": ["x0", "x1"]}, base_logZ=0.0)

    with np.errstate(divide="raise", invalid="raise"):
        result = algorithm._update_reparameterization_for_("x")

    assert result == converged
    # With equal Holder weights, the update makes both factors' normalized
    # marginals equal, while preserving their zero-probability states.
    first = algorithm._get_marginals_upper_to("x0")
    second = algorithm._get_marginals_upper_to("x1")
    np.testing.assert_allclose(first.values, second.values)
    assert first.values[-1] == second.values[-1] == 0.0
