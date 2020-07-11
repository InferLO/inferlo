# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inferlo.pairwise.inference.path_dp import prepare_path_dp
from inferlo.pairwise.utils import decode_state


if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel


def max_lh_path_dp(model: PairWiseFiniteModel) -> np.ndarray:
    """Max Likelihood for pairwise model with DP on path decomposition.

    :param model: Model for which to find most likely state.
    :return: Most likely state. np.array of ints.
    """
    decomp = prepare_path_dp(model)
    layers_cnt = len(decomp.layers)

    # Forward dynamic programming.
    # Count max likelihood for every "prefix" given fixed state of last
    # layer in prefix.
    # All operations are "+", because everything is a logarithm.
    z = [np.empty(0) for _ in range(layers_cnt)]
    z[0] = decomp.a[0]
    for i in range(1, layers_cnt):
        z[i] = np.max(z[i - 1] + decomp.b[i - 1].T, axis=1) + decomp.a[i]

    # Best state in the last layer.
    r = np.zeros(layers_cnt, dtype=np.int32)
    r[layers_cnt - 1] = np.argmax(z[layers_cnt - 1])

    # Backward dp - restoring bes state in each layer.
    for i in range(layers_cnt - 2, -1, -1):
        r[i] = np.argmax(z[i] + decomp.b[i][:, r[i + 1]])

    # Now, decode result into states in individual variables.
    result = np.zeros(model.gr_size, dtype=np.int32)
    for i in range(layers_cnt):
        result[decomp.layers[i]] = decode_state(r[i], len(decomp.layers[i]),
                                                model.al_size)
    return result
