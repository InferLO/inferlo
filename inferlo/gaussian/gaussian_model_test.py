# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np
from inferlo.gaussian import GaussianModel


def test_gaussian_BP():
    J = np.array([[4, 2], [2, 4]])
    h = np.array([6, 4])
    G = GaussianModel(J, h)
    mu, P_diag = G.infer()
    mu_true = np.dot(np.linalg.inv(G.J), G.h)
    P_true = np.linalg.inv(G.J)
    P_true_diag = np.array([P_true[i][i] for i in range(len(J))])

    assert np.allclose(mu, mu_true)
    assert np.allclose(P_diag, P_true_diag)
