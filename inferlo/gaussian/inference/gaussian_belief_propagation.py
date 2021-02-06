# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np


def gaussian_BP_iteration(model, J_delta, h_delta, J_wave, h_wave):
    """Single message passing iteration under fully parrallel schedule.
    The messages in Gaussian models can be parameterized in information form
    so that the fixed-point equations can be stated in terms of these
    information parameters.
    """
    diff_max = .0
    new_J_delta = np.zeros((model.n, model.n))
    new_h_delta = np.zeros((model.n, model.n))
    new_J_wave = np.zeros((model.n, model.n))
    new_h_wave = np.zeros((model.n, model.n))

    for i in range(model.n):
        J_wave_i = model.J[i][i]
        h_wave_i = model.h[i]
        for j in model.G[i]:
            J_wave_i += J_delta[j][i]
            h_wave_i += h_delta[j][i]
        for j in model.G[i]:
            old_J_wave = J_wave[i][j]
            old_h_wave = h_wave[i][j]
            old_J_delta = J_delta[i][j]
            old_h_delta = h_delta[i][j]
            new_J_wave[i][j] = J_wave_i - J_delta[j][i]
            new_h_wave[i][j] = h_wave_i - h_delta[j][i]
            new_J_delta[i][j] = -((model.J[i][j])**2) / new_J_wave[i][j]
            new_h_delta[i][j] =\
                -(model.J[i][j] * new_h_wave[i][j]) / new_J_wave[i][j]
            diff1 = abs(old_J_wave - new_J_wave[i][j])
            diff2 = abs(old_h_wave - new_h_wave[i][j])
            diff3 = abs(old_J_delta - new_J_delta[i][j])
            diff4 = abs(old_h_delta - new_h_delta[i][j])
            diff_max = max(diff_max, max(diff1, diff2, diff3, diff4))

    return diff_max, new_J_delta, new_h_delta, new_J_wave, new_h_wave


def gaussian_BP(model, tol=1e-9, max_iter=1e5):
    """Inference with Gaussian Belief Propagation.

    Exact results are guaranteed only for models which satisfy one
    of sufficient conditions (such as diagonal dominance, etc.)

    For details please refer to:\
        http://ssg.mit.edu/group/willsky/publ_pdfs/185_pub_MLR.pdf

    :param model: Gaussian graphical model for which to perform inference.
    :param tol: Tolerance factor for check whether we reached fixed point
    :param max_iter: Maximal number of iterations, where each iteration is
        a single message-passing round under fully parrallel schedule.
    :return: two vectors, where the first vector is a
        vector of obtained marginal means and the second vector is a vector
        of obtained marginal variances.
    """
    J_delta = np.zeros((model.n, model.n))
    h_delta = np.zeros((model.n, model.n))
    J_wave = np.zeros((model.n, model.n))
    h_wave = np.zeros((model.n, model.n))
    t = 0

    while t < max_iter:
        cur_diff, J_delta, h_delta, J_wave, h_wave =\
            gaussian_BP_iteration(model, J_delta, h_delta, J_wave, h_wave)
        if cur_diff < tol:
            break
        t += 1

    mu = np.zeros((model.n))
    P_diag = np.zeros((model.n))

    for i in range(model.n):
        J_wave_i = model.J[i][i]
        h_wave_i = model.h[i]
        for k in model.G[i]:
            J_wave_i += J_delta[k][i]
            h_wave_i += h_delta[k][i]
        mu[i] = h_wave_i / J_wave_i
        P_diag[i] = 1.0 / J_wave_i

    return mu, P_diag
