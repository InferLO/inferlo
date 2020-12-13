# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

from collections import namedtuple
import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel


LPRelaxResult = namedtuple('LPRelaxResult', ['upper_bound',
                                             'lower_bound',
                                             'rounded_solution'])


def lp_relaxation(model: PairWiseFiniteModel) -> LPRelaxResult:
    """Max Likelihood for pairwise model by solving LP relaxation.

    1) Reformulates the original problem of maximizing
    ``sum F[i][X_i] + 0.5*sum J[i][j][X[i]][X[j]])``
    as a binary optimization problem by introducing new variables
    ``y_i``, ``a`` and ``z_i,j,a,b``::

        maximize (sum_i sum_a F[i][a] * y_i,a) +
                 (sum_i sum_j sum_a sum_b 0.5*sum J[i][j][a][b] * z_i,j,a,b))
        subject to y_i,a in {0, 1}
                z_i,j,a,b in {0, 1}
                (for all i) sum_a y_i,a = 1
                z_i,j,a,b <= y_i,a
                z_i,j,a,b <= y_j,b
                z_i,j,a,b >= y_i,a + y_j,b - 1

    2) Solves the LP relaxation by relaxing binary constraints
    to::

        y_i,a in [0, 1]
        z_i,j,a,b in [0, 1]

    Note that ``z`` will be reshaped to matrix (cvxpy does not support tensor
    variables).

    :param model: Model for which to find most likely state.
    :return: Solution of LP relaxation of the Max Likelihood problem.
    """
    edge_list = model.edges

    y = cp.Variable((model.gr_size, model.al_size), nonneg=True)
    z = []
    for e in edge_list:
        z.append(cp.Variable((model.al_size, model.al_size), nonneg=True))
    obj = 0
    cons = []

    # add field
    for i in range(model.gr_size):
        for a in range(model.al_size):
            obj += model.field[i, a] * y[i, a]

    # add pairwise
    # a and b iterate over all values of the finite field
    for e in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(e[0], e[1])
                obj += J[a, b] * z[edge_list.index(e)][a, b]

    # add y constraints:
    for i in range(model.gr_size):
        cons += [sum(y[i, :]) == 1]
        for a in range(model.al_size):
            cons += [y[i, a] <= 1]

    # add z constraints
    for e in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                cons += [z[edge_list.index(e)][a, b] <= 1]
                cons += [z[edge_list.index(e)][a, b] <= y[e[0], a]]
                cons += [z[edge_list.index(e)][a, b] <= y[e[1], b]]
                cons += [z[edge_list.index(e)][a, b] >=
                         y[e[0], a] + y[e[1], b] - 1]

    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS)

    # in fact, rounding LP relaxation is sampling
    rounded = np.array(np.random.randint(low=0,
                                         high=model.al_size,
                                         size=model.gr_size))
    lower = np.log(model.evaluate(rounded))

    return LPRelaxResult(
        upper_bound=prob.value,
        lower_bound=lower,
        rounded_solution=rounded
    )
