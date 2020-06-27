from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from inferlo import PairWiseFiniteModel

from inferlo.pairwise.testing import grid_potts_model, tree_potts_model, \
    line_potts_model
from inferlo.pairwise.optimization.path_dp import max_lh_path_dp

def LPrelaxation(model: PairWiseFiniteModel):# -> np.ndarray:
    """Max Likelihood for pairwise model by solving LP relaxation.

    :param model: Model for which to find most likely state.
    :return: Solution of Linear Programming relaxation of the Max Likelihood problem.

    1) Reformulates the original problem of maximizing sum F[i][X_i] + 0.5*sum J[i][j][X[i]][X[j]])
    as a binary optimization problem by introducing new variables y_i,a and z_i,j,a,b:
        maximize (sum_i sum_a F[i][a] * y_i,a) + (sum_i sum_j sum_a sum_b 0.5*sum J[i][j][a][b] * z_i,j,a,b))
        so that y_i,a in {0, 1}
                z_i,j,a,b in {0, 1}
                (for all i) sum_a y_i,a = 1
                z_i,j,a,b <= y_i,a
                z_i,j,a,b <= y_j,b
                z_i,j,a,b >= y_i,a + y_j,b - 1

    2) Solves LP relaxation of the problem above by relaxing binary constraints to
                y_i,a in [0, 1]
                z_i,j,a,b in [0, 1]

    Note that z will be reshaped to matrix since cvxpy do not support tensor variables.
    """

    y = cp.Variable((model.gr_size, model.al_size), nonneg=True)
    z_mat = cp.Variable((model.gr_size * model.al_size, model.gr_size * model.al_size), nonneg=True)
    obj = 0
    cons = []

    #add field
    for i in range(model.gr_size):
        for a in range(model.al_size):
            obj += model.field[i, a] * y[i,a]

    #add pairwise
    for i in range(model.gr_size):
        for j in range(model.gr_size):
            for a in range(model.al_size):
                for b in range(model.al_size):
                    if (model.has_edge(i, j)):
                        J = model.get_interaction_matrix(i, j)
                        obj += 0.5 * J[a, b] * z_mat[i*model.al_size + a,j*model.al_size + b]

    #add y constraints:
    for i in range(model.gr_size):
        cons += [sum(y[i, :]) == 1]
        for a in range(model.al_size):
            cons += [y[i,a] <= 1]

    #add z constraints
    for i in range(model.gr_size):
        for j in range(model.gr_size):
            for a in range(model.al_size):
                for b in range(model.al_size):
                    cons += [z_mat[i*model.al_size + a,j*model.al_size + b] <= 1]
                    cons += [z_mat[i*model.al_size + a,j*model.al_size + b] <= y[i,a]]
                    cons += [z_mat[i*model.al_size + a,j*model.al_size + b] <= y[j,b]]
                    cons += [z_mat[i*model.al_size + a,j*model.al_size + b] >= y[i,a] + y[j,b] - 1]

    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS)

    return prob.value