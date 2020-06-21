# This is a temporary test to make sure CVXPY is properly installed.
# It should be removed once we add GM algorithms using CVXPY and tests for them.

import cvxpy
import numpy as np


def test_quadratic_form():
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(n, n)
    b = np.random.randn(n)

    for solver in ['ECOS', 'SCS', 'OSQP']:
        x = cvxpy.Variable(n)
        objective = cvxpy.Minimize(cvxpy.sum_squares(A @ x - b))
        constraints = [-100 <= x, x <= 100]
        prob = cvxpy.Problem(objective, constraints)
        f_opt = prob.solve(solver=solver)
        x_opt = x.value
        assert np.abs(f_opt) < 1e-5
        assert np.linalg.norm(x_opt - np.linalg.inv(A) @ b) < 0.5


test_quadratic_form()
