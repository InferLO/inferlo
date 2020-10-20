from collections import namedtuple
from itertools import product
import cvxpy as cp
import numpy as np


from inferlo.forney.nfg_model import NormalFactorGraphModel

map_lp_result = namedtuple('map_lp_result', ['upper_bound',
                                             'lower_bound',
                                             'factor_beliefs',
                                             'variable_beliefs',
                                             'normalization_duals',
                                             'marginalization_duals'
                                             ])


def map_lp(model: NormalFactorGraphModel) -> map_lp_result:
    """MAP linear programming relaxation for
        pairwise model with finite alphabet
    """
    al_size = model._default_domain.size()
    var_size = len(model.edges)

    factor_list = model.factors

    variable_beliefs = cp.Variable((var_size, al_size), nonneg=True)
    factor_beliefs = []
    for f in range(len(factor_list)):
        curr_factor_beliefs = {}
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[f].var_idx)))
        for x in var_vals:
            curr_factor_beliefs[x] = cp.Variable(nonneg=True)
        factor_beliefs.append(curr_factor_beliefs)
    obj = 0
    cons = []

    # add objective
    for f in range(len(factor_list)):
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[f].var_idx)))
        for x in var_vals:
            value = factor_list[f].value(list(x))
            if (value != 0):
                obj += np.log(value) * factor_beliefs[f][x]

    # normalization constraints
    for f in range(len(factor_list)):
        expr = 0
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[f].var_idx)))
        for x in var_vals:
            expr += factor_beliefs[f][x]
        cons += [expr == 1]

    # marginalization constraints
    for f in range(len(factor_list)):
        for v in range(len(factor_list[f].var_idx)):
            for a in range(al_size):
                expr = 0
                var_vals = list(product(range(al_size),
                                        repeat=len(factor_list[f].var_idx)))
                for x in var_vals:
                    temp = list(x)
                    temp[v] = a
                    x = tuple(temp)
                var_vals = list(set(var_vals))

                for x in var_vals:
                    expr += factor_beliefs[f][x]
                cons += [expr ==
                         variable_beliefs[factor_list[f].var_idx[v], a]]

    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS)

    normal_dual_vars = [cons[i].dual_value for i in range(len(factor_list))]
    marginal_dual_vars = [cons[i].dual_value
                          for i in range(len(factor_list), len(cons))]
    dual_objective = sum(normal_dual_vars)

    factor_beliefs_result = []
    for f in range(len(factor_list)):
        curr_factor_beliefs_result = []
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[f].var_idx)))
        for x in var_vals:
            curr_factor_beliefs_result.append(factor_beliefs[f][x].value)

    return map_lp_result(
        upper_bound=prob.value,
        lower_bound=dual_objective,
        factor_beliefs=factor_beliefs_result,
        variable_beliefs=variable_beliefs.value,
        normalization_duals=normal_dual_vars,
        marginalization_duals=marginal_dual_vars
    )
