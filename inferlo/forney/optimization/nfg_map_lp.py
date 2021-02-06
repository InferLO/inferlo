# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
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
    """LP relaxation of MAP problem for NFG model.

    This function implements linear programming (LP) relaxation
    of maximum a posteriori assignment problem (MAP) for
    normal factor graph with finite alphabet.

    The goal of MAP estimation is to find most probable
    assignment of original variables by maximizing probability
    density function. For the case of pairwise finite model it
    reduces to maximization of polynomial over finite
    field.

    For every variable, we introduce Q non-negative belief variables
    where Q is the size of the alphabet. Every such variable
    is our 'belief' that variable at node takes particular value.

    Analogously, for every factor we introduce Q^deg beliefs
    where deg is a number of variables in factor.

    For both node and edge beliefs we require normalization
    constraints: 1) for every variable, the sum of beliefs
    equals one and 2) for every factor the sum of beliefs
    equals one.

    We also add marginalization constraint: for every factor,
    summing factor beliefs over all except one node must equal
    to the node belief at that node.

    Finally we get a linear program and its solution is an
    upper bound on the MAP value. We restore the lower bound
    on MAP value as the solution of the dual relaxation.

    More details may be found in "MAP Estimation,
    Linear Programming and BeliefPropagation with
    Convex Free Energies" by Yair Weiss, Chen Yanover and Talya
    Meltzer. https://arxiv.org/pdf/1206.5286.pdf

    :param model: Model for which to solve MAP problem.

    :return: Object with the following fields:
      ``upper_bound`` - upper bound on MAP value (solution of LP);
      ``lower_bound`` - lower bound on MAP value (dual solution);
      ``factor_beliefs`` - optimal values of factor beliefs;
      ``variable_beliefs`` - optimal values of variable beliefs;
      ``normalization_duals`` - optimal values of dual variables that \
      correspond to normalization constraints;
      ``marginalization_duals`` - optimal values of dual variables that
      correspond to marginalization constraints.
    """

    al_size = model._default_domain.size()
    var_size = len(model.edges)

    factor_list = model.factors

    variable_beliefs = cp.Variable((var_size, al_size), nonneg=True)
    factor_beliefs = []
    for factor in range(len(factor_list)):
        curr_factor_beliefs = {}
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[factor].var_idx)))
        for x in var_vals:
            curr_factor_beliefs[x] = cp.Variable(nonneg=True)
        factor_beliefs.append(curr_factor_beliefs)
    objective = 0
    constraints = []

    # add objective
    for factor in range(len(factor_list)):
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[factor].var_idx)))
        for x in var_vals:
            value = factor_list[factor].value(list(x))
            if (value != 0):
                objective += np.log(value) * factor_beliefs[factor][x]

    # normalization constraints
    for factor in range(len(factor_list)):
        expr = 0
        var_vals = list(product(range(al_size),
                                repeat=len(factor_list[factor].var_idx)))
        for x in var_vals:
            expr += factor_beliefs[factor][x]
        constraints += [expr == 1]

    # marginalization constraints
    for factor in range(len(factor_list)):
        for variable in range(len(factor_list[factor].var_idx)):
            for a in range(al_size):
                expr = 0
                var_vals = list(
                    product(range(al_size),
                            repeat=len(factor_list[factor].var_idx)))
                for x in var_vals:
                    temp = list(x)
                    temp[variable] = a
                    x = tuple(temp)
                var_vals = list(set(var_vals))

                for x in var_vals:
                    expr += factor_beliefs[factor][x]
                constraints += [expr ==
                                variable_beliefs[
                                    factor_list[factor].var_idx[variable], a]]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS)

    normal_dual_vars = [constraints[i].dual_value
                        for i in range(len(factor_list))]
    marginal_dual_vars = [constraints[i].dual_value
                          for i in range(len(factor_list), len(constraints))]
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
