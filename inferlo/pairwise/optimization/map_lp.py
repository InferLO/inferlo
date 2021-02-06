# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from collections import namedtuple
import cvxpy as cp

from inferlo import PairWiseFiniteModel

map_lp_result = namedtuple('map_lp_result', ['upper_bound',
                                             'lower_bound',
                                             'node_beliefs',
                                             'edge_beliefs',
                                             'normalization_duals',
                                             'marginalization_duals'
                                             ])


def map_lp(model: PairWiseFiniteModel) -> map_lp_result:
    """LP relaxation of MAP problem for pairwise model.

    This function implements linear programming (LP) relaxation
    of maximum a posteriori assignment problem (MAP) for
    pairwise graphical model with finite alphabet.

    The goal of MAP estimation is to find most probable
    assignment of original variables by maximizing probability
    density function. For the case of pairwise finite model it
    reduces to maximization of quadratic function over finite
    field.

    For every node, we introduce Q non-negative belief variables
    where Q is the size of the alphabet. Every such variable
    is our 'belief' that variable at node takes particular value.

    Analogously, for every edge we introduce Q*Q pairwise beliefs.

    For both node and edge beliefs we require normalization
    constraints: 1) for every node, the sum of beliefs equals one
    and 2) for every edge the sum of edge-beliefs equals one.

    We also add marginalization constraint: for every edge,
    summing edge beliefs over one of the nodes must equal
    to the node belief of the second node.

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
      ``node_beliefs`` - optimal values of node beliefs;
      ``edge_beliefs`` - optimal values of edge beliefs;
      ``normalization_duals`` - optimal values of dual variables that
      correspond to normalization constraints;
      ``marginalization_duals`` - optimal values of dual variables that
      correspond to marginalization constraints.
    """

    edge_list = model.edges
    node_beliefs = cp.Variable((model.gr_size, model.al_size), nonneg=True)
    edge_beliefs = []
    for edge in edge_list:
        edge_beliefs.append(
            cp.Variable((model.al_size, model.al_size), nonneg=True))
    objective = 0
    constraints = []

    # add field in every node
    for node in range(model.gr_size):
        for letter in range(model.al_size):
            objective += model.field[node, letter] * node_beliefs[node, letter]

    # add pairwise interactions
    # a and b iterate over all values of the finite field
    for edge in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(edge[0], edge[1])
                objective += J[a, b] * edge_beliefs[
                    edge_list.index(edge)][a, b]

    # normalization constraints
    for edge in edge_list:
        constraints += [sum([edge_beliefs[edge_list.index(edge)][a, b]
                             for a in range(model.al_size)
                             for b in range(model.al_size)]) == 1]

    # marginalization constraints
    for edge in edge_list:
        for a in range(model.al_size):
            marginal_left = 0.0
            for b in range(model.al_size):
                marginal_left += edge_beliefs[edge_list.index(edge)][a, b]
            constraints += [marginal_left == node_beliefs[edge[0], a]]

        for a in range(model.al_size):
            marginal_right = 0.0
            for b in range(model.al_size):
                marginal_right += edge_beliefs[edge_list.index(edge)][b, a]
            constraints += [marginal_right == node_beliefs[edge[1], a]]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    normal_dual_vars = [constraints[i].dual_value
                        for i in range(len(edge_list))]
    marginal_dual_vars = [constraints[i].dual_value
                          for i in range(len(edge_list), len(constraints))]
    dual_objective = sum(normal_dual_vars)

    return map_lp_result(
        upper_bound=prob.value,
        lower_bound=dual_objective,
        node_beliefs=node_beliefs.value,
        edge_beliefs=[edge.value for edge in edge_beliefs],
        normalization_duals=normal_dual_vars,
        marginalization_duals=marginal_dual_vars
    )
