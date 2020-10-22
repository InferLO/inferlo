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
    """MAP linear programming relaxation for
            normal factor graph with finite alphabet
    """

    edge_list = model.edges
    node_beliefs = cp.Variable((model.gr_size, model.al_size), nonneg=True)
    edge_beliefs = []
    for e in edge_list:
        edge_beliefs.append(
            cp.Variable((model.al_size, model.al_size), nonneg=True))
    obj = 0
    cons = []

    # add field
    for i in range(model.gr_size):
        for a in range(model.al_size):
            obj += model.field[i, a] * node_beliefs[i, a]

    # add pairwise
    # a and b iterate over all values of the finite field
    for e in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(e[0], e[1])
                obj += J[a, b] * edge_beliefs[edge_list.index(e)][a, b]

    # normalization constraints

    for e in edge_list:
        cons += [sum([edge_beliefs[edge_list.index(e)][a, b]
                      for a in range(model.al_size)
                      for b in range(model.al_size)]) == 1]

    # marginalization constraints
    for e in edge_list:
        for a in range(model.al_size):
            expr = 0.0
            for b in range(model.al_size):
                expr += edge_beliefs[edge_list.index(e)][a, b]
            cons += [expr == node_beliefs[e[0], a]]

        for a in range(model.al_size):
            expr = 0.0
            for b in range(model.al_size):
                expr += edge_beliefs[edge_list.index(e)][b, a]
            cons += [expr == node_beliefs[e[1], a]]

    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS, eps=1e-8)

    normal_dual_vars = [cons[i].dual_value for i in range(len(edge_list))]
    marginal_dual_vars = [cons[i].dual_value
                          for i in range(len(edge_list), len(cons))]
    dual_objective = sum(normal_dual_vars)

    return map_lp_result(
        upper_bound=prob.value,
        lower_bound=dual_objective,
        node_beliefs=node_beliefs.value,
        edge_beliefs=[e.value for e in edge_beliefs],
        normalization_duals=normal_dual_vars,
        marginalization_duals=marginal_dual_vars
    )
