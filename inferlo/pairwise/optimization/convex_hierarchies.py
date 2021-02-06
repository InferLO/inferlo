# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from collections import namedtuple
from itertools import product, combinations
import cvxpy as cp
import networkx as nx

from inferlo import PairWiseFiniteModel

sherali_adams_result = namedtuple('sherali_adams_result', ['upper_bound',
                                                           'projection'
                                                           ])


def sherali_adams(model: PairWiseFiniteModel, level=3) -> sherali_adams_result:
    """
    This is an implementation of Sherali-Adams hierarchy, also called
    lift-and-project method. This method produces hierarchy of linear
    programming (LP) relaxations for the most probable state estimation
    (MAP problem).

    Let k be the level of hierarchy. Then for every cluster of variables
    of size k, we introduce a new variable. All these variables,
    called lifted variables, should satisfy normalization,
    marginalization constraints and be non-negative.

    After solving the corresponding LP, we get upper bound to the
    energy function at a most probable state and extract projected
    variable that correspond to single nodes of the model.

    More on LP hierarchies may be found in D.Sontag's thesis
    "Approximate Inference in Graphical Models using LP
    relaxations".
    https://people.csail.mit.edu/dsontag/papers/sontag_phd_thesis.pdf
    """
    al_size = model.al_size
    var_size = model.gr_size
    edge_list = model.edges

    # check level value
    if ((not isinstance(level, int)) or (level < 3) or (level > var_size)):
        print("Incorrect value of hierarchy level! Setting level to 3..")
        level = 3

    # introduce cluster variables and constraints
    clusters = {}
    constraints = []
    for cluster_size in range(1, level + 1):
        variables = list(combinations(range(var_size), cluster_size))
        values = list(product(range(al_size), repeat=cluster_size))

        for cluster_ids in variables:
            cluster = {}
            for x in values:
                cluster[x] = cp.Variable(nonneg=True)
            clusters[cluster_ids] = cluster

            # add normalization constraint
            constraints += [sum(list(cluster.values())) == 1]

            # add marginalization constraints
            for cluster_subset_size in range(1, cluster_size):
                all_cluster_subsets = list(combinations(
                    list(cluster_ids), cluster_subset_size))
                subset_values = list(product(
                    range(al_size), repeat=cluster_subset_size))

                for subset in all_cluster_subsets:
                    subset_ids = list(subset)
                    for subset_x in subset_values:
                        marginal_sum = 0.0
                        for value, cp_variable in cluster.items():
                            consistency = [value[
                                cluster_ids.index(subset_ids[i])
                            ] == subset_x[i]
                                for i in range(cluster_subset_size)]
                            if sum(consistency) == len(subset):
                                marginal_sum += cp_variable

                        constraints += [marginal_sum ==
                                        clusters[subset][subset_x]]

    # define objective
    objective = 0

    # add field in every node
    for node in range(var_size):
        for letter in range(al_size):
            objective += model.field[node, letter] * \
                clusters[(node,)][(letter,)]

    # add pairwise interactions
    # a and b iterate over all values of the finite field
    for edge in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(edge[0], edge[1])
                if (edge[0] <= edge[1]):
                    objective += J[a, b] * clusters[(edge[0], edge[1])][(a, b)]
                else:
                    objective += J[a, b] * clusters[(edge[1], edge[0])][(b, a)]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    projected_variables = []
    for node in range(var_size):
        projected_variables.append(clusters[(node,)].values())

    return sherali_adams_result(
        upper_bound=prob.value,
        projection=projected_variables
    )


def minimal_cycle(model: PairWiseFiniteModel) -> sherali_adams_result:
    """
    This is an implementation of Cycle relaxation. In fact,
    cycle relaxation is a simplified version of the third
    level of Sherali-Adams hierarchy. It may result in worse
    upper bounds but has much fewer constraints and solved
    faster.

    The idea behind this relaxation is to consider all cycles
    in the graph and to add cycle-to-edge marginalization
    constraints to the local consistency constraints.

    In some cases cycle relaxation coincides with the third
    level of Sherali-Adams. It may also be shown that instead
    of all cycles constraints, it is enough to consider only
    chordless cycles. In this code we consider the set of
    minimal cycles found by networkx.

    More on LP hierarchies may be found in D.Sontag's thesis
    "Approximate Inference in Graphical Models using LP
    relaxations".
    https://people.csail.mit.edu/dsontag/papers/sontag_phd_thesis.pdf
    """
    al_size = model.al_size
    var_size = model.gr_size
    edge_list = model.edges

    # check if graph is acyclic
    if model.is_graph_acyclic():
        print("Graph is acyclic!")
        print("Cycle relaxation is equivalent to the local LP relaxation.")

    # introduce cluster variables and constraints
    clusters = {}
    constraints = []

    # add local consistency constraints first
    for cluster_size in [1, 2]:
        variables = list(combinations(range(var_size), cluster_size))
        values = list(product(range(al_size), repeat=cluster_size))

        for cluster_ids in variables:
            cluster = {}
            for x in values:
                cluster[x] = cp.Variable(nonneg=True)
            clusters[cluster_ids] = cluster

            # add normalization constraint
            constraints += [sum(list(cluster.values())) == 1]

            # add marginalization constraints
            for cluster_subset_size in range(1, cluster_size):
                all_cluster_subsets = list(combinations(
                    list(cluster_ids), cluster_subset_size))
                subset_values = list(product(
                    range(al_size), repeat=cluster_subset_size))

                for subset in all_cluster_subsets:
                    subset_ids = list(subset)
                    for subset_x in subset_values:
                        marginal_sum = 0.0
                        for value, cp_variable in cluster.items():
                            consistency = [value[
                                cluster_ids.index(subset_ids[i])
                            ] == subset_x[i]
                                for i in range(cluster_subset_size)]
                            if sum(consistency) == len(subset):
                                marginal_sum += cp_variable

                        constraints += [marginal_sum ==
                                        clusters[subset][subset_x]]

    # add cycle consistency
    graph = model.get_graph()
    cycles = nx.cycle_basis(graph)

    for cycle in cycles:
        cycle.append(cycle[0])
        cycle_edges = []
        for i in range(len(cycle) - 1):
            edge = sorted([cycle[i], cycle[i + 1]])
            cycle_edges.append(tuple(edge))

        cycle_values = list(product(range(al_size), repeat=len(cycle)))
        cluster_ids = tuple(sorted(cycle))
        cluster = {}
        for x in cycle_values:
            cluster[x] = cp.Variable(nonneg=True)
        clusters[cluster_ids] = cluster

        # add normalization constraint
        constraints += [sum(list(cluster.values())) == 1]

        # add marginalization constraints
        for edge in cycle_edges:
            edge_values = list(product(range(al_size), repeat=2))
            first_node_position = cluster_ids.index(edge[0])
            second_node_position = cluster_ids.index(edge[1])
            for edge_value in edge_values:
                marginal_sum = 0.0
                edge_variable = clusters[edge][edge_value]
                for x in cycle_values:
                    if (x[first_node_position] == edge_value[0]) \
                            and (x[second_node_position] == edge_value[1]):
                        marginal_sum += clusters[cluster_ids][x]
                constraints += [marginal_sum == edge_variable]

    # define objective
    objective = 0

    # add field in every node
    for node in range(var_size):
        for letter in range(al_size):
            objective += model.field[node, letter] * \
                clusters[(node,)][(letter,)]

    # add pairwise interactions
    # a and b iterate over all values of the finite field
    for edge in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(edge[0], edge[1])
                if (edge[0] <= edge[1]):
                    objective += J[a, b] * clusters[(edge[0], edge[1])][(a, b)]
                else:
                    objective += J[a, b] * clusters[(edge[1], edge[0])][(b, a)]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    projected_variables = []
    for node in range(var_size):
        projected_variables.append(clusters[(node,)].values())

    return sherali_adams_result(
        upper_bound=prob.value,
        projection=projected_variables
    )


lasserre_result = namedtuple('lasserre_result', ['upper_bound',
                                                 'moment_matrix'
                                                 ])


class Indexing:
    """
    This class is used to correctly extract indices of moment
    matrices by cluster indices and corresponding values.

    The first index corresponds to empty set and equals one.
    Then we list all clusters of size one. For the first
    cluster we have q indicator variables where q is the
    size of alphabet. Totally we have n*q clusters of size
    one where n is the number of variables in model.

    Then we list clusters by size one by one. Totally
    we have q^t * Bin(n, t) indices that correspond to
    clusters of size t. We denote Bin(n, t)  the number of
    combinations with t elements.
    """
    clusters = {}
    values = {}
    first_index = [0, 1]

    def __init__(self, n, q, level=1):
        for current_level in range(1, level + 1):
            clusters = list(combinations(range(n), current_level))
            values = list(product(range(q), repeat=current_level))
            self.values[current_level] = values
            self.clusters[current_level] = clusters
            self.first_index.append(self.first_index[-1] +
                                    len(values) * len(clusters))

    def find(self, cluster, value):
        """
        Computes index of cluster in the moment matrix.
        A method to index clusters of variables described
        in the docstring for Indexing class.
        """
        info = {}
        level = len(list(cluster))
        info['level'] = level
        info['level_index'] = self.first_index[info['level']]
        info['cluster_index'] = self.clusters[level].index(cluster)
        info['value_index'] = self.values[level].index(value)
        info['value_total'] = len(self.values[level])
        info['index'] = info['level_index'] + \
            info['cluster_index'] * info['value_total'] + \
            info['value_index']

        return info


def compatible(cluster_a, value_a, cluster_b, value_b):
    """
    Checks compatibility of clusters of variables.
    Compatibility means that values agree on common
    variables.
    """
    for node in list(cluster_a):
        position_a = cluster_a.index(node)
        if node in list(cluster_b):
            position_b = cluster_b.index(node)
            if value_a[position_a] != value_b[position_b]:
                return False
    return True


def union(cluster_a, value_a, cluster_b, value_b):
    """
    Returns union of compatible clusters.
    Compatibility means that values agree on common
    variables.

    The set of variable indices is the set union
    of variables in both clusters.
    """
    cluster_a_list = list(cluster_a)
    value_a_list = list(value_a)
    cluster_b_list = list(cluster_b)
    value_b_list = list(value_b)
    value_ab = []
    cluster_ab = sorted(set(cluster_a_list + cluster_b_list))
    for i in range(len(cluster_ab)):
        if cluster_ab[i] in cluster_a_list:
            index = cluster_a_list.index(cluster_ab[i])
            value_ab.append(value_a_list[index])
        else:
            index = cluster_b_list.index(cluster_ab[i])
            value_ab.append(value_b_list[index])
    return tuple(cluster_ab), tuple(value_ab)


def lasserre(model: PairWiseFiniteModel, level=1) -> lasserre_result:
    """
    This is an implementation of Lasserre hierarchy. This method p
    roduces hierarchy of semidefinite programming (SDP) relaxations
    for the most probable state estimation (MAP problem).

    Let k be the level of hierarchy. Then we consider all clusters
    of variables of size k, and construct their moment matrix.
    Denote alphabet size by q. Then for every cluster of size t we
    introduce q^t indicator binary variables and get moment matrix.

    After solving the corresponding SDP, we get upper bound to the
    energy function at a most probable state and extract the
    resulting moment matrix.

    More on Lasserre hierarchies may be found in M.Wainwright and
    M. Jordan's book "Graphical Models, Exponential Families, and
    Variational Inference", section 9.
    https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf
    """
    var_size = model.gr_size
    al_size = model.al_size
    edge_list = model.edges

    # check level value
    if (not isinstance(level, int)) or (level < 1) or (level > var_size):
        print("Incorrect value of hierarchy level! Setting level to 1..")
        level = 1

    # introduce cluster variables
    clusters = {}
    constraints = []
    ind = Indexing(var_size, al_size, level=level)
    for cluster_size in range(1, level + 1):
        for cluster_ids in ind.clusters[cluster_size]:
            cluster = {}
            for x in ind.values[cluster_size]:
                cluster[x] = cp.Variable(nonneg=True)
            clusters[cluster_ids] = cluster

            # add normalization constraint
            constraints += [sum(list(cluster.values())) == 1]

    moment_matrix = cp.Variable((ind.first_index[-1], ind.first_index[-1]),
                                symmetric=True)
    constraints += [moment_matrix >> 0]
    constraints += [moment_matrix[0, 0] == 1]

    # add moment matrix elements constraints
    clusters_list = list(clusters.keys())
    for cluster_a in clusters_list:
        for value_a in clusters[cluster_a].keys():
            a = ind.find(cluster_a, value_a)
            a_variable = clusters[cluster_a][value_a]
            constraints += [moment_matrix[0, a['index']] == a_variable]

            for cluster_b in clusters_list:
                for value_b in clusters[cluster_b].keys():
                    b = ind.find(cluster_b, value_b)

                    if (cluster_a == cluster_b) and (value_a == value_b):
                        constraints += [moment_matrix[
                            a['index'], b['index']
                        ] == a_variable]
                    elif (cluster_a != cluster_b) and \
                            (compatible(cluster_a, value_a,
                                        cluster_b, value_b)):
                        cluster_ab, value_ab = union(cluster_a, value_a,
                                                     cluster_b, value_b)
                        if cluster_ab in clusters.keys():
                            variable_ab = clusters[cluster_ab][value_ab]
                            constraints += [moment_matrix[
                                a['index'], b['index']
                            ] == variable_ab]

    objective = 0.0

    # add field in every node
    for node in range(var_size):
        for letter in range(al_size):
            objective += model.field[node, letter] * \
                clusters[(node,)][(letter,)]

    # add pairwise interactions
    # a and b iterate over all values of the finite field
    for edge in edge_list:
        for a in range(model.al_size):
            for b in range(model.al_size):
                J = model.get_interaction_matrix(edge[0], edge[1])
                a_info = ind.find((edge[0],), (a,))
                a_index = a_info['index']
                b_info = ind.find((edge[1],), (b,))
                b_index = b_info['index']
                objective += J[a, b] * moment_matrix[a_index, b_index]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, eps=1e-8)

    return lasserre_result(
        upper_bound=prob.value,
        moment_matrix=moment_matrix.value
    )
