# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import networkx
import numpy as np

from inferlo.base.domain import DiscreteDomain
from inferlo.base.factors.discrete_factor import DiscreteFactor
from inferlo.base.generic_graph_model import GenericGraphModel
from inferlo.pairwise.pwf_model import PairWiseFiniteModel


def grid_potts_model(
        height,
        width,
        al_size=3,
        seed=111,
        zero_field=False) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a grid.

    :param height: Heigth of the grid.
    :param width: Wwidth of the grid.
    :param al_size: Alphabet size.
    :param seed: Random seed.
    :param zero_field: Whether model should be zero-field.
    :return: Generated Potts Model.
    """
    np.random.seed(seed)
    gr_size = width * height
    edges_num = 2 * width * height - width - height
    edges = []
    for x in range(height):
        for y in range(width):
            v = x * width + y
            if x != height - 1:
                edges.append((v, v + width))  # down
            if y != width - 1:
                edges.append((v, v + 1))  # right
    field = 0.1 * np.random.random(size=(gr_size, al_size))
    if zero_field:
        field *= 0
    inter = np.random.random(size=(edges_num, al_size, al_size)) * 5.0
    return PairWiseFiniteModel.create(field, edges, inter)


def tree_potts_model(gr_size=5, al_size=2, seed=111, same_j=None,
                     zero_field=False) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a random tree.

    :param gr_size: Size of the graph (number of variables).
    :param al_size: Alphabet size.
    :param seed: Random set.
    :param same_j: If set, interaction matrix for all edges.
    :param zero_field: Whether model should be zero-field.
    :return: Generated Potts Model.
    """
    np.random.seed(seed)
    tree = networkx.generators.trees.random_tree(gr_size, seed=seed)
    model = PairWiseFiniteModel(gr_size, al_size)
    if not zero_field:
        model.set_field(-3.0 + 6.0 * np.random.random((gr_size, al_size)))
    for v1, v2 in tree.edges:
        J = np.random.random((al_size, al_size)) * 5.0
        if same_j is not None:
            J = same_j
        model.add_interaction(v1, v2, J)
    return model


def line_potts_model(gr_size=5, al_size=2, seed=111, same_j=None,
                     zero_field=False) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a line graph.

    :param gr_size: Size of the graph (number of variables).
    :param al_size: Alphabet size.
    :param seed: Random seed.
    :param same_j: If set, interaction matrix for all edges.
    :param zero_field: Whether model should be zero-field.
    :return: Generated model.
    """
    np.random.seed(seed)
    field = np.zeros((gr_size, al_size))
    if not zero_field:
        field = -3.0 + 6.0 * np.random.random(field.shape)
    edges = [[i, i + 1] for i in range(gr_size - 1)]
    inter = np.random.random(size=(gr_size - 1, al_size, al_size)) * 5.0
    if same_j is not None:
        inter = np.tile(same_j, (gr_size - 1, 1, 1))
    return PairWiseFiniteModel.create(field, edges, inter)


def clique_potts_model(gr_size=5, al_size=2, seed=0) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a clique."""
    np.random.seed(seed)
    model = PairWiseFiniteModel(gr_size, al_size)
    model.set_field(-3.0 + 6.0 * np.random.random((gr_size, al_size)))
    for i in range(gr_size):
        for j in range(i + 1, gr_size):
            inter = np.random.random((al_size, al_size)) * 5.0
            model.add_interaction(i, j, inter)
    return model


def pairwise_model_on_graph(graph, al_size=2, zero_field=False, seed=0):
    """Builds random pairwise model with given interaction graph.

    :param graph: Interaction graph. Nodes must be labeled with consecutive
      integers, starting with 0.
    :param al_size: Alphabet size.
    :param zero_field: Whether model should be zero-field.
    :param seed: Random seed.
    :return: Generated model.
    """
    np.random.seed(seed)
    gr_size = len(graph.nodes())
    field = np.random.random(size=(gr_size, al_size))
    if zero_field:
        field *= 0
    edges = np.array(list(graph.edges()))
    interactions = np.random.random(size=(len(edges), al_size, al_size))
    return PairWiseFiniteModel.create(field, edges, interactions)


def ising_model_on_graph(graph: networkx.Graph,
                         field_range=0.1,
                         interaction_range=0.1,
                         seed=0) -> PairWiseFiniteModel:
    """Builds random Ising model on given graph.

    :param graph: Graph for the model. Vertices are variables,
      nodes are interactions.
    :param field_range: Fields will be sampled uniformly from
      ``[-field_range, field_range]``.
    :param interaction_range: Interactions will be sampled uniformly from
      ``[-interaction_range, interaction_range]``.
    :param seed: Random seed.
    :return: Generated model.
    """
    # Remap arbitrary variable labels to integers.
    nodes = list(graph.nodes)
    var_index = {nodes[i]: i for i in range(len(nodes))}
    edges = [(var_index[u], var_index[v]) for u, v in graph.edges()]

    np.random.seed(seed)
    field = np.random.uniform(low=-field_range, high=field_range,
                              size=(len(nodes),))
    field = np.einsum('a,b->ab', field, [-1, 1])
    inter = np.random.uniform(low=-interaction_range, high=interaction_range,
                              size=(len(edges),))
    inter = np.einsum('a,bc->abc', inter, [[1, -1], [-1, 1]])

    return PairWiseFiniteModel.create(field, edges, inter)


def make_cross(length=20, width=2) -> networkx.Graph:
    """Builds graph which looks like a cross.

    Result graph has (2*length-width)*width vertices.

    For example, this is a cross of width 3:
           ...
           +++
           +++
    ...+++++++++++...
    ...+++++++++++...
    ...+++++++++++...
           +++
           +++
           ...
    :param length: Length of a cross.
    :param width: Width of a cross.
    """
    assert width < length * 2

    nodes = set()
    for i in range(length // 2, length // 2 + width):
        for j in range(0, length):
            nodes.add((i, j))
            nodes.add((j, i))
    nodes = list(nodes)
    node_index = {nodes[i]: i for i in range(len(nodes))}
    graph = networkx.Graph()
    graph.add_nodes_from(range(len(nodes)))

    for x, y in nodes:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (x + dx, y + dy) in node_index:
                graph.add_edge(node_index[(x, y)],
                               node_index[(x + dx, y + dy)])
    return graph


def cross_potts_model(length=20, width=2, al_size=2, seed=0):
    """Builds random Potts model with cross-like interactionn graph."""
    return pairwise_model_on_graph(make_cross(length, width),
                                   al_size=al_size,
                                   seed=seed)


def random_generic_model(num_variables=10,
                         num_factors=10,
                         max_domain_size=3,
                         max_factor_size=3,
                         seed=0) -> GenericGraphModel:
    """Generates random discrete graphical model of arbitrary structure.

    You can specify number of variables and factors. Variables will have
    different domain sizes, and factors will have different number of
    variables.

    :param num_variables: Number of variables.
    :param num_factors: Number of factors.
    :param max_domain_size: Maximal size of domain. For every variable, domain
        size will be chosen at random between 2 and this value.
    :param max_factor_size: Maximal size of factor. For every factor, number of
        variables in it will be chosen at random between 1 and this value.
    """
    np.random.seed(seed)
    model = GenericGraphModel(num_variables=num_variables)
    for var_id in range(num_variables):
        domain_size = 2 + np.random.randint(max_domain_size - 1)
        model.get_variable(var_id).domain = DiscreteDomain.range(domain_size)
    for _ in range(num_factors):
        factor_size = 1 + np.random.randint(max_factor_size)
        var_idx = np.random.choice(num_variables,
                                   size=factor_size,
                                   replace=False)
        values_shape = [model.get_variable(i).domain.size() for i in var_idx]
        values = np.random.random(size=values_shape)
        factor = DiscreteFactor(model, var_idx, values)
        model.add_factor(factor)
    return model
