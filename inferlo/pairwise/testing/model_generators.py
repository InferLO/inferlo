import networkx
import numpy as np

from inferlo.pairwise.pwf_model import PairWiseFiniteModel


def grid_potts_model(
        height,
        width,
        al_size=3,
        seed=111) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a grid.

    :param height: Heigth of the grid.
    :param width: Wwidth of the grid.
    :param al_size: Alphabet size.
    :param seed: Random seed.
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
    inter = np.random.random(size=(edges_num, al_size, al_size)) * 5.0
    return PairWiseFiniteModel.create(field, edges, inter)


def tree_potts_model(gr_size=5, al_size=2, seed=111, same_j=None,
                     zero_field=False) -> PairWiseFiniteModel:
    """Generates random PairWiseFinteModel on a random tree.

    :param gr_size: Size of the graph (number of variables).
    :param al_size: Alphabet size.
    :param seed: Random set.
    :param same_j: If set, interaction matrix for all edges.
    :param zero_field: Whether base should be zero-field.
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
    :param zero_field: Whether base should be zero-field.
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
