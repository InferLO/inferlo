import networkx as nx
import numpy as np

from inferlo.pairwise.junction_tree import max_likelihood_junction_tree, \
    infer_junction_tree
from inferlo.pairwise.testing import clique_potts_model, tree_potts_model, \
    grid_potts_model, assert_results_close
from inferlo.pairwise.testing.model_generators import pairwise_model_on_graph


def _make_cross(length=20, width=2) -> nx.Graph:
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))

    for x, y in nodes:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (x + dx, y + dy) in node_index:
                graph.add_edge(node_index[(x, y)],
                               node_index[(x + dx, y + dy)])
    return graph


def test_inference_clique_10x2():
    model = clique_potts_model(gr_size=10, al_size=2, seed=0)
    ground_truth = model.infer(algorithm='bruteforce')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_tree_100x5():
    model = tree_potts_model(gr_size=100, al_size=5, seed=0)
    ground_truth = model.infer(algorithm='tree_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_grid_4x50x2():
    model = grid_potts_model(4, 50, al_size=2, seed=0)
    ground_truth = model.infer(algorithm='path_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_inference_cross_50x2x2():
    model = pairwise_model_on_graph(_make_cross(length=50, width=2), al_size=2)
    ground_truth = model.infer(algorithm='path_dp')
    result = infer_junction_tree(model)
    assert_results_close(result, ground_truth)


def test_max_likelihood_clique_10x2():
    model = clique_potts_model(gr_size=10, al_size=2, seed=0)
    true_ml = model.max_likelihood(algorithm='bruteforce')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_tree_100x5():
    model = tree_potts_model(gr_size=100, al_size=5, seed=0)
    true_ml = model.max_likelihood(algorithm='tree_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_grid_4x50x2():
    model = grid_potts_model(4, 50, al_size=2, seed=0)
    true_ml = model.max_likelihood(algorithm='path_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)


def test_max_likelihood_cross_50x2x2():
    model = pairwise_model_on_graph(_make_cross(length=50, width=2), al_size=2)
    true_ml = model.max_likelihood(algorithm='path_dp')
    ml = max_likelihood_junction_tree(model)
    assert np.allclose(ml, true_ml)
