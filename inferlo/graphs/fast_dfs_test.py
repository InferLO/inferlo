# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import networkx as nx
import numpy as np
from networkx.algorithms.traversal import depth_first_search

from inferlo.graphs.fast_dfs import fast_dfs


def _dfs_with_networkx(vert_num, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(vert_num))
    for u, v in edges:
        graph.add_edge(u, v)
    return np.array(list(depth_first_search.dfs_edges(graph, source=0)))


def test_tree():
    vert_num = 1000
    graph = nx.generators.trees.random_tree(vert_num, seed=0)
    edges = np.array(list(graph.edges()), dtype=np.int32)
    expected_dfs_edges = _dfs_with_networkx(vert_num, edges)

    result = fast_dfs(vert_num, edges)

    assert np.allclose(result.dfs_edges, expected_dfs_edges)
    assert not result.was_disconnected
    assert not result.had_cycles
    assert result.was_tree


def test_forest():
    vert_num = 10
    edges = np.array([[3, 4], [4, 8], [0, 4], [9, 0], [2, 7], [5, 1], [1, 6]])
    expected_dfs_edges = np.array([[0, 4], [4, 3], [4, 8], [0, 9], [0, 1],
                                   [1, 5], [1, 6], [0, 2], [2, 7]])

    result = fast_dfs(vert_num, edges)

    assert np.allclose(result.dfs_edges, expected_dfs_edges)
    assert result.was_disconnected
    assert not result.had_cycles
    assert not result.was_tree


def test_disconnected_with_cycles():
    vert_num = 6
    edges = np.array([[0, 1], [1, 2], [0, 2], [5, 4], [4, 3], [3, 5]])
    expected_dfs_edges = np.array([[0, 1], [1, 2], [0, 3], [3, 4], [4, 5]])

    result = fast_dfs(vert_num, edges)

    assert np.allclose(result.dfs_edges, expected_dfs_edges)
    assert result.was_disconnected
    assert result.had_cycles
    assert not result.was_tree


def test_connected_with_cycles():
    vert_num = 20
    graph = nx.generators.wheel_graph(vert_num)
    edges = np.array(list(graph.edges()), dtype=np.int32)
    expected_dfs_edges = _dfs_with_networkx(vert_num, edges)

    result = fast_dfs(vert_num, edges)

    assert np.allclose(result.dfs_edges, expected_dfs_edges)
    assert not result.was_disconnected
    assert result.had_cycles
    assert not result.was_tree


def test_long_path():
    n = 10000
    ends = np.cumsum(np.ones(n - 1), dtype=np.int32)
    starts = ends - 1
    edges = np.array([starts, ends]).T

    result = fast_dfs(n, edges)

    assert np.allclose(result.dfs_edges, edges)
    assert not result.was_disconnected
    assert not result.had_cycles
    assert result.was_tree


def test_no_edges():
    edges = np.empty((0, 2), dtype=np.int32)
    result = fast_dfs(3, edges)

    assert np.allclose(result.dfs_edges, [[0, 1], [0, 2]])
    assert result.was_disconnected
    assert not result.had_cycles
    assert not result.was_tree
