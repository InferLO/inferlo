# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from typing import List

import networkx as nx
import numpy as np

from inferlo.graphs import path_decomposition


def _normalize(layers: List[np.array]):
    return [sorted(layer.tolist()) for layer in layers]


def test_grid3x3():
    gr = nx.Graph()
    gr.add_nodes_from(range(9))
    gr.add_edges_from([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8],
                       [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8]])
    result = path_decomposition(gr)

    assert _normalize(result) == [[0], [1, 3], [2, 4, 6], [5, 7], [8]]


def test_two_cliques():
    gr = nx.Graph()
    gr.add_nodes_from(range(6))
    gr.add_edges_from([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])
    result = path_decomposition(gr)

    assert _normalize(result) == [[0], [1, 2], [3], [4, 5]]
