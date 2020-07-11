# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from typing import List

import numpy as np
from networkx import Graph


def path_decomposition(graph: Graph) -> List[np.array]:
    """Path decomposition of a graph.

    Splits vertices into layers, such that vertices in layer i are connected
    only with vertices from layers i-1, i and i+1.

    Uses simple greedy algorithm. First layer always contains single vertex
    with id=0.

    If graph is not connected, will decompose every connected component and
    concatenate results.

    This is similar, but no identical to
    https://en.wikipedia.org/wiki/Pathwidth.

    :param graph: Graph for which to compute decomposition. Its nodes must be
      labeled by consecutive integers, starting with 0.
    :return: Path decomposition of given graph. List of np.arrays, where i-th
      array contains indices of nodes in i-th layer. Guaranteed that all layers
      are disjoint and together they contain all vertices from the graph.
    """
    gr_size = len(graph.nodes)
    vertex_to_layer = -1 * np.ones(gr_size, dtype=np.int32)
    layers = []

    for i in range(gr_size):
        if vertex_to_layer[i] != -1:
            continue
        vertex_to_layer[i] = len(layers)
        layers.append(np.array([i]))

        while True:
            neighbors = set([w for v in layers[-1]
                             for w in graph.neighbors(v)])
            new_layer = []
            for v in neighbors:
                if vertex_to_layer[v] == -1:
                    new_layer.append(v)
                    vertex_to_layer[v] = len(layers)

            if len(new_layer) > 0:
                layers.append(np.array(new_layer))
            else:
                break

    return layers
