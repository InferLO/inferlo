from typing import List

import numpy as np
from networkx import Graph


def path_decomposition(graph: Graph) -> List[np.array]:
    """Path decomposition of a graph.

    Splits vertices into layers, such that vertices in layer i are connected
    only with vertices from layers i-1, i and i+1.

    Uses simple greedy algorithm. First layer always contains single vertex
    with id=0.

    This is similar, but no identical to
    https://en.wikipedia.org/wiki/Pathwidth.

    :param graph: Graph for which to compute decomposition.
    :return: Path decomposition of given graph.
    """
    # TODO: handle disconnected graphs.
    # TODO: this can be optimized with Numba!
    layers = [[0]]
    vertex_to_layer = {0: 0}

    while True:
        last_layer_id = len(layers) - 1
        neighbors = set([w for v in layers[-1] for w in graph.neighbors(v)])
        new_layer = []
        for v in neighbors:
            if v in vertex_to_layer:
                if vertex_to_layer[v] < last_layer_id - 1:
                    # TODO: this is impossible, right?
                    raise ValueError("Graph is not layered.")
            else:
                new_layer.append(v)
                vertex_to_layer[v] = last_layer_id + 1

        if len(new_layer) > 0:
            layers.append(np.array(new_layer))
        else:
            break

    return layers
