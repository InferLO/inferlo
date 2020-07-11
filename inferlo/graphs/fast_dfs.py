# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from collections import namedtuple

import numba
import numpy as np


@numba.njit("i4(i4[:],i4[:],i1[:],i4[:,:],i4,i4,i1[:],i4[:],i4[:])")
def dfs_one_vertex(flat_adj_list, fal_start, visited, dfs_edges,
                   dfs_edges_count, start_vx, flags, stack, fal_pos_stack):
    """Performs depth-first search starting from given vertex.

    This algorithm returns DFS tree for connected component containing
    ``start_vx``.

    Equivalent to ``dfs_edges`` from NetworkX library (v2.4).

    :param flat_adj_list: Flat adjacency list of length ``2 * edges_num``.
      It's obtained by listing adjacent vertices for every vertex and
      concatenating these lists.
    :param fal_start: For every vertex contains pointer to ``flat_adj_list``
      where we should start reading it's adjacent vertices. In other words,
      neighbors of vertex ``v`` are
      ``flat_adj_list[fal_start[v]: fal_start[v+1]].
      Length is ``vert_num + 1``. First element is 0, last element is
      ``len(flat_adj_list)``.
    :param dfs_edges: Array where to write DFS edges as we visit them.
    :param dfs_edges_count: How many DFS edges is already written. e need this
      to handle disconnected graphs.
    :param start_vx: From which vertex to start DFS.
    :param flags: Side output. If DFS finds cycle, it should set
      ``flags[0] = 1``.
    :param stack: Allocated memory for DFS stack. DFS stack contains nodes
      which DFS entered but not exited. It corresponds to call stack if we
      implemented DFS recursively. It's important that we don' allocate this
      memory inside this function, because then algorithm would have quadratic
      complexity if graph has a lot of small connected components.
    :param fal_stack: Allocated memory stack which would contain pointers
      (flat_adj_list indices) to currently considered children at each level.
    :return: Updated dfs_edges_count.
    """
    # Push start_vx to the stack.
    stack_head = 0
    stack[stack_head] = start_vx
    fal_pos_stack[stack_head] = fal_start[start_vx]
    visited[start_vx] = 1

    while True:
        v = stack[stack_head]
        if fal_pos_stack[stack_head] == fal_start[v + 1]:
            # Go up - pop vertex from stack.
            stack[stack_head] = -1  # debug
            stack_head -= 1
            if stack_head == -1:
                break
            # And move to the next child in parent.
            fal_pos_stack[stack_head] += 1
        else:
            # Current edge is (v, to).
            to = flat_adj_list[fal_pos_stack[stack_head]]
            if not visited[to]:
                # If `to` is not visited, add this edge to answer
                dfs_edges[dfs_edges_count, 0] = v
                dfs_edges[dfs_edges_count, 1] = to
                dfs_edges_count += 1
                # and go down by the edge.
                visited[to] = 1
                stack_head += 1
                stack[stack_head] = to
                fal_pos_stack[stack_head] = fal_start[to]
            else:
                # If `to` is visited, skip this edge.
                fal_pos_stack[stack_head] += 1
                # If it is not parent in DFS tree, we found a back edge.
                if stack_head == 0 or to != stack[stack_head - 1]:
                    flags[0] = 1

    return dfs_edges_count


@numba.njit("i4[:,:](i4,i4[:,:],i1[:])")
def _fast_dfs_internal(vert_num, edges, flags):
    """Performs depth-first search handling multiple connected components.

    :param vert_num: Number of vertices in the graph.
    :param edges: Edge list.
    :param flags: Additional output of the algorithm.
      flag[0] - whether graph has cycles (i.e. was not a forest).
      flag[1] - whether graph was disconnected and we added edges to make it
        connected.
    """
    # Number of edges.
    edge_num = edges.shape[0]
    # Calculate degree of each vertex.
    degree = np.zeros(vert_num, dtype=np.int32)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # Build compact adjacency list.
    flat_adj_list = np.zeros(2 * edge_num, dtype=np.int32)
    fal_start = np.zeros(vert_num + 1, dtype=np.int32)
    fal_start[1:] = np.cumsum(degree)
    fal_ptr = np.copy(fal_start)
    for u, v in edges:
        flat_adj_list[fal_ptr[u]] = v
        fal_ptr[u] += 1
        flat_adj_list[fal_ptr[v]] = u
        fal_ptr[v] += 1

    visited = np.zeros_like(degree, dtype=np.int8)

    # Allocate array for the answer.
    # Answer will always be a tree, so it has vert_num-1 edges.
    dfs_edges = np.empty((vert_num - 1, 2), dtype=np.int32)
    dfs_edges_count = 0

    # Allocate memory for stacks.
    stack = np.empty(vert_num, dtype=np.int32)
    fal_pos_stack = np.empty(vert_num, dtype=np.int32)

    for vertex in range(vert_num):
        if not visited[vertex]:
            if vertex != 0:
                # Add edges to connect vertex 0 to root of connected component.
                dfs_edges[dfs_edges_count, 0] = 0
                dfs_edges[dfs_edges_count, 1] = vertex
                dfs_edges_count += 1
                # Mark that graph was disconnected.
                flags[1] = 1
            dfs_edges_count = dfs_one_vertex(
                flat_adj_list, fal_start, visited, dfs_edges, dfs_edges_count,
                vertex, flags, stack, fal_pos_stack)

    assert dfs_edges_count == vert_num - 1
    return dfs_edges


FastDfsResult = namedtuple('FastDfsResult', ['dfs_edges',
                                             'had_cycles',
                                             'was_disconnected',
                                             'was_tree'])


def fast_dfs(vert_num: int, edges: np.ndarray) -> FastDfsResult:
    """Depth-first search.

    Runs the depth-first search algorithm on given simple undirected graph
    represented by an edge list and returns edges of DFS tree in order of
    visiting.

    If input graph is disconnected, completes it to connected by adding edges
    between vertex 0 and vertices with minimal index from other connected
    components. If this was done, will set ``was_disconnected=True`` in
    returned object. Thanks to this, result always has exactly ``vert_num - 1``
    edges.

    If input graph has cycles, not all edges of original graph will be in the
    output, and algorithm will indicate that by setting ``had_cycles=True`` in
    output object.

    :param vert_num: Number of vertices in the graph.
    :param edges: List of edges in the graph. ``np.array`` of type ``np.int32``
      and shape ``(num_edges, 2)``.
    :return: ``FastDfsResult`` object with result of the DFS and additional
      data.

    Fields in result object
        * ``dfs_edges`` - np.array of type ``np.int32`` and shape
          ``(vert_num-1, 2)``. Contains edges in DFS tree in DFS traversal
          order (from root to leafs). These edges are guaranteed to form a
          tree.
        * ``had_cycles`` - whether input graph had cycles.
        * ``was_disconnected`` - whether input graph was disconnected.
        * ``was_tree`` - whether input graph was a tree.

    See https://en.wikipedia.org/wiki/Depth-first_search.
    """
    edges = edges.astype(dtype=np.int32, copy=False)
    edges_count = edges.shape[0]
    assert edges.shape == (edges_count, 2)
    if edges_count > 0:
        assert np.min(edges) >= 0
        assert np.max(edges) < vert_num

    flags = np.zeros(2, dtype=np.int8)
    dfs_edges = _fast_dfs_internal(numba.types.int32(vert_num), edges, flags)

    had_cycles = (flags[0] != 0)
    was_disconnected = (flags[1] != 0)
    was_tree = not (had_cycles or was_disconnected)
    return FastDfsResult(
        dfs_edges=dfs_edges,
        had_cycles=had_cycles,
        was_disconnected=was_disconnected,
        was_tree=was_tree
    )
