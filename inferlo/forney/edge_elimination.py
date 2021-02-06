# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inferlo.base.factors import DiscreteFactor

if TYPE_CHECKING:
    from inferlo.forney.nfg_model import NormalFactorGraphModel


# TODO: Implement LogSumExp trick and return log pf to avoid overflow.
# TODO: Support convolution over several variables.

def convolve_factor(factor: DiscreteFactor, var: int) -> DiscreteFactor:
    """Convolves (integrates) factor over variable.

    Result is factor of all variables except ``var``, which at every point
    is equal to sum of values of initial factor over all possible values of
    variable ``var``

    :param factor: Factor to convolve.
    :param var: Index of variable over which summation is done. This is index
      in model, not in factor. Factor must depend on this variable.
    :return: New convolved factor.
    """
    assert var in factor.var_idx
    var_pos = factor.var_idx.index(var)
    new_vars = [v for v in factor.var_idx if v != var]
    new_values = np.sum(factor.values, axis=var_pos)
    return DiscreteFactor(factor.model, new_vars, new_values)


def convolve_two_factors(factor1: DiscreteFactor, factor2: DiscreteFactor,
                         var: int) -> DiscreteFactor:
    """Convolves two factors over variable.

    Result is factor depending on all variables of first and second factor
    except variable ``var``. It is defined as sum of products of given two
    factors, where summation is done over all possible values of variable
    ``var``.

    :param factor1: First factor to convolve.
    :param factor2: Second factor to convolve.
    :param var: Index of variable over which summation is done. This is index
      in model, not in factor. Both factors must depend on this variable.
    :return: New convolved factor.
    """
    vars1 = factor1.var_idx
    vars2 = factor2.var_idx
    assert var in vars1
    assert var in vars2

    all_vars = list(set(vars1 + vars2))
    # einsum supports only Latin letters as indices.
    assert len(all_vars) <= 26
    vars_idx = {all_vars[i]: chr(65 + i) for i in range(len(all_vars))}
    new_vars = [v for v in all_vars if v != var]

    vars1_sym = ''.join([vars_idx[v] for v in vars1])
    vars2_sym = ''.join([vars_idx[v] for v in vars2])
    new_vars_sym = ''.join([vars_idx[v] for v in new_vars])
    subscripts = vars1_sym + ',' + vars2_sym + '->' + new_vars_sym

    new_values = np.einsum(subscripts, factor1.values, factor2.values)
    return DiscreteFactor(factor1.model, new_vars, new_values)


def infer_edge_elimination(model: NormalFactorGraphModel):
    """Calculates partition function using Edge Elimination.

    Algorithm
        Repeat the following: pick variable (edge). If it's a self-loop,
        convolve (sum) the factor referencing it over one variable. If it
        connects two factors, convolve these two factors to get new factor,
        insert the new factor in factors list where first factor was, update
        edges which were connected to second factor to point to new factor.

        Strategy for picking variable to eliminate: if there is a self-loop,
        pick it. Otherwise, pick edge whose elimination would result in factor
        with least degree.

    :param model: Model, for which to compute partition function.
    :return: Partition function.
    """
    model.check_built()
    factors = [DiscreteFactor.from_factor(f) for f in model.factors]
    edges = np.array(model.edges)
    # Don't reference model from this point to ensure we don't modify it.

    num_edges = len(edges)
    edge_exists = np.ones(num_edges, dtype=bool)

    # Heuristic to find which edge to eliminate. Will pick edge of least cost.
    def edge_cost(edge_id):
        if not edge_exists[edge_id]:
            return np.inf
        u, v = edges[edge_id]
        if u == v:
            return -1
        return len(set(factors[u].var_idx) | set(factors[v].var_idx)) - 1

    while np.any(edge_exists):
        var_id = np.argmin([edge_cost(i) for i in range(num_edges)])
        f1, f2 = edges[var_id]
        if f1 == f2:
            factors[f1] = convolve_factor(factors[f1], var_id)
        else:
            # Convolve two factors and replace first factor with result.
            factors[f1] = convolve_two_factors(
                factors[f1], factors[f2], var_id)
            # Delete second factor.
            factors[f2] = None
            # Remap references to second factor on edges.
            edges = np.where(edges == f2, f1, edges)
        edge_exists[var_id] = False

    # At this point we should have one or more scalar factors.
    # Partition function is their product.
    result_factors = [f.values.item() for f in factors if f is not None]
    part_func = 1.0
    for factor in result_factors:
        part_func *= factor
    return part_func
