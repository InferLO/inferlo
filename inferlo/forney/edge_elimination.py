from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from networkx import is_connected

from inferlo.base.factors import DiscreteFactor

if TYPE_CHECKING:
    from inferlo.forney.nfg_model import NormalFactorGraphModel


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
    # Einsum supports only Latin letters as indices.
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

    :param model: Model, for which to compute partition function.
    :return: Partition function.
    """
    model.check_built()

    factors = [DiscreteFactor.from_factor(f) for f in model.factors]
    edges = np.array(model.edges)
    # Don't reference model from this point to ensure we don't modify it.

    num_edges = len(edges)
    edge_exists = [True for _ in range(num_edges)]

    # Now repeat the following: pick variable (edge). If it's a self-loop,
    # sum the factor referencing it over one variable. If it connects two
    # factors, convolve these two factors to get new factors, insert it in
    # factors list where first factor was, update variables which were
    # connected to second factor.
    # Strategy for picking variable to eliminate: if there are self-loops pick
    # them. Otherwise pick edge whose two factors has least sum of degrees.

    # print("Edges: ", edges)

    def pick_edge():
        # print("Enter pick_edge. Exists:", edge_exists)
        best_i = -1
        best_sum = 1000000000

        for i in range(num_edges):
            if not edge_exists[i]:
                continue
            u, v = edges[i]
            sum_deg = len(factors[u].var_idx) + len(factors[v].var_idx)
            if u == v:
                sum_deg = -1  # Self loop.
            if sum_deg < best_sum:
                best_i = i
                best_sum = sum_deg
        assert best_i != -1
        # print("Picked edge", best_i)
        return best_i

    for _ in range(num_edges):
        var_id = pick_edge()
        f1, f2 = edges[var_id]
        if f1 == f2:
            factors[f1] = convolve_factor(factors[f1], var_id)
        else:
            edges = np.where(edges == f2, f1, edges)
            factors[f1] = convolve_two_factors(
                factors[f1], factors[f2], var_id)
            factors[f2] = None
        edge_exists[var_id] = False

    # At this point we should have one or more scalar factors.
    # Partition function is their product.
    result_factors = [f.values.item() for f in factors if f is not None]
    part_func = 1.0
    for factor in result_factors:
        part_func *= factor
    return part_func
