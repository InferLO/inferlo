# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from typing import Callable, List

import numpy as np

from inferlo import GraphModel, DiscreteFactor, InferenceResult, \
    GenericGraphModel
from .belief_propagation import BeliefPropagation, IterativeJoinGraphPropagation
from .binary_tree_elimination import BinaryTreeElimination
from .bucket_elimination import BucketElimination, eliminate_variables
from .bucket_renormalization import BucketRenormalization
from .factor import Factor
from .graphical_model import GraphicalModel
from .mean_field import MeanField
from .mini_bucket_elimination import MiniBucketElimination
from .weighted_mini_bucket_elimination import WeightedMiniBucketElimination
from ...base.inference_result import marg_probs_to_array


def _convert(inferlo_model: GraphModel) -> GraphicalModel:
    model = GraphicalModel([], [])
    cardinalities = dict()
    for t in range(inferlo_model.num_variables):
        newvar = "V%d" % t
        model.add_variable(newvar)
        cardinalities[newvar] = inferlo_model.get_variable(t).domain.size()

    factors = list(inferlo_model.get_factors())
    for factor_id in range(len(factors)):
        factor = DiscreteFactor.from_factor(factors[factor_id])
        factor_variables = []
        for var_id in factor.var_idx:
            factor_variables.append("V%d" % var_id)
        model.add_factor(Factor(name="F%d" % factor_id,
                                variables=factor_variables,
                                values=factor.values))
    return model


def belief_propagation(model: GraphModel,
                       max_iter: int = 1000,
                       converge_thr: float = 1e-5,
                       damp_ratio: float = 0.1) -> InferenceResult:
    """Inference with (loopy) Belief Propagation.

    Estimates partition function using Loopy Belief Propagation algorithm.

    :param model: Model for which to perform inference.
    :param max_iter: Number of iterations.
    :param converge_thr: Convergence threshold.
    :param damp_ratio: Damp ratio.
    :return: Inference result.

    References
        [1] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/belief_propagation.py>`__.
    """
    algo = BeliefPropagation(_convert(model))
    algo.run(max_iter=max_iter,
             converge_thr=converge_thr,
             damp_ratio=damp_ratio)
    return algo.get_inference_result()


def bucket_elimination(model: GraphModel,
                       elimination_order_method: str = "random") -> float:
    """Inference with Bucket Elimination.

    Estimates partition function using Bucket Elimination algorithm.

    :param model: Model for which to perform inference.
    :param elimination_order_method: Elimination order. Can be 'random',
      'not_random' or 'given'. If 'given', elimination order should be passed in
      'elimination_order' argument.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Rina Dechter.
        Bucket elimination: A unifying framework for reasoning. 1999.
        https://arxiv.org/abs/1302.3572

        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/bucket_elimination.py>`__.
    """
    algo = BucketElimination(_convert(model))
    return algo.run(elimination_order_method=elimination_order_method)


def bucket_renormalization(model: GraphModel,
                           ibound: int = 10,
                           max_iter: int = 1) -> float:
    """Inference with Bucket Renormalization.

    Estimates partition function using Bucket Renormalization algorithm.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    :param max_iter: Number of optimization iterations. 0 corresponds to
        Mini-Bucket Renormalization. 1 corresponds to Global-Bucket
        Renormalization. Default is 1.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Sungsoo Ahn, Michael Chertkov, Adrian Weller, Jinwoo Shin.
        Bucket Renormalization for Approximate Inference. 2018.
        https://arxiv.org/abs/1803.05104

        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/bucket_renormalization.py>`__.
    """
    algo = BucketRenormalization(_convert(model), ibound=ibound)
    algo.run(max_iter=max_iter)
    return algo.get_log_z()


def iterative_join_graph_propagation(
        model: GraphModel,
        ibound: int = 10,
        max_iter: int = 1000,
        converge_thr: float = 1e-5,
        damp_ratio: float = 0.1) -> float:
    """Inference with Iterative Join Graph Propagation.

    Estimates partition function using Iterative Join Graph Propagation
      algorithm, which belongs to the class of Generalized Belief Propagation
      methods.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    :param max_iter: Number of iterations.
    :param converge_thr: Convergence threshold.
    :param damp_ratio: Damp ratio.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Rina Dechter, Kalev Kask, Robert Mateescu.
        Iterative Join-Graph Propagation. 2012.
        https://arxiv.org/abs/1301.0564

        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/belief_propagation.py>`__.
    """
    algo = IterativeJoinGraphPropagation(_convert(model), ibound=ibound)
    algo.run(max_iter=max_iter,
             converge_thr=converge_thr,
             damp_ratio=damp_ratio)
    return algo.get_log_z()


def mean_field(model: GraphModel,
               max_iter: int = 1000,
               converge_thr: float = 1e-2) -> float:
    """Inference with Mean Field.

    Estimates partition function using Mean Field approximation.

    :param model: Model for which to perform inference.
    :param max_iter: Maximal number of iterations.
    :param converge_thr: Convergence threshold.
    :return: Natural logarithm of estimated partition function.

    References
        [1] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/mean_field.py>`__.
    """
    return MeanField(_convert(model)).run()


def mini_bucket_elimination(model: GraphModel,
                            ibound: int = 10) -> float:
    """Inference with Mini Bucket Elimination.

    Estimates partition function using Mini Bucket Elimination algorithm.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Rina Dechter, Irina Rish.
        Mini-buckets: A general scheme for bounded inference. 2003.
        https://dl.acm.org/doi/abs/10.1145/636865.636866

        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/mini_bucket_elimination.py>`__.
    """
    algo = MiniBucketElimination(_convert(model), ibound=ibound)
    algo.run()
    return algo.get_log_z()


def weighted_mini_bucket_elimination(model: GraphModel,
                                     ibound: int = 10) -> float:
    """Inference with Weighted Mini Bucket Elimination.

    Estimates partition function using Weighted Mini Bucket Elimination
    algorithm.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Qiang Liu, Alexander T. Ihler.
        Bounding the Partition Function using Holder's Inequality. 2011.
        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/weighted_mini_bucket_elimination.py>`__.
    """
    return WeightedMiniBucketElimination(_convert(model), ibound=ibound).run()


# TODO: Move to the new model class.
def _restrict_model(model: GenericGraphModel, var_id: int, val: int):
    """Makes a copy of model in which value of given variable is fixed."""
    assert 0 <= val < model.get_variable(var_id).domain.size()
    new_model = model.copy()
    for i in range(len(new_model.factors)):
        if var_id in new_model.factors[i].var_idx:
            new_model.factors[i] = new_model.factors[i].restrict(var_id, val)
    return new_model


def get_marginals(
        model: GenericGraphModel,
        log_pf_algo: Callable[[GenericGraphModel], float],
        var_ids: List[int] = None,
        skip_last: bool = False) -> InferenceResult:
    """Calculates marginal probabilities using provided algorithm for computing
    partition function.

    For every value of every variable builds new model where value of that
    variable is fixed, and computes partition function for new model. Then
    calculates marginal probability.

    This is high-level abstraction, agnostic of underlying algorithm. It will
    get exact results as long as underlying algorithm is exact.

    :param model: Graphical model.
    :param log_pf_algo: Function which computes.
    :param var_ids: If set, marginal probabilities will be only calculated for
        these variables.
    :param skip_last: If True, marginal probabilities will be calculated for
        all but one values of variables, and last value will be inferred from
        condition that all probabilities add up to 1. This saves computation,
        but not recommended for approximate algorithms, as it may yield negative
        values. If False, all marginal probabilities will be computed
        independently, and then normalized - so they won't depend on the global
        partition function.
    """
    log_pf = log_pf_algo(model)
    marg_probs = []
    if var_ids is None:
        var_ids = list(range(model.num_variables))
    for var_id in var_ids:
        card = model.get_variable(var_id).domain.size()
        mp = np.zeros(card)
        for j in range(card):
            if skip_last and j == card - 1:
                break
            restr_model = _restrict_model(model, var_id, j)
            mp[j] = log_pf_algo(restr_model)
        if skip_last:
            mp = np.exp(mp - log_pf)
            mp[card - 1] = 1.0 - np.sum(mp[:card - 1])
        else:
            mp = np.exp(mp - np.max(mp))
            mp /= np.sum(mp)
        assert (np.sum(mp) - 1.0) < 1e-5
        marg_probs.append(mp)
    return InferenceResult(log_pf=log_pf,
                           marg_prob=marg_probs_to_array(marg_probs))


"""
Binary Tree Elimination algorithms.
"""


def bucket_elimination_bt(model: GraphModel) -> InferenceResult:
    """Inference with Bucket Elimination on binary tree.

    It runs Bucket elimination multiple times in different order to get marginal
    probabilities for every variable. However, it partially reuses results
    using "divide & conquer" technique, which allows to compute all marginal
    probabilities by doing only O(N logN) eliminations.
    """
    model = _convert(model)
    bt_algo = BinaryTreeElimination(eliminate_variables)
    return bt_algo.run(model)


def mini_bucket_elimination_bt(
        model: GraphModel,
        ibound: int = 10) -> InferenceResult:
    """Inference with Mini-Bucket Elimination on binary tree.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    """

    def eliminate(model, order):
        algo = MiniBucketElimination(model,
                                     ibound=ibound,
                                     elimination_order=order)
        algo.run()
        algo.working_model.add_factor(Factor("", [], log_values=np.array(algo.base_logZ)))
        return algo.working_model

    bt_algo = BinaryTreeElimination(eliminate)
    return bt_algo.run(_convert(model))


def mini_bucket_renormalization_bt(
        model: GraphModel,
        ibound: int = 10) -> InferenceResult:
    """Inference with Mini-Bucket Renormalization on binary tree.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    """

    def eliminate(model, order):
        br_algo = BucketRenormalization(model,
                                        ibound=ibound,
                                        elimination_order=order)
        eliminated_model = eliminate_variables(br_algo.renormalized_model,
                                               br_algo.renormalized_elimination_order)
        eliminated_model.add_factor(Factor("", [], log_values=np.array(br_algo.base_logZ)))
        return eliminated_model

    bt_algo = BinaryTreeElimination(eliminate)
    return bt_algo.run(_convert(model))
