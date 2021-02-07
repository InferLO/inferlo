# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from inferlo import GraphModel, DiscreteFactor, InferenceResult
from .belief_propagation import BeliefPropagation, IterativeJoinGraphPropagation
from .bucket_elimination import BucketElimination
from .bucket_renormalization import BucketRenormalization
from .factor import Factor
from .graphical_model import GraphicalModel
from .mean_field import MeanField
from .mini_bucket_elimination import MiniBucketElimination
from .weighted_mini_bucket_elimination import WeightedMiniBucketElimination


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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/belief_propagation.py>`_.
    """
    algo = BeliefPropagation(_convert(model))
    algo.run(max_iter=max_iter,
             converge_thr=converge_thr,
             damp_ratio=damp_ratio)
    return algo.get_inference_result()


def bucket_elimination(model: GraphModel) -> float:
    """Inference with Bucket Elimination.

    Estimates partition function using Bucket Elimination algorithm.

    :param model: Model for which to perform inference.
    :return: Natural logarithm of estimated partition function.

    References
        [1] Rina Dechter.
        Bucket elimination: A unifying framework for reasoning. 1999.
        https://arxiv.org/abs/1302.3572

        [2] `Original implementation
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/bucket_elimination.py>`_.
    """
    return BucketElimination(_convert(model)).run()


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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/bucket_renormalization.py>`_.
    """
    algo = BucketRenormalization(_convert(model), ibound=ibound)
    return algo.run(max_iter=max_iter)


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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/belief_propagation.py>`_.
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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/mean_field.py>`_.
    """
    return MeanField(_convert(model)).run()


def mini_bucket_elimination(model: GraphModel, ibound: int = 10) -> float:
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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/mini_bucket_elimination.py>`_.
    """
    return MiniBucketElimination(_convert(model), ibound=ibound).run()


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
        <https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/weighted_mini_bucket_elimination.py>`_.
    """
    return WeightedMiniBucketElimination(_convert(model), ibound=ibound).run()
