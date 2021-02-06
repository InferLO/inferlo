# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from inferlo import GraphModel, DiscreteFactor
from .belief_propagation import BeliefPropagation, IterativeJoinGraphPropagation
from .bucket_elimination import BucketElimination
from .bucket_renormalization import BucketRenormalization
from .factor import Factor
from .graphical_model import GraphicalModel
from .mean_field import MeanField
from .mini_bucket_elimination import MiniBucketElimination
from .weighted_mini_bucket_elimination import WeightedMiniBucketElimination


def _convert(inferlo_model: GraphModel) -> GraphicalModel:
    model = GraphicalModel()
    cardinalities = dict()
    for t in range(inferlo_model.num_variables):
        newvar = "V" + str(t)
        model.add_variable(newvar)
        cardinalities[newvar] = inferlo_model.get_variable(t).domain.size()

    factors = list(inferlo_model.get_factors())
    for factor_id in range(len(factors)):
        factor = DiscreteFactor.from_factor(factors[factor_id])
        factor_variables = []
        for var_id in factor.var_idx:
            factor_variables.append("V" + str(var_id))
        model.add_factor(Factor(name="F" + str(factor_id),
                                variables=factor_variables,
                                values=factor.values))
    return model


def belief_propagation(model: GraphModel) -> float:
    """Inference with Belief Propagation.

    Estimates partition function using Loopy Belief Propagation algorithm.

    Original implementation from https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/belief_propagation.py

    :param model: Model for which to perform inference.
    :return: Natural logarithm of estimated partition function.
    """
    return BeliefPropagation(_convert(model)).run()


def bucket_elimination(model: GraphModel) -> float:
    """Inference with Bucket Elimination.

    Estimates partition function using Bucket Elimination algorithm.

    Original implementation from https://github.com/sungsoo-ahn/bucket-renormalization/blob/master/inference/bucket_elimination.py

    :param model: Model for which to perform inference.
    :return: Natural logarithm of estimated partition function.

    Reference
        Dechter, Rina.
        Bucket elimination: A unifying framework for reasoning.
        Artificial Intelligence, 113(1):41–85, 1999.
        https://arxiv.org/abs/1302.3572
    """
    return BucketElimination(_convert(model)).run()


def bucket_renormalization(model: GraphModel,
                           ibound: int = 10,
                           max_iter: int = 1) -> float:
    """Inference with Bucket Renormalization.

    :param model: Model for which to perform inference.
    :param ibound: Maximal size of mini-bucket.
    :param max_iter: Number of optimization iterations. 0 corresponds to
        Mini-Bucket Renormalization. 1 corresponds to Global-Bucket
        Renormalization. Default is 1.
    :return: Natural logarithm of estimated partition function.

    Reference
        Sungsoo Ahn, Michael Chertkov, Adrian Weller, Jinwoo Shin
        Bucket Renormalization for Approximate Inference, 2018.
        https://arxiv.org/abs/1803.05104
    """
    algo = BucketRenormalization(_convert(model), ibound=ibound)
    return algo.run(max_iter=max_iter)


def iterative_join_graph_propagation(model: GraphModel,
                                     ibound: int = 10) -> float:
    """Inference with Iterative Join Graph Propagation.

    :param model: Model for which to perform inference.
    :return: Natural logarithm of estimated partition function.


    https://arxiv.org/abs/1301.0564
    """
    return IterativeJoinGraphPropagation(_convert(model), ibound=ibound).run()


def mean_field(model: GraphModel) -> float:
    """Inference with Mean Field.

    :param model: Model for which to perform inference.
    :return: Natural logarithm of estimated partition function.
    """
    return MeanField(_convert(model)).run()


def mini_bucket_elimination(model: GraphModel, ibound: int = 10) -> float:
    """Inference with Mean Field.

    :param model: Model for which to perform inference.
    :param ibound:
    :return: Natural logarithm of estimated partition function.
    """
    return MiniBucketElimination(_convert(model), ibound=ibound).run()


def weighted_mini_bucket_elimination(model: GraphModel,
                                     ibound: int = 10) -> float:
    """Inference with Weighted Mini Bucket Elimination.

    :param model: Model for which to perform inference.
    :param ibound:
    :return: Natural logarithm of estimated partition function.
    """
    return WeightedMiniBucketElimination(_convert(model), ibound=ibound).run()
