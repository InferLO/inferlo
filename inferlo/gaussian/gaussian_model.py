# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from __future__ import annotations
from functools import partial
from collections import defaultdict
import numpy as np
from inferlo.base.domain import RealDomain
from inferlo.base.generic_graph_model import GenericGraphModel
from inferlo.base.factors.function_factor import FunctionFactor
from .inference.gaussian_belief_propagation import gaussian_BP


class GaussianModel(GenericGraphModel):
    """Gaussian graphical model

    A Gaussian graphical model is defined by an undirected graph G = (V;E),
    where V is the set of nodes and E is the set of edges,
    and a collection of jointly Gaussian random variables x= (x_i; i in V).
    The probability density is given by p(x) ∝ exp{-1/2*x^T*J*x + h^T*x}
    where J is a symmetric, positive definite matrix, that is sparse so as to
    respect the graph G: if {i,j} not in E, then J_{i,j} = 0

    For more information please refer to
    'Walk-SumsandBeliefPropagationinGaussianGraphicalModels' by
    Dmitry M. Malioutov, Jason K. Johnson, Alan S. Willsky :
    http://ssg.mit.edu/group/willsky/publ_pdfs/185_pub_MLR.pdf
    """
    J = np.array([])
    h = np.array([])
    G = defaultdict(list)
    n = 0

    def __init__(self, J: np.array, h: np.array, domain=None):
        if domain is None:
            domain = RealDomain()
        self.n = J.shape[0]
        super().__init__(self.n, domain)
        self.J = J
        self.h = h

        for i in range(self.n):
            factor_h = FunctionFactor(
                self, [i], partial(
                    lambda x, k: k * x[0], k=self.h[i]))
            self.add_factor(factor_h)
            factor_J_sq = FunctionFactor(self, [i], partial(
                lambda x, k: (-0.5) * k * (x[0] * x[0]), k=self.J[i][i]))
            self.add_factor(factor_J_sq)

            for j in range(i + 1, self.n):
                if abs(self.J[i][j]) > 0:
                    self.G[i].append(j)
                    self.G[j].append(i)
                    factor_J = FunctionFactor(self, [i, j], partial(
                        lambda x, k: -k * x[0] * x[1], k=self.J[i][j]))
                    self.add_factor(factor_J)

    def infer(self, **kwargs):
        return gaussian_BP(self, **kwargs)
