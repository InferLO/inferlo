# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import warnings

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .bucket_elimination import BucketElimination
from .factor import Factor, default_factor_name, product_over_
from .graphical_model import GraphicalModel
from .mini_bucket_elimination import MiniBucketElimination


class BucketRenormalization(MiniBucketElimination):
    """Bucket Renormalization algorithm."""

    def __init__(self, model: GraphicalModel, **kwargs):
        super(BucketRenormalization, self).__init__(model, **kwargs)
        self._initialize_projectors()

    def _initialize_projectors(self):
        replications = dict()
        working_model = self.renormalized_model.copy()
        for var in self.elimination_order:
            main_rvar = self.variables_replicated_from_[var][-1]
            main_projectors = []
            for (i, rvar) in enumerate(self.variables_replicated_from_[var]):
                if i < len(self.variables_replicated_from_[var]) - 1:
                    fac = product_over_(*working_model.get_adj_factors(rvar))
                    replicated_projector = self._get_svd_projector(fac, rvar)
                    replicated_projector.name = "RP_{}".format(rvar)
                    projector = replicated_projector.copy()
                    projector.variables = [main_rvar]
                    projector.name = "P_{}".format(rvar)

                    replications[rvar] = (
                        main_rvar, replicated_projector, projector)
                    main_projectors.append(projector)

                    working_model.add_factors_from(
                        [replicated_projector.copy(), projector.copy()])
                    self.renormalized_model.add_factors_from(
                        [replicated_projector, projector])

                working_model.contract_variable(rvar)

        self.replications = replications

    def _optimize(self):
        for var in reversed(self.renormalized_elimination_order):
            if var in self.replications.keys():
                mb_var, projector, mb_projector = self.replications[var]
                self.renormalized_model.remove_factors_from(
                    [projector, mb_projector])
                be = BucketElimination(self.renormalized_model)
                marginal_factor = be.get_marginal_factor(
                    elimination_order_method="given",
                    elimination_order=self.renormalized_elimination_order,
                    exception_variables=[var, mb_var],
                )
                new_mb_projector = self._get_svd_projector(marginal_factor,
                                                           mb_var)
                new_projector = Factor(
                    name=default_factor_name(),
                    variables=[var],
                    log_values=new_mb_projector.log_values,
                )

                self.renormalized_model.add_factors_from(
                    [new_projector, new_mb_projector])
                self.replications[var] = (
                    mb_var, new_projector, new_mb_projector)

    def run(self, max_iter=10):
        """Runs the algorithm, returns log(Z)."""
        for _ in range(max_iter):
            self._optimize()

    def get_log_z(self):
        """Calculates log Z."""
        be = BucketElimination(self.renormalized_model)
        logZ = self.base_logZ
        logZ += be.run(
            elimination_order_method="given",
            elimination_order=self.renormalized_elimination_order
        )

        return logZ

    def _get_svd_projector(self, factor, variable):
        factor.transpose_by_(
            [variable, *sorted(set(factor.variables) - set([variable]))])
        flattened_factor_log_values = factor.log_values.reshape(
            factor.get_cardinality_for_(variable), -1
        )

        max_log = np.max(flattened_factor_log_values)
        if np.isnan(max_log):
            warnings.warn('Got nan in flattened_factor_log_values')
            np.nan_to_num(flattened_factor_log_values, copy=False, nan=-np.inf)
            max_log = np.max(flattened_factor_log_values)
        if not np.isfinite(max_log):
            warnings.warn('Got infinite value in flattened_factor_log_values')
            max_log = 0.0
        flattened_factor_values = np.exp(flattened_factor_log_values - max_log)

        U, _, _ = randomized_svd(flattened_factor_values, n_components=1)

        # U,_,_ = np.linalg.svd(flattened_factor_values)
        u = U[:, 0]
        if np.sum(u) < 0:
            u = -u

        u[u < 0] = 0.0
        u /= np.linalg.norm(u)

        return Factor(name=default_factor_name(), variables=[variable],
                      values=u)
