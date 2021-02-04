import numpy as np
from copy import copy
import random
import sys

sys.path.extend(["graphical_model/"])
from factor import Factor, default_factor_name, product_over_
from bucket_elimination import BucketElimination
from mini_bucket_elimination import MiniBucketElimination
from sklearn.utils.extmath import randomized_svd


class BucketRenormalization(MiniBucketElimination):
    def __init__(self, model, **kwargs):
        super(BucketRenormalization, self).__init__(model, **kwargs)
        self.initialize_projectors()

    def initialize_projectors(self):
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

                    replications[rvar] = (main_rvar, replicated_projector, projector)
                    main_projectors.append(projector)

                    working_model.add_factors_from([replicated_projector.copy(), projector.copy()])
                    self.renormalized_model.add_factors_from([replicated_projector, projector])

                working_model.contract_variable(rvar)

        self.replications = replications

    def optimize(self):
        for var in reversed(self.renormalized_elimination_order):
            if var in self.replications.keys():
                mb_var, projector, mb_projector = self.replications[var]
                self.renormalized_model.remove_factors_from([projector, mb_projector])
                be = BucketElimination(self.renormalized_model)
                marginal_factor = be.get_marginal_factor(
                    elimination_order_method="given",
                    elimination_order=self.renormalized_elimination_order,
                    exception_variables=[var, mb_var],
                )
                new_mb_projector = self._get_svd_projector(marginal_factor, mb_var)
                new_projector = Factor(
                    name=default_factor_name(),
                    variables=[var],
                    log_values=new_mb_projector.log_values,
                )

                self.renormalized_model.add_factors_from([new_projector, new_mb_projector])
                self.replications[var] = (mb_var, new_projector, new_mb_projector)

    def run(self, max_iter=10):
        for t in range(max_iter):
            self.optimize()

        return self.get_logZ()

    def get_logZ(self):
        be = BucketElimination(self.renormalized_model)
        logZ = self.base_logZ
        logZ += be.run(
            elimination_order_method="given", elimination_order=self.renormalized_elimination_order
        )

        return logZ

    def _get_svd_projector(self, factor, variable):
        factor.transpose_by_([variable, *sorted(set(factor.variables) - set([variable]))])
        flattened_factor_log_values = factor.log_values.reshape(
            factor.get_cardinality_for_(variable), -1
        )

        if np.isfinite(np.max(flattened_factor_log_values)):
            flattened_factor_values = np.exp(
                flattened_factor_log_values - np.max(flattened_factor_log_values)
            )
        else:
            raise ValueError()

        U, _, _ = randomized_svd(flattened_factor_values, n_components=1)

        # U,_,_ = np.linalg.svd(flattened_factor_values)
        u = U[:, 0]
        if np.sum(u) < 0:
            u = -u

        u[u < 0] = 0.0
        u /= np.linalg.norm(u)

        return Factor(name=default_factor_name(), variables=[variable], values=u)
