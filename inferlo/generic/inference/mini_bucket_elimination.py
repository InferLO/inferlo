# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import random
from copy import copy
from functools import reduce

import numpy as np

from inferlo.base.graph_model import GraphModel
from .factor import Factor, product_over_
from .graphical_model import GraphicalModel


class MiniBucketElimination:
    def __init__(self, model: GraphicalModel = None, **kwargs):
        self.base_logZ = 0.0
        self.model = model.copy()
        if "elimination_order" in kwargs:
            self.elimination_order = copy(kwargs["elimination_order"])
        else:
            self.elimination_order = []

        if "renormalized_model" not in kwargs:
            self.renormalize_model(**kwargs)
        else:
            self.renormalized_model = kwargs["renormalized_model"].copy()
            self.renormalized_elimination_order = kwargs[
                "renormalized_elimination_order"]
            self.variables_replicated_from_ = kwargs[
                "variables_replicated_from_"]
            self.base_logZ = kwargs["base_logZ"]

        self.initialize_relation()

    @classmethod
    def initialize_with_(cls, mbr):
        return cls(
            mbr.model,
            elimination_order=mbr.elimination_order,
            renormalized_model=mbr.renormalized_model,
            renormalized_elimination_order=mbr.renormalized_elimination_order,
            variables_replicated_from_=mbr.variables_replicated_from_,
            base_logZ=mbr.base_logZ,
        )

        return mbr

    def initialize_relation(self):
        variable_upper_to_ = {var: None for var in
                              self.renormalized_model.variables}
        variables_lower_to_ = {var: [] for var in
                               self.renormalized_model.variables}
        factor_upper_to_ = {var: Factor.scalar(1.0) for var in
                            self.renormalized_model.variables}
        upper_candidate_for_ = {var: set() for var in
                                self.renormalized_model.variables}

        for fac in self.renormalized_model.factors:
            lower_var = min(fac.variables,
                            key=self.renormalized_elimination_order.index)
            factor_upper_to_[lower_var] = fac
            upper_candidate_for_[lower_var] = upper_candidate_for_[
                lower_var].union(
                set(fac.variables)
            )

        for var in self.renormalized_elimination_order:
            m_vars = sorted(
                upper_candidate_for_[var],
                key=self.renormalized_elimination_order.index
            )
            upper_candidate_for_[var] = copy(m_vars[m_vars.index(var) + 1:])
            if m_vars.index(var) + 1 < len(m_vars):
                upper_var = m_vars[m_vars.index(var) + 1]
                variable_upper_to_[var] = upper_var
                variables_lower_to_[upper_var].append(var)
                upper_candidate_for_[upper_var] = upper_candidate_for_[
                    upper_var].union(
                    m_vars[m_vars.index(var) + 1:]
                )

        self.variable_upper_to_ = variable_upper_to_
        self.variables_lower_to_ = variables_lower_to_

        self.factors_adj_to_ = {
            var: self.renormalized_model.get_adj_factors(var)
            for var in self.renormalized_model.variables
        }
        self.factor_upper_to_ = factor_upper_to_
        # self.upper_candidate_for_ = upper_candidate_for_

    def renormalize_model(self, **kwargs):
        ibound = kwargs["ibound"]
        if "elimination_order_method" in kwargs:
            if kwargs["elimination_order_method"] == "random":
                elimination_order = copy(self.model.variables)
                use_min_fill = False
                random.shuffle(elimination_order)
            elif kwargs["elimination_order_method"] == "not_random":
                elimination_order = copy(self.model.variables)
                use_min_fill = False
            elif kwargs["elimination_order_method"] == "min_fill":
                elimination_order = []
                use_min_fill = True
        elif "elimination_order" in kwargs:
            elimination_order = copy(kwargs["elimination_order"])
            use_min_fill = False
        else:
            elimination_order = copy(self.model.variables)
            use_min_fill = False
            random.shuffle(elimination_order)
            # elimination_order = []
            # use_min_fill = True

        renormalized_model = self.model.copy()
        renormalized_elimination_order = []

        variables_replicated_from_ = {var: [] for var in self.model.variables}
        factors_adj_to_ = dict()
        working_factorss = [[fac] for fac in renormalized_model.factors]
        eliminated_variables = []
        for t in range(len(self.model.variables)):
            uneliminated_variables = sorted(
                set(self.model.variables) - set(eliminated_variables))
            candidate_mini_buckets_for_ = dict()

            bucket_for_ = {cand_var: [] for cand_var in uneliminated_variables}
            for facs in working_factorss:
                for cand_var in self.get_variables_in_(
                        [[fac] for fac in facs], eliminated=eliminated_variables
                ):
                    bucket_for_[cand_var].append(facs)

            for cand_var in uneliminated_variables:
                candidate_mini_buckets_for_[cand_var] = []
                for facs in bucket_for_[cand_var]:
                    mini_bucket = next(
                        (
                            mb
                            for mb in candidate_mini_buckets_for_[cand_var]
                            if self.get_bucket_size(
                                mb + [facs], eliminated=eliminated_variables + [cand_var]
                            )
                            < ibound
                        ),
                        False,
                    )
                    if mini_bucket:
                        mini_bucket.append(facs)
                    else:
                        candidate_mini_buckets_for_[cand_var].append([facs])

            if use_min_fill:
                var, mini_buckets = min(
                    candidate_mini_buckets_for_.items(), key=lambda x: len(x[1])
                )
                elimination_order.append(var)
            else:
                var = elimination_order[t]
                mini_buckets = candidate_mini_buckets_for_[var]

            eliminated_variables.append(var)
            mini_buckets.sort(
                key=lambda mb: self.get_bucket_size(mb,
                                                    eliminated=eliminated_variables)
            )

            remove_idx = []
            for working_facs_idx, working_facs in enumerate(working_factorss):
                if var in self.get_variables_in_(
                        [[fac] for fac in working_facs]):
                    remove_idx.append(working_facs_idx)

            for i in reversed(sorted(remove_idx)):
                working_factorss.pop(i)

            for (i, mb) in enumerate(mini_buckets):
                mb_facs = [fac for facs in mb for fac in facs]
                working_factorss.append(mb_facs)

                replicated_var = var + "_" + str(i)
                variables_replicated_from_[var].append(replicated_var)
                factors_adj_to_[replicated_var] = [fac for fac in mb_facs if
                                                   var in fac.variables]

        for var in elimination_order:
            for replicated_var in variables_replicated_from_[var]:
                renormalized_model.add_variable(replicated_var)
                renormalized_elimination_order.append(replicated_var)
                for fac in factors_adj_to_[replicated_var]:
                    fac.variables[fac.variables.index(var)] = replicated_var

            renormalized_model.remove_variable(var)

        factors_upper_to_ = {var: [] for var in renormalized_model.variables}
        for fac in renormalized_model.factors:
            lower_var = min(fac.variables,
                            key=renormalized_elimination_order.index)
            factors_upper_to_[lower_var].append(fac)

        for var, facs in factors_upper_to_.items():
            if facs:
                new_fac = product_over_(*facs)
                for fac in facs:
                    renormalized_model.remove_factor(fac)
                renormalized_model.add_factor(new_fac)

        base_logZ = 0.0
        for fac in renormalized_model.factors:
            base_logZ += np.max(fac.log_values)
            fac.log_values -= np.max(fac.log_values)

        self.elimination_order = elimination_order
        self.renormalized_model = renormalized_model
        self.renormalized_elimination_order = renormalized_elimination_order
        self.variables_replicated_from_ = variables_replicated_from_
        self.base_logZ = base_logZ

    def run(self):
        working_model = self.renormalized_model.copy()
        for var in self.elimination_order:
            for i, rvar in enumerate(self.variables_replicated_from_[var]):
                if i < len(self.variables_replicated_from_[var]) - 1:
                    working_model.contract_variable(rvar, operator="max")
                else:
                    working_model.contract_variable(rvar, operator="sum")

        logZ = self.base_logZ
        for fac in working_model.factors:
            logZ += fac.log_values

        return logZ

    def get_variables_in_(self, bucket, eliminated=[]):
        if [fac.variables for facs in bucket for fac in facs]:
            variables_in_bucket = reduce(
                lambda vars1, vars2: set(vars1).union(set(vars2)),
                [fac.variables for facs in bucket for fac in facs],
            )
            variables_in_bucket = sorted(
                set(variables_in_bucket) - set(eliminated))
            return list(variables_in_bucket)
        else:
            return []

    def get_bucket_size(self, bucket, eliminated=[]):
        variables_in_bucket = self.get_variables_in_(bucket, eliminated)
        if variables_in_bucket:
            return len(
                variables_in_bucket
            )  # np.prod([self.model.get_cardinality_for_(var) for var in variables_in_bucket])
        else:
            return 0
