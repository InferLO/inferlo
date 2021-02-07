# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import random
from copy import copy
from typing import Dict, Tuple, List

import numpy as np

from .factor import Factor, product_over_, entropy
from .graphical_model import GraphicalModel
from ... import InferenceResult


def _default_message_name(prefix="_M"):
    _default_message_name.cnt += 1
    return prefix + str(_default_message_name.cnt)


_default_message_name.cnt = 0


class BeliefPropagation:
    """Belief Propagation algorithm."""

    def __init__(self, model):
        self.model = model.copy()
        init_np_func = np.ones
        self.factors_adj_to_ = {var: self.model.get_adj_factors(
            var) for var in self.model.variables}

        self.messages = dict()  # type: Dict[Tuple, Factor]
        for fac in model.factors:
            for var in fac.variables:
                self.messages[(fac, var)] = Factor.initialize_with_(
                    _default_message_name(), [var], init_np_func,
                    model.get_cardinality_for_(var)
                )
                self.messages[(fac, var)].normalize()

        for fac in model.factors:
            for var in fac.variables:
                self.messages[(var, fac)] = Factor.initialize_with_(
                    _default_message_name(), [var], init_np_func,
                    model.get_cardinality_for_(var)
                )
                self.messages[(var, fac)].normalize()

    def run(self, max_iter=1000, converge_thr=1e-5, damp_ratio=0.1):
        """Runs the algorithm, returns log(Z)."""
        for _ in range(max_iter):
            old_messages = {key: item.copy() for key, item in
                            self.messages.items()}
            self._update_messages(damp_ratio)
            if self._is_converged(converge_thr, self.messages, old_messages):
                break

        self.beliefs = {}
        for var in self.model.variables:
            self.beliefs[var] = product_over_(
                *self._message_to_(var)).normalize(inplace=False)

        for fac in self.model.factors:
            self.beliefs[fac] = product_over_(fac, *self._message_to_(
                fac)).normalize(inplace=False)

    def get_log_z(self):
        """Calculates partition function based on beliefs."""
        log_z = 0.0
        for var in self.model.variables:
            log_z += (1 - self.model.degree(var)) * entropy(self.beliefs[var])
        for fac in self.model.factors:
            log_z += entropy(self.beliefs[fac], fac)
        return log_z

    def get_marginal_probabilities(self) -> np.array:
        """Calculates marginal probabilities based on beliefs."""

        def to_np_array(marginals: List[np.array]):
            max_card = max(len(m) for m in marginals)
            result = np.zeros((len(marginals), max_card))
            for i in range(len(marginals)):
                result[i, 0:len(marginals[i])] = marginals[i]
            return result

        return to_np_array(
            [self.beliefs[var].values for var in self.model.variables])

    def get_inference_result(self) -> InferenceResult:
        """Extracts inference result (partition functions and marginals)."""
        return InferenceResult(log_pf=self.get_log_z(),
                               marg_prob=self.get_marginal_probabilities())

    def _update_messages(self, damp_ratio):
        dr = damp_ratio
        factor_order = copy(self.model.factors)
        random.shuffle(factor_order)

        def update_message(key, msg):
            self.messages[key] = dr * self.messages[key] + (1 - dr) * msg

        for fac in factor_order:
            for var in fac.variables:
                messages = [msg for msg in
                            self._message_to_(fac, except_objs=[var])]
                next_message = (product_over_(fac, *messages)
                                .marginalize_except_([var], inplace=False)
                                .normalize(inplace=False))
                update_message((fac, var), next_message)

        variable_order = copy(self.model.variables)
        random.shuffle(variable_order)
        for var in variable_order:
            for fac in self.factors_adj_to_[var]:
                messages_to_var_except_fac = self._message_to_(var,
                                                               except_objs=[
                                                                   fac])
                if messages_to_var_except_fac:
                    next_message = product_over_(
                        *messages_to_var_except_fac).normalize(
                        inplace=False
                    )
                    update_message((var, fac), next_message)

    def _is_converged(self, converge_thr, messages, new_messages):
        for var in self.model.variables:
            blf = product_over_(
                *[messages[(fac, var)] for fac in self.factors_adj_to_[var]]
            ).normalize(inplace=False)
            new_blf = product_over_(
                *[new_messages[(fac, var)] for fac in self.factors_adj_to_[var]]
            ).normalize(inplace=False)
            if np.sum(np.abs(blf.values - new_blf.values)) > converge_thr:
                return False

        return True

    def _message_to_(self, obj, except_objs=None):
        if except_objs is None:
            except_objs = []
        if obj in self.model.factors:
            return [self.messages[(var, obj)] for var in obj.variables if
                    var not in except_objs]
        elif obj in self.model.variables:
            return [
                self.messages[(fac, obj)]
                for fac in self.factors_adj_to_[obj]
                if fac not in except_objs
            ]
        else:
            raise TypeError("Object {obj} not in the model.".format(obj=obj))


class IterativeJoinGraphPropagation(BeliefPropagation):
    """Iterative Join Graph Propagation algorithm."""

    def __init__(self, model: GraphicalModel, ibound: int):
        self.org_model = model.copy()
        model = model.copy()

        unelminated_variables = copy(model.variables)
        while unelminated_variables:

            def get_bucket_size(facs):
                adj_adj_vars = [fac.variables for fac in facs]
                a = set()
                for variables in adj_adj_vars:
                    a = a.union(variables)
                return len(a)

            def get_key(var):
                return get_bucket_size(model.get_adj_factors(var))

            var = min(unelminated_variables, key=get_key)
            unelminated_variables.pop(unelminated_variables.index(var))
            if get_key(var) == 1:
                pass
            elif get_key(var) < ibound:
                model.contract_variable(var)
            else:
                facs = model.get_adj_factors(var)
                model.remove_factors_from(facs)
                mini_buckets = []
                for fac in facs:
                    mini_bucket = next(
                        (mb for mb in mini_buckets if
                         get_bucket_size(mb + [fac]) < ibound), False
                    )
                    if mini_bucket:
                        mini_bucket.append(fac)
                    else:
                        mini_buckets.append([fac])

                for mini_bucket in mini_buckets:
                    if mini_bucket:
                        model.add_factor(product_over_(*mini_bucket))
                    else:
                        print("empty mini-bucket")

        super(IterativeJoinGraphPropagation, self).__init__(model)
