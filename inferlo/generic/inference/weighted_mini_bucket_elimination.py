# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import numpy as np

from .factor import product_over_, Factor
from .graphical_model import GraphicalModel
from .mini_bucket_elimination import MiniBucketElimination


class WeightedMiniBucketElimination(MiniBucketElimination):
    """Weighted Mini Bucket Elimination algorithm."""

    def __init__(self, model: GraphicalModel, **kwargs):
        super(WeightedMiniBucketElimination, self).__init__(model, **kwargs)

        self._initialize_holder_weights()
        self.reparameterization_step_size = 0.1
        self.holder_weight_step_size = 0.1
        self.messages = dict()

    def _initialize_holder_weights(self):
        holder_weights_for_ = dict()
        for _, rep_vars in self.variables_replicated_from_.items():
            for rep_var in rep_vars:
                holder_weights_for_[rep_var] = 1.0 / len(rep_vars)

        self.holder_weights_for_ = holder_weights_for_

    def run(self, max_iter=10, update_weight=True, update_reparam=True,
            verbose=False):
        """Runs the algorithm, returns log(Z)."""
        for var in self.elimination_order:
            for rvar in self.variables_replicated_from_[var]:
                self._forward_pass_for_(rvar)

        if max_iter > 0:
            for var in reversed(self.elimination_order):
                for rvar in self.variables_replicated_from_[var]:
                    if self.variable_upper_to_[rvar]:
                        self._backward_pass_for_(self.variable_upper_to_[rvar],
                                                 rvar)

        for t in range(max_iter):
            if verbose:
                print("{}/{}".format(t, max_iter))
                print(self._get_log_z())
            converge_flag = self._update_parameters(
                update_weight=update_weight, update_reparam=update_reparam
            )
            if converge_flag:
                print(t)
                break

        for var in self.renormalized_elimination_order:
            self._forward_pass_for_(var)

        return self._get_log_z()

    def _update_parameters(self, update_weight=False, update_reparam=True):
        converge_flag = False
        for var in self.elimination_order:
            # if len(self.variables_replicated_from_[var]) > 1:
            if update_weight:
                self._update_holder_weights_for_(var)
            if update_reparam:
                reparam_converge_flag = self._update_reparameterization_for_(
                    var)
                if not reparam_converge_flag:
                    converge_flag = False

            for rvar in self.variables_replicated_from_[var]:
                self._forward_pass_for_(rvar)

        for var in reversed(self.elimination_order):
            # if len(self.variables_replicated_from_[var]) > 1:
            if update_weight:
                self._update_holder_weights_for_(var)
            if update_reparam:
                reparam_converge_flag = self._update_reparameterization_for_(
                    var)
                if not reparam_converge_flag:
                    converge_flag = False
                    # print(converge_flag)

            for rvar in self.variables_replicated_from_[var]:
                if self.variable_upper_to_[rvar]:
                    self._backward_pass_for_(self.variable_upper_to_[rvar],
                                             rvar)

        return converge_flag

    def _get_log_z(self):
        logZ = self.base_logZ
        for var in self.renormalized_elimination_order:
            if not self.variable_upper_to_[var]:
                upper_var = self.variable_upper_to_[var]
                logZ += self.messages[var, upper_var].log_values

        return logZ

    def _get_marginals_upper_to(self, variable):
        marginal = product_over_(self.factor_upper_to_[variable],
                                 *self._messages_to_(variable))
        marginal.pow(1 / self.holder_weights_for_[variable])
        marginal.normalize()
        marginal.name = "q_c_{}".format(variable)
        return marginal

    def _forward_pass_for_(self, variable):
        upper_variable = self.variable_upper_to_[variable]
        message = product_over_(
            self.factor_upper_to_[variable],
            *self._upper_messages_to_(variable))
        message.marginalize(
            [variable], operator="weighted_sum",
            weight=self.holder_weights_for_[variable]
        )
        message.name = "M_{}->{}".format(variable, upper_variable)

        self.messages[(variable, upper_variable)] = message

    def _backward_pass_for_(self, variable, lower_variable):
        message = product_over_(self.factor_upper_to_[variable],
                                *self._messages_to_(variable))
        message.pow(
            self.holder_weights_for_[lower_variable] /
            self.holder_weights_for_[variable])
        message.div(self.messages[(lower_variable, variable)])

        lower_upper_factor = self.factor_upper_to_[lower_variable]
        variables_to_marginalize = list(
            set(message.variables) - set(lower_upper_factor.variables))
        message.marginalize(
            variables_to_marginalize,
            operator="weighted_sum",
            weight=self.holder_weights_for_[lower_variable],
        )
        message.name = "M_{}<-{}".format(lower_variable, variable)
        self.messages[(variable, lower_variable)] = message

    def _update_reparameterization_for_(self, variable):
        belief_from_ = dict()
        log_average_belief = 0.0
        for rvar in self.variables_replicated_from_[variable]:
            belief_from_[rvar] = self._get_marginals_upper_to(rvar)
            belief_from_[rvar].marginalize_except_([rvar])
            belief_from_[rvar].variables = [variable]
            belief_from_[rvar].normalize()
            log_average_belief = (
                log_average_belief + self.holder_weights_for_[rvar] *
                belief_from_[rvar].log_values
            )

        converge_flag = True
        for rvar in self.variables_replicated_from_[variable]:
            if np.sum(np.abs(-belief_from_[rvar].log_values +
                             log_average_belief)) > 1e-2:
                converge_flag = False
                temp_log_val = (self.holder_weights_for_[rvar]) * (
                    -belief_from_[rvar].log_values + log_average_belief
                )
                temp = Factor("", [rvar], log_values=temp_log_val)
                self.factor_upper_to_[rvar].product(temp)

        return converge_flag

    def _update_holder_weights_for_(self, variable):
        entropy_for_ = dict()
        average_entropy = 0.0
        for rvar in self.variables_replicated_from_[variable]:
            belief_from_rvar = self._get_marginals_upper_to(rvar)
            b_values = belief_from_rvar.values
            cb_values = belief_from_rvar.normalize(
                [rvar], inplace=False).values

            b_values.ravel()
            cb_values.ravel()
            index_to_keep = b_values != 0
            entropy_for_[rvar] = -np.sum(
                b_values[index_to_keep] * np.log(cb_values[index_to_keep]))

            average_entropy += self.holder_weights_for_[rvar] * entropy_for_[
                rvar]

        holder_weight_sum = 0.0
        for rvar in self.variables_replicated_from_[variable]:
            self.holder_weights_for_[rvar] *= np.exp(
                -self.holder_weight_step_size * (
                    entropy_for_[rvar] - average_entropy)
            )
            holder_weight_sum += self.holder_weights_for_[rvar]

        for rvar in self.variables_replicated_from_[variable]:
            self.holder_weights_for_[rvar] /= holder_weight_sum

    def _messages_to_(self, variable):
        if self.variable_upper_to_[variable]:
            upper_variable = self.variable_upper_to_[variable]
            return [self.messages[
                (upper_variable, variable)]] + self._upper_messages_to_(
                variable)
        else:
            return self._upper_messages_to_(variable)

    def _upper_messages_to_(self, variable):
        return [
            self.messages[(lower_var, variable)] for lower_var in
            self.variables_lower_to_[variable]
        ]
