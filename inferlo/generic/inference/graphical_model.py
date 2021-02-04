from copy import copy

import numpy as np

from inferlo.base.factors.discrete_factor import DiscreteFactor
from inferlo.base.graph_model import GraphModel
from .factor import Factor, product_over_


class GraphicalModel:
    """
    Graphical model representation used by algorithms taken from
    https://github.com/sungsoo-ahn/bucket-renormalization
    TODO: use Inferlo's GraphModel instead.
    """
    def __init__(self, variables=[], factors=[]):
        if variables:
            self.variables = variables
        else:
            self.variables = []

        self.factors = []

        if factors:
            for factor in factors:
                self.add_factor(factor)

    """
    Variable related operations
    """

    def add_variable(self, variable):
        self.variables.append(variable)

    def add_variables_from(self, variables):
        for variable in variables:
            self.add_variable(variable)

    def remove_variable(self, variable):
        self.variables.remove(variable)

    def remove_variables_from(self, variables):
        for variable in variables:
            self.variables.remove(variable)

    def get_cardinality_for_(self, variable):
        factor = next(factor for factor in self.factors if variable in factor.variables)
        if factor:
            return factor.get_cardinality_for_(variable)
        else:
            raise ValueError("variable not in the model")

    def get_cardinalities_from(self, variables):
        cardinalities = []
        for variable in variables:
            cardinalities.append(self.get_cardinality_for_(variable))
        return cardinalities

    def contract_variable(self, variable, operator="sum", **kwargs):
        adj_factors = self.get_adj_factors(variable)
        new_factor = product_over_(*adj_factors).copy(rename=True)
        new_factor.marginalize([variable], operator=operator, **kwargs)
        for factor in adj_factors:
            self.remove_factor(factor)

        self.remove_variable(variable)
        self.add_factor(new_factor)

        return new_factor

    def get_adj_factors(self, variable):
        factor_list = []
        for factor in self.factors:
            if variable in factor.variables:
                factor_list.append(factor)

        return factor_list

    def degree(self, variable):
        return len(self.get_adj_factors(variable))

    """
    Factor related operations
    """

    def add_factor(self, factor):
        if set(factor.variables) - set(factor.variables).intersection(set(self.variables)):
            raise ValueError("Factors defined on variable not in the model.")
        self.factors.append(factor)

    def add_factors_from(self, factors):
        for factor in factors:
            self.add_factor(factor)

    def remove_factor(self, factor):
        self.factors.remove(factor)

    def remove_factors_from(self, factors):
        for factor in factors:
            self.factors.remove(factor)

    def get_factor(self, name):
        for factor in self.factors:
            if factor.name == name:
                return factor

    def get_factors_from(self, names):
        factors = []
        for factor in self.factors:
            if factor.name in names:
                factors.append(factor)
        return factors

    """
    GM related operations
    """

    def copy(self) -> 'GraphicalModel':
        return GraphicalModel(copy(self.variables), [factor.copy() for factor in self.factors])

    def summary(self):
        print(np.max([self.get_cardinality_for_(var) for var in self.variables]))
        print(np.max([len(fac.variables) for fac in self.factors]))
        print(len([fac for fac in self.factors if len(fac.variables) > 1]))

    def display_factors(self):
        for fac in self.factors:
            print(fac)

    @staticmethod
    def from_inferlo_model(inferlo_model: GraphModel) -> 'GraphicalModel':
        model = GraphicalModel()
        model.name = 'generated_from_inferlo'

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

def check_forney(gm):
    for variable in gm.variables:
        if gm.degree(variable) != 2:
            return False
    return True
