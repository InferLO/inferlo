# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
from copy import copy
from typing import List

import numpy as np

from .factor import product_over_, Factor


class GraphicalModel:
    """
    Graphical model representation used by algorithms taken from
    https://github.com/sungsoo-ahn/bucket-renormalization
    TODO(fedimser): use Inferlo's GraphModel instead.
    """

    def __init__(self, variables: List[str], factors: List[Factor]):
        self.variables = variables
        self.factors = factors

    """
    Variable related operations
    """

    def add_variable(self, variable: str):
        """Adds variable."""
        self.variables.append(variable)

    def add_variables_from(self, variables):
        """Adds multiple variables."""
        for variable in variables:
            self.add_variable(variable)

    def remove_variable(self, variable):
        """Removes variable."""
        self.variables.remove(variable)

    def remove_variables_from(self, variables):
        """Removes multiple variables."""
        for variable in variables:
            self.variables.remove(variable)

    def get_cardinality_for_(self, variable):
        """Returns cardinality of a variable."""
        factor = next(
            factor for factor in self.factors if variable in factor.variables)
        if factor:
            return factor.get_cardinality_for_(variable)
        else:
            raise ValueError("variable not in the model")

    def get_cardinalities_from(self, variables):
        """Returns cardinalities for given variables."""
        cardinalities = []
        for variable in variables:
            cardinalities.append(self.get_cardinality_for_(variable))
        return cardinalities

    def contract_variable(self, variable, operator="sum", **kwargs):
        """Contracts variable."""
        adj_factors = self.get_adj_factors(variable)
        new_factor = product_over_(*adj_factors).copy(rename=True)
        new_factor.marginalize([variable], operator=operator, **kwargs)
        for factor in adj_factors:
            self.remove_factor(factor)

        self.remove_variable(variable)
        self.add_factor(new_factor)

        return new_factor

    def get_adj_factors(self, variable):
        """Retruns all factors depending on given variable."""
        factor_list = []
        for factor in self.factors:
            if variable in factor.variables:
                factor_list.append(factor)

        return factor_list

    def degree(self, variable):
        """Returns number of vactors depending on given variable."""
        return len(self.get_adj_factors(variable))

    """
    Factor related operations
    """

    def add_factor(self, factor: Factor):
        """Adds a factor."""
        if set(factor.variables) - \
                set(factor.variables).intersection(set(self.variables)):
            raise ValueError("Factors defined on variable not in the model.")
        self.factors.append(factor)

    def add_factors_from(self, factors):
        """Adds multiple factors."""
        for factor in factors:
            self.add_factor(factor)

    def remove_factor(self, factor):
        """Removes a factor."""
        self.factors.remove(factor)

    def remove_factors_from(self, factors):
        """Removes multiple factors."""
        for factor in factors:
            self.factors.remove(factor)

    def get_factor(self, name):
        """Gets factor by name."""
        for factor in self.factors:
            if factor.name == name:
                return factor

    def get_factors_from(self, names):
        """Gets factors by their names."""
        factors = []
        for factor in self.factors:
            if factor.name in names:
                factors.append(factor)
        return factors

    """
    GM related operations
    """

    def copy(self) -> 'GraphicalModel':
        """Makes copy of this model."""
        return GraphicalModel(copy(self.variables),
                              [factor.copy() for factor in self.factors])

    def summary(self):
        """Print summary about this model."""
        print(np.max([self.get_cardinality_for_(var)
                      for var in self.variables]))
        print(np.max([len(fac.variables) for fac in self.factors]))
        print(len([fac for fac in self.factors if len(fac.variables) > 1]))

    def display_factors(self):
        """Prints all factors."""
        for fac in self.factors:
            print(fac)
