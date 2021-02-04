import sys
import numpy as np
import random
from functools import reduce
from copy import copy

sys.path.extend(["graphical_model/"])
from factor import Factor, product_over_


class BucketElimination:
    def __init__(self, model):
        self.model = model.copy()

    def run(self, elimination_order_method="random", **kwargs):
        if elimination_order_method == "random":
            elimination_order = copy(self.model.variables)
            random.shuffle(elimination_order)
        elif elimination_order_method == "not_random":
            elimination_order = copy(self.model.variables)
        elif elimination_order_method == "given":
            elimination_order = kwargs["elimination_order"]

        eliminated_model = self.model.copy()
        max_ibound = 0
        for var in elimination_order:
            max_ibound = max(
                max_ibound, get_bucket_size([fac for fac in eliminated_model.get_adj_factors(var)])
            )
            eliminated_model.contract_variable(var)
        Z = Factor.scalar(1.0)
        for fac in eliminated_model.factors:
            Z = Z * fac

        return Z.log_values

    def get_marginal_factor(self, elimination_order_method="random", **kwargs):
        if elimination_order_method == "random":
            elimination_order = copy(self.model.variables)
            random.shuffle(elimination_order)
        elif elimination_order_method == "not_random":
            elimination_order = copy(self.model.variables)
        elif elimination_order_method == "given":
            elimination_order = kwargs["elimination_order"]

        if "exception_variables" in kwargs:
            exception_variables = kwargs["exception_variables"]
        else:
            exception_variables = []

        eliminated_model = self.model.copy()
        max_ibound = 0
        for var in elimination_order:
            if var not in exception_variables:
                max_ibound = max(
                    max_ibound,
                    get_bucket_size([fac for fac in eliminated_model.get_adj_factors(var)]),
                )
                eliminated_model.contract_variable(var)

        final_factor = product_over_(*eliminated_model.factors)
        if "exception_variables" in kwargs:
            final_factor.transpose_by_(exception_variables)

        return final_factor


def get_bucket_size(bucket):
    s = set()
    for fac in bucket:
        s = s.union(set(fac.variables))

    return len(s)
