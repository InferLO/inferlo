from copy import copy
from functools import reduce

import numpy as np
from numpy import exp, log, amax, amin, squeeze


class Factor:
    """
    Factor representation used by algorithms taken from
    https://github.com/sungsoo-ahn/bucket-renormalization
    TODO: use Inferlo's DiscreteFactor instead.
    """

    def __init__(self, name=None, variables=[], **kwargs):
        if name:
            self.name = name
        else:
            self.name = default_factor_name()

        self.variables = variables

        if "values" in kwargs:
            values = kwargs["values"]

            with np.errstate(divide="ignore", invalid="ignore"):
                self.log_values = log(values)
        elif "log_values" in kwargs:
            self.log_values = kwargs["log_values"]
        else:
            self.log_values = np.array(0.0)

        if len(variables) != self.log_values.ndim:
            raise ValueError(
                "Shape {shape} of {name} does not match number {variable_num}.".format(
                    shape=self.log_values.shape, name=name,
                    variable_num=len(variables)
                )
            )

        self.cardinality = list(self.log_values.shape)

    @property
    def values(self):
        with np.errstate(over="raise"):
            values = exp(self.__log_values)

        return values

    @values.setter
    def values(self, values):
        if (values < -1e-15).any():
            raise ValueError(
                "value of factor cannot be negative:{}".format(values))
        elif (values < 0).any():
            values[values < 0.0] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            self.log_values = log(values)

    @property
    def log_values(self):
        return self.__log_values

    @log_values.setter
    def log_values(self, log_values):
        self.__log_values = log_values

    @classmethod
    def initialize_with_(cls, name, variables, numpy_func, cardinality,
                         **kwargs):
        return cls(name=name, variables=variables,
                   values=numpy_func(cardinality), **kwargs)

    @classmethod
    def full_like_(cls, factor, value):
        return cls(
            name=copy(factor.name),
            variables=copy(factor.variables),
            values=np.full_like(factor.values, value),
        )

    @classmethod
    def scalar(cls, value=1.0):
        return cls("", [], value=np.array(1.0))

    def get_cardinality_for_(self, variable):
        return self.cardinality[self.variables.index(variable)]

    def get_cardinalities_for_(self, variables):
        return [self.cardinality[self.variables.index(variable)] for variable in
                variables]

    def copy(self, rename=False):
        if rename:
            return Factor(
                name=default_factor_name(),
                variables=copy(self.variables),
                log_values=np.copy(self.log_values),
            )
        else:
            return Factor(
                name=copy(self.name),
                variables=copy(self.variables),
                log_values=np.copy(self.log_values),
            )

    def pow(self, w, inplace=True):
        fac = self if inplace else self.copy()
        fac.log_values *= w

        if not inplace:
            return fac

    def log(self, inplace=True):
        fac = self if inplace else self.copy()
        fac.log_values = log(fac.log_values)
        if not inplace:
            return fac

    def exp(self, inplace=True):
        fac = self if inplace else self.copy()
        fac.log_values = exp(fac.log_values)
        if not inplace:
            return fac

    def transpose_by_(self, variables, inplace=True):
        fac = self if inplace else self.copy()
        new_axes = [fac.variables.index(variable) for variable in variables]
        fac.log_values = np.transpose(fac.log_values, axes=new_axes)
        fac.cardinality = list(fac.log_values.shape)
        fac.variables = variables
        if not inplace:
            return fac

    ## Numerically unstable
    def invT(self, inplace=True):
        fac = self if inplace else self.copy()
        fac.values = np.linalg.inv(fac.values).transpose(1, 0)
        if not inplace:
            return fac

    def marginalize(self, variables=None, operator="sum", inplace=True,
                    **kwargs):
        fac = self if inplace else self.copy()
        if not variables:
            variables = self.variables
        for var in variables:
            if var not in fac.variables:
                raise ValueError(
                    "{variable} not in scope.".format(variable=var))

        variable_indices = tuple(
            [fac.variables.index(var) for var in variables])
        index_to_keep = sorted(
            set(range(len(self.variables))) - set(variable_indices))

        fac.variables = [fac.variables[index] for index in index_to_keep]
        fac.cardinality = [fac.cardinality[index] for index in index_to_keep]

        if operator == "sum":
            fac.log_values = logsumexp(fac.log_values, axis=variable_indices)
        elif operator == "weighted_sum":
            w = kwargs["weight"]
            if w != 0:
                fac.log_values /= w
                fac.log_values = logsumexp(fac.log_values,
                                           axis=variable_indices)
                fac.log_values *= w
            else:
                fac.log_values = amax(fac.log_values, axis=variable_indices)
        elif operator == "max":
            fac.log_values = amax(fac.log_values, axis=variable_indices)
        elif operator == "min":
            fac.log_values = amin(fac.log_values, axis=variable_indices)
        else:
            raise ValueError("Unsupported operator.")

        if not inplace:
            return fac

    def marginalize_except_(self, variables, operator="sum", inplace=True,
                            **kwargs):
        fac = self if inplace else self.copy()
        variables_to_marginalize = [var for var in fac.variables if
                                    var not in variables]
        if variables_to_marginalize:
            fac.marginalize(variables=variables_to_marginalize,
                            operator=operator, **kwargs)

        fac.transpose_by_(variables)

        if not inplace:
            return fac

    def normalize(self, variables=None, inplace=True):
        if not variables:
            variables = self.variables
        org_variables = copy(self.variables)

        fac = self if inplace else self.copy()

        variable_indices = tuple(
            [fac.variables.index(variable) for variable in variables])

        zero_indices = ~np.isfinite(fac.log_values)
        with np.errstate(invalid="ignore"):
            fac.log_values = fac.log_values - logsumexp(
                fac.log_values, axis=variable_indices, keepdims=True
            )
        fac.log_values[zero_indices] = -np.inf

        fac.transpose_by_(org_variables)

        if not inplace:
            return fac

    def add(self, fac1, inplace=True):
        fac = self if inplace else self.copy()
        if isinstance(fac1, (int, float)):
            max_a = amax(fac.log_values)
            fac.log_values = np.log(exp(fac.log_values - max_a) + fac1) + max_a
            fac.log_values += max_a
        else:
            a1 = fac1.transpose_by_(fac.variables, inplace=False).log_values
            max_a = max(amax(a1), amax(fac.log_values))
            with np.errstate(invalid="raise"):
                try:
                    fac.log_values = log(
                        exp(a1 - max_a) + exp(fac.log_values - max_a))
                except:
                    print(a1)
                    print(fac.log_values)
                    print(max_a)

            fac.log_values += max_a

        if not inplace:
            fac.name = default_factor_name()
            return fac

    def sub(self, fac1, name=None, inplace=True):
        fac = self if inplace else self.copy()
        if isinstance(fac1, (int, float)):
            max_a = amax(fac.log_values)
            fac.log_values = np.log(exp(fac.log_values - max_a) - fac1) + max_a
            fac.log_values += max_a
        else:
            a1 = fac1.transpose_by_(fac.variables, inplace=False).log_values
            max_a = max(amax(a1), amax(fac.log_values))
            fac.log_values = log(exp(fac.log_values - max_a) - exp(a1 - max_a))
            fac.log_values += max_a

        if not inplace:
            fac.name = default_factor_name()
            return fac

    def product(self, fac1, name=None, inplace=True):
        fac = self if inplace else self.copy()

        if isinstance(fac1, (int, float)):
            with np.errstate(divide="ignore", invalid="ignore"):
                fac.log_values += log(fac1)
        else:
            fac1 = fac1.copy()
            extra_variables = set(fac1.variables) - set(fac.variables)
            if extra_variables:
                fac.log_values = add_dims(fac.log_values, len(extra_variables))
                fac.variables.extend(extra_variables)
                new_variable_card = fac1.get_cardinalities_for_(extra_variables)
                fac.cardinality = np.append(fac.cardinality, new_variable_card)

            extra_variables = set(fac.variables) - set(fac1.variables)
            if extra_variables:
                fac1.log_values = add_dims(fac1.log_values,
                                           len(extra_variables))
                fac1.variables.extend(extra_variables)
            for axis in range(fac.log_values.ndim):
                exchange_index = fac1.variables.index(fac.variables[axis])
                fac1.variables[axis], fac1.variables[exchange_index] = (
                    fac1.variables[exchange_index],
                    fac1.variables[axis],
                )
                fac1.log_values = fac1.log_values.swapaxes(axis, exchange_index)

            fac.log_values = fac.log_values + fac1.log_values

        if not inplace:
            fac.name = default_factor_name()
            return fac

    def div(self, fac1, name=None, inplace=True):
        fac = self if inplace else self.copy()

        if isinstance(fac1, (int, float)):
            with np.errstate(divide="ignore", invalid="ignore"):
                fac.log_values += log(fac1)
        else:
            fac1 = fac1.copy()
            extra_variables = set(fac1.variables) - set(fac.variables)
            if extra_variables:
                fac.log_values = add_dims(fac.log_values, len(extra_variables))
                fac.variables.extend(extra_variables)
                new_variable_card = fac1.get_cardinalities_for_(extra_variables)
                fac.cardinality = np.append(fac.cardinality, new_variable_card)

            extra_variables = set(fac.variables) - set(fac1.variables)
            if extra_variables:
                fac1.log_values = add_dims(fac1.log_values,
                                           len(extra_variables))
                fac1.variables.extend(extra_variables)
            for axis in range(fac.log_values.ndim):
                exchange_index = fac1.variables.index(fac.variables[axis])
                fac1.variables[axis], fac1.variables[exchange_index] = (
                    fac1.variables[exchange_index],
                    fac1.variables[axis],
                )
                fac1.log_values = fac1.log_values.swapaxes(axis, exchange_index)

            zero_indices = np.logical_and(
                ~np.isfinite(fac.log_values), ~np.isfinite(fac1.log_values)
            )

            with np.errstate(invalid="ignore"):
                fac.log_values = fac.log_values - fac1.log_values

            fac.log_values[zero_indices] = 1.0

    def __pow__(self, w):
        return self.pow(w, inplace=True)

    def __add__(self, other):
        return self.add(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.sub(other, inplace=False)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __div__(self, other):
        if type(other) == type(self):
            return self.product(other.pow(-1.0, inplace=False), inplace=False)
        else:
            return self.product(1 / other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv(self, other):
        return self.__div__(other)

    def __eq__(self, other):
        if not issubclass(type(self), type(other)):
            return False
        if self.name == other.name and tuple(sorted(self.variables)) == tuple(
                sorted(other.variables)
        ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name.join(sorted(self.variables)))

    def __repr__(self):
        return "Factor: " + self.name + " Variables: " + ", ".join(
            self.variables)


def add_dims(a: np.array, extra_dims):
    """Reshapes array by adding dimensions on the right."""
    return a.reshape(*a.shape, *([1] * extra_dims))


def logsumexp(a, axis=None, keepdims=False):
    if axis is None:
        a = a.ravel()

    a_max = amax(a, axis=axis, keepdims=True)
    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0

    tmp = exp(a - a_max)
    with np.errstate(divide="ignore"):
        out = log(np.sum(tmp, axis=axis, keepdims=keepdims))

    if not keepdims:
        a_max = squeeze(a_max, axis=axis)
    out += a_max
    return out


def default_factor_name(prefix="_F"):
    default_factor_name.cnt += 1
    return prefix + str(default_factor_name.cnt)


default_factor_name.cnt = 0


def default_variable_name(prefix="_V"):
    default_variable_name.cnt += 1
    return prefix + str(default_variable_name.cnt)


default_variable_name.cnt = 0


def product_over_(*args):
    factor_list = [tensor for tensor in args if tensor]
    if len(factor_list) > 1:
        return reduce(lambda phi1, phi2: phi1 * phi2, factor_list)
    elif isinstance(factor_list[0], (int, float)):
        return factor_list[0]
    else:
        return factor_list[0].copy()


def entropy(p, q=None):
    p_values = np.copy(p.values)
    p_values.ravel()
    index_to_keep = p_values != 0
    p_values = p_values[index_to_keep]
    if q:
        q.transpose_by_(p.variables)
        q_values = np.copy(q.values)
        q_values.ravel()
        q_values = q_values[index_to_keep]
        return np.sum(p_values * (log(q_values) - log(p_values)))
    else:
        return np.sum(-p_values * log(p_values))
