from copy import copy
import numpy as np

from .factor import Factor, product_over_, entropy


def default_message_name(prefix="_M"):
    default_message_name.cnt += 1
    return prefix + str(default_message_name.cnt)


default_message_name.cnt = 0


class MeanField:
    def __init__(self, model, **kwargs):
        self.model = model.copy()

        mean_field_init_method = kwargs.get("mean_field_init_method")
        if mean_field_init_method == "random":
            init_np_func = np.ones
        elif mean_field_init_method == "uniform":
            init_np_func = np.random.random
        else:
            init_np_func = np.ones

        self.mean_fields = {}
        for var in self.model.variables:
            self.mean_fields[var] = Factor.initialize_with_(
                default_message_name(), [var], init_np_func, model.get_cardinality_for_(var)
            )
            self.mean_fields[var].normalize()

    def run(self, max_iter=1000, converge_thr=1e-2, verbose=False):
        for t in range(max_iter):
            old_mean_field = {var: copy(self.mean_fields[var]) for var in self.model.variables}
            self._update_mean_fields()
            if self._is_converged(self.mean_fields, old_mean_field, converge_thr):
                # if verbose:
                #    print("Mean field converged in {} iterations.".format(t+1))
                break

        # if verbose and not self._is_converged(self.mean_fields, old_mean_field, converge_thr):
        #    print("Mean field did not converge after {} iterations.".format(max_iter))

        return self.get_logZ()

    def get_logZ(self):
        logZ = 0
        for var in self.model.variables:
            logZ += entropy(self.mean_fields[var])

        for fac in self.model.factors:
            m = product_over_(*[self.mean_fields[var] for var in fac.variables])
            index_to_keep = m.values != 0
            logZ += np.sum(m.values[index_to_keep] * fac.log_values[index_to_keep])

        return logZ

    def _update_mean_fields(self):
        variable_order = np.random.permutation(self.model.variables)
        for var in variable_order:
            next_mean_field = Factor.full_like_(self.mean_fields[var], 0.0)
            for fac in self.model.get_adj_factors(var):
                tmp = Factor(
                    name="tmp",
                    variables=[var],
                    values=np.ones(self.model.get_cardinality_for_(var)),
                )
                tmp = product_over_(
                    tmp, *[self.mean_fields[var1] for var1 in fac.variables if var1 != var]
                )
                tmp.transpose_by_(fac.variables)
                tmp.log_values = fac.log_values * tmp.values
                next_mean_field = next_mean_field + tmp.marginalize_except_([var], inplace=False)

            self.mean_fields[var] = next_mean_field.exp(inplace=False).normalize(inplace=False)
            self.mean_fields[var].log_values = np.nan_to_num(self.mean_fields[var].log_values)

    def _is_converged(self, mean_field, old_mean_field, converge_thr):
        for var in self.model.variables:
            if np.sum(np.abs(mean_field[var].values - old_mean_field[var].values)) > converge_thr:
                return False
        return True
