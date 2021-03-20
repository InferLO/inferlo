from scipy.special import softmax

from inferlo import GenericGraphModel, InferenceResult
from inferlo.generic.inference.factor import product_over_
from inferlo.generic.inference.inference import _convert
from inferlo.base.inference_result import marg_probs_to_array


class BackwardBucketElimination:
    """Backward Bucket Elimination algorithm.

    This is the regular Bucket Elimination algorithm followed by a "backward pass" for computing
    marginal probabilities.

    Implemented as described in [1] (pages 31-32), code uses notation from [1].

    References
        [1] Qiang Liu.
        Reasoning and Decisions in Probabilistic Graphical Models – A Unified Framework. 2014.
        http://sli.ics.uci.edu/pmwiki/uploads/Group/liu_phd.pdf
    """

    @staticmethod
    def infer(model: GenericGraphModel) -> InferenceResult:
        """Performs inference for graphical model."""
        model = _convert(model)
        elimination_order = model.variables
        buckets = dict()
        new_factors = dict()
        marg_prob = dict()  # Factors - marg prob for bucket corresponding to variable.
        i_to_j = dict()  # var_id -> j (id to whose bucket received new factor for var_id).
        c = dict()  # For every bucket - set of variables on which it depends.

        for var_id in elimination_order:
            # Find the bucket, and store it.
            bucket = model.get_adj_factors(var_id)
            # Eliminate the variable.
            new_factor = product_over_(*bucket)
            new_factor.marginalize(variables=[var_id])
            new_factor.name = 'new_' + var_id
            # Update the factor list.
            model.remove_factors_from(bucket)
            model.add_factor(new_factor)
            # Store.
            buckets[var_id] = bucket
            new_factors[var_id] = new_factor
            for factor in bucket:
                if factor.name.startswith('new_'):
                    i_to_j[factor.name[4:]] = var_id
            # c_i - set of variables on which bucket depends.
            pi_i = set(new_factors[var_id].variables)
            c[var_id] = pi_i.union({var_id})

        # Backward pass.
        # Find marginals for the last variable.
        last_var = elimination_order[-1]
        marg_prob[last_var] = product_over_(*buckets[last_var])

        for var_id in elimination_order[::-1][1:]:
            # Find conditional probability.
            p = product_over_(*buckets[var_id]) / new_factors[var_id]
            # Multiply by marg.prob for variable whose bucket received new factor for var_id.
            j = i_to_j[var_id]
            p = p * marg_prob[j]
            # Marginalize by extra variables.
            to_marg = list((c[j] - c[var_id]).intersection(set(p.variables)))
            if len(to_marg) > 0:
                p.marginalize(variables=to_marg)
            assert (set(p.variables) == c[var_id])
            # Store the result
            marg_prob[var_id] = p

        # Process results - reduce to 1-variable marginals and convert to probabilties.
        ans_mp = []
        for var_id in model.variables:
            p = marg_prob[var_id]
            p.marginalize_except_([var_id])
            p = softmax(p.log_values)
            assert len(p.shape) == 1
            ans_mp.append(p)

        log_pf = product_over_(*model.factors).log_values.item()
        return InferenceResult(
            log_pf=log_pf,
            marg_prob=marg_probs_to_array(ans_mp)
        )
