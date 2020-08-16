# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Callable, Dict

import numpy as np

from inferlo.base.factors.discrete_factor import DiscreteFactor
from inferlo.base import InferenceResult

if TYPE_CHECKING:
    from inferlo import GraphModel

recordSentMessages = True


class Prob:
    """Equivalent of dai::Prob.

    Wrapper around a vector - represents probability distribution.
    """

    @staticmethod
    def uniform(n):
        """Creates unifom probability distribution."""
        return Prob.same_value(n, 1.0 / n)

    @staticmethod
    def same_value(n: int, val: float):
        """Creates vector filled with the same value."""
        return Prob(np.ones(n, dtype=np.float64) * val)

    def __init__(self, p: np.ndarray):
        self.p = p

    def fill(self, x):
        """Sets all entries to x."""
        self.p = np.ones_like(self.p) * x

    def clone(self):
        """Makes a copy."""
        return Prob(np.array(self.p))

    def __imul__(self, other):
        self.p *= other.p
        return self

    def __iadd__(self, other):
        self.p += other.p
        return self

    def normalize(self):
        """Normalize distribution."""
        self.p /= np.sum(self.p)

    def entropy(self) -> float:
        """Calculate entropy of the distribution."""
        return - np.sum(self.p * np.log(self.p))

    def __str__(self):
        return str(self.p)


def dist_kl(p: Prob, q: Prob):
    """Kullback-Leibler divergence between two probability distributions."""
    kl_div = p.p * (np.log(p.p + (p == 0)) - np.log(q.p + (p.p == 0)))
    return np.sum(kl_div)


def dist_linf(p: Prob, q: Prob):
    """Distance between two probability distributions in L_infinity norm."""
    return np.max(np.abs(p.p - q.p))


@dataclass
class Neighbor:
    """Describes the neighbor relationship of two nodes in a graph.

    Corresponds to dai::Neighbor.
    """
    # Corresponds to the index of this Neighbor entry in the vector of
    # neighbors.
    iter: int
    # Contains the absolute index of the neighboring node.
    node: int
    # Contains the "dual" index (i.e., the index of this node in the Neighbors
    # vector of the neighboring node)
    dual: int


@dataclass
class EdgeProp:
    """Type used for storing edge properties."""
    index: np.ndarray  # Index cached for this edge.
    message: Prob  # Old message living on this edge.
    new_message: Prob  # New message living on this edge
    residual: float  # Residual for this edge


class LDFactor:
    """Equivalent of dai::Factor.

    Consists of set of variables and flattened values assigned to all var
    combinations. Variables are assigned like in Inferlo, but tensor is
    transposed before flattening.
    """

    def __init__(self, model: GraphModel, var_idx: List[int], p: Prob):
        self.model = model
        self.var_idx = var_idx
        self.p = p

    @staticmethod
    def uniform(model: GraphModel, var_idx: List[int]):
        """Creates factor defining uniform distribution."""
        total_domain_size = 1
        for i in var_idx:
            total_domain_size *= model.get_variable(i).domain.size()
        return LDFactor(model, var_idx, Prob.uniform(total_domain_size))

    @staticmethod
    def from_inferlo_factor(f: DiscreteFactor):
        """Converts inferlo.DiscreteFactor to LDFactor."""
        rev_perm = list(range(len(f.var_idx)))[::-1]
        prob = f.values.transpose(rev_perm).reshape(-1)
        return LDFactor(f.model, f.var_idx, Prob(prob))

    def to_inferlo_factor(self) -> DiscreteFactor:
        """Converts LDFactor to inferlo.DiscreteFactor."""
        sizes = [self.model.get_variable(i).domain.size()
                 for i in self.var_idx[::-1]]
        libdai_tensor = self.p.p.reshape(sizes)
        rev_perm = list(range(len(self.var_idx)))[::-1]
        inferlo_tensor = libdai_tensor.transpose(rev_perm)
        return DiscreteFactor(self.model, self.var_idx, inferlo_tensor)

    def combine_with_factor(self, other: LDFactor,
                            func: Callable[[float, float], float]):
        """Applies binary function to two factors."""
        # Check that variables of the other factor are subset of variables of
        # the given factor.
        for i in other.var_idx:
            assert i in self.var_idx

        # Now, update every value of given factor with corresponding value of
        # the other factor.
        for idx in range(len(self.p.p)):
            j = other._encode_value_index(self._decode_value_index(idx))
            self.p.p[idx] = func(self.p.p[idx], other.p.p[j])
        return self

    def __iadd__(self, other: LDFactor):
        return self.combine_with_factor(other, lambda x, y: x + y)

    def __imul__(self, other: LDFactor):
        return self.combine_with_factor(other, lambda x, y: x * y)

    def marginal(self, new_var_idx, normed=True) -> LDFactor:
        """Sums factor over some variables."""
        result = self.to_inferlo_factor().marginal(new_var_idx)
        result = LDFactor.from_inferlo_factor(result)
        if normed:
            result.p.normalize()
        return result

    def max_marginal(self, new_var_idx, normed=True) -> LDFactor:
        """Eleiminates certain variables by finding maximum."""
        result = self.to_inferlo_factor().max_marginal(new_var_idx)
        result = LDFactor.from_inferlo_factor(result)
        if normed:
            result.p.normalize()
        return result

    def clone(self):
        """Makes a copy of this factor."""
        return LDFactor(self.model, self.var_idx, self.p.clone())

    def _decode_value_index(self, idx):
        """Returns dict from variable id to variable value."""
        ans = dict()
        for var_id in self.var_idx:
            size = self.model.get_variable(var_id).domain.size()
            ans[var_id] = idx % size
            idx //= size
        return ans

    def _encode_value_index(self, var_values: Dict[int, int]):
        ans = 0
        base = 1
        for var_id in self.var_idx:
            size = self.model.get_variable(var_id).domain.size()
            ans += base * var_values[var_id]
            base *= size
        return ans

    def __str__(self):
        return "%s %s" % (self.var_idx, self.p.p)


class BP:
    """Belief propagation algorithm, equivalent to dai::BP.

    This class is ported from libDAI's dai::BP class. It runs belief
    propagation algorithm for graphical model with discrete variables with
    arbitrary factor graph.

    At the moment MAXPROD algorithm (for finding MAP state) is not supported.

    Use BP.infer() to perform inference.
    """

    @staticmethod
    def infer(model, options=None):
        """Runs inference BP algorithm for given model.

        Supports all options which libdai::BP supports. Refer to libDAI
        documentation for options descritpion.
        """
        if options is None:
            options = {'tol': 1e-9, 'logdomain': 0, 'updates': 'SEQRND'}
        inf_alg = BP(model, options)
        inf_alg.init()
        inf_alg.run()
        return InferenceResult(inf_alg.log_z(), inf_alg.marg_prob())

    def __init__(self, model: GraphModel, props: Dict[str, str]):
        # Stores all edge properties
        self._edges: List[List[EdgeProp]] = []
        # Maximum difference between variable beliefs encountered so far
        self._maxdiff = 0.0
        # Number of iterations needed
        self._iters = 0
        # The history of message updates (only recorded if \a
        # recordSentMessages is \c true)
        self._sentMessages = []
        # Stores variable beliefs of previous iteration
        self._oldBeliefsV: List[LDFactor] = []
        # Stores factor beliefs of previous iteration
        self._old_beliefs_f: List[LDFactor] = []
        # Stores the update schedule
        self._update_seq = []

        self.model = model
        self.factors = [
            LDFactor.from_inferlo_factor(
                DiscreteFactor.from_factor(f)) for f in model.get_factors()]
        self.nrVars = model.num_variables
        self.nrFactors = len(self.factors)

        # Prepare Neighbors.
        # For every variable - factors, referencing it.
        self.nbV: List[List[Neighbor]] = [[] for _ in range(self.nrVars)]
        # For every factor - variables it references.
        self.nbF: List[List[Neighbor]] = [[] for _ in range(self.nrFactors)]
        for factor_id in range(len(self.factors)):
            factor = self.factors[factor_id]
            for var_iter_index in range(len(factor.var_idx)):
                var_id = factor.var_idx[var_iter_index]
                nbv_len = len(self.nbV[var_id])
                nbf_len = len(self.nbF[factor_id])
                assert var_iter_index == nbf_len
                self.nbV[var_id].append(
                    Neighbor(
                        iter=nbv_len,
                        node=factor_id,
                        dual=nbf_len))
                self.nbF[factor_id].append(
                    Neighbor(
                        iter=nbf_len,
                        node=var_id,
                        dual=nbv_len))

        # Parse properties.
        self.logdomain = bool(int(props.get('logdomain', 0)))
        self.updates = props['updates']
        self.inference = props.get('inference', 'SUMPROD')
        self.verbose = int(props.get('verbose', 0))
        self.damping = float(props.get('damping', 0.0))
        self.maxiter = int(props.get('maxiter', 10000))
        self.maxtime = float(props.get('maxtime', np.inf))
        self.tol = float(props['tol'])

        self._construct()

    def _construct(self):
        """Helper function for constructors."""
        # Create edge properties
        self._edges = []
        for i in range(self.nrVars):
            self._edges.append([])
            for _ in self.nbV[i]:
                size = self._var_size(i)
                new_ep = EdgeProp(
                    index=None,
                    message=Prob.uniform(size),
                    new_message=Prob.uniform(size),
                    residual=0.0)
                self._edges[i].append(new_ep)

        # Create old beliefs
        self._oldBeliefsV = []
        for i in range(self.nrVars):
            self._oldBeliefsV.append(LDFactor.uniform(self.model, [i]))
        self._old_beliefs_f = []
        for ii in range(self.nrFactors):
            self._old_beliefs_f.append(
                LDFactor.uniform(
                    self.model,
                    self.factors[ii].var_idx))

        # Create update sequence
        self._update_seq = []
        for ii in range(self.nrFactors):
            for i in self.nbF[ii]:
                self._update_seq.append((i.node, i.dual))

    def init(self):
        """Initializes messages awith default values."""
        c = 0.0 if self.logdomain else 1.0
        for i in range(self.nrVars):
            for ii in self.nbV[i]:
                self._edges[i][ii.iter].message.fill(c)
                self._edges[i][ii.iter].new_message.fill(c)
                if self.updates == 'SEQMAX':
                    self._update_residual(i, ii.iter, 0.0)
        self._iters = 0

    def find_max_residual(self):
        """Find max residual."""
        # TODO: optimize with a lookup table.
        max_r = -np.inf
        best_edge = None
        for i in range(self.nrVars):
            for _I in range(len(self.nbV[i])):
                if self._edges[i][_I].residual > max_r:
                    max_r = self._edges[i][_I].residual
                    best_edge = i, _I
        return best_edge

    def _calc_incoming_message_product(
            self,
            ii: int,
            without_i: bool,
            i: int) -> Prob:
        """Calculate the product of factor \a I and the incoming messages.

        If without_i == True, the message coming from variable i is omitted
        from the product.

        This function is used by calc_new_message and calc_belief_f.
        """
        f_prod = self.factors[ii].clone()
        if self.logdomain:
            f_prod.p.p = np.log(f_prod.p.p)

        # Calculate product of incoming messages and factor I
        for j in self.nbF[ii]:
            if without_i and (j.node == i):
                continue

            # prod_j will be the product of messages coming into j
            size = self._var_size(j.node)
            default_val = 0.0 if self.logdomain else 1.0
            prod_j = Prob.same_value(size, default_val)
            for J in self.nbV[j.node]:
                if J.node != ii:  # for all J in nb(j) \ I
                    if self.logdomain:
                        prod_j += self._edges[j.node][J.iter].message
                    else:
                        prod_j *= self._edges[j.node][J.iter].message

            # multiply prod with prod_j
            if self.logdomain:
                f_prod += LDFactor(self.model, [j.node], prod_j)
            else:
                f_prod *= LDFactor(self.model, [j.node], prod_j)
        return f_prod.p

    def _calc_new_message(self, i: int, _I: int):
        # calculate updated message I->i
        ii = self.nbV[i][_I].node

        if len(self.factors[ii].var_idx) == 1:  # optimization
            marg = self.factors[ii].p.clone()
        else:
            Fprod = self.factors[ii].clone()
            Fprod.p = self._calc_incoming_message_product(ii, True, i)

            if self.logdomain:
                Fprod.p.p = np.exp(Fprod.p.p - np.max(Fprod.p.p))

            # Marginalize onto i
            if self.inference == 'SUMPROD':
                marg = Fprod.marginal([i]).p
            else:
                marg = Fprod.max_marginal([i]).p

        # Store result
        if self.logdomain:
            self._edges[i][_I].new_message = Prob(np.log(marg.p))
        else:
            self._edges[i][_I].new_message = marg

        # Update the residual if necessary
        if self.updates == 'SEQMAX':
            self._update_residual(
                i,
                _I,
                dist_linf(
                    self._edges[i][_I].new_message,
                    self._edges[i][_I].message))

    # BP::run does not check for NANs for performance reasons
    # Somehow NaNs do not often occur in BP...
    def run(self):
        """Runs BP algorithm."""
        tic = time.time()

        # Do several passes over the network until maximum number of iterations
        # has been reached or until the maximum belief difference is smaller
        # than tolerance.
        max_diff = np.inf
        while (self._iters < self.maxiter) and (
                max_diff > self.tol) and (time.time() - tic) < self.maxtime:
            if self.updates == 'SEQMAX':
                if self._iters == 0:
                    # do the first pass
                    for i in range(self.nrVars):
                        for ii in self.nbV[i]:
                            self._calc_new_message(i, ii.iter)
                # Maximum-Residual BP [\ref EMK06]
                for _ in range(len(self._update_seq)):
                    # Update the message with the largest residual.
                    i, _I = self.find_max_residual()
                    self._update_message(i, _I)

                    # I->i has been updated, which means that residuals for all
                    # J->j with J in nb[i]\I and j in nb[J]\i have to be
                    # updated
                    for J in self.nbV[i]:
                        if J.iter != _I:
                            for j in self.nbF[J.node]:
                                _J = j.dual
                                if j != i:
                                    self._calc_new_message(j.node, _J)
            elif self.updates == 'PARALL':
                # Parallel updates
                for i in range(self.nrVars):
                    for ii in self.nbV[i]:
                        self._calc_new_message(i, ii.iter)

                for i in range(self.nrVars):
                    for ii in self.nbV[i]:
                        self._update_message(i, ii.iter)
            else:
                # Sequential updates
                if self.updates == 'SEQRND':
                    random.shuffle(self._update_seq)

                for e in self._update_seq:
                    self._calc_new_message(e[0], e[1])
                    self._update_message(e[0], e[1])

            # Calculate new beliefs and compare with old ones
            max_diff = -np.inf
            for i in range(self.nrVars):
                b = self._belief_v(i).clone()
                max_diff = max(max_diff,
                               dist_linf(b.p, self._oldBeliefsV[i].p))
                self._oldBeliefsV[i] = b
            for ii in range(self.nrFactors):
                b = self._belief_f(ii).clone()
                max_diff = max(max_diff,
                               dist_linf(b.p, self._old_beliefs_f[ii].p))
                self._old_beliefs_f[ii] = b
            self._iters += 1

        if max_diff > self._maxdiff:
            self._maxdiff = max_diff
        return max_diff

    def _calc_belief_v(self, i: int) -> Prob:
        p = Prob.same_value(self.model.get_variable(i).domain.size(),
                            0.0 if self.logdomain else 1.0)
        for ii in self.nbV[i]:
            if self.logdomain:
                p += self._edges[i][ii.iter].new_message
            else:
                p *= self._edges[i][ii.iter].new_message
        return p

    def _belief_v(self, i: int) -> LDFactor:
        p = self._calc_belief_v(i)

        if self.logdomain:
            p.p = np.exp(p.p - np.max(p.p))
        p.normalize()
        return LDFactor(self.model, [i], p)

    def _belief_f(self, ii) -> LDFactor:
        p = self._calc_belief_f(ii)

        if self.logdomain:
            p.p = np.exp(p.p - np.max(p.p))
        p.normalize()

        return LDFactor(self.model, self.factors[ii].var_idx, p)

    def _calc_belief_f(self, ii: int) -> Prob:
        return self._calc_incoming_message_product(ii, False, 0)

    def log_z(self) -> float:
        """Calculates logarithm of the partition function."""
        ans = 0.0
        for i in range(self.nrVars):
            ans += (1.0 - len(self.nbV[i])) * self._belief_v(i).p.entropy()
        for ii in range(self.nrFactors):
            ans -= dist_kl(self._belief_f(ii).p, self.factors[ii].p)
        return ans

    def marg_prob(self) -> np.ndarray:
        """Calculates marginal probabilities."""
        max_domain_size = np.max([self._var_size(i)
                                  for i in range(self.nrVars)])
        ans = np.zeros((self.nrVars, max_domain_size), dtype=np.float64)
        for var_id in range(self.nrVars):
            ans[var_id, 0:self._var_size(var_id)] = self._belief_v(var_id).p.p
        return ans

    def _var_size(self, var_idx):
        return self.model.get_variable(var_idx).domain.size()

    def _update_message(self, i: int, _I: int):
        if recordSentMessages:
            self._sentMessages.append((i, _I))
        if self.damping == 0.0:
            self._edges[i][_I].message = self._edges[i][_I].new_message.clone()
            if self.updates == 'SEQMAX':
                self._update_residual(i, _I, 0.0)
        else:
            d = self.damping
            old_msg = self._edges[i][_I].message.p
            new_msg = self._edges[i][_I].new_message.p
            if self.logdomain:
                self._edges[i][_I].message.p = (
                    (old_msg * d) + (new_msg * (1.0 - d)))
            else:
                self._edges[i][_I].message.p = (
                    (old_msg ** d) * (new_msg ** (1.0 - d)))
            if self.updates == 'SEQMAX':
                new_res = dist_linf(
                    self._edges[i][_I].new_message,
                    self._edges[i][_I].message)
                self._update_residual(i, _I, new_res)

    def _update_residual(self, i, _I, r):
        self._edges[i][_I].residual = r
