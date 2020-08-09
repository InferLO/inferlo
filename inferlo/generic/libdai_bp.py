from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Callable, Dict

import numpy as np
import time
import random
from inferlo.base.factors.discrete_factor import DiscreteFactor, FunctionFactor

if TYPE_CHECKING:
    from inferlo import GraphModel

DAI_BP_FAST = False
recordSentMessages = True


class Prob:
    @staticmethod
    def uniform(n):
        return Prob(np.ones(n, dtype=np.float64) / n)

    @staticmethod
    def same_value(n: int, val: float):
        return Prob(np.ones(n, dtype=np.float64) * val)

    def __init__(self, p: np.ndarray):
        self._p = p

    def fill(self, x):
        """Sets all entries to x."""
        self._p = np.ones_like(self._p) * x

    def clone(self):
        return Prob(np.array(self._p))

    def __imul__(self, other):
        self._p *= other._p
        return self

    def __iadd__(self, other):
        self._p += other._p
        return self

    def normalize(self):
        self._p /= np.sum(self._p)

    def takeLog(self):
        self._p = np.log(self._p)

    def takeExp(self):
        self._p = np.log(self._p)

    def max(self):
        return np.max(self._p)

    def log(self):
        return Prob(np.log(self._p))

    def entropy(self) -> float:
        return - np.sum(self._p * np.log(self._p))

    def __str__(self):
        return str(self._p)


def kl_div(p, q):
    return p * (np.log(p + (p == 0)) - np.log(q + (p == 0)))


# Distance between probability distributions.
def dist(p: Prob, q: Prob, dt):
    if dt == 'DISTL1':
        return np.sum(np.abs(p._p - q._p))
    elif dt == 'DISTLINF':
        return np.max(np.abs(p._p - q._p))
    elif dt == 'DISTTV':
        return np.sum(np.abs(p._p - q._p)) / 2
    elif dt == 'DISTKL':
        return np.sum(kl_div(p._p, q._p))
    elif dt == 'DISTHEL':
        return np.sum(np.square(np.sqrt(p._p) - np.sqrt(q._p))) / 2
    else:
        raise ValueError('UNKNOWN_ENUM_VALUE')


@dataclass
class Neighbor:
    iter: int  # Corresponds to the index of this Neighbor entry in the vector of neighbors.
    node: int  # Contains the absolute index of the neighboring node.
    dual: int  # Contains the "dual" index (i.e., the index of this node in the Neighbors vector of the neighboring node)


@dataclass
class EdgeProp:
    """Type used for storing edge properties."""
    index: np.ndarray  # Index cached for this edge.
    message: Prob  # Old message living on this edge.
    newMessage: Prob  # New message living on this edge
    residual: float  # Residual for this edge


# LibDai's factor. Consists of set of variables and flattened values assigned to all var combinations.
# Variables are assigned linke in Inferlo, but tensor is transposed before flattening.
class LDFactor:

    def __init__(self, model: GraphModel, var_idx: List[int], p: Prob):
        self.model = model
        self.var_idx = var_idx
        self.p = p

        if len(var_idx) > 0:
            assert isinstance(var_idx[0], int)

    @staticmethod
    def uniform(model: GraphModel, var_idx: List[int]):
        total_domain_size = 1
        for i in var_idx:
            total_domain_size *= model.get_variable(i).domain.size()
        return LDFactor(model, var_idx, Prob.uniform(total_domain_size))

    @staticmethod
    def from_inferlo_factor(f: DiscreteFactor):
        return LDFactor(f.model, f.var_idx, Prob(f.libdai_vs()))

    def to_inferlo_factor(self) -> DiscreteFactor:
        sizes = [self.model.get_variable(i).domain.size() for i in self.var_idx[::-1]]
        libdai_tensor = self.p._p.reshape(sizes)
        rev_perm = list(range(len(self.var_idx)))[::-1]
        inferlo_tensor = libdai_tensor.transpose(rev_perm)
        return DiscreteFactor(self.model, self.var_idx, inferlo_tensor)

    @staticmethod
    def combine_factors(factor1: LDFactor, factor2: LDFactor, func: Callable[[float, float], float]) -> LDFactor:
        result = FunctionFactor.combine_factors(
            factor1.to_inferlo_factor().as_function_factor(),
            factor2.to_inferlo_factor().as_function_factor(),
            func
        )
        return LDFactor.from_inferlo_factor(DiscreteFactor.from_factor(result))

    def __iadd__(self, other: LDFactor):
        return LDFactor.combine_factors(self, other, lambda x, y: x + y)

    def __imul__(self, other: LDFactor):
        return LDFactor.combine_factors(self, other, lambda x, y: x * y)

    def marginal(self, new_var_idx, normed=True) -> LDFactor:
        result = self.to_inferlo_factor().marginal(new_var_idx)
        result = LDFactor.from_inferlo_factor(result)
        if normed:
            result.p.normalize()
        return result

    def maxMarginal(self, new_var_idx, normed=True) -> LDFactor:
        result = self.to_inferlo_factor().max_marginal(new_var_idx)
        result = LDFactor.from_inferlo_factor(result)
        if normed:
            result.p.normalize()
        return result

    def clone(self):
        return LDFactor(self.model, self.var_idx, self.p.clone())

    def __str__(self):
        return "%s %s" % (self.var_idx, self.p._p)


class BP:
    def __init__(self, model: GraphModel, props: Dict[str, str]):
        # Stores all edge properties
        self._edges = []
        # Lookup table (only used for maximum-residual BP)
        self._edge2lut = []
        # Lookup table (only used for maximum-residual BP)
        self._lut = []
        # Maximum difference between variable beliefs encountered so far
        self._maxdiff = 0.0
        # Number of iterations needed
        self._iters = 0
        # The history of message updates (only recorded if \a recordSentMessages is \c true)
        self._sentMessages = []
        # Stores variable beliefs of previous iteration
        self._oldBeliefsV: List[LDFactor] = []
        # Stores factor beliefs of previous iteration
        self._oldBeliefsF: List[LDFactor] = []
        # Stores the update schedule
        self._updateSeq = []

        self.model = model
        self.factors = [LDFactor.from_inferlo_factor(DiscreteFactor.from_factor(f)) for f in model.get_factors()]
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
                self.nbV[var_id].append(Neighbor(iter=nbv_len, node=factor_id, dual=nbf_len))
                self.nbF[factor_id].append(Neighbor(iter=nbf_len, node=var_id, dual=nbv_len))

        # Parse properties.
        self.logdomain = bool(int(props.get('logdomain', 0)))
        self.updates = props['updates']
        self.inference = props.get('inference', 'SUMPROD')
        self.verbose = int(props.get('verbose', 0))
        self.damping = float(props.get('damping', 0.0))
        self.maxiter = int(props.get('maxiter', 10000))
        self.maxtime = float(props.get('maxtime', np.inf))
        self.tol = float(props['tol'])

        self.construct()

    def construct(self):
        # Create edge properties
        self._edges = []
        self._edge2lut = []

        for i in range(self.nrVars):
            self._edges.append([])
            if self.updates == 'SEQMAX':
                self._edge2lut.append([])
            for I in self.nbV[i]:
                newEP = EdgeProp(index=None,
                                 message=Prob.uniform(self.model.get_variable(i).domain.size()),
                                 newMessage=Prob.uniform(self.model.get_variable(i).domain.size()),
                                 residual=0.0)

                if DAI_BP_FAST:
                    raise ValueError('Fast not supported yet.')

                self._edges[i].append(newEP)
                if self.updates == 'SEQMAX':
                    self._edge2lut[i].append(self._lut.insert((newEP.residual, (i, len(self._edges[i]) - 1))))

        # Create old beliefs
        self._oldBeliefsV = []
        for i in range(self.nrVars):
            self._oldBeliefsV.append(LDFactor.uniform(self.model, [i]))
        self._oldBeliefsF = []
        for I in range(self.nrFactors):
            self._oldBeliefsF.append(LDFactor.uniform(self.model, self.factors[I].var_idx))

        # Create update sequence
        self._updateSeq = []
        for I in range(self.nrFactors):
            for i in self.nbF[I]:
                self._updateSeq.append((i.node, i.dual))

    def init(self):
        c = 0.0 if self.logdomain else 1.0
        for i in range(self.nrVars):
            for I in self.nbV[i]:
                self._edges[i][I.iter].message.fill(c)
                self._edges[i][I.iter].newMessage.fill(c)
                if self.updates == 'SEQMAX':
                    self.updateResidual(i, I.iter, 0.0)
        self._iters = 0

    def findMaxResidual(self):
        assert len(self._lut) > 0
        largestEl = self._lut[-1]
        return largestEl.second

    def calcIncomingMessageProduct(self, I: int, without_i: bool, i: int) -> Prob:
        Fprod = self.factors[I].clone()
        if self.logdomain:
            Fprod.p.takeLog()

        print("CIMP %d %s" % (I, Fprod.p))

        # Calculate product of incoming messages and factor I
        for j in self.nbF[I]:
            if not (without_i and (j == i)):
                # prod_j will be the product of messages coming into j
                prod_j = Prob.same_value(self.model.get_variable(j.node).domain.size(),
                                         0.0 if self.logdomain else 1.0)
                for J in self.nbV[j.node]:
                    if J != I:  # for all J in nb(j) \ I
                        if self.logdomain:
                            prod_j += self._edges[j.node][J.iter].message
                        else:
                            prod_j *= self._edges[j.node][J.iter].message

                # multiply prod with prod_j
                if not DAI_BP_FAST:
                    # UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
                    if self.logdomain:
                        Fprod += LDFactor(self.model, [j.node], prod_j)
                    else:
                        Fprod *= LDFactor(self.model, [j.node], prod_j)
                else:
                    raise ValueError('Fast not supported yet.')
            print("CIMP after nb %d: %s" % (j.node, Fprod.p))
        return Fprod.p

    def calcNewMessage(self, i: int, _I: int):
        # calculate updated message I->i
        I = self.nbV[i][_I].node

        marg = None
        if len(self.factors[I].var_idx) == 1:  # optimization
            marg = self.factors[I].p.clone()
        else:
            Fprod = self.factors[I].clone()
            prod = self.calcIncomingMessageProduct(I, True, i)

            if self.logdomain:
                prod -= prod.max()
                prod.takeExp()

            # Marginalize onto i
            if not DAI_BP_FAST:
                # UNOPTIMIZED (SIMPLE TO READ, BUT SLOW) VERSION
                if self.inference == 'SUMPROD':
                    marg = Fprod.marginal([i]).p
                else:
                    marg = Fprod.maxMarginal([i]).p
            else:
                raise ValueError('Fast not supported yet.')

        # Store result
        if self.logdomain:
            self._edges[i][_I].newMessage = marg.log()
        else:
            self._edges[i][_I].newMessage = marg

        # Update the residual if necessary
        if self.updates == 'SEQMAX':
            self.updateResidual(i, _I, dist(self._edges[i][_I].newMessage, self._edges[i][_I].message, 'DISTLINF'))

    # BP::run does not check for NANs for performance reasons
    # Somehow NaNs do not often occur in BP...
    def run(self):
        if self.verbose >= 1:
            print("Starting " + self.identify() + "...")

        tic = time.time()

        # do several passes over the network until maximum number of iterations has
        # been reached or until the maximum belief difference is smaller than tolerance
        maxDiff = np.inf
        while (self._iters < self.maxiter) and maxDiff > self.tol and (time.time() - tic) < self.maxtime:
            if self.updates == 'SEQMAX':
                if self._iters == 0:
                    # do the first pass
                    for i in range(self.nrVars):
                        for I in self.nbV[i]:
                            self.calcNewMessage(i, I.iter)
                # Maximum-Residual BP [\ref EMK06]
                for t in range(len(self._updateSeq)):
                    # Update the message with the largest residual.
                    i, _I = self.findMaxResidual()
                    self.updateMessage(i, _I)

                    # I->i has been updated, which means that residuals for all
                    # J->j with J in nb[i]\I and j in nb[J]\i have to be updated
                    for J in self.nbV[i]:
                        if J.iter != _I:
                            for j in self.nbF[J.node]:
                                _J = j.dual
                                if j != i:
                                    self.calcNewMessage(j.node, _J)
            elif self.updates == 'PARALL':
                # Parallel updates
                for i in range(self.nrVars):
                    for I in self.nbV[i]:
                        self.calcNewMessage(i, I.iter)

                for i in range(self.nrVars):
                    for I in self.nbV[i]:
                        self.updateMessage(i, I.iter)
            else:
                # Sequential updates
                if self.updates == 'SEQRND':
                    random.shuffle(self._updateSeq)

                for e in self._updateSeq:
                    self.calcNewMessage(e[0], e[1])
                    self.updateMessage(e[0], e[1])

            # calculate new beliefs and compare with old ones
            maxDiff = -np.inf
            for i in range(self.nrVars):
                b = self.beliefV(i).clone()
                maxDiff = max(maxDiff, dist(b.p, self._oldBeliefsV[i].p, 'DISTLINF'))
                self._oldBeliefsV[i] = b
            for I in range(self.nrFactors):
                b = self.beliefF(I).clone()
                maxDiff = max(maxDiff, dist(b.p, self._oldBeliefsF[I].p, 'DISTLINF'))
                self._oldBeliefsF[I] = b

            if self.verbose >= 3:
                print("%s::run:  maxdiff %f after %d passes" % (self.name(), maxDiff, self._iters + 1))

            self._iters += 1

        if maxDiff > self._maxdiff:
            self._maxdiff = maxDiff

        if self.verbose >= 1:
            if maxDiff > self.tol:
                if self.verbose >= 1:
                    print("%s::run: WARNING: not converged after %d passes (%f seconds)...final maxdiff:%f" % (
                        self.name(), self._iters, time.time() - tic, maxDiff))
            else:
                if self.verbose >= 3:
                    print('%s::run: converged in %d passes (%f seconds).' % (
                        self.name(), self._iters, time.time() - tic))
        return maxDiff

    def calcBeliefV(self, i: int) -> Prob:
        p = Prob.same_value(self.model.get_variable(i).domain.size(),
                            0.0 if self.logdomain else 1.0)
        for I in self.nbV[i]:
            if self.logdomain:
                p += self._edges[i][I.iter].newMessage
            else:
                p *= self._edges[i][I.iter].newMessage
        return p

    def beliefV(self, i: int) -> LDFactor:
        p = self.calcBeliefV(i)

        if self.logdomain:
            p -= p.max()
            p.takeExp()
        p.normalize()
        return LDFactor(self.model, [i], p)

    def beliefF(self, I) -> LDFactor:
        p = self.calcBeliefF(I)

        if self.logdomain:
            p -= p.max()
            p.takeExp()
        p.normalize()

        return LDFactor(self.model, self.factors[I].var_idx, p)

    def calcBeliefF(self, I: int) -> Prob:
        return self.calcIncomingMessageProduct(I, False, 0)

    def beliefs(self):
        result = []
        for i in range(self.nrVars):
            result.append(self.beliefV(i))
        for I in range(self.nrFactors):
            result.append(self.beliefF(I))
        return result

    def logZ(self):
        ans = 0.0
        for i in range(self.nrVars):
            ans += (1.0 - len(self.nbV[i])) * self.beliefV(i).p.entropy()
        for I in range(self.nrFactors):
            ans -= dist(self.beliefF(I).p, self.factors[I].p, 'DISTKL')
        return ans

    def updateMessage(self, i: int, _I: int):
        if recordSentMessages:
            print("%d->%d:%s" % (i, _I, self._edges[i][_I].newMessage._p))
            self._sentMessages.append((i, _I))
        if self.damping == 0.0:
            self._edges[i][_I].message = self._edges[i][_I].newMessage.clone()
            if self.updates == 'SEQMAX':
                self.updateResidual(i, _I, 0.0)
        else:
            d = self.damping
            if self.logdomain:
                self._edges[i][_I].message = (self._edges[i][_I].message * d) + (
                        self._edges[i][_I].newMessage * (1.0 - d))
            else:
                self._edges[i][_I].message = (self._edges[i][_I].message ** d) * (
                        self._edges[i][_I].newMessage ** (1.0 - d))
            if self.updates == 'SEQMAX':
                self.updateResidual(i, _I, dist(self._edges[i][_I].newMessage, self._edges[i][_I].message, 'DISTLINF'))

    def updateResidual(self, i, _I, r):
        self._edges[i][_I].residual = r

        # rearrange look-up table (delete and reinsert new key)
        self._lut.erase(self._edge2lut[i][_I])
        self._edge2lut[i][_I] = self._lut.insert((r, (i, _I)))

    def identify(self) -> str:
        return 'BP'

    def name(self) -> str:
        return 'BP'
