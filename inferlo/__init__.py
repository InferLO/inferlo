"""Root InferLO package."""
from inferlo.base import (
    Domain,
    Factor,
    DiscreteFactor,
    FunctionFactor,
    GenericGraphModel,
    GraphModel,
    Variable,
)
from inferlo.forney import NormalFactorGraphModel
from inferlo.pairwise import PairWiseFiniteModel
