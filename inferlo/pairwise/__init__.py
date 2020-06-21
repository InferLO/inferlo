"""Pairwise graphcal models (such as Potts or Ising).

Contains model definition and algorithms specific to it.
"""
from .pwf_model import PairWiseFiniteModel
from .inference_result import InferenceResult
from .junction_tree import to_junction_tree_model
