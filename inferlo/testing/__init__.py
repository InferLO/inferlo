"""Testing utils."""
from .model_generators import (
    clique_potts_model,
    cross_potts_model,
    grid_potts_model,
    line_potts_model,
    pairwise_model_on_graph,
    tree_potts_model,
)
from .test_utils import assert_results_close
