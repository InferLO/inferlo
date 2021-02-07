"""Testing utils."""
from .experiment_runner import ExperimentRunner
from .model_generators import (
    clique_potts_model,
    cross_potts_model,
    grid_potts_model,
    ising_model_on_graph,
    line_potts_model,
    pairwise_model_on_graph,
    random_generic_model,
    tree_potts_model,
)
from .test_utils import assert_results_close
