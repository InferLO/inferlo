API Reference
=============


Graphical models
''''''''''''''''

Classes representing different kinds of graphical models.

.. autosummary::
    :toctree: generated/

    inferlo.GenericGraphModel
    inferlo.GraphModel
    inferlo.NormalFactorGraphModel
    inferlo.PairWiseFiniteModel



Auxiliary classes
'''''''''''''''''

Classes representing elementary concepts, such as variables and factors.
Also contains data classes representing results of computations.

.. autosummary::
    :toctree: generated/

    inferlo.DiscreteDomain
    inferlo.Domain
    inferlo.Factor
    inferlo.DiscreteFactor
    inferlo.FunctionFactor
    inferlo.RealDomain
    inferlo.Variable
    inferlo.pairwise.InferenceResult


Algorithms on graphical models
''''''''''''''''''''''''''''''

All algorithms are available through methods on graphical models. To invoke
specific algorithm, you should pass its name as "algorithm" argument to
corresponding methods. That's why algorithms are not exported in top-level
package.

However, we list all algorithms here for reference. Some algorithms may take
additional parameters (such as number of iterations for iterative algorithm), which
should be passed as keyword arguments. Use references below to get information on
algorithm-specific parameters.

Inference algorithms
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.bruteforce.infer_bruteforce
    inferlo.pairwise.junction_tree.infer_junction_tree
    inferlo.pairwise.inference.mean_field.infer_mean_field
    inferlo.pairwise.inference.message_passing.infer_message_passing
    inferlo.pairwise.inference.path_dp.infer_path_dp
    inferlo.pairwise.inference.tree_dp.infer_tree_dp
    inferlo.forney.edge_elimination.infer_edge_elimination

Optimization algorithms
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.bruteforce.max_lh_bruteforce
    inferlo.pairwise.junction_tree.max_likelihood_junction_tree
    inferlo.pairwise.optimization.tree_dp.max_likelihood_tree_dp
    inferlo.pairwise.optimization.path_dp.max_lh_path_dp

Sampling algorithms
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.bruteforce.sample_bruteforce
    inferlo.pairwise.sampling.tree_dp.sample_tree_dp

Model conversion algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.junction_tree.to_junction_tree_model

Algorithm on graphs
'''''''''''''''''''

In most cases we use NetworkX to represent and manipulate graphs. However,
in some cases, if the algorithm is too specific and is not available in
NetworkX, or if we need it to be faster than in NetworkX, we implement it
in our library. Those implementations are collected in this section. They are
intended for internal usage by other algorithms rather than to be used directly
by library users.

.. autosummary::
    :toctree: generated/

    inferlo.graphs.fast_dfs
    inferlo.graphs.path_decomposition

Model generators
''''''''''''''''

These function generate random models of certain structure. They are
useful for testing and benchmarking.

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.testing.clique_potts_model
    inferlo.pairwise.testing.grid_potts_model
    inferlo.pairwise.testing.line_potts_model
    inferlo.pairwise.testing.tree_potts_model
    inferlo.pairwise.testing.pairwise_model_on_graph