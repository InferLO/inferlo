API Reference
=============


Graphical models
''''''''''''''''

Classes representing different kinds of graphical models

.. autosummary::
    :toctree: generated/

    inferlo.GenericGraphModel
    inferlo.GraphModel
    inferlo.NormalFactorGraphModel
    inferlo.PairWiseFiniteModel



Auxiliary classes
'''''''''''''''''

Classes representing elementary concepts, such as
variables and factors.

.. autosummary::
    :toctree: generated/

    inferlo.DiscreteDomain
    inferlo.Domain
    inferlo.Factor
    inferlo.DiscreteFactor
    inferlo.FunctionFactor
    inferlo.RealDomain
    inferlo.Variable


Algorithms
''''''''''

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
    inferlo.pairwise.inference.mean_field.infer_mean_field
    inferlo.pairwise.inference.message_passing.infer_message_passing
    inferlo.pairwise.inference.path_dp.infer_path_dp
    inferlo.pairwise.inference.tree_dp.infer_tree_dp

Optimization algorithms
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.bruteforce.max_lh_bruteforce
    inferlo.pairwise.optimization.tree_dp.max_likelihood_tree_dp

Sampling algorithms
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.sampling.tree_dp.sample_tree_dp