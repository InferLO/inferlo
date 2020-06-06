API Reference
=============


Basic concepts
''''''''''''''''''

Classes describing basic building blocks, such as variables and factors.

.. autosummary::
    :toctree: generated/

    inferlo.Domain
	inferlo.Factor
	inferlo.Variable


Kinds of graphical models
'''''''''''''''''''''''''

Classes describing different kinds of graphical models

.. autosummary::
    :toctree: generated/

    inferlo.GraphModel
	inferlo.PairWiseFiniteModel
	inferlo.NormalFactorGraphModel

Algorithms
''''''''''''''''''

These algorithms can be invoked directly or by corresponding methods of graphical model
classes.

.. autosummary::
    :toctree: generated/

    inferlo.pairwise.inference.mean_field.infer_mean_field

