# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferlo.base import Domain, GraphModel


class Variable:
    """Random variable."""

    def __init__(
            self,
            model: GraphModel,
            index: int,
            domain: Domain):
        """Initializies a random variable.

        :param model: Model this variable belongs to.
        :param index: Index (position) of this variable in a model.
        :param domain: Domain of this variable (set of values it can take).
        """
        self.model = model
        self.index = index
        self.domain = domain
        self.name = 'x_%d' % index

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
