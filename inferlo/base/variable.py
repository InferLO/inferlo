from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferlo.model import Domain, GraphModel


class Variable:
    """A variable"""

    def __init__(
            self,
            model: GraphModel,
            index: int,
            domain: Domain):
        self.model = model
        self.index = index
        self.domain = domain
        self.name = 'x_%d' % index

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
