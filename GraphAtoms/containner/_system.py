from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.containner._graph import GraphContainner


class System(GraphContainner):
    """The whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        assert self.move_fix_tag is None
        assert self.coordination is None
        assert self.pressure is None
        return self
