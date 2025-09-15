from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.containner._graph import GraphContainner


class System(GraphContainner):
    """The whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        assert self.pressure is None, "pressure should be None for System."
        assert self.sticking is None, "sticking should be None for System."
        assert self.move_fix_tag is None, (
            "move_fix_tag should be None for System."
        )
        assert self.coordination is None, (
            "coordination should be None for System."
        )
        return self
