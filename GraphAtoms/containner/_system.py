from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.containner._atomic import ATOM_KEY
from GraphAtoms.containner._graph import Graph


class System(Graph):
    """The whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        msg = "The key of `{:s}` should be None for System."
        for k in (ATOM_KEY.MOVE_FIX_TAG, ATOM_KEY.COORDINATION):
            assert getattr(self, k) is None, msg.format(k)
        return self
