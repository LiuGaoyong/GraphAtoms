from typing import Self, override

from pydantic import NonNegativeFloat, model_validator

from graphatoms.dataclasses import OurBaseModel


class GasMixin(OurBaseModel):
    sticking: NonNegativeFloat | None = None
    pressure: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def __check_gas(self) -> Self:
        if not self.is_gas:
            object.__setattr__(self, "sticking", None)
            object.__setattr__(self, "pressure", None)
        else:
            if self.sticking is None:
                object.__setattr__(self, "sticking", 1.0)
            assert isinstance(self.sticking, float)
            assert 0.0 <= self.sticking <= 10.0
            assert isinstance(self.pressure, float)
            assert 0.0 <= self.pressure
        return self

    @property
    def is_gas(self) -> bool:
        return self.pressure is not None

    @override
    def _string(self) -> str:
        return "GAS" if self.is_gas else ""


#######################################################################
#                                   Test
#######################################################################


def test_BondGraph() -> None:
    assert len(GasMixin.__abstractmethods__) == 0, (
        GasMixin.__abstractmethods__,
        GasMixin.__name__,
    )
