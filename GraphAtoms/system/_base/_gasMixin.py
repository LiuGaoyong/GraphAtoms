from pydantic import NonNegativeFloat, model_validator
from typing_extensions import Self

from GraphAtoms.dataclasses import OurFrozenModel


class GasMixin(OurFrozenModel):
    sticking: NonNegativeFloat | None = None
    pressure: NonNegativeFloat | None = None

    @property
    def is_gas(self) -> bool:
        return self.pressure is not None

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
