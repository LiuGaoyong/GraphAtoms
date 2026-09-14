"""The Recorder for the completeness of event table.


C = 1 / (alpha * Nr)    # from reference paper 1
    Where Nr is the number of continuous new events.
Note: The `Nr` must be ensured by repeated exploration.

factor = fmin / ftotal  # from reference paper 2
    fmin: the minimum time found for all events.
    ftotal: the total time exploration for all events.
factor2 = 1 / (1 + factor)
C = 1/ (1 + factor2**m)

Ref:
    1)  Xu, Lijun, and Graeme Henkelman. "Adaptive kinetic Monte Carlo for
    first-principles accelerated dynamics." The Journal of Chemical Physics
    129.11 (2008): 114104.
    2) Williams, C. J. Off-lattice kinetic Monte Carlo methods and the
    Fe-H system. PhD thesis, University of Cambridge, 2024.
"""

from collections import defaultdict
from typing import override

import pydantic

from graphatoms.dataclasses import OurBaseModel  # type: ignore


class OldNewRecorder(OurBaseModel):
    old: pydantic.NonNegativeInt = 0
    new: pydantic.NonNegativeInt = 0
    fail: pydantic.NonNegativeInt = 0
    skip: pydantic.NonNegativeInt = 0
    continuous_old: pydantic.NonNegativeInt = 0  # Nr

    @pydantic.computed_field
    @property
    def total(self) -> int:
        return sum([self.old, self.new, self.fail, self.skip])

    @pydantic.validate_call
    def exploration_can_be_finished(
        self,
        confidence: pydantic.PositiveFloat = 5,
    ) -> pydantic.StrictBool:
        if confidence <= 0:
            raise KeyError("The confidence must be positive.")
        elif confidence < 1:
            raise NotImplementedError("The confidence must be 1 or greater.")
            value = 1 / (alpha * self.continuous_old)  # noqa: F821
            value = 0 if self.new == 0 else 1 - self.new / self.total
        else:
            value = self.continuous_old

        return value > confidence

    @override
    def _string(self) -> str:  # type: ignore
        return ",".join(
            [
                f"{self.old}o",
                f"{self.new}n",
                f"{self.fail}f",
                f"{self.skip}s",
                f"{self.continuous_old}c",
            ]
        )


class Recorder(OurBaseModel):
    cluster: set[str] = set()
    system: set[str] = set()
    exploration: dict[str, OldNewRecorder] = defaultdict(OldNewRecorder)

    @override
    def _string(self) -> str:  # type: ignore
        return (
            f"{len(self.cluster)} cluster "
            + f"& {len(self.system)} system "
            + "have been explored"
        )


if __name__ == "__main__":
    from pathlib import Path

    obj = Recorder()
    obj.system.add("fdsafs")
    obj.system.add("fdsafs")
    obj.exploration["fdsafs"].new += 1
    print(obj)
    print(repr(obj))
    obj.write_json(Path("a.json"))
