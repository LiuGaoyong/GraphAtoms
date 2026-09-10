from collections import defaultdict
from typing import override

import pydantic

from graphatoms.dataclasses import OurBaseModel  # type: ignore


class OldNewRecorder(OurBaseModel):
    old: pydantic.NonNegativeInt = 0
    new: pydantic.NonNegativeInt = 0
    fail: pydantic.NonNegativeInt = 0
    skip: pydantic.NonNegativeInt = 0
    continuous_old: pydantic.NonNegativeInt = 0

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
