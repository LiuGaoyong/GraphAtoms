"""The Recorder for the completeness of event table.


C = 1 / (alpha * Nr)    # from reference paper 1
    Where Nr is the number of continuous new events.
Note: The `Nr` must be ensured by repeated exploration.

factor = fmin / ftotal  # from reference paper 2
    fmin: the minimum time found for all events.
    ftotal: the total time exploration for all events.
factor2 = 1 / (1 + factor)
C = 1/ (1 + factor2**m)
    m: consecutive failed searches (CFS). Where, in this
        context, a 'failed' search is one that fails to discover
        an unknown SP (e.g. finds a known SP or fails to converge).

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


class _OldNewRecorder(OurBaseModel):
    old: pydantic.NonNegativeInt = 0
    new: pydantic.NonNegativeInt = 0
    fail: pydantic.NonNegativeInt = 0
    skip: pydantic.NonNegativeInt = 0
    continuous_old: pydantic.NonNegativeInt = 0  # Nr
    continuous_nonnew: pydantic.NonNegativeInt = 0  # m

    @pydantic.computed_field
    @property
    def total(self) -> int:
        """The total number of events (old + new + fail).

        Note: The `skip` is not included in the total number of events.
        """
        return sum([self.old, self.new, self.fail])

    @pydantic.validate_call
    def exploration_can_be_finished(
        self,
        confidence: pydantic.PositiveFloat = 5,
        min_found: pydantic.NonNegativeInt | None = None,
    ) -> pydantic.StrictBool:
        if confidence <= 0:
            raise KeyError("The confidence must be positive.")
        elif confidence < 1:
            # Use Williams formula
            assert min_found is not None, (
                "min_found must be provided for Williams"
                + " formula (i.e. confidence < 1)."
            )
            m = self.continuous_nonnew
            factor = min_found / self.total
            factor2 = 1 / (1 + factor)
            value = 1 / (1 + factor2**m)
        else:
            # Use Xu-Henkelman formula
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
                f"{self.continuous_old}Nr",
                f"{self.continuous_nonnew}M",
                f"{self.total}Tot",
            ]
        )

    def found_skip(self) -> None:
        self.skip += 1

    def found_new(self) -> None:
        self.continuous_nonnew = 0
        self.continuous_old = 0
        self.new += 1

    def found_old(self) -> None:
        self.continuous_nonnew += 0
        self.continuous_old += 1
        self.old += 1

    def found_fail(self) -> None:
        self.continuous_nonnew += 1
        self.fail += 1


class Recorder(OurBaseModel):
    system: set[str] = set()
    cluster: dict[str, _OldNewRecorder] = defaultdict(_OldNewRecorder)
    adsorption: set[str] = set()
    bulk: set[str] = set()

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
    obj.cluster["fdsafs"].new += 1
    obj.cluster["fdsafs"].new += 1
    print(obj)
    print(repr(obj))
    obj.write_json(Path("a.json"))
