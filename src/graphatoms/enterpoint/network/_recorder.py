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

from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Annotated, Any, Self, override

import pydantic

from graphatoms.dataclasses import OurBaseModel


class RecorderItemBase(pydantic.BaseModel):
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


class OldNewRecorder4Confidence(OurBaseModel, RecorderItemBase):
    @pydantic.validate_call
    def get_williams_confidence(
        self, min_found: pydantic.NonNegativeInt
    ) -> Annotated[float, pydantic.Field(ge=0.5, lt=1.0)]:
        """The Williams confidence formula.

        Args:
            min_found: The minimum time found for all events.

        Returns:
            The confidence value.
        """
        m = self.continuous_nonnew
        factor = min_found / self.total
        factor2 = 1 / (1 + factor)
        value = 1 / (1 + factor2**m)
        return value

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
            value = self.get_williams_confidence(min_found)
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


class RecorderInfo(RecorderItemBase):
    create_at: datetime = pydantic.Field(default_factory=lambda: datetime.now())
    confidence: Annotated[float, pydantic.Field(ge=0.5, lt=1.0)]
    for_cluster: str = pydantic.Field(default="")
    for_system: str = pydantic.Field(default="")
    for_gas: str = pydantic.Field(default="")

    @classmethod
    def from_oldnew(
        cls,
        oldnew: OldNewRecorder4Confidence,
        min_found: pydantic.NonNegativeInt,
        *,
        for_cluster: str = "",
        for_system: str = "",
        for_gas: str = "",
    ) -> Self:
        return cls(
            old=oldnew.old,
            new=oldnew.new,
            fail=oldnew.fail,
            skip=oldnew.skip,
            continuous_old=oldnew.continuous_old,
            continuous_nonnew=oldnew.continuous_nonnew,
            confidence=oldnew.get_williams_confidence(min_found),
            for_cluster=for_cluster,
            for_system=for_system,
            for_gas=for_gas,
        )

    @classmethod
    def get_csv_title(cls) -> str:
        msg = "old,new,fail,total,skip,continuous_old,continuous_nonnew"
        return f"{msg},confidence,for_cluster,for_system,for_gas,create_at"

    def to_csv_line(self) -> str:
        msg = f"{self.old},{self.new},{self.fail},{self.total},"
        msg += f"{self.skip},{self.continuous_old},{self.continuous_nonnew},"
        msg += f"{self.confidence},{self.for_cluster},{self.for_system}"
        create_at = self.create_at.strftime("%Y-%m-%d %H:%M:%S.%f")
        return f"{msg},{self.for_gas},{create_at}"


class RecorderBase(OurBaseModel):
    cluster: dict[str, OldNewRecorder4Confidence] = defaultdict(
        OldNewRecorder4Confidence
    )
    adsorption: Any
    bulk: Any

    @override
    def _string(self) -> str:  # type: ignore
        return (
            f"{len(self.cluster)} cluster(dimer) "
            + f"& {len(self.adsorption)} adsorption "
            + f"& {len(self.bulk)} bulk "
            + "have been explored"
        )

    @abstractmethod
    def clear(self) -> None:
        self.cluster.clear()


class Recorder(RecorderBase):
    adsorption: set[str] = set()
    bulk: set[str] = set()

    @override
    def clear(self) -> None:
        self.adsorption.clear()
        self.bulk.clear()
        super().clear()


class RecorderV2(RecorderBase):
    adsorption: dict[tuple[str, str], set[str]] = defaultdict(set)
    bulk: dict[str, set[str]] = defaultdict(set)

    @override
    def clear(self) -> None:
        self.adsorption.clear()
        self.bulk.clear()
        super().clear()

    def exploration_can_be_finished(
        self,
        cluster_key: str,
        *args,
        other_cluster_key: str | None = None,
        adsorption_gas_key: str | None = None,
        confidence: pydantic.PositiveFloat = 5.0,
        min_found: pydantic.NonNegativeInt | None = None,
        **kwargs,
    ) -> bool:
        if adsorption_gas_key is not None:
            assert other_cluster_key is not None, (
                "The `other_cluster_key` must be not None"  #
                " when `adsorption_gas_key` is provided."
            )
            return self.__explore_adsorption_can_be_finished(
                cluster_key=cluster_key,
                gas_key=adsorption_gas_key,
                initial_try_key=other_cluster_key,
            )
        elif other_cluster_key is not None:
            return self.__explore_bulk_can_be_finished(
                cluster_key=cluster_key,
                other_cluster_key=other_cluster_key,
            )
        else:
            return self.__explore_cluster_can_be_finished(
                cluster_key=cluster_key,
                confidence=confidence,
                min_found=min_found,
            )

    def __explore_adsorption_can_be_finished(
        self,
        gas_key: str,
        cluster_key: str,
        initial_try_key: str,
    ) -> bool:
        if (cluster_key, gas_key) not in self.adsorption:
            self.adsorption[(cluster_key, gas_key)] = set()
        return initial_try_key in self.adsorption[(cluster_key, gas_key)]

    def __explore_bulk_can_be_finished(
        self,
        cluster_key: str,
        other_cluster_key: str,
    ) -> bool:
        if cluster_key not in self.bulk:
            self.bulk[cluster_key] = set()
        return other_cluster_key in self.bulk[cluster_key]

    def __explore_cluster_can_be_finished(
        self,
        cluster_key: str,
        confidence: pydantic.PositiveFloat,
        min_found: pydantic.NonNegativeInt | None = None,
    ) -> bool:
        if cluster_key not in self.cluster:
            self.cluster[cluster_key] = OldNewRecorder4Confidence()
        return self.cluster[cluster_key].exploration_can_be_finished(
            confidence=confidence,
            min_found=min_found,
        )


if __name__ == "__main__":
    from io import StringIO
    from pathlib import Path

    import pandas as pd

    obj = Recorder()
    obj.adsorption.add("fdsafs")
    obj.cluster["fdsafs"].new += 1
    obj.cluster["fdsafs"].new += 1
    print(obj)
    print(repr(obj))
    obj.write_json(Path("a.json"))

    io = StringIO()
    io.write(RecorderInfo.get_csv_title() + "\n")
    a = RecorderInfo(confidence=0.6)
    io.write(RecorderInfo(confidence=0.5).to_csv_line() + "\n")
    io.write(RecorderInfo(confidence=0.5).to_csv_line() + "\n")
    io.write(a.to_csv_line() + "\n")

    s = io.getvalue()
    print(s)
    df = pd.read_csv(StringIO(s))
    print("df:")
    print(df)
    print(df.sort_values(by="create_at", ascending=True))
