import dataclasses as dc
from pathlib import Path
from typing import Any, Self, override

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    computed_field,
    model_validator,
)

from graphatoms.dataclasses import OurBaseModel
from graphatoms.enterpoint.config import EventConfig
from graphatoms.reaction import EventBase, EventInfo

__all__ = ["MetaData"]


class _GasInfo(BaseModel):
    name: str
    sticking: float
    pressure: float


class _MetaDataBasic(OurBaseModel, EventConfig):
    gas_sticking: dict[str, NonNegativeFloat] = {}
    gas_pressure: dict[str, NonNegativeFloat] = {}
    temperature: float = 300

    @override
    def _string(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.model_dump().items()])

    @computed_field
    @property
    def gas_info_lst(self) -> list[_GasInfo]:
        result: list[_GasInfo] = []
        for name, sticking in self.gas_sticking.items():
            result.append(
                _GasInfo(
                    name=name,
                    sticking=sticking,
                    pressure=self.default_pressure,
                )
            )
        for name, pressure in self.gas_pressure.items():
            result.append(
                _GasInfo(
                    name=name,
                    sticking=self.default_sticking,
                    pressure=pressure,
                )
            )
        return result


class _MetaDataTable(BaseModel):
    key_rxn: list[str] = []
    key_r: list[str] = []
    key_g: list[str | None] = []
    key_t: list[str | None] = []
    key_p: list[str] = []
    Ea_forword: list[float] = []
    rate_forword: list[float] = []
    rate_reversed: list[float] = []
    Ea_reversed: list[float] = []
    for_cluster: list[str] = []
    for_system: list[str] = []
    count: list[int] = []
    dE: list[float] = []

    @model_validator(mode="before")
    @classmethod
    def __check(cls, data: Any) -> Any:
        msg = "The fields of EventInfo must be in the metadata table."
        a = set(EventInfo.__pydantic_fields__)
        b = set(cls.__pydantic_fields__)
        assert a <= b, msg
        return data

    def __to_dict(self) -> dict[str, list[str | float | None]]:
        result: dict[str, list[str | float | None]] = self.model_dump()
        assert len(set(len(v) for v in result.values())) == 1
        return result

    def __len__(self) -> int:
        self._check_length_same()
        return len(self.key_rxn)

    def _check_length_same(self) -> None:
        msg = "The length of each field must be the same."
        assert len(self.key_r) == len(self.key_rxn), msg
        assert len(self.key_g) == len(self.key_rxn), msg
        assert len(self.key_t) == len(self.key_rxn), msg
        assert len(self.key_p) == len(self.key_rxn), msg
        assert len(self.Ea_forword) == len(self.key_rxn), msg
        assert len(self.rate_forword) == len(self.key_rxn), msg
        assert len(self.rate_reversed) == len(self.key_rxn), msg
        assert len(self.Ea_reversed) == len(self.key_rxn), msg
        assert len(self.for_cluster) == len(self.key_rxn), msg
        assert len(self.for_system) == len(self.key_rxn), msg
        assert len(self.count) == len(self.key_rxn), msg
        assert len(self.dE) == len(self.key_rxn), msg

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.__to_dict())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(**{k: df[k].to_list() for k in df.columns})

    def get_minconut_for(
        self,
        *,
        cluster_key: str | None = None,
        system_key: str | None = None,
        **kwargs,
    ) -> int:
        if cluster_key is not None:
            return min(
                [
                    self.count[i]
                    for i, ck in enumerate(self.for_cluster)
                    if ck == cluster_key
                ]
                + [0]
            )
        elif system_key is not None:
            return min(
                [
                    self.count[i]
                    for i, sk in enumerate(self.for_system)
                    if sk == system_key
                ]
                + [0]
            )
        else:
            return min(self.count + [0])


class MetaData(BaseModel):
    basic: _MetaDataBasic = _MetaDataBasic()
    table: _MetaDataTable = _MetaDataTable()

    def persistence(self, path: Path | str) -> None:
        self.basic.write_json(Path(path) / "metadata-basic.json")
        self.table.dataframe.to_feather(Path(path) / "metadata-table.feather")

    @classmethod
    def from_storage(cls, path: Path | str) -> Self:
        Path(path).mkdir(parents=True, exist_ok=True)
        basic = _MetaDataBasic.read_json(Path(path) / "metadata-basic.json")
        df = pd.read_feather(Path(path) / "metadata-table.feather")
        table = _MetaDataTable.from_dataframe(df)
        return cls(basic=basic, table=table)

    def __len__(self) -> int:
        return self.table.__len__()

    def _bkl_solver(
        self,
        forward_nmatched: list[int] | np.ndarray,
        reversed_nmatched: list[int] | np.ndarray,
        *args,
        **kwargs,
    ) -> tuple[EventInfo, float]:
        n_events = len(self.table)
        forward_nmatched = np.asarray(forward_nmatched, dtype=int).flatten()
        reversed_nmatched = np.asarray(reversed_nmatched, dtype=int).flatten()
        assert len(forward_nmatched) == len(reversed_nmatched) == n_events, (
            "The length of forward_nmatched "
            + "and reversed_nmatched must be "
            + "same as the number of events "
            + "in the metadata table."
        )
        matched = np.append(forward_nmatched, reversed_nmatched)
        forward_rates = np.asarray(self.table.rate_forword)
        reversed_rates = np.asarray(self.table.rate_reversed)
        rates = np.append(forward_rates, reversed_rates).flatten() * matched

        k_tot = rates.sum()
        if k_tot <= 0:
            raise ValueError("The sum of rates must be positive.")

        # two independent uniform random numbers in [0, 1)
        rho1, rho2 = np.random.random(2)

        # cumulative rate table, binary search the selected event
        index: int = np.searchsorted(np.cumsum(rates), rho1 * k_tot)

        # prevent floating-point error from causing index out of range
        # if the index is out of range, set it to the last index
        if index >= len(rates):
            index = len(rates) - 1

        # time increment: -ln(rho2) / k_tot
        dt: float = -np.log(rho2) / k_tot

        if index >= n_events:
            key_rxn = self.table.key_rxn[index - n_events]
            einfo = self.read(key_rxn).reversed
        else:
            key_rxn = self.table.key_rxn[index]
            einfo = self.read(key_rxn)
        return einfo, dt

    def rxn_count_add_one(self, value: EventBase | str) -> None:
        """Add the count of the reaction with the hash value.

        Note:
            This typically means a previously discovered reaction
            (i.e. old event) was encountered during exploration.
        """
        if isinstance(value, EventBase):
            value = value.hash
        index = self.table.key_rxn.index(value)
        self.table.count[index] += 1

    def read(self, value: str | int) -> EventInfo:
        """Read the event metadata from the Table.

        Returns:
            EventBase: The event metadata.
        """
        if isinstance(value, int):
            assert 0 <= value < len(self.table), (
                "The index must be between 0 and the "
                + "number of events in the metadata table."
            )
            index = int(value)
        elif isinstance(value, str):
            index = self.table.key_rxn.index(value)
        else:
            raise ValueError(f"Unknown event type: {type(value)}")
        return EventInfo(
            key_rxn=self.table.key_rxn[index],
            key_r=self.table.key_r[index],
            key_g=self.table.key_g[index],
            key_t=self.table.key_t[index],
            key_p=self.table.key_p[index],
            Ea_forword=self.table.Ea_forword[index],
            rate_forword=self.table.rate_forword[index],
            rate_reversed=self.table.rate_reversed[index],
            Ea_reversed=self.table.Ea_reversed[index],
            for_cluster=self.table.for_cluster[index],
            for_system=self.table.for_system[index],
            dE=self.table.dE[index],
        )

    def has(self, event: EventBase | str) -> bool:
        """Check if the event is in the Table.

        Returns:
            bool: True if the event is in the Table, False otherwise.
        """
        if isinstance(event, str):
            return event in self.table.key_rxn
        elif isinstance(event, EventBase):
            return event.hash in self.table.key_rxn
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

    def write(
        self,
        event: EventBase,
        for_cluster: str,
        for_system: str,
    ) -> None:
        """Write the event metadata to the Table."""
        if not self.has(event):
            reversed_event = event.reversed
            if reversed_event.hash != event.hash:
                raise ValueError(
                    "The hash of the reversed event must "
                    + "be the same as the original event."
                    + " Please contact the developer."
                )
            self.table.count.append(1)
            temperature = float(self.basic.temperature)
            info = EventInfo.from_event(
                event=event,
                for_system=for_system,
                for_cluster=for_cluster,
                temperature=temperature,
            )
            for k, v in info.to_dict().items():
                lst: list = getattr(self.table, k)
                lst.append(v)
            try:
                self.table._check_length_same()
            except AssertionError as e:
                raise ValueError(f"{e} Please contact the developer.")


if __name__ == "__main__":
    import dataclasses as dc
    from pprint import pprint

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print(df)
    print({k: df[k].values for k in df.columns})

    meta = MetaData(basic=_MetaDataBasic(**dc.asdict(EventConfig())))
    pprint(meta)
    print(meta.basic)
    print("-----------------")
    print(meta.table.rate_reversed)

    Path("./zzz").mkdir(parents=True, exist_ok=True)
    meta.persistence(Path("./zzz"))
    meta2 = MetaData.from_storage(Path("./zzz"))
    pprint(meta2)
    print(meta2.basic)
    print("-----------------")
    print(meta2.table)
