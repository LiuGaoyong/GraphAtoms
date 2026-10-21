import dataclasses as dc
import types
from pathlib import Path
from typing import Any, Self, Union, get_args, get_origin, override

import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeFloat, model_validator

from graphatoms.dataclasses import OurBaseModel
from graphatoms.enterpoint.config import Config, EventConfig
from graphatoms.reaction import EventBase, EventInfo
from graphatoms.system.bonds import matchmode2nmatch
from graphatoms.utils.bytestool import atomic_write

__all__ = ["MetaData"]


def _is_list_of_optional_str(annotation: Any) -> bool:
    """Whether the annotation is ``list[str | None]``."""
    if get_origin(annotation) is not list:
        return False
    element = get_args(annotation)[0]
    if get_origin(element) not in (Union, types.UnionType):
        return False
    return set(get_args(element)) == {str, type(None)}


def _missing_to_none(value: Any) -> Any:
    """Convert a missing value marker of pandas to ``None``."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


class _MetaDataBasic(OurBaseModel, EventConfig):
    gas_sticking: dict[str, NonNegativeFloat] = {}

    @classmethod
    def from_config(cls, config: Config | EventConfig) -> Self:
        if isinstance(config, Config):
            config = config.event
        if not isinstance(config, dict):
            if isinstance(config, EventConfig):
                config = dc.asdict(config)  # type: ignore
            elif isinstance(config, _MetaDataBasic):
                config = config.to_dict()
        return cls(**config)  # type: ignore

    def check_consistency(
        self,
        config: EventConfig | Self,
        exclude_keys: set[str] = {"db_format"},
    ) -> None:
        msg = r"The keys of {} must be the same as input, "
        msg += r"but we got {}(from self) and {}(from input)"
        for k in self.__pydantic_fields__:
            if k in exclude_keys:
                continue
            v1 = getattr(self, k)
            v2 = getattr(config, k)
            if not isinstance(v1, dict):
                assert v1 == v2 or np.isclose(v1, v2), msg.format(k, v1, v2)
            else:
                for kk, vv1 in v1.items():
                    assert kk in v2.keys(), msg.format(k + "." + kk, vv1, None)
                    assert vv1 == v2[kk] or np.isclose(vv1, v2[kk]), msg.format(
                        k + "." + kk, vv1, v2[kk]
                    )

    @override
    def _string(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.model_dump().items()])


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
        """Create the table from a DataFrame.

        Note:
            Since pandas 3.0, a string column containing missing values is
            read back as the new default string dtype (``str``), whose
            missing value marker is ``numpy.nan`` instead of ``None``.
            Such ``nan`` values are rejected by the ``str | None`` fields,
            so they are converted back to ``None`` here.
        """
        data: dict[str, list[Any]] = {}
        for k in df.columns:
            values: list[Any] = df[k].to_list()
            field = cls.__pydantic_fields__.get(k)
            if field is not None and _is_list_of_optional_str(field.annotation):
                values = [_missing_to_none(v) for v in values]
            data[k] = values
        return cls(**data)

    def merge(self, other: Self, idxs: list[int] | None = None) -> None:
        if idxs is None:
            idxs = list(range(len(other)))
        for i in idxs:
            k = other.key_rxn[i]
            if k not in self.key_rxn:
                self.key_rxn.append(k)
                self.key_r.append(other.key_r[i])
                self.key_g.append(other.key_g[i])
                self.key_t.append(other.key_t[i])
                self.key_p.append(other.key_p[i])
                self.Ea_forword.append(other.Ea_forword[i])
                self.rate_forword.append(other.rate_forword[i])
                self.rate_reversed.append(other.rate_reversed[i])
                self.Ea_reversed.append(other.Ea_reversed[i])
                self.for_cluster.append(other.for_cluster[i])
                self.for_system.append(other.for_system[i])
                self.count.append(other.count[i])
                self.dE.append(other.dE[i])
            else:
                idx = self.key_rxn.index(k)
                self.count[idx] += other.count[i]
                len(self.key_r[idx])
                print(
                    f"Find duplicate key: {k}\n"
                    f"  dE:   (old={self.dE[idx]},new={other.dE[i]})\n"
                    f"  Ea:   (old={self.Ea_forword[idx]},"
                    f"new={other.Ea_forword[i]})\n"
                    f"  key_r:(old={self.key_r[idx]},new={other.key_r[i]})\n"
                    f"  key_g:(old={self.key_g[idx]},new={other.key_g[i]})\n"
                    f"  key_t:(old={self.key_t[idx]},new={other.key_t[i]})\n"
                    f"  key_p:(old={self.key_p[idx]},new={other.key_p[i]})\n",
                    flush=True,
                )

    def get_minconut_for(
        self,
        *,
        cluster_key: str | None = None,
        system_key: str | None = None,
        gas_key: str | None = None,
        **kwargs,
    ) -> int:
        if cluster_key is not None:
            return min(
                [
                    self.count[i]
                    for i, ck in enumerate(self.for_cluster)
                    if (
                        ck == cluster_key
                        and ((gas_key is None) or self.key_g[i] is gas_key)
                    )
                ]
                + [0]
            )
        elif system_key is not None:
            return min(
                [
                    self.count[i]
                    for i, sk in enumerate(self.for_system)
                    if (
                        sk == system_key
                        and ((gas_key is None) or self.key_g[i] is gas_key)
                    )
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
        atomic_write(
            Path(path) / "metadata-table.feather",
            writer=lambda p: self.table.dataframe.to_feather(p),
        )

    @classmethod
    def from_storage(cls, path: Path | str) -> Self:
        Path(path).mkdir(parents=True, exist_ok=True)
        basic = _MetaDataBasic.read_json(Path(path) / "metadata-basic.json")
        df = pd.read_feather(Path(path) / "metadata-table.feather")
        table = _MetaDataTable.from_dataframe(df)
        return cls(basic=basic, table=table)

    def __len__(self) -> int:
        return self.table.__len__()

    def bkl_solver(
        self,
        matching_dct: dict[tuple[str, bool], np.ndarray],
        *args,
        env_pressure: dict[str, float] = {},
        **kwargs,
    ) -> tuple[pd.DataFrame, str, bool, float, float]:
        """Solve the BKL equation to select the reaction.

        Args:
            matching_dct: The dictionary of the matching results.
                (rxn_key, is_forward) -> the matched np.ndarray

        Returns:
            1. The dataframe of the matching results.
            2. The key of the selected reaction.
            3. Whether the reaction is forward.
            4. The time increment.
            5. The total rate of the selected reaction.
        """
        lst: list[tuple[str, bool, int, float]] = []
        names = ["rxn_key", "is_forward", "nmatched", "rate"]
        for (rxn_key, is_forward), single_match_res in matching_dct.items():
            i: int = self.table.key_rxn.index(rxn_key)
            # Eliminate the overestimation caused by permutation symmetry.
            nmatched = matchmode2nmatch(single_match_res)
            if is_forward:
                rate = self.table.rate_forword[i]
                key_g = self.table.key_g[i]
                if key_g is not None:
                    rate *= env_pressure.get(key_g, 0.0)
            else:
                rate = self.table.rate_reversed[i]
            lst.append((rxn_key, is_forward, nmatched, rate))
        df = pd.DataFrame(lst, columns=names)

        rates = df["rate"].to_numpy() * df["nmatched"].to_numpy()
        k_tot = rates.sum()
        if k_tot <= 0:
            raise ValueError("The sum of rates must be positive.")

        # two independent uniform random numbers in [0, 1)
        rho1, rho2 = np.random.random(2)

        # cumulative rate table, binary search the selected event
        index: int = np.searchsorted(np.cumsum(rates), rho1 * k_tot)

        # time increment: -ln(rho2) / k_tot
        dt = float(-np.log(rho2) / k_tot)

        rxn_key = str(df["rxn_key"].iloc[index])
        is_forward = bool(df["is_forward"].iloc[index])
        return df, rxn_key, is_forward, dt, k_tot

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
