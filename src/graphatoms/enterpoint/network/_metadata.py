import dataclasses as dc
from pathlib import Path
from typing import Self, override

import pandas as pd
from pydantic import BaseModel, NonNegativeFloat, computed_field

from graphatoms.dataclasses import OurBaseModel
from graphatoms.enterpoint.config import EventConfig
from graphatoms.reaction import EventBase
from graphatoms.system.database import DatabaseABC


class GasInfo(BaseModel):
    name: str
    sticking: float
    pressure: float


class MetaDataBasic(OurBaseModel, EventConfig):
    gas_sticking: dict[str, NonNegativeFloat] = {}
    gas_pressure: dict[str, NonNegativeFloat] = {}
    temperature: float = 300

    @override
    def _string(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self.model_dump().items()])

    @computed_field
    @property
    def gas_info_lst(self) -> list[GasInfo]:
        result: list[GasInfo] = []
        for name, sticking in self.gas_sticking.items():
            result.append(
                GasInfo(
                    name=name,
                    sticking=sticking,
                    pressure=self.default_pressure,
                )
            )
        for name, pressure in self.gas_pressure.items():
            result.append(
                GasInfo(
                    name=name,
                    sticking=self.default_sticking,
                    pressure=pressure,
                )
            )
        return result


class MetaDataTable(BaseModel):
    key_rxn: list[str] = []
    key_r: list[str] = []
    key_g: list[str | None] = []
    key_t: list[str | None] = []
    key_p: list[str] = []
    Ea: list[float] = []
    dE: list[float] = []
    rate: list[float] = []

    def __to_dict(self) -> dict[str, list[str | float | None]]:
        result: dict[str, list[str | float | None]] = self.model_dump()
        assert len(set(len(v) for v in result.values())) == 1
        return result

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.__to_dict())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        return cls(**{k: df[k].to_list() for k in df.columns})

    def write(self, event: EventBase, temperature: float = 300.0) -> bool:
        """Write the event metadata to the Table.

        Returns:
            bool: True if the event is new, False otherwise.
        """
        if event.hash in self.key_rxn:
            return False  # event is already in the database
        else:
            self.key_rxn.append(event.hash)
            self.key_r.append(DatabaseABC.get_key_of(event.R))
            if event.G is not None:
                self.key_g.append(DatabaseABC.get_key_of(event.G))
            else:
                self.key_g.append(None)
            if event.T is not None:
                self.key_t.append(DatabaseABC.get_key_of(event.T))
            else:
                self.key_t.append(None)
            self.key_p.append(DatabaseABC.get_key_of(event.P))
            self.Ea.append(event.get_Ea(temperature))
            self.dE.append(event.get_dE(temperature))
            self.rate.append(event.get_rate(temperature))
            return True


class MetaData(BaseModel):
    basic: MetaDataBasic = MetaDataBasic()
    table: MetaDataTable = MetaDataTable()

    def persistence(self, path: Path | str) -> None:
        self.basic.write_json(Path(path) / "metadata-basic.json")
        self.table.dataframe.to_feather(Path(path) / "metadata-table.feather")

    @classmethod
    def from_storage(cls, path: Path | str) -> Self:
        Path(path).mkdir(parents=True, exist_ok=True)
        basic = MetaDataBasic.read_json(Path(path) / "metadata-basic.json")
        df = pd.read_feather(Path(path) / "metadata-table.feather")
        table = MetaDataTable.from_dataframe(df)
        return cls(basic=basic, table=table)


if __name__ == "__main__":
    import dataclasses as dc
    from pprint import pprint

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print(df)
    print({k: df[k].values for k in df.columns})

    meta = MetaData(basic=MetaDataBasic(**dc.asdict(EventConfig())))
    pprint(meta)
    print(meta.basic)
    print("-----------------")

    Path("./zzz").mkdir(parents=True, exist_ok=True)
    meta.persistence(Path("./zzz"))
    meta2 = MetaData.from_storage(Path("./zzz"))
    pprint(meta2)
    print(meta2.basic)
    print("-----------------")
    print(meta2.table)
