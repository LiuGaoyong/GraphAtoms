import pandas as pd
from pydantic import BaseModel, NonNegativeFloat

from graphatoms.enterpoint.config import EventConfig


class MetaDataBasic(EventConfig):
    gas_sticking: dict[str, NonNegativeFloat] = {}
    gas_pressure: dict[str, NonNegativeFloat] = {}
    temperature: float = 300


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


class MetaData(BaseModel):
    basic: MetaDataBasic = MetaDataBasic()
    table: MetaDataTable = MetaDataTable()


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print(df)
    print({k: df[k].values for k in df.columns})

    meta = MetaData()
