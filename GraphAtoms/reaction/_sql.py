from typing import Annotated

from sqlmodel import Field, SQLModel


class EventSQL(SQLModel):
    hash: Annotated[str, Field(primary_key=True, index=True)]
    is_adsorption: Annotated[str, Field(index=True)]
    is_desorption: Annotated[str, Field(index=True)]
    is_reaction: Annotated[str, Field(index=True)]
    r_hash: Annotated[str, Field(index=True)]
    p_hash: Annotated[str, Field(index=True)]
    ts_hash: Annotated[str, Field(index=True)]
    gas_hash: str | None = Field(default=None, index=True)
    event: bytes
