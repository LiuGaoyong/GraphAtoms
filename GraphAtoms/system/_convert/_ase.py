from collections.abc import Mapping
from typing import Any

from ase import Atoms
from typing_extensions import Self

from .._base import Base
from .._base._atomsAttr import NumbersMixin
from .._base._bondsAttr import BondsAttr
from .._base._boxMixin import BoxMixin
from .._base._engMixin import EnergeticsMixin


class ASEConverter(Base):
    @classmethod
    def from_ase(
        cls,
        atoms: Atoms,
        parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        return cls.from_dict(
            {
                "numbers": atoms.numbers,
                "positions": atoms.positions,
                "box": atoms.cell.array if atoms.pbc.any() else None,
            }
            | atoms.info,
            parse_bonds=parse_bonds,
            parse_bonds_distance=parse_bonds_distance,
            parse_bonds_order=parse_bonds_order,
            **kwargs,
        )

    def to_ase(
        self,
        *,
        exclude_energetics: bool = False,
        exclude_bond_attibutes: bool = False,
        **kwargs,
    ) -> Atoms:
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            pbc=self.is_periodic,
            cell=self.ase_cell,
            info=self.to_dict(
                exclude_none=True,
                exclude_computed_fields=True,
                exclude=(
                    {"positions"}
                    | BoxMixin.__pydantic_fields__.keys()
                    | NumbersMixin.__pydantic_fields__.keys()
                    | (
                        set()
                        if not exclude_energetics
                        else EnergeticsMixin.__pydantic_fields__.keys()
                    )
                    | (
                        set()
                        if not exclude_bond_attibutes
                        else BondsAttr.__pydantic_fields__.keys()
                    )
                ),
                numpy_ndarray_compatible=True,
                numpy_convert_to_list=False,
            ),
        )
