from collections.abc import Sequence
from functools import cached_property
from typing import Annotated, override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.symbols import Symbols
from pydantic import model_validator
from pymatgen.core.structure import Molecule as PmgMol
from pymatgen.core.structure import Structure as PmgStrct
from typing_extensions import Self

from ...dataclasses import NDArray, OurBaseModel, numpy_validator
from ._box import Box
from ._eng import Energetics


class NumbersMixin(OurBaseModel):
    numbers: Annotated[NDArray, numpy_validator(int)]

    def __len__(self) -> int:
        return self.numbers.shape[0]

    @cached_property
    @override
    def natoms(self) -> int:
        return len(self.numbers)

    @property
    def Z(self) -> NDArray:
        return self.numbers

    @cached_property
    def formula(self) -> str:
        return self.symbols.get_chemical_formula("metal")

    @cached_property
    def symbols(self) -> Symbols:
        return Symbols(self.numbers)

    @cached_property
    def is_nonmetal(self) -> bool:
        #          He, Ne, Ar, Kr, Xe, Rn
        gas_elem = [2, 10, 18, 36, 54, 86]
        #            H, C, N, O, F, P,  S,  Cl, Br, I
        gas_elem += [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        return bool(all(np.isin(self.numbers, gas_elem)))


class Structure(NumbersMixin, Box, Energetics):
    positions: Annotated[NDArray, numpy_validator(float, (-1, 3))]

    @model_validator(mode="after")
    def __check_positions(self) -> Self:
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        return self

    @override
    def _string(self) -> str:
        return ",".join(
            [
                self.formula,
                Box._string(self),
                Energetics._string(self),
            ]
        )

    @property
    def R(self) -> NDArray:
        return self.positions

    @classmethod
    def SUPPORTED_CONVERT_FORMATS(cls) -> Sequence[str]:
        result = super().SUPPORTED_CONVERT_FORMATS()
        return tuple(result) + ("ase", "pymatgen")

    @classmethod
    def from_ase(cls, atoms: Atoms, **kw) -> Self:
        dct: dict[str, np.ndarray] = {
            k: v
            for k, v in atoms.todict().items()  #
            if isinstance(v, np.ndarray)
        } | atoms.info
        if isinstance(atoms.calc, Calculator):
            result = atoms.calc.results
            if "energy" not in dct | kw:
                try:
                    dct["energy"] = result["energy"]
                except Exception:
                    pass
            if "fmax" not in dct | kw:
                try:
                    forces = result["forces"]
                    fnorm = np.linalg.norm(forces, axis=1)
                    dct["fmax"] = np.max(fnorm)
                except Exception:
                    pass
        return super().from_dict(dct, **kw)

    def to_ase(
        self,
        *,
        exclude_energetics: bool = False,
        **kwargs,
    ) -> Atoms:
        return Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.ase_cell,
            pbc=self.pbc,
            info=self.to_dict(
                exclude_none=True,
                exclude_computed_fields=True,
                exclude=(
                    {"positions"}
                    | Box.__pydantic_fields__.keys()
                    | NumbersMixin.__pydantic_fields__.keys()
                    | (
                        set()
                        if not exclude_energetics
                        else Energetics.__pydantic_fields__.keys()
                    )
                ),
                numpy_ndarray_compatible=True,
                numpy_convert_to_list=False,
            ),
        )

    def to_pymatgen(
        self,
        *,
        exclude_energetics: bool = False,
        **kwargs,
    ) -> PmgStrct | PmgMol:
        raise NotImplementedError

    @classmethod
    def from_pymatgen(cls, obj: PmgStrct | PmgMol, **kw) -> Self:
        raise NotImplementedError
