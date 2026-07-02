import warnings
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Annotated, Any, Self, override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.cell import Cell
from ase.geometry import cell as cellutils
from ase.symbols import Symbols
from ase.thermochemistry import BaseThermoChem, HarmonicThermo
from ase.units import invcm
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
    validate_call,
)
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule as PmgMol
from pymatgen.core.structure import Structure as PmgStrct

from graphatoms.dataclasses import NDArray, OurBaseModel, numpy_validator

__all__ = ["Box", "Energetics", "Structure"]


class Box(OurBaseModel):
    cell: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None
    pbc: Annotated[NDArray, numpy_validator(bool, (3,))] | None = None

    @override
    def _string(self) -> str:
        return "PBC" if self.is_periodic else "NOPBC"

    @model_validator(mode="after")
    def __check_cell(self) -> Self:
        if not self.is_periodic:
            object.__setattr__(self, "cell", None)
            object.__setattr__(self, "pbc", None)
        return self

    @cached_property
    def is_periodic(self) -> bool:
        if self.pbc is None:
            return False
        elif self.cell is None:
            return False
        elif self.ase_cell.volume < 1e-3:
            return False
        else:
            pbc = self.cell.any(1) & self.pbc
            return bool(np.sum(pbc) > 0)

    @cached_property
    def cellpar(self) -> tuple[float, float, float, float, float, float]:
        return tuple(cellutils.cell_to_cellpar(self.ase_cell.array).tolist())

    @cached_property
    def a(self) -> float:
        return self.cellpar[0]

    @cached_property
    def b(self) -> float:
        return self.cellpar[1]

    @cached_property
    def c(self) -> float:
        return self.cellpar[2]

    @cached_property
    def alpha(self) -> float:
        return self.cellpar[3]

    @cached_property
    def beta(self) -> float:
        return self.cellpar[4]

    @cached_property
    def gamma(self) -> float:
        return self.cellpar[5]

    @cached_property
    def ase_cell(self) -> Cell:
        return Cell.new(self.cell)

    @cached_property
    def pmg_lattice(self) -> Lattice:
        return Lattice(self.ase_cell.array)

    @cached_property
    def is_orthorhombic(self) -> bool:
        if self.cell is None:
            return True
        return cellutils.is_orthorhombic(self.cell)


class Energetics(OurBaseModel):
    """Mixin for energetics (energy, forces, frequencies) and thermochemistry.

    Attributes:
        frequencies: Vibrational frequencies (cm^-1).
        fmax: Maximum force (eV/Å).
        energy: Potential energy (eV).
    """

    frequencies: Annotated[NDArray, numpy_validator()] | None = None
    fmax: NonNegativeFloat | None = None
    energy: float | None = None

    @override
    def _string(self) -> str:
        lst: list[str] = []
        if self.energy is None:
            lst.append("NOSPE")
        else:
            if abs(self.energy) >= 1:
                lst.append(f"E={self.energy:.2f}eV")
            elif abs(self.energy * 1000) >= 1:
                lst.append(f"E={self.energy * 1000:.2f}meV")
            else:
                lst.append(f"E={self.energy * 1000:.3e}meV")
        if self.frequencies is None:
            lst.append("NOVIB")
        else:
            lst.append("VIB")
        if self.check_minima():
            lst.append("Minima")
        if self.check_ts():
            lst.append("TS")
        return ",".join(lst)

    @cached_property
    @abstractmethod
    def natoms(self) -> int: ...

    @cached_property
    def E(self) -> float:
        return np.nan if self.energy is None else self.energy

    @cached_property
    def FM(self) -> float:
        return np.nan if self.fmax is None else self.fmax

    @cached_property
    def FQ(self) -> NDArray:
        if self.frequencies is None:
            return np.full(self.natoms * 3, np.nan)
        else:
            return self.frequencies

    @validate_call
    def check_minima(
        self,
        fmax: PositiveFloat = 0.05,
        fqmin: PositiveFloat = 30.0,
    ) -> bool:
        """Whether the system is a minima.

        The criteria for a minima are:
            a) The energy is not None.
            b) The maximum force is not None and less than the threshold.
            c) The frequencies are not None and all the frequencies are
                greater than zero or the absolute value of the maximum
                imaginary frequencies are less than the threshold.
        """
        if self.energy is None:
            return False
        elif self.fmax is None:
            return False
        elif self.fmax > abs(float(fmax)):
            return False
        elif self.frequencies is None:
            return False
        else:
            min_abs_freq = abs(float(fqmin))
            min_freq = np.min(self.frequencies)
            return min_freq > 0 or abs(min_freq) < min_abs_freq

    @validate_call
    def check_ts(
        self,
        fmax: PositiveFloat = 0.1,
        fqmin: PositiveFloat = 20.0,
    ) -> bool:
        """Whether the system is a transition state.

        The criteria for a transition state are:
            a) The energy is not None.
            b) The maximum force is not None and less than the threshold.
            c) The frequencies are not None and the absolute value of the
                maximum imaginary frequencies are less than the threshold
                and there is only one imaginary frequency.
        """
        if self.energy is None:
            return False
        elif self.fmax is None:
            return False
        elif self.fmax > abs(float(fmax)):
            return False
        elif self.frequencies is None:
            return False
        else:
            min_abs_freq = abs(float(fqmin))
            min_freq = self.frequencies[0]
            if min_freq > -min_abs_freq:
                return False
            if self.frequencies.size != 1:
                min_freq_1, min_freq_2 = np.sort(self.frequencies)[:2]
                return min_freq_1 < -min_abs_freq and min_freq_2 > 0
            else:
                return True

    @validate_call
    def _get_thermo(self, fqmin: PositiveFloat = 50.0) -> BaseThermoChem:
        if self.frequencies is None:
            evib = np.array([])
        else:
            cond = self.frequencies >= abs(float(fqmin))
            freqs = self.frequencies[cond]
            evib = freqs * invcm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return HarmonicThermo(
                vib_energies=evib,  # type: ignore
                potentialenergy=self.E,
                ignore_imag_modes=True,
            )

    @validate_call
    def get_vibrational_energy_contribution(
        self,
        fqmin: PositiveFloat = 50.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the change in internal energy due to vibrations.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 50.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the internal energy change 0->T in eV.
        """
        thermo: BaseThermoChem = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_vib_energy_contribution(temperature=temp)

    @validate_call
    def get_vibrational_entropy_contribution(
        self,
        fqmin: PositiveFloat = 50.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the entropy due to vibrations.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 50.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the internal energy change 0->T in eV.
        """
        thermo: BaseThermoChem = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_vib_entropy_contribution(
            temperature=float(temp),
            return_list=False,
        )  # type: ignore

    @validate_call
    def get_enthalpy(
        self,
        fqmin: PositiveFloat = 50.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the enthalpy in in the harmonic approximation.

        Note: In the harmonic approximation, the
            enthalpy is equal the interal energy.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 50.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the enthalpy in eV.
        """
        thermo: HarmonicThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_internal_energy(temperature=temp, verbose=False)

    @validate_call
    def get_entropy(
        self,
        fqmin: PositiveFloat = 50.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the entropy in in the harmonic approximation.

        Note: In the harmonic approximation, the
            entropy is equal the vibrational entropy.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 50.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the entropy in eV/K.
        """
        thermo: HarmonicThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_entropy(
            temperature=float(temp),
            verbose=False,
        )  # type: ignore

    @validate_call
    def get_free_energy(
        self,
        fqmin: PositiveFloat = 50.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the free energy.

        Note:
            a) In the harmonic approximation, the free energy
                is equal the Helmholtz free energy.
            b) In the ideal gas approximation, the free energy
                is equal the Gibbs free energy.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 50.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the Helmholtz/Gibbs free energy in eV.
        """
        thermo: HarmonicThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_helmholtz_energy(temperature=temp, verbose=False)


class Matter(OurBaseModel):
    numbers: Annotated[NDArray, numpy_validator(int)]

    def __len__(self) -> int:
        return self.numbers.shape[0]

    @override
    def _string(self) -> str:
        return self.formula

    @cached_property
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


class Structure(Matter, Box, Energetics):
    positions: Annotated[NDArray, numpy_validator(float, (-1, 3))]

    @model_validator(mode="after")
    def __check_positions(self) -> Self:
        assert self.positions.shape == (self.natoms, 3), self.positions.shape
        return self

    @override
    def _string(self) -> str:
        return ",".join(
            [
                Matter._string(self),
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

    @staticmethod
    def _ase2dct(atoms: Atoms, **kw) -> dict[str, Any]:
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
        return dct

    @classmethod
    def from_ase(cls, atoms: Atoms, **kw) -> Self:
        dct = cls._ase2dct(atoms=atoms, **kw)
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
                    | Matter.__pydantic_fields__.keys()
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


def test_class() -> None:
    for cls in (Box, Energetics, Matter, Structure):
        print(cls.__name__)
        assert len(cls.__abstractmethods__) == 0, cls.__abstractmethods__


def test_Structure() -> None:
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule("CH4")
    atoms.calc = EMT()
    # atoms.get_potential_energy()
    struct = Structure.from_ase(atoms)
    new = Structure.from_dict(struct.to_dict())
    print(struct, new)
    assert new == struct
