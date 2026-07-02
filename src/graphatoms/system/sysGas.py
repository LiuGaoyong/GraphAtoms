import warnings
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Self, override

import numpy as np
import pydantic
from ase import Atoms
from ase.build import molecule
from ase.geometry import get_angles
from ase.thermochemistry import _GEOMETRY_OPTIONS as IDEAL_GAS_GEOMETRY_OPTIONS
from ase.thermochemistry import BaseThermoChem, IdealGasThermo
from ase.units import invcm
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
    validate_call,
)

from graphatoms.system.system import System
from graphatoms.utils import rdutils as rdtool


class Gas(System):
    """Gas molecule with thermochemistry support.

    Such as enthalpy, entropy, free energy.
    """

    @model_validator(mode="after")
    def __some_keys_should_be_none(self) -> Self:
        assert self.is_gas, "The `is_gas` should be True for Gas."
        msg = "The `{:s}` should be None for " + f"{self.__class__.__name__}."
        assert self.move_fix_tag is None, msg.format("move_fix_tag")
        assert self.is_outer is None, msg.format("is_outer")
        if self.is_adsorbate is None:
            arr = np.ones_like(self.numbers, dtype=bool)
            object.__setattr__(self, "is_adsorbate", arr)
        return self

    @cached_property
    def smiles(self) -> str:
        """Return the canonical SMILES."""
        return rdtool.get_smiles(
            numbers=self.numbers,
            source=self.source,
            target=self.target,
            canonical=True,
        )

    @override
    @classmethod
    def from_ase(  # type: ignore
        cls,
        atoms: Atoms,
        *,
        sticking: float = 1.0,
        pressure: float = 101325.0,
        parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        return super().from_ase(
            atoms=atoms,
            parse_bonds=parse_bonds,
            parse_bonds_order=parse_bonds_order,
            parse_bonds_distance=parse_bonds_distance,
            **(
                kwargs
                | dict(
                    sticking=sticking,
                    pressure=pressure,
                )
            ),
        )

    @classmethod
    def from_molecule(
        cls,
        name: str,
        *,
        sticking: float = 1.0,
        pressure: float = 101325.0,
        parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        return cls.from_ase(
            atoms=molecule(name),
            sticking=sticking,
            pressure=pressure,
            parse_bonds=parse_bonds,
            parse_bonds_distance=parse_bonds_distance,
            parse_bonds_order=parse_bonds_order,
            **kwargs,
        )

    @classmethod
    def new_from_smiles(
        cls,
        smiles: str,
        *,
        sticking: float = 1.0,
        pressure: float = 101325.0,
        parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
        parse_bonds_distance: bool = False,
        parse_bonds_order: bool = False,
        **kwargs,
    ) -> Self:
        rdmol = rdtool.smiles2rdmol(smiles)
        atoms: Atoms = rdtool.rdmol2ase(rdmol)
        return cls.from_ase(
            atoms=atoms,
            sticking=sticking,
            pressure=pressure,
            parse_bonds=parse_bonds,
            parse_bonds_distance=parse_bonds_distance,
            parse_bonds_order=parse_bonds_order,
            **kwargs,
        )

    @property
    def __geometry_type(self) -> IDEAL_GAS_GEOMETRY_OPTIONS:
        na, R = self.natoms, self.positions
        if na == 1:
            geometry_type = "monatomic"
        elif na == 2:
            geometry_type = "linear"
        else:
            r01 = R[1] - R[0]  # vector
            r02 = R[2:] - R[0]  # matrix
            r03 = np.vstack([r01] * (na - 2))
            angles = get_angles(r02, r03)
            if np.all(angles <= 1e-3):
                geometry_type = "linear"
            else:
                geometry_type = "nonlinear"
        return geometry_type

    @property
    def __spin(self) -> pydantic.NonNegativeFloat:
        # the total electronic spin.
        #   0   for molecules in which all electronsare paired;
        #   0.5 for a free radical with a single unpaired electron;
        #   1.0 for a triplet with two unpaired electrons, such as O2.
        na, Z = self.natoms, self.numbers
        if na == 2 and np.all(Z == 8):  # "O2" is triplet and spin is 1
            return 1.0
        else:
            return 0.0

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
            return IdealGasThermo(
                vib_energies=evib,  # type: ignore
                geometry=self.__geometry_type,
                symmetrynumber=self.nsymmetry,
                potentialenergy=self.E,
                ignore_imag_modes=True,
                natoms=self.natoms,
                spin=self.__spin,
            )

    @pydantic.validate_call
    @override
    def get_enthalpy(
        self,
        fqmin: PositiveFloat = 10.0,
        temp: NonNegativeFloat = 300,
    ) -> float:
        """Calculates the enthalpy in in the ideal gas approximation.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 10.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the enthalpy in eV.
        """
        thermo: IdealGasThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_enthalpy(temperature=temp, verbose=False)

    @pydantic.validate_call
    @override
    def get_entropy(
        self,
        fqmin: PositiveFloat = 10.0,
        temp: NonNegativeFloat = 300,
        pressure: NonNegativeFloat = 101325,
    ) -> float:
        """Calculates the entropy in in the ideal gas approximation.

        Args:
            fqmin (PositiveFloat, optional):
                a frequency threshold in cm^-1. Defaults to 10.0.
            temp (NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.
            pressure (pydantic.NonNegativeFloat, optional):
                a pressure given in Pa. Defaults to 101325.

        Returns:
            float: the entropy in eV/K.
        """
        thermo: IdealGasThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_entropy(
            temperature=float(temp),
            pressure=pressure,
            verbose=False,
        )

    @pydantic.validate_call
    @override
    def get_free_energy(
        self,
        fqmin: PositiveFloat = 10.0,
        temp: NonNegativeFloat = 300,
        pressure: NonNegativeFloat = 101325,
    ) -> float:
        thermo: IdealGasThermo = self._get_thermo(fqmin=fqmin)  # type: ignore
        return thermo.get_gibbs_energy(
            temperature=float(temp),
            pressure=pressure,
            verbose=False,
        )
