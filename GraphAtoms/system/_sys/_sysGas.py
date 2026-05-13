import warnings
from functools import cached_property

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
from typing_extensions import Self, override

from ...utils import rdutils as rdtool
from ._sysAllThing import SysGraph


class Gas(SysGraph):
    @model_validator(mode="after")
    def __some_keys_should_be_none(self) -> Self:
        msg = "The `{:s}` should be None for System."
        assert self.is_outer is None, msg.format("is_outer")
        assert self.move_fix_tag is None, msg.format("move_fix_tag")
        assert self.coordination is None, msg.format("coordination")
        assert self.is_gas, "The `is_gas` should be True for Gas."
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

    @classmethod
    def from_molecule(cls, name: str, **kw) -> Self:
        return cls.from_ase(molecule(name), **kw)

    @classmethod
    def new_from_smiles(cls, smiles: str, **kw) -> Self:
        rdmol = rdtool.smiles2rdmol(smiles)
        atoms: Atoms = rdtool.rdmol2ase(rdmol)
        return cls.from_ase(atoms, **kw)

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
