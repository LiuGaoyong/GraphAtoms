from functools import cached_property
from typing import Annotated

import numpy as np
import pydantic
from ase import Atoms, units
from ase.build import molecule
from ase.data import atomic_masses as ATOM_MASS
from ase.geometry import get_angles
from numpy.typing import ArrayLike
from typing_extensions import Self, override

from GraphAtoms.containner._aOther import OTHER_KEY
from GraphAtoms.containner._system import System


class Gas(System):
    """The gas molecular system."""

    sticking: Annotated[float, pydantic.Field(ge=0, le=100)] = 1.0
    pressure: pydantic.PositiveFloat = 101325.0

    @pydantic.model_validator(mode="after")
    def __some_keys_should_xxx2(self) -> Self:
        assert self.is_nonmetal, "is_nonmetal should be True for Gas."
        assert not self.is_periodic, "is_periodic should be False for Gas."
        return self

    @classmethod
    @override
    def from_ase(
        cls,
        atoms: Atoms,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        sticking: float = 1.0,
        pressure: float = 101325.0,
        energy: float | None = None,
        frequencies: ArrayLike | None = None,
        **kw,
    ) -> Self:
        obj = System.from_ase(
            atoms,
            infer_conn,
            infer_order=infer_order,
            plus_factor=plus_factor,
            multiply_factor=multiply_factor,
            charge=charge,
        )
        dct = obj.model_dump(mode="python", exclude_none=True)
        dct["sticking"] = float(sticking)
        dct["pressure"] = float(pressure)
        if energy is not None:
            dct[OTHER_KEY.ENERGY] = float(energy)
        if frequencies is not None:
            dct[OTHER_KEY.FREQS] = np.asarray(frequencies, float).flatten()
        return cls.model_validate(dct)

    @classmethod
    def from_molecule(
        cls,
        name: str,
        sticking: float = 1.0,
        pressure: float = 101325.0,
        energy: float | None = None,
        frequencies: ArrayLike | None = None,
        infer_order: bool = False,
        infer_conn: bool = True,
        **kw,
    ) -> Self:
        return cls.from_ase(
            molecule(name),
            frequencies=frequencies,
            infer_order=infer_order,
            infer_conn=infer_conn,
            pressure=pressure,
            sticking=sticking,
            energy=energy,
            **kw,
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        energy: float,
        frequencies: ArrayLike,
        pressure: float = 101325.0,
    ) -> Self:
        raise NotImplementedError

    @cached_property
    def __geometry_type(self) -> str:
        na, R = self.natoms, self.R
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

    @cached_property
    def __spin(self) -> pydantic.NonNegativeFloat:
        # the total electronic spin.
        #   0   for molecules in which all electronsare paired;
        #   0.5 for a free radical with a single unpaired electron;
        #   1.0 for a triplet with two unpaired electrons, such as O2.
        na, Z = self.natoms, self.Z
        if na == 2 and np.all(Z == 8):  # "O2" is triplet and spin is 1
            return 1.0
        else:
            return 0.0

    @property
    @override
    def vib_energies(self) -> np.ndarray:
        vib_energies: np.ndarray = super().vib_energies
        if self.__geometry_type == "nonlinear":
            vib_energies = vib_energies[-(3 * self.natoms - 6) :]
        elif self.__geometry_type == "linear":
            vib_energies = vib_energies[-(3 * self.natoms - 5) :]
        elif self.__geometry_type == "monatomic":
            vib_energies = np.array([])
        else:
            raise ValueError(f"Unsupported geometry: {self.__geometry_type}")
        return vib_energies

    @pydantic.validate_call
    @override
    def get_enthalpy(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the enthalpy in in the ideal gas approximation.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the enthalpy in eV.
        """
        hvib = self.get_vibrational_energy_contribution(temperature)
        if self.__geometry_type == "nonlinear":  # rotational heat capacity
            Cv_r = 1.5 * units.kB
        elif self.__geometry_type == "linear":
            Cv_r = units.kB
        elif self.__geometry_type == "monatomic":
            Cv_r = 0.0
        else:
            raise ValueError(f"Unknown geometry type: {self.__geometry_type}.")
        Cv_t = 1.5 * units.kB  # translational heat capacity (3-d gas)
        Cp_corr = units.kB  # correction term for heat capacity
        h0 = (Cv_t + Cv_r + Cp_corr) * temperature
        return self.energy_plus_zpe + h0 + hvib

    @pydantic.validate_call
    @override
    def get_entropy(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
        pressure: pydantic.NonNegativeFloat = 101325,
    ) -> float:
        """Calculates the entropy in in the ideal gas approximation.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.
            pressure (pydantic.NonNegativeFloat, optional):
                a pressure given in Pa. Defaults to 101325.

        Returns:
            float: the entropy in eV/K.
        """
        referencepressure = 1e5  # Pa

        # Translational entropy (term inside the log is in SI units).
        mass = sum(ATOM_MASS[self.Z]) * units._amu  # kg/molecule
        pimkT = np.pi * mass * units._k * temperature  # unit: ???
        S_t = (2 * pimkT / units._hplanck**2) ** (3.0 / 2)
        S_t *= units._k * temperature / referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)

        # Rotational entropy (term inside the log is in SI units).
        if self.__geometry_type == "monatomic":
            S_r = 0.0
        else:
            atoms = Atoms(numbers=self.Z, positions=self.R)
            inertias: np.ndarray = atoms.get_moments_of_inertia()  # type: ignore
            inertias *= units._amu / (10.0**10) ** 2  # kg m^2
            x = 8.0 * np.pi**2 * units._k * temperature
            x *= self.nsymmetry / units._hplanck**2
            if self.__geometry_type == "nonlinear":
                S_r = np.sqrt(np.pi * np.prod(inertias)) * x ** (3.0 / 2.0)
            elif self.__geometry_type == "linear":
                S_r = x * max(inertias)  # should be two identical and one zero
            else:
                raise ValueError(
                    f"Unknown geometry type: {self.__geometry_type}."
                )
            S_r = units.kB * (np.log(S_r) + 1.0)

        # Vibrational entropy.
        S_v = self.get_vibrational_entropy_contribution(temperature)

        # Electronic entropy.
        S_e = units.kB * np.log(2 * self.__spin + 1)

        # Pressure correction to translational entropy.
        S_p = -units.kB * np.log(pressure / referencepressure)

        return S_t + S_r + S_v + S_e + S_p
