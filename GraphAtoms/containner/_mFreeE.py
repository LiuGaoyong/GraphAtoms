"""Calculate free energy on the given temperature."""

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from ase.geometry import get_angles
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.units import invcm

from GraphAtoms.containner._atomic import TOTAL_KEY


class FreeEnergyMixin(ABC):
    @property
    @abstractmethod
    def THERMO_ATOMS(self) -> Atoms: ...

    @property
    def __energy(self) -> float:
        return float(self.THERMO_ATOMS.info.get(TOTAL_KEY.ENERGY, np.nan))

    @property
    def __frequencies(self) -> np.ndarray:
        v = self.THERMO_ATOMS.info.get(TOTAL_KEY.FREQS, [])
        return np.asarray(v, dtype=float).flatten()

    @property
    def __vib_energies_positive(self) -> np.ndarray:
        return self.__vib_energies[self.__vib_energies > 1e-5]

    @property
    def __vib_energies(self) -> np.ndarray:
        return self.__frequencies * invcm

    @property
    def __thermo(self) -> HarmonicThermo:
        return HarmonicThermo(
            vib_energies=self.__vib_energies_positive,
            potentialenergy=self.__energy,
        )

    @property
    def __nsymmetry(self) -> int:
        return self.THERMO_ATOMS.info.get("nsymmetry", 1)

    @property
    def __thermo_gas(self) -> IdealGasThermo:
        vib_energies = self.__vib_energies_positive
        R = self.THERMO_ATOMS.positions
        Z = self.THERMO_ATOMS.numbers
        e = self.__energy
        na = len(Z)

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

        # the total electronic spin.
        #   0   for molecules in which all electronsare paired;
        #   0.5 for a free radical with a single unpaired electron;
        #   1.0 for a triplet with two unpaired electrons, such as O2.
        if na == 2 and np.all(Z == 8):  # "O2" is triplet and spin is 1
            spin = 1.0
        else:
            spin = 0.0

        return IdealGasThermo(
            vib_energies=vib_energies[vib_energies > 1e-5],
            geometry=geometry_type,
            potentialenergy=e,
            atoms=Atoms(numbers=Z, positions=R),
            spin=spin,
            symmetrynumber=self.__nsymmetry,
            ignore_imag_modes=True,
        )

    @property
    def __pressure(self) -> float | None:
        return self.THERMO_ATOMS.info.get(TOTAL_KEY.PRESSURE, None)

    @property
    def __is_gas_system(self) -> bool:
        return self.__pressure is not None

    @property
    def THERMO(self) -> IdealGasThermo | HarmonicThermo:
        return self.__thermo_gas if self.__is_gas_system else self.__thermo

    def get_free_energy(
        self,
        temperature: float,
        verbose: bool = False,
    ) -> float:
        """Calculate free energy on the given temperature."""
        if isinstance(self.THERMO, HarmonicThermo):
            return self.THERMO.get_helmholtz_energy(
                temperature=temperature,
                verbose=verbose,
            )
            HarmonicThermo._vibrational_entropy_contribution
        else:
            return self.THERMO.get_gibbs_energy(
                pressure=self.__pressure,
                temperature=temperature,
                verbose=verbose,
            )
