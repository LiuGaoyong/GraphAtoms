from typing import Self, override

import numpy as np
from ase import Atoms
from ase.data import atomic_masses as MASS
from pydantic import model_validator

from graphatoms.reaction._event import EventBase, h, kB
from graphatoms.system.system import System


class Adsorption(EventBase):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert self.G is not None, "The gas must be not None."
        assert self.T is None, "The transition state must be None."
        assert len(self.P) == len(self.R) + len(self.G), (
            "The product state must be the sum of the "  #
            "reactant state and the gas molecule."
        )
        return self

    @property
    @override
    def reversed(self) -> "Desorption":
        return Desorption(R=self.P, G=self.G, T=self.T, P=self.R)

    @override
    def apply(
        self,
        atoms: System | Atoms,
        *args,
        matched_indxs: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Atoms, float]:
        atoms, rmsd = super().apply(atoms, *args, matched_indxs, **kwargs)
        patoms: Atoms = self.P.to_ase(exclude_energetics=True)
        mask = np.arange(len(self.P)) >= len(self.R)
        atoms.extend(patoms[mask])
        return atoms, rmsd

    @override
    def get_Ea(self, *args, **kwargs) -> float:  # type: ignore
        """The adsorption reaction does not have an activation energy.

        Returns:
            np.inf
        """
        return np.inf

    @override
    def get_dE(
        self,
        temperature: float = 300.0,
        *args,
        pressure: float | None = None,
        **kwargs,
    ) -> float:
        assert self.G is not None, "The gas must be not None."
        e_P = self.P.get_free_energy(fqmin=30.0, temp=temperature)
        e_R = self.R.get_free_energy(fqmin=30.0, temp=temperature)
        if pressure is None:
            pressure = self.G.pressure
        assert pressure is not None, "The pressure must be not None."
        e_G = self.G.get_free_energy(
            fqmin=30.0,
            temp=temperature,
            pressure=pressure,
        )
        return e_P - e_R - e_G

    @override
    def get_rate(
        self,
        temperature: float = 300.0,
        *args,
        pressure: float | None = None,
        sticking: float | None = None,
        **kwargs,
    ) -> float:
        """Get the rate of the reaction by the collision theory.

        Ref: Dominic R. Alfonso; Kinetic Monte Carlo Simul-
            ation of CO Adsorption on Sulfur-Covered Pd(100).
            J. Phys. Chem. A  2014, 118, 7306-7313.
        Eq:
                          S*A
            rate = -------------------
                    sqrt(2*pi*m*kB*T)
        """
        assert self.G is not None, "The gas must be not None."
        kBT = kB * temperature
        m = np.sum(MASS[self.G.numbers])
        A = abs(self.G.area + self.R.area - self.P.area) / 2.0

        if sticking is None:
            sticking = self.G.sticking
        assert sticking is not None, (
            "The sticking coefficient must be not None."
        )
        if pressure is None:
            pressure = self.G.pressure
        assert pressure is not None, "The pressure must be not None."
        return (sticking * A * pressure) / np.sqrt(2 * np.pi * m * kBT)


class Desorption(EventBase):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert self.G is not None, "The gas must be not None."
        assert self.T is None, "The transition state must be None."
        assert len(self.R) == len(self.P) + len(self.G), (
            "The reactant state must be the sum of the "  #
            "product state and the gas molecule."
        )
        return self

    @property
    @override
    def reversed(self) -> "Adsorption":
        return Adsorption(R=self.P, G=self.G, T=self.T, P=self.R)

    @override
    def apply(
        self,
        atoms: System | Atoms,
        *args,
        matched_indxs: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Atoms, float]:
        atoms, rmsd = super().apply(atoms, *args, matched_indxs, **kwargs)
        del atoms[np.arange(len(self.R)) >= len(self.P)]
        return atoms, rmsd

    @override
    def get_Ea(self, *args, **kwargs) -> float:  # type: ignore
        """The desorption reaction does not have an activation energy.

        Returns:
            np.inf
        """
        return np.inf

    @override
    def get_dE(
        self,
        temperature: float = 300.0,
        *args,
        pressure: float | None = None,
        **kwargs,
    ) -> float:
        assert self.G is not None, "The gas must be not None."
        e_P = self.P.get_free_energy(fqmin=30.0, temp=temperature)
        e_R = self.R.get_free_energy(fqmin=30.0, temp=temperature)
        if pressure is None:
            pressure = self.G.pressure
        assert pressure is not None, "The pressure must be not None."
        e_G = self.G.get_free_energy(
            fqmin=30.0,
            temp=temperature,
            pressure=pressure,
        )
        return e_R + e_G - e_P

    @override
    def get_rate(
        self,
        temperature: float = 300.0,
        *args,
        pressure: float | None = None,
        sticking: float | None = None,
        **kwargs,
    ) -> float:
        """Get the rate of the reaction by the collision theory.

        Ref: Dominic R. Alfonso; Kinetic Monte Carlo Simul-
            ation of CO Adsorption on Sulfur-Covered Pd(100).
            J. Phys. Chem. A  2014, 118, 7306-7313.
        Eq:
                     S*A*2*pi*m*(kB*T)**2        -dE
            rate = -----------------------*exp(-------)
                            h**3                 kB*T
        """
        assert self.G is not None, "The gas must be not None."
        kBT = kB * temperature
        m = np.sum(MASS[self.G.numbers])
        A = abs(self.G.area + self.R.area - self.P.area) / 2.0

        if pressure is None:
            pressure = self.G.pressure
        assert pressure is not None, "The pressure must be not None."
        dE = self.get_dE(temperature=temperature, pressure=pressure)
        exp = np.exp(-dE / kBT)

        if sticking is None:
            sticking = self.G.sticking
        assert sticking is not None, "The sticking must be not None."
        return sticking * A * 2.0 * np.pi * m * kBT**2 / h**3 * exp
