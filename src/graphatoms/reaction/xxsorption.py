from typing import Any, Self, override

import numpy as np
from ase import Atoms
from ase.data import atomic_masses as MASS
from ase.units import _amu as atomic_mass_unit
from ase.units import _e as electron_charge
from pydantic import model_validator

from graphatoms.reaction._event import EventBase, kB


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
    def apply_once(
        self,
        atoms: Atoms,
        matched_indxs: list[int] | np.ndarray,
        info: dict[str, Any] = {},
    ) -> tuple[Atoms, float]:
        """Apply the event once to the system."""
        result, rmsd = super().apply_once(atoms, matched_indxs, info=info)
        patoms: Atoms = self.P.to_ase(exclude_energetics=True)
        mask = np.arange(len(self.P)) >= len(self.R)
        if "is_fix" in result.info:
            result.info["is_fix"] = np.append(
                result.info["is_fix"],
                np.zeros(np.sum(mask), dtype=bool),
            )
        result.info["is_adsorbate"] = np.append(
            result.info.get("is_adsorbate", np.zeros(len(atoms), dtype=bool)),
            np.ones(np.sum(mask), dtype=bool),
        )
        result.info.pop("is_outer", None)
        result.info.pop("is_core", None)
        result.extend(patoms[mask])
        return result, rmsd

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
        """Get the change in energy (in eV) of the adsorption reaction.

        Args:
            temperature (float, optional):
                a temperature given in Kelvin. Defaults to 300.0.
            pressure (float | None, optional):
                a pressure given in Pa. Defaults to None.

        Returns:
            float: the change in energy in eV.
        """
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
        """Get the rate (in 1/s) of the reaction by the collision theory.

        Args:
            temperature (float, optional):
                a temperature given in Kelvin. Defaults to 300.0.
            pressure (float | None, optional):
                a pressure given in Pa. Defaults to None.
            sticking (float | None, optional):
                a sticking coefficient. Defaults to None.

        Returns:
            float: the rate in 1/s.

        Ref: Dominic R. Alfonso; Kinetic Monte Carlo Simul-
            ation of CO Adsorption on Sulfur-Covered Pd(100).
            J. Phys. Chem. A  2014, 118, 7306-7313.
        Eq:
                          S*A
            rate = -------------------
                    sqrt(2*pi*m*kB*T)
        """
        # for Oxygen gas
        # A=4,P=1atm,T=300K --> rate=1.09e8
        assert self.G is not None, "The gas must be not None."
        kBT = kB * temperature  # energy in eV
        kBT *= electron_charge  # Convert to J
        m = np.sum(MASS[self.G.numbers])
        m *= atomic_mass_unit  # Convert amu to kg
        A = abs(self.G.area + self.R.area - self.P.area) / 2.0
        A *= 1e-20  # Convert Å^2 to m^2

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
    def apply_once(
        self,
        atoms: Atoms,
        matched_indxs: list[int] | np.ndarray,
        info: dict[str, Any] = {},
    ) -> tuple[Atoms, float]:
        """Apply the event once to the system."""
        atoms, rmsd = super().apply_once(atoms, matched_indxs, info=info)
        del atoms[np.arange(len(self.R)) >= len(self.P)]
        for k, v in atoms.info.items():
            if k.startswith("is_") and isinstance(v, np.ndarray):
                atoms.info[k] = v[np.arange(len(self.R)) >= len(self.P)]
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

        Eq:
                                   -dE
            rate = rate(ads)*exp(-------)
                                   kB*T
        """
        assert self.G is not None, "The gas must be not None."
        if pressure is None:
            pressure = self.G.pressure
        assert pressure is not None, "The pressure must be not None."
        dE = self.get_dE(temperature=temperature, pressure=pressure)
        exp = np.exp(-dE / kB * temperature)

        if sticking is None:
            sticking = self.G.sticking
        assert sticking is not None, "The sticking must be not None."
        factor = self.reversed.get_rate(
            pressure=1.0e5,  # the standard pressure is 1bar
            temperature=temperature,
            sticking=sticking,
        )
        return factor * exp
