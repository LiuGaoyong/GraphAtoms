from typing import Self, override

import numpy as np
from ase.units import _e, _hplanck, kB
from pydantic import model_validator

from graphatoms.reaction._event import EventBase

h = _hplanck / _e  # Planck constant in eV
kB = kB  # Boltzmann constant in eV/K


class _Reaction(EventBase):
    @model_validator(mode="after")
    def __check_something(self) -> Self:
        n = int(max(len(self.R), len(self.P)))
        assert self.T is not None, "The transition state must be not None."
        assert n == len(self.T), (
            "The number of atoms must be equal. But got "  #
            f"T={len(self.T)}, R={len(self.R)} "  #
            f"and P={len(self.P)}."
        )
        return self


class ReactionLH(_Reaction):
    """The reaction by Langmuir-Hinshelwood mechanism."""

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert self.G is None, "The gas must be None."
        assert len(self.R) == len(self.P), (
            "The number of atoms must be equal. But got "
            f"R={len(self.R)} and P={len(self.P)}."
        )
        return self

    @override
    def get_Ea(self, temperature: float = 300.0) -> float:
        assert self.T is not None, "The transition state must be not None."
        e_T = self.T.get_free_energy(30, temp=temperature)
        e_R = self.R.get_free_energy(30, temp=temperature)
        return e_T - e_R

    @override
    def get_dE(self, temperature: float = 300.0) -> float:
        e_P = self.P.get_free_energy(30, temp=temperature)
        e_R = self.R.get_free_energy(30, temp=temperature)
        return e_P - e_R

    def get_rate(self, temperature: float = 300.0) -> float:
        exp = np.exp(-self.get_Ea(temperature) / kB * temperature)
        return kB * temperature / h * exp


class ReactionER(_Reaction):
    """The reaction by Eley-Rideal mechanism."""

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        raise NotImplementedError()
        assert self.G is not None, "The gas must be not None."
        assert (
            len(self.P) == len(self.R) + len(self.G)  #
            or len(self.R) == len(self.P) + len(self.G)
        )
        return self


if __name__ == "__main__":
    from scipy import constants

    print(constants.Boltzmann)
    print(constants.Planck)
    print(constants.eV)

    temperature = 300.0  # K
    print(constants.Boltzmann * temperature / constants.Planck)
    print(kB * temperature / h)
