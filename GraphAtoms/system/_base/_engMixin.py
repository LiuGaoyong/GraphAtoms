import warnings
from abc import abstractmethod
from functools import cached_property
from typing import Annotated

import numpy as np
from ase.thermochemistry import BaseThermoChem, HarmonicThermo
from ase.units import invcm
from pydantic import NonNegativeFloat, PositiveFloat, validate_call

from GraphAtoms.dataclasses import (
    NDArray,
    OurFrozenModel,
    numpy_validator,
)


class EnergeticsMixin(OurFrozenModel):
    frequencies: Annotated[NDArray, numpy_validator()] | None = None
    fmax: NonNegativeFloat | None = None
    energy: float | None = None

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
