from functools import cached_property

import numpy as np
import pydantic
from ase.units import invcm, kB
from typing_extensions import Self, override

from GraphAtoms.common import NpzPklBaseModel, XxxKeyMixin
from GraphAtoms.utils.ndarray import NDArray, Shape


class __EnergeticsKey(XxxKeyMixin):
    FMAX = "fmax_nonconstraint"
    FMAXC = "fmax_constraint"
    FREQS = "frequencies"
    ENERGY = "energy"


ENERGETICS_KEY = __EnergeticsKey()
__all__ = ["ENERGETICS_KEY", "Energetics"]


class Energetics(NpzPklBaseModel):
    frequencies: NDArray[Shape["*"], float] = np.array([])  # type: ignore
    fmax_nonconstraint: pydantic.NonNegativeFloat = float("inf")
    fmax_constraint: pydantic.NonNegativeFloat = float("inf")
    energy: float = float("inf")

    @classmethod
    @override
    def _convert(cls) -> dict[str, tuple[tuple, str]]:
        result: dict[str, tuple[tuple, str]] = super()._convert()
        result[ENERGETICS_KEY.FREQS] = ((None,), "float64")
        return result

    @pydantic.model_validator(mode="after")
    def __check_keys(self) -> Self:
        fields = self.__pydantic_fields__.keys()
        assert set(fields) >= set(ENERGETICS_KEY._DICT.values()), (
            "Invalid fields or ENERGETICS_KEY."
        )
        assert set(fields) >= set(self._convert().keys()), (
            "Invalid _convert dictionary."
        )
        return self

    @cached_property
    def is_minima(self) -> bool:
        return min(self.fmax_constraint, self.fmax_nonconstraint) < 0.05

    @cached_property
    def vib_energies(self) -> NDArray[Shape["*"], float]:  # type: ignore
        vib_energies = np.asarray(self.frequencies, float) * invcm  # in eV
        return vib_energies[vib_energies > 1e-5]  # type: ignore

    @cached_property
    def ZPE(self) -> float:
        """Returns the zero-point vibrational energy correction in eV."""
        return float(np.sum(self.vib_energies) / 2.0)

    @pydantic.validate_call
    def get_vibrational_energy_contribution(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the change in internal energy due to vibrations.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the internal energy change 0->T in eV.
        """
        x = self.vib_energies / (kB * temperature)  # type: ignore
        return float(np.sum(self.vib_energies / (np.exp(x) - 1.0)))

    @pydantic.validate_call
    def get_vibrational_entropy_contribution(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the entropy due to vibrations.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the entropy change 0->T in eV/K.
        """
        x = self.vib_energies / (kB * temperature)  # type: ignore
        Sv0 = x / (np.exp(x) - 1.0)
        Sv1 = np.log(1.0 - np.exp(-x))
        return np.sum(kB * (Sv0 - Sv1)).item()

    @pydantic.validate_call
    def get_enthalpy(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the enthalpy in in the harmonic approximation.

        Note: In the harmonic approximation, the
            enthalpy is equal the interal energy.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the enthalpy in eV.
        """
        v = self.get_vibrational_energy_contribution(temperature)
        return self.energy + self.ZPE + v

    @pydantic.validate_call
    def get_entropy(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the entropy in in the harmonic approximation.

        Note: In the harmonic approximation, the
            entropy is equal the vibrational entropy.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the entropy in eV/K.
        """
        return self.get_vibrational_entropy_contribution(temperature)

    @pydantic.validate_call
    def get_free_energy(
        self,
        temperature: pydantic.NonNegativeFloat = 300,
    ) -> float:
        """Calculates the free energy in in the harmonic approximation.

        Note: a) In the harmonic approximation, the free energy is equal the
            Helmholtz free energy. b) In the ideal gas approximation, the
            free energy is equal the Gibbs free energy.

        Args:
            temperature (pydantic.NonNegativeFloat, optional):
                a temperature given in Kelvin. Defaults to 300.

        Returns:
            float: the free energy in eV.
        """
        H = self.get_enthalpy(temperature)
        S = self.get_entropy(temperature)
        return H - temperature * S
