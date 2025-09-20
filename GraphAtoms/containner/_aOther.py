import warnings
from functools import cached_property
from typing import Annotated

import numpy as np
import pydantic
from ase.cell import Cell
from ase.geometry import cell as cellutils
from ase.thermochemistry import _clean_vib_energies
from ase.units import invcm, kB
from pymatgen.core.lattice import Lattice
from typing_extensions import override

from GraphAtoms.common import NpzPklBaseModel, XxxKeyMixin
from GraphAtoms.utils.ndarray import NDArray, numpy_validator


class __OtherMixinKey(XxxKeyMixin):
    FMAX = "fmax_nonconstraint"
    FMAXC = "fmax_constraint"
    FREQS = "frequencies"
    ENERGY = "energy"
    BOX = "box"


OTHER_KEY = __OtherMixinKey()
__all__ = ["OTHER_KEY", "OtherMixin"]


class _BoxMixin(NpzPklBaseModel):
    box: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None

    @cached_property
    def __par(self) -> tuple[float, float, float, float, float, float]:
        return cellutils.cell_to_cellpar(self.ase_cell.array)  # type: ignore

    @property
    def a(self) -> float:
        return self.__par[0]

    @property
    def b(self) -> float:
        return self.__par[1]

    @property
    def c(self) -> float:
        return self.__par[2]

    @property
    def alpha(self) -> float:
        return self.__par[3]

    @property
    def beta(self) -> float:
        return self.__par[4]

    @property
    def gamma(self) -> float:
        return self.__par[5]

    @override
    def _string(self) -> str:
        return "PBC" if self.is_periodic else "NOPBC"

    @cached_property
    def ase_cell(self) -> Cell:
        return Cell.new(self.box)

    @cached_property
    def pmg_lattice(self) -> Lattice:
        return Lattice(self.ase_cell.array)

    @cached_property
    def is_periodic(self) -> bool:
        if self.box is None:
            return False
        v = np.array([self.a, self.b, self.c])
        return not np.all(np.abs(v) < 1e-7)

    @cached_property
    def is_orthorhombic(self) -> bool:
        if self.box is None:
            return True
        return cellutils.is_orthorhombic(self.box)


class OtherMixin(_BoxMixin):
    frequencies: Annotated[NDArray, numpy_validator()] | None = None
    fmax_nonconstraint: pydantic.NonNegativeFloat | None = None
    fmax_constraint: pydantic.NonNegativeFloat | None = None
    energy: float | None = None

    @override
    def _string(self) -> str:
        lst: list[str] = [super()._string()]
        if self.energy is None:
            lst.append("NOSPE")
        else:
            e = self.energy * 1000  # convert to meV
            lst.append(f"E={int(e) if abs(e) > 1 else f'{e:.3e}'}meV")
        if self.frequencies is None:
            lst.append("NOVIB")
        else:
            lst.append("VIB")
        fmaxs = (self.fmax_constraint, self.fmax_nonconstraint)
        if all(i is not None for i in fmaxs) and self.is_minima:
            lst.append("Minima")
        return ",".join(lst)

    @cached_property
    def is_minima(self) -> bool:
        fmaxs = (self.fmax_constraint, self.fmax_nonconstraint)
        return min((f if f is not None else np.inf) for f in fmaxs) < 0.05

    @property
    def vib_energies(self) -> np.ndarray:  # type: ignore
        if self.frequencies is None:
            result = np.array([], float)
        else:
            result = np.asarray(self.frequencies, float) * invcm  # in eV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return _clean_vib_energies(result, True)[0]  # type: ignore

    @cached_property
    def ZPE(self) -> float:
        """Returns the zero-point vibrational energy correction in eV."""
        return float(np.sum(self.vib_energies) / 2.0)

    @property
    def energy_plus_zpe(self) -> float:
        """Returns the energy plus the zero-point vibrational energy in eV."""
        return np.inf if self.energy is None else self.energy + self.ZPE

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
        if self.energy is None:
            return np.inf
        else:
            v = self.get_vibrational_energy_contribution(temperature)
            return float(self.energy_plus_zpe + v)

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
