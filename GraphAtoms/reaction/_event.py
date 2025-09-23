from functools import cached_property
from typing import override

import numpy as np
import pydantic
from ase import units as U
from ase.data import atomic_masses as ATOMIC_MASSES
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from GraphAtoms.common.base import BaseModel
from GraphAtoms.containner import Cluster, Gas, Graph, System
from GraphAtoms.utils.rotation import kabsch, rotate
from GraphAtoms.utils.string import hash as hash_string


class Event(BaseModel):
    r: Cluster | System
    p: Cluster | System
    ts: Cluster | System
    gas: Gas | None = None

    # type: ignore[prop-decorator] # ignore for mypy
    @cached_property
    def is_adsorption(self) -> bool:
        return self.gas is not None and self.p.natoms > self.r.natoms

    # type: ignore[prop-decorator] # ignore for mypy
    @cached_property
    def is_desorption(self) -> bool:
        return self.gas is not None and self.p.natoms < self.r.natoms

    # type: ignore[prop-decorator] # ignore for mypy
    @cached_property
    def is_reaction(self) -> bool:
        return self.gas is None

    # type: ignore[prop-decorator] # ignore for mypy
    @cached_property
    def hash(self) -> str:
        features = sorted([self.r.hash, self.p.hash]) + [self.ts.hash]
        features.append("nogas" if self.gas is None else self.gas.hash)
        return hash_string(",".join(features), digest_size=8)

    @property
    def forward(self) -> Self:
        if sorted([self.r.hash, self.p.hash])[0] == self.r.hash:
            return self
        else:
            return self.__class__(
                r=self.p,
                p=self.r,
                ts=self.ts,
                gas=self.gas,
            )

    @property
    def reverse(self) -> Self:
        if sorted([self.r.hash, self.p.hash])[0] != self.r.hash:
            return self
        else:
            return self.__class__(
                r=self.p,
                p=self.r,
                ts=self.ts,
                gas=self.gas,
            )

    @override
    def _string(self) -> str:
        formulas = [
            system.symbols.get_chemical_formula()
            for system in [self.r, self.p, self.ts]
        ]
        gas = (
            "none"
            if self.gas is None
            else "_".join(
                [
                    self.gas.symbols.get_chemical_formula(),
                    self.gas.hash,
                ]
            )
        )
        return (
            f"r={'_'.join([formulas[0], self.r.hash])},"
            f"p={'_'.join([formulas[1], self.p.hash])},"
            f"ts={'_'.join([formulas[2], self.ts.hash])},"
            f"gas={gas}"
        )

    @pydantic.model_validator(mode="after")
    def __check_natoms(self) -> Self:
        natoms = self.ts.natoms
        try:
            if self.gas is not None:
                if natoms == self.r.natoms:
                    assert self.gas.natoms == natoms - self.p.natoms
                elif natoms == self.p.natoms:
                    assert self.gas.natoms == natoms - self.r.natoms
                else:
                    raise Exception()
            else:
                assert self.r.natoms == self.p.natoms == natoms
        except Exception:
            raise RuntimeError(
                f"Invalid natoms: r={self.r.natoms},"
                f"p={self.p.natoms},ts={self.ts.natoms},"
                f"gas={self.gas.natoms if self.gas is not None else 'None'}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def __check_type(self) -> Self:
        cls = type(self.ts)
        try:
            assert issubclass(cls, Graph)
            assert isinstance(self.r, cls)
            assert isinstance(self.p, cls)
        except Exception:
            raise TypeError(
                f"Invalid type: r={type(self.r)},p={type(self.p)},"
                f"ts={cls}. Only Cluster and System are allowed."
            )
        return self

    @override
    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.hash == other.hash

    def get_active_energy(self) -> float:
        assert self.ts.energy is not None, "ts.energy must be set."
        assert self.r.energy is not None, "r.energy must be set."
        if self.gas is not None:
            assert self.gas.energy is not None, "gas.energy must be set."
            if self.is_adsorption:
                return self.ts.energy - self.r.energy - self.gas.energy
        return self.ts.energy - self.r.energy

    def get_active_free_energy(self, temperature: float) -> float:
        result = self.ts.get_free_energy(temperature)
        result -= self.r.get_free_energy(temperature)
        if self.gas is not None:
            if self.is_adsorption:
                result -= self.gas.get_free_energy(temperature)
        return result

    def get_rate_constant(
        self,
        temperature: float,
        use_collision_theory: bool = True,
    ) -> float:
        kBT = float(U.kB * temperature)  # kB * T in eV
        planck = float(U._hplanck / U.J)  # Plank'area constant in eV * s
        expfactor = np.exp(-self.get_active_free_energy(temperature) / kBT)
        if use_collision_theory and not self.is_reaction:
            area = float(self.__get_area_for_adsorption())  # in Å^2
            assert self.gas is not None, "gas must be set for adsorption."
            assert self.gas.sticking is not None, "Invalid sticking factor."
            gas_mass = np.sum(ATOMIC_MASSES[self.gas.numbers])  # type: ignore
            assert not np.isnan(area), "Invalid area for adsorption."
            lmd = planck / np.sqrt(2 * np.pi * gas_mass * kBT)
            if self.is_adsorption:
                return area * self.gas.sticking * lmd / planck
            else:
                denominator = planck * lmd**2
                numerator = area * self.gas.sticking * kBT
                return expfactor * numerator / denominator
        else:
            return kBT / planck * expfactor  # in 1/s

    def __get_area_for_adsorption(self) -> float:
        if self.gas is None or self.is_reaction:
            return np.nan
        else:
            s_gas = self.gas.get_atomic_sasa()
            s_r = self.r.get_atomic_sasa()
            s_p = self.p.get_atomic_sasa()
            if len(s_r) < len(s_p):
                safterads = np.asarray(s_p)
                sbeforeads = np.append(s_r, s_gas)
            else:
                safterads = np.asarray(s_r)
                sbeforeads = np.append(s_p, s_gas)
            assert safterads.ndim == sbeforeads.ndim == 1
            assert safterads.shape == sbeforeads.shape
            return sum((sbeforeads - safterads)[: -len(s_gas)]) / 2

    @pydantic.validate_call(config={"arbitrary_types_allowed": True})
    def get_position_after_occur(
        self,
        system: System,
        matched_indxs: ArrayLike,
    ) -> tuple[np.ndarray, float]:
        """This event occur on the given system."""
        natoms: int = min(self.r.natoms, self.p.natoms)
        refgeom: np.ndarray = self.r.positions[:natoms, :]
        matched_indxs = np.asarray(matched_indxs)[:natoms]
        _i = np.vectorize(lambda x: np.argwhere(matched_indxs == x).item())(
            np.arange(natoms)
        )
        rot, t, rmsd = kabsch(
            A=refgeom,
            B=system.positions[_i, :],
        )
        assert isinstance(rot, Rotation)
        rot_inv, t_inv = rot.inv(), -t

        # 1. geom --> geom reactant
        geom = rotate(system.positions.copy(), rot) + t
        # 2. geom reactant --> geom product
        geom[_i, :] += self.p.positions[:natoms, :] - refgeom
        # 3. geom product --> result
        geom: np.ndarray = rotate(geom, rot_inv) + t_inv
        return geom, rmsd


class Reaction(Event):
    @pydantic.model_validator(mode="after")
    def __check(self) -> Self:
        assert self.is_reaction, "is_reaction must be True for Reaction."
        try:
            np.testing.assert_array_equal(self.r.numbers, self.ts.numbers)
            np.testing.assert_array_equal(self.p.numbers, self.ts.numbers)
        except Exception:
            raise ValueError("Numbers of reactant and product must be same.")
        return self

    @override
    def _string(self) -> str:
        return (
            f"{self.ts.symbols.get_chemical_formula()},"
            f"r={self.r.hash},p={self.p.hash},"
            f"ts={self.ts.hash},"
        )


class Adsorption(Event):
    @property
    @override
    def forward(self) -> Self:
        return self

    @property
    @override
    def reverse(self) -> "Desorption":
        return Desorption(
            r=self.p,
            p=self.r,
            ts=self.ts,
            gas=self.gas,
        )

    @pydantic.model_validator(mode="after")
    def __check(self) -> Self:
        assert self.is_adsorption, "is_adsorption must be True for Adsorption."
        try:
            assert self.gas is not None
            np.testing.assert_array_equal(self.p.numbers, self.ts.numbers)
            np.testing.assert_array_equal(
                np.append(self.r.numbers, self.gas.numbers),
                self.ts.numbers,
            )
        except Exception:
            raise ValueError(
                "The set sum of reactant and gas must equal to "
                "transition state and product."
            )
        return self


class Desorption(Event):
    @pydantic.model_validator(mode="after")
    def __check(self) -> Self:
        assert self.is_desorption, "is_desorption must be True for Desorption."
        try:
            assert self.gas is not None
            np.testing.assert_array_equal(self.r.numbers, self.ts.numbers)
            np.testing.assert_array_equal(
                np.append(self.p.numbers, self.gas.numbers),
                self.ts.numbers,
            )
        except Exception:
            raise ValueError(
                "The set sum of product and gas must equal to "
                "transition state and reactant."
            )
        return self

    @property
    @override
    def forward(self) -> Self:
        return self

    @property
    @override
    def reverse(self) -> "Adsorption":
        return Adsorption(
            r=self.p,
            p=self.r,
            ts=self.ts,
            gas=self.gas,
        )
