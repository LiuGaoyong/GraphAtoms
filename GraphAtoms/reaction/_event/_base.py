from functools import cached_property
from typing import override

import numpy as np
from ase import Atoms
from pydantic import model_validator
from typing_extensions import Self

from ...system import DEFAULT_WH_HASH_DEPTH, Gas, SysGraph
from ...utils.bytestool import hash_string
from ...dataclasses import OurFrozenModel
from .._amove import MoveABC

DEFAULT_CHECK_MINIMA_FMAX = 0.05  #    eV/Å
DEFAULT_CHECK_MINIMA_FQMIN = 30.0  #   cm^-1
DEFAULT_CHECK_TS_FQMIN = 50.0  #       cm^-1
DEFAULT_CHECK_TS_FMAX = 0.1  #         eV/Å


class RTGP(OurFrozenModel, MoveABC):
    R: SysGraph
    T: SysGraph | None = None
    G: Gas | None = None
    P: SysGraph

    ########################################################################
    #           Properties for checking the type of the event.
    ########################################################################

    @cached_property
    def is_reaction(self) -> bool:
        """Whether the event is a reaction."""
        n = int(max(len(self.R), len(self.P)))
        return self.T is not None and len(self.T) == n

    @cached_property
    def is_reaction_LH(self) -> bool:
        """Whether the event is a reaction based on Langmuir-Hinsher model."""
        return self.is_reaction and self.G is None

    @cached_property
    def is_reaction_ER(self) -> bool:
        """Whether the event is a reaction based on Eley-Rideal model."""
        return (
            self.is_reaction
            and self.G is not None
            and (
                len(self.P) == len(self.R) + len(self.G)  #
                or len(self.R) == len(self.P) + len(self.G)
            )
        )

    @cached_property
    def is_adsorption(self) -> bool:
        """Whether the event is an adsorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.P) == len(self.R) + len(self.G)
        )

    @cached_property
    def is_desorption(self) -> bool:
        """Whether the event is a desorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.R) == len(self.P) + len(self.G)
        )

    ########################################################################
    #                       Validation for the event.
    ########################################################################

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        self.__check_basic()
        self.__chech_gas()
        self.__check_ts()
        return self

    def __check_basic(self) -> None:
        assert isinstance(self.R, self.P.__class__), (
            "The `R` and `P` should be of the same class."
        )
        assert self.R.check_minima(
            fmax=DEFAULT_CHECK_MINIMA_FMAX,
            fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
        ), "The reactant should be a minima."
        assert self.P.check_minima(
            fmax=DEFAULT_CHECK_MINIMA_FMAX,
            fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
        ), "The product should be a minima."
        assert self.R.is_connected, "The `R` should be a connected graph."
        assert self.P.is_connected, "The `P` should be a connected graph."
        assert self.R.is_periodic == self.P.is_periodic, (
            "The `R` and `P` should be of the same periodicity."
        )
        assert self.R.is_orthorhombic == self.P.is_orthorhombic, (
            "The `R` and `P` should be of the same orthorhombicity."
        )
        assert np.allclose(self.R.ase_cell, self.P.ase_cell), (
            "The `R` and `P` should have the same cell."
        )
        assert self.R.hash != self.P.hash, (
            "The `R` and `P` should have different hash."
        )

        assert any(i is not None for i in [self.T, self.G]), (
            "At least one of `T` and `G` should be not None."
        )
        assert any(
            [
                self.is_reaction_LH,
                self.is_reaction_ER,
                self.is_adsorption,
                self.is_desorption,
            ]
        ), (
            "The event should be either a reaction based on "
            "Langmuir-Hinsher model, a reaction based on "
            "Eley-Rideal model, an adsorption or a desorption."
        )

    def __chech_gas(self) -> None:
        if self.G is not None:
            assert self.G.check_minima(
                fmax=DEFAULT_CHECK_MINIMA_FMAX,
                fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
            ), "The gas should be a minima."
            n = abs(len(self.P) - len(self.R))
            assert len(self.G) == int(n), (
                "The number of gas atoms should match the difference in "
                "the number of atoms between the product and reactant."
            )

            small, big = sorted([self.R, self.P], key=lambda x: len(x))
            z: np.ndarray = np.append(small.numbers, self.G.numbers)
            assert np.array_equal(z, big.numbers), (
                "The combined numbers of the small `R/P` and "
                "gas should match the numbers of the big `R/P`."
            )

    def __check_ts(self) -> None:
        if self.T is not None:
            assert self.T.check_ts(
                fmax=DEFAULT_CHECK_TS_FMAX,
                fqmin=DEFAULT_CHECK_TS_FQMIN,
            ), "The `T` should be a transition state."
            assert isinstance(self.T, self.R.__class__), (
                "The `R` and `T` should be of the same class."
            )
            assert len(self.T) == max(len(self.R), len(self.P)), (
                "The number of atoms in the transition state "
                "should be equal to the maximum number of "
                "atoms in the reactant and product."
            )

    ########################################################################
    #                   the magic methods for the event.
    ########################################################################

    @override  # for __str__ method of the base class
    def _string(self) -> str:
        r_fml = self.R.symbols.get_chemical_formula("metal")
        p_fml = self.P.symbols.get_chemical_formula("metal")
        before, after = f"{r_fml}:{self.R.hash}", f"{p_fml}:{self.P.hash}"
        if self.G is not None:
            gas_fml = self.G.symbols.get_chemical_formula("metal")
            if len(self.R) < len(self.P):
                before = f"{gas_fml} + {before}"
            else:
                after = f"{gas_fml} + {after}"
        if self.T is None:
            ts = "none"
        else:
            ts = self.T.symbols.get_chemical_formula("metal")
            ts = f"{ts}:{self.T.hash}"
        return f"{before} --> {ts} --> {after}"

    @cached_property
    @override  # for __hash__ method of the base class
    def hash(self) -> str:
        t = self.T.hash if self.T is not None else ""
        g = self.G.hash if self.G is not None else ""
        v = ",".join([*sorted([self.R.hash, self.P.hash]), t, g])
        return hash_string(v, digest_size=DEFAULT_WH_HASH_DEPTH)

    @override
    def __call__(self, atoms: Atoms, *args, **kwargs) -> Atoms:
        r = self.R.__class__.from_ase(atoms, *args, **kwargs)
        raise NotImplementedError()

    def __reversed__(self) -> Self:  # type: ignore
        return self.__class__(R=self.P, G=self.G, T=self.T, P=self.R)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif self.hash == other.hash:
            return False
        else:
            return self.R.hash == other.R.hash
