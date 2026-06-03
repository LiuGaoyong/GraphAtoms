from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from pydantic import BaseModel
from typing_extensions import override

from ...commands.runner3.abc import SimState as MonteCarloState


@dataclass(slots=True)
class CanonicalState(MonteCarloState):
    U: float = 0.0

    @override
    def __post_init__(self) -> None:
        MonteCarloState.__post_init__(self)
        self.U = float(self.U)


@dataclass(slots=True)
class GrandCanonicalState(CanonicalState):
    N: int = 0

    @override
    def __post_init__(self) -> None:
        self.N = len(self.atoms)
        super().__post_init__()


@dataclass(slots=True)
class IsothermalIsobaricState(CanonicalState):
    V: float = -1

    @override
    def __post_init__(self) -> None:
        if self.V < 0:
            self.V = self.atoms.cell.volume
        if self.V < 1e-3:
            raise ValueError(
                "The volume is too small. The Cell "
                f"Matrix is:\n {self.atoms.cell}"
            )
        super().__post_init__()


class MonteCarloRecord(BaseModel, arbitrary_types_allowed=True):
    """A class to store the record of the Monte Carlo move."""

    how_to_change: str = "---"
    energy_change: float = np.nan
    state_old: CanonicalState | None = None
    state_new: CanonicalState | None = None
    random_number: float = np.random.rand()
    accept_probability: float = np.nan
    fmax_old: float = np.nan
    fmax_new: float = np.nan
    cost: float = np.nan


class MonteCarloMove(ABC):
    SUPPORT_ENSEMBLE: tuple[Literal["NVT", "NPT", "uVT"], ...] = (
        "NVT",
        "NPT",
        "uVT",
    )

    def __init__(
        self,
        state: MonteCarloState,
        calculator: Calculator | None = None,
    ) -> None:
        self.__t0 = perf_counter()
        if not isinstance(calculator, Calculator):
            if not isinstance(state.atoms.calc, Calculator):
                raise ValueError("The calculator is not set")
            calculator = state.atoms.calc
            calculator.results.clear()
        else:
            state.atoms.calc = calculator
        if self.SUPPORT_ENSEMBLE[0] == "NVT":
            assert isinstance(state, CanonicalState)
        elif self.SUPPORT_ENSEMBLE[0] == "NPT":
            assert isinstance(state, GrandCanonicalState)
        elif self.SUPPORT_ENSEMBLE[0] == "uVT":
            assert isinstance(state, IsothermalIsobaricState)
        else:
            raise ValueError(
                f"The ensemble {self.SUPPORT_ENSEMBLE[0]} is not supported."
            )
        self.record = MonteCarloRecord()
        self.record.random_number = np.random.random()
        self.record.state_old = self.state = state
        self.atoms = state.atoms
        self.calc = calculator

    @abstractmethod
    def _get_new_atoms(self, **kwargs) -> tuple[Atoms, str]:
        """Return the new atoms and the `how to change` string."""

    @abstractmethod
    def _check_accept(self, *, new_atoms: Atoms, **kwargs) -> bool:
        """Calculate the accept probability & check accept or reject."""

    def __call__(self, **kwargs) -> None:
        new_atoms, self.record.how_to_change = self._get_new_atoms(**kwargs)
        accept = self._check_accept(new_atoms=new_atoms, **kwargs)
        assert self.record.state_new is not None
        assert self.record.state_old is not None
        assert not np.isnan(self.record.fmax_old)
        assert not np.isnan(self.record.fmax_new)
        assert not np.isnan(self.record.energy_change)
        assert not np.isnan(self.record.accept_probability)
        assert 0 <= self.record.random_number < 1
        if accept:
            self.state = self.record.state_new
        else:
            self.state = self.record.state_old
        self.record.cost = perf_counter() - self.__t0

    @classmethod
    def run(
        cls,
        *,
        state: MonteCarloState,
        calculator: Calculator | None = None,
        **kwargs,
    ) -> tuple[MonteCarloState, MonteCarloRecord]:
        obj = cls(state, calculator)
        obj.__call__(**kwargs)
        return obj.state, obj.record
