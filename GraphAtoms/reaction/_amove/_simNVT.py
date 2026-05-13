from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator
from typing_extensions import override

from ...commands.runner3.abc import SimABC
from ._base import CanonicalState
from ._base import MonteCarloRecord as Record
from ._moveFB import FBDispC
from ._moveSP import SPDispC
from ._moveSwap import Swap, SwapCutoff


class MonteCarloNVT(SimABC):
    def __init__(
        self,
        *,
        atoms: Path | str | None = None,
        calculator: str | Calculator = "emt",
        record_file: str | Path = "record.table",
        trajectory: str | Path = "ase.traj",
        trajectory_interval: int = 1,
        record_interval: int = 1,
        cutoff: float = np.inf,
        workdir: str | Path = ".",
        stepsize_max: float = 0.5,
        temperature: float = 300.0,
        max_steps: int = 100_000_000,
        lambda_for_force_bias: float = np.nan,
        swap_for_connected_pair_only: bool = True,
        swap_interval: int = 100,
    ) -> None:
        super().__init__(
            calculator=calculator,
            atoms=atoms,
            record_interval=record_interval,
            record_file=record_file,
            trajectory=trajectory,
            trajectory_interval=trajectory_interval,
            max_steps=max_steps,
            workdir=workdir,
        )
        self.__swap_interval = swap_interval
        self.__stepsize_max: float = stepsize_max
        self.__temperature: float = temperature
        self.__cutoff: float = cutoff
        self.__fmax: float = np.nan
        self.__isteps: int = 0

        self.__for_single = np.isinf(cutoff)  # move for single or multiple
        if not self.__for_single:
            self.__e = self.atoms.get_potential_energy()
            self.__swap_cls: type[SwapCutoff] = Swap
        else:
            self.__e: float = 0.0
            assert bool(swap_for_connected_pair_only), (
                "Swap for connected pair only is not allowed when cutoff"
                " is not inf (i.e. multi-atom displacement)."
            )
            self.__swap_cls: type[SwapCutoff] = SwapCutoff

        self.__lambda_for_force_bias = float(lambda_for_force_bias)
        if np.isnan(self.__lambda_for_force_bias):
            self.__disp_cls: type[SPDispC] | type[FBDispC] = SPDispC
        elif 0.5 <= self.__lambda_for_force_bias <= 1.0:
            raise NotImplementedError("Force-bias move overflow ???")
            self.__disp_cls: type[SPDispC] | type[FBDispC] = FBDispC
        else:
            raise ValueError(
                "The `lambda_for_force_bias` must be between 0.5 and 1.0"
                f" or it is NaN. But got {self.__lambda_for_force_bias}."
                " The value of nan means that the force bias is not used."
            )

    @override
    def step(self) -> None:
        state = CanonicalState.from_ase(self.atoms, U=self.__e)

        # try swap atoms
        if (
            self.__swap_interval != 0
            and self.nsteps % self.__swap_interval == 0
        ):
            try:
                state, record = self.__swap_cls.run(
                    state=state,
                    calculator=None,
                    temperature=self.__temperature,
                    cutoff=self.__cutoff,
                )
                swap_success = True
            except RuntimeError as e:
                if str(e).startswith("No suitable pair found"):
                    swap_success = False
                else:
                    raise e
        else:
            swap_success = False

        # displace atoms
        if not swap_success:
            kwargs = dict(
                state=state,
                calculator=self.atoms.calc,
                stepsize_max=self.__stepsize_max,
                temperature=self.__temperature,
                cutoff=self.__cutoff,
            )
            if not np.isnan(self.__lambda_for_force_bias):
                kwargs["lambda_param"] = self.__lambda_for_force_bias
            state, record = self.__disp_cls.run(*[], **kwargs)
        assert isinstance(state, CanonicalState)
        assert isinstance(record, Record)  # type: ignore
        assert record.state_old is not None
        assert record.state_new is not None
        assert record.how_to_change != "---"
        assert not np.isnan(record.energy_change)
        assert not np.isnan(record.accept_probability)
        assert not np.isnan(record.fmax_new)
        assert not np.isnan(record.fmax_old)
        assert not np.isnan(record.cost)

        # use Metropolis rule
        self.__cost_dtime = record.cost
        self.__delta_energy = record.energy_change
        self.__boltzmann_factor = record.accept_probability
        self.__random_factor = record.random_number
        self.__how_to_change = record.how_to_change
        self.__accept = state is record.state_new
        if self.__accept:
            self.__isteps += 1
            self.__e = record.state_new.U  # type: ignore
            self.__fmax = record.fmax_new
        else:
            self.__e = record.state_old.U  # type: ignore
            self.__fmax = record.fmax_old
        self.atoms = state.to_ase()

    @override
    def log_header(self) -> list[str]:
        return [
            f"{'N(accept)':>9s}",
            f"{'E':>10s}",
            f"{'dE':>10s}",
            f"{'Boltzmann':>10s}",
            f"{'Random':>10s}",
            f"{'Fmax':>10s}",
            f"{'DtimeCost':>10s}",
            f"{'Accept?':>7s}",
            f"{'(%)':>5s}",
            f"{'How2Change':<24s}",
        ]

    @override
    def log_lst(self) -> list[str]:
        if self.nsteps == 0:
            self.__cost_dtime = np.nan
            self.__boltzmann_factor = np.nan
            self.__random_factor = np.nan
            self.__delta_energy = np.nan
            self.__how_to_change = "---"
            self.__accept = "nan"
            acc = np.nan
        else:
            acc = self.__isteps / self.nsteps * 100

        return [
            f"{self.__isteps:9d}",
            f"{self.__e:10.4f}",
            f"{self.__delta_energy:10.4f}",
            f"{self.__boltzmann_factor:10.4f}",
            f"{self.__random_factor:10.4f}",
            f"{self.__fmax:10.4f}",
            f"{self.__cost_dtime:10.4f}",
            f"{str(self.__accept):>7s}",
            f"{acc:5.1f}",
            f"{self.__how_to_change:<24s}",
        ]
