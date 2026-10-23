"""Ray-native on-the-fly KMC runner."""

from __future__ import annotations

import json
from time import perf_counter
from typing import override

import numpy as np
import pandas as pd
import ray
from ase import Atoms
from ase.io import read
from ase.io.trajectory import TrajectoryWriter

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner.general.otfkmc import (
    OTFKMCInfo,
    update_gas_number,
)
from graphatoms.reaction import Adsorption, Desorption, Reaction
from graphatoms.system import System
from graphatoms.utils.bytestool import atomic_write

from ._expl import RayExplorationMixin
from ._runner import RayRunnerBase
from ._tasks import actor_call, submit_task, task_apply, task_match

__all__ = ["RayOTFKMC", "OTFKMCInfo"]


class RayOTFKMC(RayExplorationMixin, RayRunnerBase):
    """On-the-fly KMC using a Ray-native runner and a network actor."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)

    @override
    def _init_run_state(self) -> None:
        self._traj_path = self.path.joinpath("otfkmc.traj")
        self._info_path = self.path.joinpath("info.csv")
        if self.config.restart and self._info_path.exists():
            self._df_data = [
                OTFKMCInfo(**row.to_dict())  # type: ignore
                for _, row in pd.read_csv(self._info_path).iterrows()
            ]
            self._traj = TrajectoryWriter(self._traj_path, mode="a")
            if len(self._df_data) > 0:
                atoms = read(self._traj_path, index=-1)
                assert isinstance(atoms, Atoms)
                self._atoms: Atoms | None = atoms
            else:
                self._atoms = None
        else:
            self._traj = TrajectoryWriter(self._traj_path, mode="w")
            self._df_data: list[OTFKMCInfo] = [OTFKMCInfo()]
            self._atoms = None

    @property
    def istep(self) -> int:
        return len(self._df_data) - 1

    def _solve_matching(self, system: System) -> dict:
        futures = []
        for rxn_key in actor_call(self.actor, "key_rxn"):
            for rxn_is_forward in [True, False]:
                futures.append(
                    submit_task(
                        task_match,
                        actor=self.actor,
                        config=self.config,
                        system=system,
                        rxn_is_forward=rxn_is_forward,
                        rxn_key=rxn_key,
                    )
                )
        matching_dct: dict[tuple[str, bool], np.ndarray] = {}
        while len(futures) > 0:
            done, _ = ray.wait(list(futures), num_returns=1)
            ref = done[0]
            futures.remove(ref)
            rxn_key, rxn_is_forward, single_match_res = ray.get(ref)
            if single_match_res is not None:
                matching_dct[(rxn_key, rxn_is_forward)] = single_match_res
        return matching_dct

    def _apply_reaction(
        self,
        system: System,
        selected_rxn_key: str,
        match_mode: np.ndarray,
        rxn_is_forward: bool,
    ) -> tuple[Atoms | str, float]:
        ref = submit_task(
            task_apply,
            actor=self.actor,
            config=self.config,
            rxn_key=selected_rxn_key,
            system=system,
            match_mode=match_mode,
            raise_when_fail=True,
            forward=rxn_is_forward,
        )
        _, _, apply_result, rmsd = ray.get(ref)
        return apply_result, rmsd

    @override
    def run(self) -> None:
        atoms = self._atoms
        df_data = self._df_data

        if self.istep == 0:
            system = self.get_system_for(None)
            self._traj.write(
                Atoms(
                    numbers=system.numbers,
                    positions=system.positions,
                    pbc=system.is_periodic,
                    cell=system.ase_cell,
                )
            )

        while True:
            otfkmc_info = OTFKMCInfo()
            self.logger.info("=" * self._log_length)
            self.logger.info(f"Step {self.istep} Start")
            self.logger.info("-" * self._log_length)

            system = self.get_system_for(atoms)
            data = [i.model_dump() for i in df_data]
            atomic_write(
                self._info_path,
                writer=lambda p: pd.DataFrame(data).to_csv(p),
            )
            self._traj.write(system.to_ase())
            if df_data[-1].time > float(self.config.max_times):
                self.logger.info(
                    f"Max time {self.config.max_times:.2f}"
                    f"(now={df_data[-1].time:.2f}) is reached, stop."
                )
                self._traj.close()
                break
            elif self.istep >= int(self.config.max_steps):
                self.logger.info(
                    f"Max steps {self.config.max_steps} is reached, stop."
                )
                self._traj.close()
                break

            start = perf_counter()
            expl_success, energy_real = self.explore(system)
            otfkmc_info.energy_real = energy_real
            end = perf_counter()
            otfkmc_info.cost_exploration = end - start
            if not expl_success:
                self.logger.warning(
                    f"Exploration failed by {end - start:.2f} s "
                    f"for system={system.get_key_for_metadata()}"
                )
                continue
            self.logger.info(
                f"Exploration finished by {end - start:.2f} s "
                f"for system={system.get_key_for_metadata()}"
            )
            self.logger.info("-" * self._log_length)

            # 3 step: graph matching
            start = perf_counter()
            matching_dct = self._solve_matching(system)
            otfkmc_info.cost_matching = perf_counter() - start
            self.logger.info(
                f"Matching finished by {perf_counter() - start:.2f} seconds, "
                f"got {len(matching_dct)} reactions can be applied."
            )
            if len(matching_dct) == 0:
                msg = "No reaction can be applied."
                self.logger.error(msg)
                raise AssertionError(msg)

            # 4 step: select a reaction (BKL)
            start = perf_counter()
            bkl_result = actor_call(
                self.actor,
                "bkl_solver",
                matching_dct,
                self.gas_pressure_dct,
            )
            df, selected_rxn_key, rxn_is_forward, dt, rate_tot = bkl_result
            otfkmc_info.select_rxn_key = selected_rxn_key
            otfkmc_info.select_rxn_forward = rxn_is_forward
            otfkmc_info.time = df_data[-1].time + dt
            selected_info, selected_rxn = actor_call(
                self.actor,
                "read_event",
                selected_rxn_key,
            )
            match_mode = matching_dct[(selected_rxn_key, rxn_is_forward)]
            assert isinstance(selected_rxn, Adsorption | Reaction)
            if not rxn_is_forward:
                selected_rxn = selected_rxn.reversed
                selected_info = selected_info.reversed
            assert isinstance(selected_rxn, Adsorption | Reaction | Desorption)
            if not isinstance(match_mode, np.ndarray):
                raise AssertionError("The match_mode is not a numpy array.")
            self.logger.info(f"Matched dataframe: \n{df}")
            self.logger.info(f"KMC delta time: {dt} second")
            self.logger.info(f"KMC total rate: {rate_tot} /s")
            self.logger.info(f"Selected reaction is forward: {rxn_is_forward}")
            self.logger.info(f"Selected reaction: {selected_rxn_key}")
            self.logger.info(f"Selected reaction info: {selected_info}")
            self.logger.info(f"Reaction match indexes: {match_mode}")

            # 5. update the system
            apply_result, rmsd = self._apply_reaction(
                system=system,
                selected_rxn_key=selected_rxn_key,
                match_mode=match_mode,
                rxn_is_forward=rxn_is_forward,
            )
            if not isinstance(apply_result, Atoms):
                msg = "Apply result is not a ase.Atoms object."
                msg += f"Its type is {type(apply_result)}."
                raise AssertionError(msg)
            atoms = apply_result
            if isinstance(selected_rxn, Adsorption | Desorption):
                update_gas_number(
                    selected_rxn,
                    self.gas_number_dct,
                    self.logger,
                )
            self.logger.info(
                f"Apply Rxn {selected_rxn_key} Successfully, "
                f"RMSD: {rmsd:.4f}.\n {selected_info}"
            )
            otfkmc_info.cost_other = perf_counter() - start
            otfkmc_info.energy = df_data[-1].energy + selected_info.dE
            otfkmc_info.env_pressure_dct = json.dumps(self.gas_pressure_dct)
            df_data.append(otfkmc_info)
            self.logger.info(f"Step {self.istep} End")
            self.logger.info("=" * 50)
