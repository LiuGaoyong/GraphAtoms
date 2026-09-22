"""The steps for the on-the-fly KMC simulation."""

from time import perf_counter
from typing import override

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from pydantic import BaseModel

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner._base import ExplorationABC
from graphatoms.enterpoint.runner._helper import helper_apply, helper_match

__all__ = ["OTFKMC"]


class OTFKMCInfo(BaseModel):
    time: float = 0.0
    energy: float = 0
    energy_real: float = np.nan
    select_rxn_key: str = "Initial"
    select_rxn_forward: bool = True
    cost_exploration: float = 0
    cost_matching: float = 0
    cost_other: float = 0


class OTFKMC(ExplorationABC):
    """The class for the on-the-fly KMC simulation."""

    @override
    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.__traj_path = self.path.joinpath("otfkmc.traj")
        self.__info_path = self.path.joinpath("info.csv")
        if not self.config.restart:
            self.__traj = TrajectoryWriter(self.__traj_path, mode="w")
            self.__df_data: list[OTFKMCInfo] = [OTFKMCInfo()]
            self.__atoms: Atoms | None = None
        else:
            self.__traj = TrajectoryWriter(self.__traj_path, mode="a")
            self.__atoms = read(self.__traj_path, index=-1)  # type:ignore
            assert isinstance(self.__atoms, Atoms)
            self.__df_data = [
                OTFKMCInfo(**row.to_dict())  # type: ignore
                for _, row in pd.read_csv(self.__info_path).iterrows()
            ]
        # Note: the `istep=len(self.__df_data) - 1` is the current step number

    @override
    def run(self) -> None:
        """Run the simulation for the given number of steps."""

        atoms: Atoms | None = self.__atoms
        df_data: list[OTFKMCInfo] = self.__df_data
        self.istep: int = len(df_data) - 1

        if self.istep == 0:
            system = self.get_system_for(None)
            self.__traj.write(
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

            # -------------------------------------------
            # prepare: System, Persistence, Criteria
            # -------------------------------------------
            system = self.get_system_for(atoms)
            data = [i.model_dump() for i in df_data]
            pd.DataFrame(data).to_csv(self.__info_path)
            self.__traj.write(system.to_ase())
            if df_data[-1].time > float(self.config.max_times):
                self.logger.info(
                    self._reformat_message(
                        f"Max time {self.config.max_times:.2f}"
                        + f"(now={df_data[-1].time:.2f}) is "
                        + "reached, stop the KMC simulation."
                    )
                )
                self.__traj.close()
                break
            elif self.istep >= int(self.config.max_steps):
                self.logger.info(
                    self._reformat_message(
                        f"Max steps {self.config.max_steps} is "
                        + "reached, stop the KMC simulation."
                    )
                )
                self.__traj.close()
                break

            # -------------------------------------------
            # 1-2 step: analyze the system & exploration
            # -------------------------------------------
            start = perf_counter()
            self.explore(system)
            end = perf_counter()
            otfkmc_info.cost_exploration = end - start
            msg = f"Exploration finished by {end - start:.2f} s"
            self.logger.info(self._reformat_message(msg))
            self.logger.info(self._reformat_message("-" * self._log_length))

            # -------------------------------------------
            # 3 step: graph matching
            # -------------------------------------------
            # TODO: deduplicate the matching submission
            futures: list = []
            start, nsubmit = perf_counter(), 0
            for rxn_key in self.network.metadata.table.key_rxn:
                for rxn_is_forward in [True, False]:
                    nsubmit += 1
                    futures.append(
                        self.executor.submit(
                            helper_match,
                            rxn_key=rxn_key,
                            system=system,
                            rxnet=self.network,
                            rxn_is_forward=rxn_is_forward,
                        )
                    )
            matching_dct: dict[tuple[str, bool], np.ndarray] = {}
            while len(futures) > 0:
                _result, futures = self.executor.wait(futures)  # type: ignore
                rxn_key, rxn_is_forward, single_match_res = _result
                if single_match_res is not None:
                    if not isinstance(single_match_res, np.ndarray):
                        msg = "Matching result is not a numpy array."
                        msg += f"Its type is {type(single_match_res)}."
                        self.logger.error(self._reformat_message(msg))
                        raise AssertionError(msg)
                    matching_dct[(rxn_key, rxn_is_forward)] = single_match_res
            otfkmc_info.cost_matching = perf_counter() - start
            msg = f"Matching finished for {nsubmit} reactions "
            msg += f"by {perf_counter() - start:.2f} seconds, "
            msg += f"got {len(matching_dct)} reactions can be applied."
            self.logger.info(self._reformat_message(msg))
            if len(matching_dct) == 0:
                msg = "No reaction can be applied."
                self.logger.error(self._reformat_message(msg))
                raise AssertionError(msg)
            self.logger.info(self._reformat_message("-" * self._log_length))

            # -------------------------------------------
            # 4 step: select a reaction (BKL)
            # -------------------------------------------
            start = perf_counter()
            bkl_result = self.network.metadata.bkl_solver(matching_dct)
            df, selected_rxn_key, rxn_is_forward, dt, rate_tot = bkl_result
            otfkmc_info.select_rxn_key = selected_rxn_key
            otfkmc_info.select_rxn_forward = rxn_is_forward
            otfkmc_info.time = df_data[-1].time + dt
            selected_info, selected_rxn = self.network.read(selected_rxn_key)
            match_mode = matching_dct[(selected_rxn_key, rxn_is_forward)]
            if not rxn_is_forward:
                selected_rxn = selected_rxn.reversed
                selected_info = selected_info.reversed
            if not isinstance(match_mode, np.ndarray):
                msg = "The match_mode is not a numpy array."
                self.logger.error(self._reformat_message(msg))
                raise AssertionError(msg)
            self.logger.info(f"Matched dataframe: \n{df}")
            self.logger.info(f"KMC delta time: {dt} second")
            self.logger.info(f"KMC total rate: {rate_tot} /s")
            self.logger.info(f"Selected reaction is forward: {rxn_is_forward}")
            self.logger.info(f"Selected reaction: {selected_rxn_key}")
            self.logger.info(f"Selected reaction info: {selected_info}")

            # -------------------------------------------
            # 5. update the system
            # -------------------------------------------
            (_, _, atoms, rmsd) = helper_apply(
                system=system,
                match_mode=match_mode,
                rxn_key=selected_rxn_key,
                forward=rxn_is_forward,
                rxnet=self.network,
            )
            self.logger.info(
                f"Apply Rxn {selected_rxn_key} "
                + f"Successfully, RMSD: {rmsd:.4f}.\n {selected_info}"
            )
            otfkmc_info.cost_other = perf_counter() - start
            otfkmc_info.energy = df_data[-1].energy + selected_info.dE
            df_data.append(otfkmc_info)  # `istep += 1`
            self.logger.info(f"Step {self.istep} End")
            self.logger.info("=" * 50)
