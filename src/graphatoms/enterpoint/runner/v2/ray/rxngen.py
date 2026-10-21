"""Ray-native reaction-network generation runner."""

from __future__ import annotations

from time import perf_counter
from typing import override

import ray
from ase import Atoms

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner.general.rxngen import MatchApplyRecorder
from graphatoms.system import System

from ._expl import RayExplorationMixin
from ._runner import RayRunnerBase
from ._tasks import actor_call, submit_task, task_apply

__all__ = ["RayReactionNetworkGenerator"]


class RayReactionNetworkGenerator(RayExplorationMixin, RayRunnerBase):
    """Reaction-network generation using the Ray-native runner."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)

    @override
    def _init_run_state(self) -> None:
        fname = "match_apply_recorder.json"
        self._match_apply_path = self.path.joinpath(fname)
        if not self.config.restart:
            self._match_apply_recorder = MatchApplyRecorder()
            self._match_apply_recorder.persist(self._match_apply_path)
            self._initial_system_dct: dict[str, System] = {
                sys.get_key_for_metadata(use_positions_uuid=False): sys
                for sys in self.get_batch_system_for()
            }
        else:
            self._match_apply_recorder = (
                MatchApplyRecorder.model_validate_json(
                    json_data=self._match_apply_path.read_text(),
                )
            )
            self._initial_system_dct = self._system_dct_from_db

    @property
    def _explored_systems(self) -> set[str]:
        return set(self._match_apply_recorder.record.keys())

    @property
    def _system_dct_from_db(self) -> dict[str, System]:
        return {
            k: System.from_ase(
                v,
                parse_bonds=self.config.bonds,  # type: ignore
                parse_atoms_is_outer_or_not=True,
            )
            for k, v in actor_call(self.actor, "db_system_items").items()
        }

    @override
    def run(self) -> None:
        system_dct: dict[str, System] = self._initial_system_dct
        for self._istep in range(self.config.max_steps):
            unexplored = {
                k: v
                for k, v in system_dct.items()
                if k not in self._explored_systems
            }
            if len(unexplored) == 0:
                self.logger.info(
                    f"Istep={self._istep}: No system left to explore."
                )
                break
            self.logger.info(
                f"Istep={self._istep}: Explore {len(unexplored)} systems."
            )
            for sys_key, system in unexplored.items():
                start = perf_counter()
                expl_success, _ = self.explore(system)
                end = perf_counter()
                if expl_success:
                    self.logger.info(
                        f"Istep={self._istep}: Exploration finished by "
                        f"{end - start:.2f} s for system={sys_key}"
                    )
                else:
                    self.logger.warning(
                        f"Istep={self._istep}: Exploration failed by "
                        f"{end - start:.2f} s for system={sys_key}"
                    )
            system_dct = self._match_and_apply(system_dct)
        self.logger.info("-" * self._log_length)
        self.logger.info(f"Istep={self._istep}: Simulation finished.")

    def _match_and_apply(
        self,
        dct: dict[str, System],
    ) -> dict[str, System]:
        futures = []
        record = self._match_apply_recorder.record
        for sys_key, system in dct.items():
            rec = record[sys_key]
            for is_forward in [True, False]:
                for rxn_key in actor_call(self.actor, "key_rxn"):
                    if (rxn_key, is_forward) not in rec:
                        futures.append(
                            submit_task(
                                task_apply,
                                actor=self.actor,
                                config=self.config,
                                rxn_key=rxn_key,
                                system=system,
                                match_mode=None,
                                raise_when_fail=False,
                                forward=is_forward,
                            )
                        )
                        rec.add((rxn_key, is_forward))
        self._match_apply_recorder.persist(self._match_apply_path)
        self.logger.info(
            f"Istep={self._istep}: Submit {len(futures)} match and apply tasks."
        )

        while len(futures) > 0:
            done, _ = ray.wait(list(futures), num_returns=1)
            ref = done[0]
            futures.remove(ref)
            rxn_key, forward, apply_result, rmsd = ray.get(ref)
            if isinstance(apply_result, Atoms):
                new_system = System.from_ase(
                    apply_result,
                    parse_bonds=self.config.bonds,  # type: ignore
                    parse_atoms_is_outer_or_not=True,
                )
                new_sys_key = new_system.get_key_for_metadata(False)
                dct[new_sys_key] = new_system
                self.logger.info(
                    f"Apply for rxn_key={rxn_key}(fwd={forward}) "
                    f"Successfully, RMSD={rmsd:.4f}, System={new_sys_key}"
                )
            elif isinstance(apply_result, str):
                self.logger.info(
                    f"Apply for rxn_key={rxn_key}(fwd={forward}) failed. "
                    f"Its error message is '{apply_result}'."
                )
            else:
                raise AssertionError(
                    "Apply result is not a ase.Atoms/str object. "
                    f"Its type is {type(apply_result)}."
                )
        return dct
