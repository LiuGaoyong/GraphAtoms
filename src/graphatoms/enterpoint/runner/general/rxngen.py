"""The steps for the on-the-fly KMC simulation."""

from collections import defaultdict
from time import perf_counter
from typing import override

from ase import Atoms
from pydantic import BaseModel

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner.common import helper_apply
from graphatoms.system import System

from ._expl import ExplorationBase

__all__ = ["ReactionNetworkGenerator"]


class MatchApplyRecorder(BaseModel):
    #          system_key:  (rxn_key, is_forward)
    record: dict[str, set[tuple[str, bool]]] = defaultdict(set)


class ReactionNetworkGenerator(ExplorationBase):
    """The class for the reaction network exploration."""

    @override
    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        fname = "match_apply_recorder.json"
        self.__path = self.path.joinpath(fname)
        if not self.config.restart:
            self.__match_apply_recorder = MatchApplyRecorder()
        else:
            data = self.__path.read_text()
            obj = MatchApplyRecorder.model_validate_json(data)
            self.__match_apply_recorder: MatchApplyRecorder = obj

    @override
    def run(self) -> None:
        """Run the simulation for the given number of steps."""
        system_dct: dict[str, System] = {
            sys.get_key_for_metadata(use_positions_uuid=False): sys
            for sys in self.get_batch_system_for()
        }
        for self.__istep in range(self.config.max_steps):
            for sys_key, system in system_dct.items():
                start = perf_counter()
                expl_success, _ = self.explore(system)
                end = perf_counter()
                if expl_success:
                    msg = f"Istep={self.__istep}: Exploration finished by "
                    msg += f"{end - start:.2f} s for system={sys_key}"
                    self.logger.info(self._reformat_message(msg))
                    msg = "-" * self._log_length
                    self.logger.info(self._reformat_message(msg))
                else:
                    msg = f"Istep={self.__istep}: Exploration failed by "
                    msg += f"{end - start:.2f} s for system={sys_key}"
                    self.logger.warning(self._reformat_message(msg))
            system_dct = self.__match_and_apply(system_dct)

    def __match_and_apply(self, dct: dict[str, System]) -> dict[str, System]:
        # ---------------------------------------------
        # submit adsorption tasks to executor
        # ---------------------------------------------
        futures: list = []
        start = perf_counter()
        for sys_key, system in dct.items():
            record = self.__match_apply_recorder.record[sys_key]
            for is_forward in [True, False]:
                for rxn_key in self.network.metadata.table.key_rxn:
                    if (rxn_key, is_forward) not in record:
                        futures.append(
                            self.executor.submit(
                                helper_apply,
                                system=system,
                                rxn_key=rxn_key,
                                forward=is_forward,
                                network=self.network,
                                raise_when_fail=False,
                                match_mode=None,
                            )
                        )
                        record.add((rxn_key, is_forward))
        msg = f"Istep={self.__istep}: Submit {len(futures)} match and"
        msg += f" apply tasks by {perf_counter() - start:.2f} seconds"
        self.logger.info(self._reformat_message(msg))

        # -----------------------------------------------------------
        # wait for the adsorption tasks to finish
        # -----------------------------------------------------------
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            rxn_key, forward, apply_result, rmsd = future_result
            if isinstance(apply_result, Atoms):
                new_system = System.from_ase(
                    apply_result,
                    parse_bonds=self.config.bonds,  # type: ignore
                    parse_atoms_is_outer_or_not=True,
                )
                new_sys_key = new_system.get_key_for_metadata(False)
                dct[new_sys_key] = new_system
                msg = f"Apply for rxn_key={rxn_key}(fwd={forward}) "
                msg += f"Successfully, RMSD={rmsd:.4f}, System={new_sys_key}"
                self.logger.info(self._reformat_message(msg))
            elif isinstance(apply_result, str):
                msg = f"Apply for rxn_key={rxn_key}(fwd={forward}) "
                msg += f"failed. Its error message is '{apply_result}'."
                self.logger.info(self._reformat_message(msg))
            else:
                msg = "Apply result is not a ase.Atoms/str object."
                msg += f"Its type is {type(apply_result)}."
                self.logger.error(self._reformat_message(msg))
                raise AssertionError(msg)
        return dct
