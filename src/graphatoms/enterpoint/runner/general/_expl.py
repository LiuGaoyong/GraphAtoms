import os
from time import perf_counter
from typing import Any

os.environ["LOGURU_FORMAT"] = LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    + " | <level>{level: ^8}</level> | "
    + "<level>{message}</level>"
)  # the length of the line is 37 characters in loguru
# LOGURU_FORMAT = env(
#     "LOGURU_FORMAT",
#     str,
#     "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>"
#     "{line}</cyan> - <level>{message}</level>",
# )

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.runner.common import (
    RunnerABC,
    helper_adsorption,
    helper_dimer,
    helper_optimization,
)
from graphatoms.system import Cluster, Gas, SysGraph, System
from graphatoms.utils import asetools
from graphatoms.utils.adsorption import Helper
from graphatoms.utils.parser import hydra_parse


class ExplorationBase(RunnerABC):
    def _first_step(
        self,
        system: System | Atoms | None,
    ) -> tuple[dict[tuple[bool, int, str], Cluster], str]:
        """Return the dictionary of clusters & system key.

        Keys:
            (is_surface, ncore, hash)
        Values:
            Cluster: the cluster of the core
        """
        if not isinstance(system, System):
            system = self.get_system_for(system)
        if not isinstance(system, System):
            msg = f"Unknown type of input: {type(system)}"
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

        oesc = bool(self.config.exploration.surface_only_explore_single_core)
        if oesc and len(self.gas_lst) == 0:
            max_ncore = 1
        else:
            max_ncore = int(self.config.exploration.max_ncore_for_surface)

        values: list[Cluster] = []
        keys: list[tuple[bool, int, str]] = []
        for core in system.get_site_core(max_ncore=max_ncore):  # type: ignore
            idx_core = np.unique(np.where(core)).astype(int)
            values.append(
                Cluster.from_select(
                    system,
                    idx_core,
                    env_threshold=self.config.site.env_threshold,
                    max_moved_threshold=self.config.site.max_moved_threshold,
                    method=self.config.site.method,
                )
            )
            keys.append((True, len(idx_core), values[-1].hash))
        if bool(self.config.exploration.allow_explore_bulk):
            if system.is_outer is None:
                msg = "System.is_outer is None."
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            if system.is_fix is None:
                is_moved = np.ones_like(system.is_outer, dtype=bool)
            else:
                is_moved = np.logical_not(system.is_fix)
            is_inner = np.logical_not(system.is_outer)
            mask = np.logical_and(is_inner, is_moved)
            for idx in np.unique(np.where(mask)).astype(int):
                values.append(
                    Cluster.from_select(
                        system,
                        np.array([idx]),
                        env_threshold=self.config.site.env_threshold,
                        max_moved_threshold=self.config.site.max_moved_threshold,
                        method=self.config.site.method,
                    )
                )
                keys.append((False, 1, values[-1].hash))

        # remove duplicate cluster
        _, idxs = np.unique([i.hash for i in values], return_index=True)
        self.logger.info(
            self._reformat_message(
                f"Find {len(values)} cluster for system="
                f"{system.hash}. And {len(idxs)} unique cluster."
            )
        )

        # optimize the cluster in parallel mode
        result: dict[tuple[bool, int, str], Cluster] = (  # type: ignore
            self.__batch_optimization_parallel(
                container={keys[int(id)]: values[int(id)] for id in idxs},
                raise_on_failed=False,
                is_minima=True,
            )
        )

        # persist the network for restart. [minima list]
        self.network.persistence()
        return result, system.get_key_for_metadata()

    def __batch_optimization_parallel(
        self,
        container: list[SysGraph] | dict[Any, SysGraph],
        raise_on_failed: bool = True,
        is_minima: bool = True,
    ) -> list[SysGraph] | dict[Any, SysGraph]:
        """Optimize the container in parallel mode."""
        if isinstance(container, list):
            dct = self.__batch_optimization_parallel(
                container={
                    i: cluster  # type: ignore
                    for i, cluster in enumerate(container)
                },
                is_minima=is_minima,
                raise_on_failed=raise_on_failed,
            )
            if not isinstance(dct, dict):
                msg = f"Unknown type of output: {type(dct)}"
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            return [dct[i] for i in sorted(dct.keys())]

        elif isinstance(container, dict):
            start, msg = perf_counter(), "cluster" if is_minima else "gas"
            msg = f"Start to optimize the {len(container)} {msg}."
            self.logger.info(self._reformat_message(msg))
            result: dict[Any, SysGraph] = {}
            futures: list = []

            # -------------------------------------------------
            # Submit the sysgraph optimization to the executor
            # -------------------------------------------------
            for label, sysgraph in container.items():
                key: str = sysgraph.get_key_for_metadata()
                if is_minima and sysgraph in self.network.db_minima:
                    atoms: Atoms = self.network.db_minima[key]
                    result[label] = v = Cluster.from_ase(atoms)
                    msg = f"Read '{key}' from the DB for {v}."
                    self.logger.info(self._reformat_message(msg))
                elif not is_minima and sysgraph in self.network.db_gas:
                    atoms: Atoms = self.network.db_gas[key]
                    result[label] = v = Gas.from_ase(atoms)
                    msg = f"Read '{key}' from the DB for {v}."
                    self.logger.info(self._reformat_message(msg))
                else:
                    futures.append(
                        self.executor.submit(
                            helper_optimization,
                            config=self.config,
                            graph_label=label,
                            graph=sysgraph,
                            allow_hash_change=False,
                            allow_not_connected=False,
                            raise_when_fail=False,
                        )
                    )
            msg = f"Submit the optimization {len(futures)} jobs"
            msg += f" by {perf_counter() - start:.2f} seconds."
            self.logger.info(self._reformat_message(msg))
            # -------------------------------------------------
            # Wait for the sysgraph optimization to finish
            # -------------------------------------------------
            while len(futures) > 0:
                future_result, futures = self.executor.wait(futures)  # type: ignore
                sysgraph_or_msg, label, cost_time = future_result
                if isinstance(sysgraph_or_msg, Gas | Cluster):
                    msg: str = f"Optimization (success): {sysgraph_or_msg}."
                    if isinstance(sysgraph_or_msg, Cluster):
                        self.network.db_minima.add(sysgraph_or_msg)
                    else:
                        self.network.db_gas.add(sysgraph_or_msg)
                    result[label] = sysgraph_or_msg
                elif isinstance(sysgraph_or_msg, str):
                    msg = str(sysgraph_or_msg)
                    if raise_on_failed:
                        self.logger.error(self._reformat_message(msg))
                        raise RuntimeError(msg)
                else:
                    msg = f"Unknown type: {type(sysgraph_or_msg)}"
                    self.logger.error(self._reformat_message(msg))
                    raise ValueError(msg)
                msg = f"CostTime={cost_time:.2f} for {msg}"
                self.logger.info(self._reformat_message(msg))
            self.logger.info(
                self._reformat_message(
                    "All optimization jobs are done in "
                    f"{perf_counter() - start:.2f} seconds."
                )
            )
            return result
        else:
            msg = f"Unknown type of container: {type(container)}"
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

    def explore(self, system: System | Atoms | None = None) -> None:
        # 1 step: analyze the system
        dct, system_key = self._first_step(system)

        # 2 step: exploration
        ncore_4_adspt = int(self.config.exploration.max_ncore_for_surface)
        if bool(self.config.exploration.surface_only_explore_single_core):
            ncore_4_surface = 1
        else:
            ncore_4_surface = ncore_4_adspt
        # 2.1 group the clusters ... ...
        lst_4bulk: list[Cluster] = []
        lst_4surface: list[Cluster] = []
        lst_4adsorption: list[Cluster] = []
        for (is_surface, ncore, _), v in dct.items():  # type: ignore
            if is_surface:
                if ncore <= ncore_4_surface:
                    lst_4surface.append(v)
                elif ncore <= ncore_4_adspt:
                    lst_4adsorption.append(v)
            else:
                lst_4bulk.append(v)
        # 2.2 exploration for each cluster ... ...
        if len(lst_4bulk) > 0:
            self._second_step_bulk(
                lst_4bulk,
                system_key=system_key,
            )
        if len(lst_4adsorption) > 0:
            for cluster in lst_4adsorption:
                for gas in self.gas_lst:
                    self._second_step_adsorption(
                        gas=gas,
                        cluster=cluster,
                        system_key=system_key,
                    )
        if len(lst_4surface) > 0:
            for cluster in lst_4surface:
                self._second_step_surface(
                    cluster=cluster,
                    system_key=system_key,
                )
        self.network.recorder.system.add(system_key)
        self.network.persistence()

    def _second_step_surface(
        self,
        cluster: Cluster,
        system_key: str = "",
    ) -> None:
        # -----------------------------------------------------------
        # check if the cluster has been explored
        # -----------------------------------------------------------
        cluster_key = cluster.get_key_for_metadata()
        confidence = self.config.exploration.maxconfidence
        oldnew = self.network.recorder.cluster[cluster_key]
        if oldnew.exploration_can_be_finished(
            confidence=confidence,
            min_found=self.network.metadata.table.get_minconut_for(
                cluster_key=cluster_key,
            ),
        ):
            msg = f"the cluster {cluster_key} has been explored."
            self.logger.info(self._reformat_message(msg))
            return

        # -----------------------------------------------------------
        # check if the cluster is at a minimum & prepare something
        # -----------------------------------------------------------
        if not cluster.check_minima(
            fmax=float(self.config.event.max_force),
            fqmin=float(self.config.event.min_frequency),
        ):
            msg = f"Cluster {cluster_key} is not at a minimum."
            self.logger.error(self._reformat_message(msg))
            raise AssertionError(msg)
        calc: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )
        start: float = perf_counter()
        futures: list = []

        # -----------------------------------------------------------
        # submit dimer tasks to executor
        # -----------------------------------------------------------
        for _ in range(int(self.config.exploration.maxtry)):
            thetacutoff = float(self.config.exploration.thetacutoff)
            if thetacutoff < 0:
                futures.append(
                    self.executor.submit(
                        helper_dimer,
                        config=self.config,
                        graph_label=cluster_key,
                        allow_fixed_bonds_change=False,
                        graph=cluster.model_copy(deep=True),
                        raise_when_fail=False,
                        displacement=None,
                    )
                )
            else:
                disp = asetools.call_dimer_displace(
                    atoms=cluster.to_ase().copy(),
                    calc=calc,
                    mask=None,
                    parse_mask_from_atoms=True,
                    start=start,
                )
                can_be_skip, cosine = self.network.scheduler.can_be_skip(
                    cluster_key,
                    diffpositions=disp,
                    thetacutoff=float(self.config.exploration.thetacutoff),
                )
                if can_be_skip:
                    oldnew.found_skip()
                    msg = "Skip to submit dimer task for "
                else:
                    msg = "Submit dimer task for "
                    futures.append(
                        self.executor.submit(
                            helper_dimer,
                            config=self.config,
                            graph_label=cluster_key,
                            allow_fixed_bonds_change=False,
                            graph=cluster.model_copy(deep=True),
                            raise_when_fail=False,
                            displacement=disp,
                        )
                    )
                msg = f"{msg}{cluster_key}, cosine={cosine:.2f}"
                self.logger.info(self._reformat_message(msg))
        self.logger.info(
            self._reformat_message(
                f"Submit {len(futures)} dimer tasks by "
                f"{perf_counter() - start:.2f} seconds"
            )
        )

        # -----------------------------------------------------------
        # wait for the dimer tasks to finish
        # -----------------------------------------------------------
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, _, cost_time, _ = future_result
            label = self.network.found(
                event,
                for_gas=None,
                for_cluster=cluster_key,
                for_system=system_key,
                persist=True,
            )
            if label.startswith("fail"):
                msg = f"Dimer search {label} by {cost_time:.2f}"
                msg += f" seconds because of {event}"
                self.logger.info(self._reformat_message(msg))
            else:
                msg = "Dimer search successfully, and got "
                msg += f"{label} by {cost_time:.2f} seconds."
                if str(event) not in label:
                    msg += f" Simplify original {event} by threshold "
                    msg += f"{self.config.event.simplified_threshold:.2f}"
                self.logger.info(self._reformat_message(msg))
            if oldnew.exploration_can_be_finished(
                confidence=confidence,
                min_found=self.network.metadata.table.get_minconut_for(
                    cluster_key=cluster_key,
                    exclude_gas=True,
                ),
            ):
                msg = f"Finish exploration for {cluster_key} with "
                msg += f"confidence {confidence:.2f}. {len(futures)}"
                msg += " dimer tasks left. They will be canceled."
                self.logger.info(self._reformat_message(msg))
                for future in futures:
                    self.executor.cancel(future)
                break
        self.network.persistence()

    def _second_step_adsorption(
        self,
        gas: Gas,
        cluster: Cluster,
        system_key: str = "",
    ) -> None:
        """The second step for the on-the-fly KMC simulation."""
        cluster_key = cluster.get_key_for_metadata(True)
        gas_key = gas.get_key_for_metadata(False)

        # ---------------------------------------------
        # check if the adsorption has been explored
        # ---------------------------------------------
        graph_label = f"{cluster_key}_{gas_key}"
        oldnew = self.network.recorder.adsorption[graph_label]
        if oldnew.exploration_can_be_finished(
            confidence=self.config.exploration.maxconfidence,
            min_found=self.network.metadata.table.get_minconut_for(
                cluster_key=cluster_key,
                exclude_gas=False,
            ),
        ):
            msg = f"The adsorption for {cluster_key} with "
            msg += f"{gas_key} has been explored."
            self.logger.info(self._reformat_message(msg))
            return

        # ---------------------------------------------
        # build helper for adsorption
        # ---------------------------------------------
        assert cluster.ncore > 0, "graph must have core"
        adsorption_helper = Helper(
            atoms=cluster.to_ase(
                exclude_bond_attibutes=True,
                exclude_energetics=True,
            ).copy(),
            adsorbate=gas,
            core=np.unique(cluster.idx_core),
            nfibonacci=int(self.config.exploration.nfibonacci),
            use_direct=True,
            use_raw=True,
        )

        # ---------------------------------------------
        # submit adsorption tasks to executor
        # ---------------------------------------------
        futures: list = []
        start: float = perf_counter()
        lst = np.arange(1, adsorption_helper.nrun)
        maxtry = int(self.config.exploration.maxtry)
        if adsorption_helper.nrun > maxtry:
            lst = np.random.choice(lst, size=maxtry - 1, replace=False)
        np.random.shuffle(lst)
        lst = np.append([0], lst)
        for irun in lst:
            futures.append(
                self.executor.submit(
                    helper_adsorption,
                    config=self.config,
                    graph=cluster,
                    gas=gas,
                    irun=irun,
                    graph_label=graph_label,
                    allow_hash_change=True,
                    raise_when_fail=False,
                    deep_copy=True,
                )
            )
        self.logger.info(
            self._reformat_message(
                f"Submit {len(lst)} adsorption tasks by "
                f"{perf_counter() - start:.2f} seconds"
                f" for {cluster_key} with {gas_key}"
            )
        )

        # -----------------------------------------------------------
        # wait for the adsorption tasks to finish
        # -----------------------------------------------------------
        confidence = self.config.exploration.maxconfidence
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, _, cost_time, _ = future_result
            event_label = self.network.found(
                event,
                for_gas=gas_key,
                for_cluster=cluster_key,
                for_system=system_key,
                persist=True,
            )
            if event_label.startswith("fail"):
                msg = f"Adsorption search {event_label} by {cost_time:.2f}"
                msg += f" seconds because of {event}"
                self.logger.info(self._reformat_message(msg))
            else:
                msg = "Adsorption search successfully, and got "
                msg += f"{event_label} by {cost_time:.2f} seconds."
                if str(event) not in event_label:
                    msg += f" Simplify original {event} by threshold "
                    msg += f"{self.config.event.simplified_threshold:.2f}"
                self.logger.info(self._reformat_message(msg))
            if oldnew.exploration_can_be_finished(
                confidence=confidence,
                min_found=self.network.metadata.table.get_minconut_for(
                    cluster_key=cluster_key,
                    exclude_gas=False,
                ),
            ):
                msg = f"Finish exploration for {cluster_key} and {gas_key}"
                msg += f"with confidence {confidence:.2f}. {len(futures)}"
                msg += " adsorption tasks left. They will be canceled."
                self.logger.info(self._reformat_message(msg))
                for future in futures:
                    self.executor.cancel(future)
                break
        self.network.persistence()

    def _second_step_bulk(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> None:
        """The second step for the on-the-fly KMC simulation."""
        raise NotImplementedError
