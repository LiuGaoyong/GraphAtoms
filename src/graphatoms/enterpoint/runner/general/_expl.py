from collections.abc import Callable
from time import perf_counter
from typing import Any

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
    ) -> tuple[dict[tuple[bool, int, str], Cluster], str, float]:
        """Return the dictionary of clusters & system key.

        Returns:
            dct (dict[tuple[bool, int, str], Cluster]):
            system_key (str): the system key.
            energy_opt (float): the optimized energy.
        """
        dct, system_key, energy = self.analysis(system=system)
        result: dict[tuple[bool, int, str], Cluster] = (  # type: ignore
            self.__batch_optimization_parallel(
                container=dct,  # type: ignore
                raise_on_failed=False,
            )
        )
        self.network.persistence()
        return result, system_key, energy

    def __batch_optimization_parallel(
        self,
        container: list[SysGraph] | dict[Any, SysGraph],
        raise_on_failed: bool = True,
    ) -> list[SysGraph] | dict[Any, SysGraph]:
        """Optimize the container in parallel mode."""
        if isinstance(container, list):
            dct = self.__batch_optimization_parallel(
                container={
                    i: cluster  # type: ignore
                    for i, cluster in enumerate(container)
                },
                raise_on_failed=raise_on_failed,
            )
            if not isinstance(dct, dict):
                msg = f"Unknown type of output: {type(dct)}"
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            return [dct[i] for i in sorted(dct.keys())]

        elif isinstance(container, dict):
            start, msg = perf_counter(), "sysgraph"
            msg = f"Start to optimize the {len(container)} {msg}."
            self.logger.info(self._reformat_message(msg))
            result: dict[Any, SysGraph] = {}
            futures: list = []

            # -------------------------------------------------
            # Submit the sysgraph optimization to the executor
            # -------------------------------------------------
            for label, sysgraph in container.items():
                key: str = sysgraph.get_key_for_metadata()
                sysgraph_opt = self.network.object_has_been_optimized(sysgraph)
                if sysgraph_opt is not None:
                    result[label] = v = sysgraph_opt
                    msg = f"Read '{key}' from the DB for {v}."
                    self.logger.info(self._reformat_message(msg))
                    continue
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
                # split result from the function of `helper_optimization`
                sysgraph_or_msg, (label, _), cost_time = future_result
                if isinstance(sysgraph_or_msg, Gas | Cluster | System):
                    msg: str = f"Optimization (success): {sysgraph_or_msg}."
                    if isinstance(sysgraph_or_msg, Gas):
                        self.network.write_sysgraph(sysgraph_or_msg, "gas")
                    elif isinstance(sysgraph_or_msg, Cluster):
                        self.network.write_sysgraph(sysgraph_or_msg, "cluster")
                    else:
                        self.network.write_sysgraph(sysgraph_or_msg, "system")
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

    def explore(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float]:
        """Explore the system, and return whether it is explored successfully and the final energy."""  # noqa: E501
        # 1 step: analyze the system
        dct, system_key, energy = self._first_step(system)
        if system_key == "":
            return False, energy

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
                if ncore <= ncore_4_adspt:
                    lst_4adsorption.append(v)
            else:
                lst_4bulk.append(v)
        msg = f"The system {system_key} will be explored"
        msg += f" with {len(lst_4bulk)}-bulk,{len(lst_4adsorption)}"
        msg += f"-adsorption,{len(lst_4surface)}-surface."
        self.logger.info(self._reformat_message(msg))
        # 2.2 exploration for each cluster ... ...
        if len(lst_4bulk) > 0:
            raise NotImplementedError(
                "Bulk exploration is not implemented yet."
            )
            self.__second_step_bulk(
                lst_4bulk,
                system_key=system_key,
            )
        if len(lst_4adsorption) > 0:
            for cluster in lst_4adsorption:
                for gas in self.gas_lst:
                    self.__second_step_adsorption(
                        gas=gas,
                        cluster=cluster,
                        system_key=system_key,
                    )
        if len(lst_4surface) > 0:
            for cluster in lst_4surface:
                self.__second_step_surface(
                    cluster=cluster,
                    system_key=system_key,
                )
        self.network.persistence()
        return True, energy

    def __second_step_surface(
        self,
        cluster: Cluster,
        system_key: str = "",
    ) -> None:
        # -----------------------------------------------------------
        # check if the cluster has been explored
        # -----------------------------------------------------------
        cluster_key = cluster.get_key_for_metadata(False)
        if self.network.exploration_can_be_finished(
            confidence=self.config.exploration.maxconfidence,
            cluster_key=cluster_key,
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
                can_be_skip, cosine = self.network.exploration_can_be_skipped(
                    cluster_key=cluster_key,
                    diffpositions=disp,
                    thetacutoff=thetacutoff,
                )
                if can_be_skip:
                    self.network.recorder.cluster[cluster_key].found_skip()
                    msg = "Skip to submit dimer task for "
                else:
                    msg = "Submit dimer task for "
                    futures.append(
                        self.executor.submit(
                            helper_dimer,
                            config=self.config,
                            network=self.network,
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
            if self.network.exploration_can_be_finished(
                confidence=self.config.exploration.maxconfidence,
                cluster_key=cluster_key,
            ):
                msg = "Finish exploration for "
                msg += f"{cluster_key} with confidence "
                msg += f"{self.config.exploration.maxconfidence:.2f}."
                msg += f" {len(futures)} dimer tasks left."
                msg += " They will be canceled."
                self.logger.info(self._reformat_message(msg))
                for future in futures:
                    self.executor.cancel(future)
                break
        self.network.persistence()

    def _2step_ads_submit(
        self,
        gas: Gas,
        cluster: Cluster,
        *args,
        helper_adsorption_func: Callable = helper_adsorption,
        **kwargs,
    ) -> list[Any]:
        """The second step for the on-the-fly KMC simulation."""
        cluster_key = cluster.get_key_for_metadata(False)
        gas_key = gas.get_key_for_metadata(False)
        graph_label = f"{cluster_key}_{gas_key}"

        # ---------------------------------------------
        # check if the adsorption has been explored
        # ---------------------------------------------
        if self.network.exploration_can_be_finished(
            confidence=self.config.exploration.maxconfidence,
            cluster_key=cluster_key,
            gas_key=gas_key,
        ):
            msg = f"The adsorption for {cluster_key} with "
            msg += f"{gas_key} has been explored."
            self.logger.info(self._reformat_message(msg))
            return []

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
            distance_lst=np.arange(
                self.config.exploration.min_distance,
                self.config.exploration.max_distance,
                self.config.exploration.step_distance,
            ),
            use_direct=True,
            use_raw=True,
        )

        # ---------------------------------------------
        # submit adsorption tasks to executor
        # ---------------------------------------------
        futures: list = []
        start: float = perf_counter()
        maxtry = int(self.config.exploration.maxtry)
        lst = np.arange(1, adsorption_helper.nrun).astype(int)
        if adsorption_helper.nrun > maxtry:
            lst = np.random.choice(lst, size=maxtry - 1, replace=False)
        np.random.shuffle(lst)
        lst = np.append([0], lst)
        for irun in lst:
            futures.append(
                self.executor.submit(
                    helper_adsorption_func,
                    config=self.config,
                    graph=cluster,
                    gas=gas,
                    initial_atoms=adsorption_helper.__call__(irun=irun).copy(),
                    graph_label=graph_label,
                    raise_when_fail=False,
                    deep_copy=True,
                )
            )
        self.logger.info(
            self._reformat_message(
                f"Submit {len(lst)}(nrun={adsorption_helper.nrun})"
                f" adsorption tasks by {perf_counter() - start:.2f}"
                f" seconds for {cluster_key} with {gas_key}"
            )
        )
        return futures

    def __second_step_adsorption(
        self,
        gas: Gas,
        cluster: Cluster,
        system_key: str = "",
    ) -> None:
        gas_key = gas.get_key_for_metadata(False)
        cluster_key = cluster.get_key_for_metadata(False)
        futures = self._2step_ads_submit(gas, cluster)
        graph_label = f"{cluster_key}_{gas_key}"

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
            if self.network.exploration_can_be_finished(
                confidence=confidence,
                cluster_key=cluster_key,
                gas_key=gas_key,
            ):
                msg = f"Finish exploration for {cluster_key} and {gas_key}"
                msg += f"with confidence {confidence:.2f}. {len(futures)}"
                msg += " adsorption tasks left. They will be canceled."
                self.logger.info(self._reformat_message(msg))
                for future in futures:
                    self.executor.cancel(future)
                break
        self.network.recorder.adsorption.add(graph_label)
        self.network.persistence()

    def __second_step_bulk(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> None:
        """The second step for the on-the-fly KMC simulation."""
        raise NotImplementedError
