from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import ray
from ase import Atoms

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.runner.v2._helperv2 import (
    HelperAdsorptionResult,
    HelperDimerResult,
    HelperResult,
)
from graphatoms.enterpoint.runner.v2.general._expl import (
    ExplorationBase as _ExplorationBase,
)
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.adsorption import Helper

from ._helper import (
    RayReactionNetwork,
    RayReactionNetworkHandle,
    helper_adsorption,
    helper_dimer,
    helper_optimization,
)

__all__ = ["ExplorationRayBase"]


class ExplorationRayBase(_ExplorationBase):
    """Exploration driven by Ray tasks and a shared network actor.

    The v2 exploration algorithm is kept as-is, but every reader/writer of
    the reaction network goes through :class:`RayReactionNetwork`.  The
    driver process keeps a local, read-only copy that is refreshed after
    each exploration round.
    """

    def __init__(self, *, config: Config) -> None:
        super().__init__(config=config)
        self.network_ray: RayReactionNetworkHandle | None = None

    # -----------------------------------------------------------
    #                 actor / local network synchronisation
    # -----------------------------------------------------------
    def sync_network_to_ray(self) -> RayReactionNetworkHandle:
        """Persist the local network and (re)create its Ray actor."""
        self.network.persistence()
        if self.network_ray is not None:
            ray.kill(self.network_ray, no_restart=True)
        # The actor only guards the network state; it must not reserve a CPU
        # slot that a dimer/adsorption task could otherwise use.
        self.network_ray = RayReactionNetwork.options(num_cpus=0).remote(
            config=self.config
        )
        return self.network_ray

    def sync_network_to_local(self) -> None:
        """Persist the actor network and reload the local read-only copy."""
        if self.network_ray is not None:
            ray.get(self.network_ray.persistence.remote())
        self.network = ReactionNetwork.from_config(self.config, restart=True)

    # -----------------------------------------------------------
    #                        task submission
    # -----------------------------------------------------------
    def _submit_task(self, task: Any, **kwargs) -> ray.ObjectRef:  # type: ignore
        ncpus = int(self.config.parallel_ncpus_for_tasks)
        if ncpus > 0:
            task = task.options(num_cpus=ncpus)
        return task.remote(**kwargs)

    def __batch_optimization_ray(
        self,
        container: dict[Any, Cluster],
        network: RayReactionNetworkHandle,
    ) -> dict[Any, Cluster]:
        """Optimize every cluster that is not in the network database."""
        start = perf_counter()
        msg = f"Start to optimize the {len(container)} cluster."
        self.logger.info(self._reformat_message(msg))
        result: dict[Any, Cluster] = {}
        pending: dict[ray.ObjectRef, Any] = {}  # type: ignore

        for label, cluster in container.items():
            key = cluster.get_key_for_metadata()
            sysgraph_opt = ray.get(
                network.object_has_been_optimized.remote(cluster)
            )
            if sysgraph_opt is not None:
                result[label] = sysgraph_opt  # type: ignore[assignment]
                msg = f"Read '{key}' from the DB for {sysgraph_opt}."
                self.logger.info(self._reformat_message(msg))
                continue
            ref = self._submit_task(
                helper_optimization,
                config=self.config,
                graph=cluster,
                network=network,
                allow_hash_change=False,
                allow_not_connected=False,
                raise_when_fail=False,
                run_vibration=True,
                deep_copy=True,
            )
            pending[ref] = label

        msg = f"Submit the optimization {len(pending)} jobs"
        msg += f" by {perf_counter() - start:.2f} seconds."
        self.logger.info(self._reformat_message(msg))

        while len(pending) > 0:
            done, _ = ray.wait(list(pending), num_returns=1)  # type: ignore
            ref = done[0]
            label = pending.pop(ref)
            future_result = ray.get(ref)
            msg = future_result.message
            if future_result.result is None:
                self.logger.warning(self._reformat_message(msg))
                continue
            graph_opt = future_result.result
            if isinstance(graph_opt, Gas):
                type_ = "gas"
            elif isinstance(graph_opt, Cluster):
                type_ = "cluster"
            elif isinstance(graph_opt, System):
                type_ = "system"
            else:
                msg = f"Unknown type: {type(graph_opt)}"
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            ray.get(network.write_sysgraph.remote(graph_opt, type_))
            result[label] = graph_opt
            self.logger.info(self._reformat_message(msg))

        ray.get(network.persistence.remote())
        msg = "All optimization jobs are done in "
        msg += f"{perf_counter() - start:.2f} seconds."
        self.logger.info(self._reformat_message(msg))
        return result

    def _2step_ads_submit_ray(
        self,
        gas: Gas,
        cluster: Cluster,
        network: RayReactionNetworkHandle,
    ) -> list[ray.ObjectRef]:  # type: ignore
        """Submit the adsorption tasks for one cluster/gas pair."""
        cluster_key = cluster.get_key_for_metadata(False)
        gas_key = gas.get_key_for_metadata(False)
        graph_label = f"{cluster_key}_{gas_key}"

        # ---------------------------------------------
        # check if the adsorption has been explored
        # ---------------------------------------------
        if ray.get(
            network.exploration_can_be_finished.remote(
                confidence=self.config.exploration.maxconfidence,
                cluster_key=cluster_key,
                gas_key=gas_key,
            )
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
        # submit adsorption tasks
        # ---------------------------------------------
        futures: list[ray.ObjectRef] = []  # type: ignore
        start = perf_counter()
        maxtry = int(self.config.exploration.maxtry)
        lst = np.arange(1, adsorption_helper.nrun).astype(int)
        if adsorption_helper.nrun > maxtry:
            lst = np.random.choice(lst, size=maxtry - 1, replace=False)
        np.random.shuffle(lst)
        lst = np.append([0], lst)
        for irun in lst:
            futures.append(
                self._submit_task(
                    helper_adsorption,
                    config=self.config,
                    graph=cluster,
                    gas=gas,
                    network=network,
                    initial_atoms=adsorption_helper(irun=irun).copy(),
                    graph_label=graph_label,
                    raise_when_fail=False,
                    deep_copy=True,
                )
            )
        msg = f"Submit {len(lst)}(nrun={adsorption_helper.nrun})"
        msg += f" adsorption tasks by {perf_counter() - start:.2f}"
        msg += f" seconds for {cluster_key} with {gas_key}"
        self.logger.info(self._reformat_message(msg))
        return futures

    def submit(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float, list]:
        """Submit the system exploration.

        Returns:
            whether it is explored successfully
            the final energy
            the pending Ray ObjectRefs.
        """
        dct, system_key, energy = self._first_step(system)
        if system_key == "":
            return False, energy, []

        # 2 step: exploration (only submit)
        network = self.network_ray
        assert network is not None, "The network actor is not initialized."
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
        futures: list[ray.ObjectRef] = []  # type: ignore
        if len(lst_4bulk) > 0:
            self.__second_step_bulk_ray(lst_4bulk, system_key=system_key)
        if len(lst_4adsorption) > 0:
            for cluster in lst_4adsorption:
                for gas in self.gas_lst:
                    futures.extend(
                        self._2step_ads_submit_ray(
                            cluster=cluster,
                            gas=gas,
                            network=network,
                        )
                    )
        if len(lst_4surface) > 0:
            maxtry = int(self.config.exploration.maxtry)
            for cluster in lst_4surface:
                for _ in range(maxtry):
                    futures.append(
                        self._submit_task(
                            helper_dimer,
                            config=self.config,
                            cluster=cluster,
                            network=network,
                            allow_metal_element_in_gas=False,
                            allow_fixed_bonds_change=False,
                            allow_desorption="all",
                            deep_copy=True,
                        )
                    )
        return True, energy, futures

    def collect(self, futures: list) -> None:
        """Collect the results from the Ray tasks."""
        network = self.network_ray
        if network is None:
            return
        threshold_str = f"{self.config.event.simplified_threshold:.2f}"
        graph_label_seq: set[str] = set()
        pending = set(futures)
        while len(pending) > 0:
            done, _ = ray.wait(list(pending), num_returns=1)  # type: ignore
            ref = done[0]
            pending.discard(ref)
            future_result = ray.get(ref)
            if not isinstance(future_result, HelperResult):
                msg = f"Unknown type {type(future_result)}, "
                msg += f"and the value is {future_result}."
                raise NotImplementedError(msg)
            msg = future_result.message
            if isinstance(future_result, HelperAdsorptionResult):
                if future_result.result is not None:
                    event = future_result.result
                    event_label = ray.get(
                        network.found.remote(
                            event=event,
                            for_cluster=future_result.for_cluster,
                            for_gas=future_result.for_gas,
                            persist=True,
                        )
                    )
                    if str(event) not in event_label:
                        msg += f" Simplify original {event} by "
                        msg += f"threshold {threshold_str}"
                gas_key = future_result.for_gas
                cluster_key = future_result.for_cluster
                graph_label = f"{cluster_key}_{gas_key}"
                graph_label_seq.add(graph_label)
            elif isinstance(future_result, HelperDimerResult):
                if future_result.result is not None:
                    event = future_result.result
                    event_label = ray.get(
                        network.found.remote(
                            event=event,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            persist=True,
                        )
                    )
                    if str(event) not in event_label:
                        msg += f" Simplify original {event} by threshold "
                        msg += f"{threshold_str}"
                elif future_result.is_failed:
                    ray.get(
                        network.found.remote(
                            future_result.message,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            persist=True,
                        )
                    )
                elif future_result.is_skipped:
                    ray.get(
                        network.record_skip.remote(future_result.for_cluster)
                    )
            else:
                raise NotImplementedError(f"Unknown type {type(future_result)}")
            self.logger.info(self._reformat_message(msg))

        for graph_label in graph_label_seq:
            ray.get(network.add_adsorption.remote(graph_label))
        ray.get(network.persistence.remote())
        self.sync_network_to_local()

    # -----------------------------------------------------------
    #                       first step
    # -----------------------------------------------------------
    def _first_step(
        self,
        system: System | Atoms | None,
    ) -> tuple[dict[tuple[bool, int, str], Cluster], str, float]:
        """Analyze the system, then optimize its clusters with Ray tasks."""
        dct, system_key, energy = self.analysis(system=system)
        if system_key == "":
            return dct, system_key, energy
        network = self.sync_network_to_ray()
        dct = self.__batch_optimization_ray(dct, network)
        self.sync_network_to_local()
        return dct, system_key, energy

    def __second_step_bulk_ray(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> None:
        """Bulk exploration is not supported by the v2 algorithm yet."""
        raise NotImplementedError("Bulk exploration is not implemented yet.")
