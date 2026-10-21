from ase import Atoms

from graphatoms.enterpoint.parallel import executor_map
from graphatoms.enterpoint.runner.general._expl import ExplorationBase as _Base
from graphatoms.enterpoint.runner.v2._helperv2 import (
    HelperAdsorptionResult,
    HelperDimerResult,
    HelperNebResult,
    HelperResult,
    helper_adsorption,
    helper_dimer,
    helper_neb,
)
from graphatoms.system import Cluster, System


class ExplorationBase(_Base):
    def explore(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float]:
        is_, eng, futures = self.submit(system)
        self.collect(futures)
        return is_, eng

    def collect(self, futures: list) -> None:
        """Collect the results from the futures."""
        threshold_str = f"{self.config.event.simplified_threshold:.2f}"
        graph_label_seq: set[str] = set()
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            if isinstance(future_result, HelperResult):
                msg = future_result.message
                if isinstance(future_result, HelperAdsorptionResult):
                    if future_result.result is not None:
                        event = future_result.result
                        event_label = self.network.found(
                            for_cluster=future_result.for_cluster,
                            for_gas=future_result.for_gas,
                            persist=True,
                            event=event,
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
                        event_label = self.network.found(
                            event=event,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            persist=True,
                        )
                        if str(event) not in event_label:
                            msg += f" Simplify original {event} by threshold "
                            msg += (
                                f"{self.config.event.simplified_threshold:.2f}"
                            )
                    elif future_result.is_failed:
                        self.network.found(
                            future_result.message,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            persist=True,
                        )
                    elif future_result.is_skipped:
                        self.network.recorder.cluster[
                            future_result.for_cluster
                        ].found_skip()
                elif isinstance(future_result, HelperNebResult):
                    if future_result.result is not None:
                        event = future_result.result
                        event_label = self.network.found(
                            event=event,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            for_system=future_result.for_system,
                            persist=True,
                        )
                        if str(event) not in event_label:
                            msg += " Simplify original "
                            msg += f"{event} by threshold {threshold_str}"
                    else:
                        self.network.found(
                            future_result.message,
                            for_cluster=future_result.for_cluster,
                            for_gas=None,
                            for_system=future_result.for_system,
                            persist=True,
                        )
                    # record the explored candidate, so a restart will skip it
                    graph_label = future_result.for_cluster
                    graph_label += f"@{future_result.candidate_key}"
                    self.network.recorder.bulk.add(graph_label)
                else:
                    raise NotImplementedError(
                        f"Unknown type {type(future_result)}"  #
                    )
                self.logger.info(self._reformat_message(msg))
            else:
                raise NotImplementedError(
                    f"Unknown type {type(future_result)}, "  #
                    f"and the value is {future_result}."
                )
        for graph_label in graph_label_seq:
            self.network.recorder.adsorption.add(graph_label)
        self.network.persistence()

    def submit(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float, list]:
        """Submit the system exploration.

        Returns:
            whether it is explored successfully
            the final energy.
        """
        dct, system_key, energy = self._first_step(system)
        if system_key == "":
            return False, energy, []

        # 2 step: exploration (only submit)
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
        futures = []
        if len(lst_4bulk) > 0:
            futures.extend(
                self.__second_step_bulk(
                    lst_4bulk,
                    system_key=system_key,
                )
            )
        if len(lst_4adsorption) > 0:
            for cluster in lst_4adsorption:
                for gas in self.gas_lst:
                    futures.extend(
                        self._2step_ads_submit(
                            helper_adsorption_func=helper_adsorption,
                            cluster=cluster,
                            gas=gas,
                        )
                    )
        if len(lst_4surface) > 0:
            for cluster in lst_4surface:
                futures.extend(
                    executor_map(
                        fn=helper_dimer,
                        executor=self.executor,
                        fn_params=[
                            dict(
                                cluster=cluster,
                                config=self.config,
                                allow_metal_element_in_gas=False,
                                allow_fixed_bonds_change=False,
                                allow_desorption="all",
                                deep_copy=True,
                            )
                            for _ in range(int(self.config.exploration.maxtry))
                        ],
                        only_submit=True,
                    )
                )
        return True, energy, futures

    def __second_step_bulk(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> list:
        """Submit the NEB tasks of the bulk exploration.

        Note: the exploration of a bulk cluster is finished when all the
        NEB candidates (i.e. the `aseqhull` list of that cluster) have been
        explored, so every remaining candidate is submitted at once and no
        task is canceled.
        """
        futures: list = []
        for cluster in lst:
            cluster_key = cluster.get_key_for_metadata(False)
            candidates = self._2step_bulk_candidates(cluster)
            if len(candidates) == 0:
                msg = f"The bulk exploration for {cluster_key} is finished."
                self.logger.info(self._reformat_message(msg))
                continue
            futures.extend(
                executor_map(
                    fn=helper_neb,
                    executor=self.executor,
                    fn_params=[
                        dict(
                            cluster=cluster,
                            config=self.config,
                            intermediate_atoms=intermediate_atoms,
                            final_atoms=final_atoms,
                            candidate_key=key,
                            for_system=system_key,
                            allow_fixed_bonds_change=False,
                            deep_copy=True,
                        )
                        for key, intermediate_atoms, final_atoms in candidates
                    ],
                    only_submit=True,
                )
            )
            msg = f"Submit {len(candidates)} NEB tasks for "
            msg += f"the bulk cluster {cluster_key}."
            self.logger.info(self._reformat_message(msg))
        return futures
