"""Exploration orchestration for the Ray-native runners.

The mixin assumes the concrete runner provides ``analysis``, ``gas_lst``,
``gas_pressure_dct`` and ``actor``; it does not use ``self.network`` or an
executor.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import ray
from ase import Atoms

from graphatoms.enterpoint.runner.v2._helperv2 import (
    HelperAdsorptionResult,
    HelperDimerResult,
    HelperResult,
)
from graphatoms.enterpoint.runner.v2.general._expl import classify_clusters
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.adsorption import Helper

from ._tasks import (
    actor_call,
    submit_task,
    task_adsorption,
    task_dimer,
    task_optimization,
    wait_one,
)

__all__ = ["RayExplorationMixin"]


class RayExplorationMixin:
    """Ray-native implementation of the v2 exploration algorithm."""

    def explore(
        self,
        system: System | Atoms | None = None,
    ) -> tuple[bool, float]:
        dct, system_key, energy = self.analysis(system)
        if system_key == "":
            return False, energy

        self._optimize_clusters(dct)
        lst_4bulk, lst_4surface, lst_4adsorption = classify_clusters(
            self.config,
            dct,
        )
        msg = f"The system {system_key} will be explored"
        msg += f" with {len(lst_4bulk)}-bulk,{len(lst_4adsorption)}"
        msg += f"-adsorption,{len(lst_4surface)}-surface."
        self.logger.info(msg)

        if len(lst_4bulk) > 0:
            raise NotImplementedError(
                "Bulk exploration is not implemented yet."
            )

        refs: list = []
        for cluster in lst_4adsorption:
            for gas in self.gas_lst:
                refs.extend(self._submit_adsorption(cluster, gas))
        for cluster in lst_4surface:
            for _ in range(int(self.config.exploration.maxtry)):
                refs.append(
                    submit_task(
                        task_dimer,
                        actor=self.actor,
                        config=self.config,
                        cluster=cluster,
                        allow_metal_element_in_gas=False,
                        allow_fixed_bonds_change=False,
                        allow_desorption="all",
                        deep_copy=True,
                    )
                )
        self._collect(refs)
        return True, energy

    # ------------------------------------------------------------------
    #                          optimization
    # ------------------------------------------------------------------
    def _optimize_clusters(
        self,
        dct: dict[tuple[bool, int, str], Cluster],
    ) -> None:
        start = perf_counter()
        self.logger.info(f"Start to optimize the {len(dct)} cluster.")
        pending: dict = {}
        for label, cluster in dct.items():
            key = cluster.get_key_for_metadata()
            cached = actor_call(
                self.actor,
                "object_has_been_optimized",
                cluster,
            )
            if cached is not None:
                dct[label] = cached
                self.logger.info(f"Read '{key}' from the DB for {cached}.")
                continue
            ref = submit_task(
                task_optimization,
                actor=self.actor,
                config=self.config,
                graph=cluster,
                graph_label=key,
                allow_hash_change=False,
                allow_not_connected=False,
                run_vibration=True,
                deep_copy=True,
            )
            pending[ref] = label

        while len(pending) > 0:
            done, _ = ray.wait(list(pending), num_returns=1)
            ref = done[0]
            label = pending.pop(ref)
            result = ray.get(ref)
            self.logger.info(result.message)
            if result.result is None:
                continue
            graph_opt = result.result
            if isinstance(graph_opt, Gas):
                type_ = "gas"
            elif isinstance(graph_opt, Cluster):
                type_ = "cluster"
            elif isinstance(graph_opt, System):
                type_ = "system"
            else:
                raise ValueError(f"Unknown type: {type(graph_opt)}")
            actor_call(self.actor, "write_sysgraph", graph_opt, type_)
            dct[label] = graph_opt

        actor_call(self.actor, "persistence")
        self.logger.info(
            "All optimization jobs are done in "
            f"{perf_counter() - start:.2f} seconds."
        )

    # ------------------------------------------------------------------
    #                          submission
    # ------------------------------------------------------------------
    def _submit_adsorption(
        self,
        cluster: Cluster,
        gas: Gas,
    ) -> list:
        cluster_key = cluster.get_key_for_metadata(False)
        gas_key = gas.get_key_for_metadata(False)
        graph_label = f"{cluster_key}_{gas_key}"

        if actor_call(
            self.actor,
            "exploration_can_be_finished",
            float(self.config.exploration.maxconfidence),
            cluster_key,
            gas_key,
        ):
            self.logger.info(
                f"The adsorption for {cluster_key} with {gas_key} "
                "has been explored."
            )
            return []

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
        maxtry = int(self.config.exploration.maxtry)
        lst = np.arange(1, adsorption_helper.nrun).astype(int)
        if adsorption_helper.nrun > maxtry:
            lst = np.random.choice(lst, size=maxtry - 1, replace=False)
        np.random.shuffle(lst)
        lst = np.append([0], lst)
        refs = []
        for irun in lst:
            refs.append(
                submit_task(
                    task_adsorption,
                    actor=self.actor,
                    gas=gas,
                    config=self.config,
                    graph=cluster,
                    initial_atoms=adsorption_helper(irun=irun).copy(),
                    graph_label=graph_label,
                    raise_when_fail=False,
                    deep_copy=True,
                )
            )
        self.logger.info(
            f"Submit {len(lst)}(nrun={adsorption_helper.nrun})"
            f" adsorption tasks for {cluster_key} with {gas_key}"
        )
        return refs

    # ------------------------------------------------------------------
    #                           collection
    # ------------------------------------------------------------------
    def _collect(self, refs: list) -> None:
        graph_label_seq: set[str] = set()
        pending = list(refs)
        while len(pending) > 0:
            result, pending = wait_one(pending)
            if not isinstance(result, HelperResult):
                raise NotImplementedError(
                    f"Unknown type {type(result)}, and the value is {result}."
                )
            self.logger.info(result.message)
            if isinstance(result, HelperAdsorptionResult):
                graph_label_seq.add(
                    f"{result.for_cluster}_{result.for_gas}"
                )
            elif isinstance(result, HelperDimerResult):
                pass
            else:
                raise NotImplementedError(f"Unknown type {type(result)}")

        for graph_label in graph_label_seq:
            actor_call(self.actor, "add_adsorption", graph_label)
        actor_call(self.actor, "persistence")
