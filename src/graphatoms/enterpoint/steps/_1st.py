from time import perf_counter
from typing import Any, override

import numpy as np
from ase import Atoms

from graphatoms.enterpoint.config import Config
from graphatoms.system import Cluster, Gas, System  # type: ignore
from graphatoms.system.graph import SysGraph
from graphatoms.utils.parser import hydra_parse

from ._0abc import BaseABC
from ._helper import helper_optimization as helper_cluster_optimization


class FirstStep(BaseABC):
    """The abstract base class for the runner."""

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.__gas_lst: list[Gas] = self.__batch_optimization_parallel(
            container=[
                Gas.from_name(
                    gas_info.name,
                    sticking=gas_info.sticking,
                    pressure=gas_info.pressure,
                    parse_bonds=self.config.bonds,  # type: ignore
                )
                for gas_info in self.network.metadata.basic.gas_info_lst
            ],
            raise_on_failed=True,
            is_minima=False,
        )
        # persist the network for restart. [gas list]
        self.network.persistence()

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
            assert isinstance(dct, dict), f"Unknown type of output: {type(dct)}"
            return [dct[i] for i in sorted(dct.keys())]

        elif isinstance(container, dict):
            start, msg = perf_counter(), "cluster" if is_minima else "gas"
            self.logger.info(f"Start to optimize the {len(container)} {msg}.")
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
                    self.logger.info(f"Read '{key}' from the DB for {v}.")
                elif not is_minima and sysgraph in self.network.db_gas:
                    atoms: Atoms = self.network.db_gas[key]
                    result[label] = v = Gas.from_ase(atoms)
                    self.logger.info(f"Read '{key}' from the DB for {v}.")
                else:
                    futures.append(
                        self.executor.submit(
                            helper_cluster_optimization,
                            config=self.config,
                            graph_label=label,
                            graph=sysgraph,
                            allow_hash_change=False,
                            raise_when_fail=False,
                        )
                    )
            self.logger.info(
                f"Submit the optimization {len(futures)} jobs"
                f" by {perf_counter() - start:.2f} seconds."
            )
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
                        raise RuntimeError(msg)
                else:
                    msg = f"Unknown type: {type(sysgraph_or_msg)}"
                    raise ValueError(msg)
                self.logger.info(f"CostTime={cost_time:.2f} for {msg}")
            self.logger.info(
                "All optimization jobs are done in "
                f"{perf_counter() - start:.2f} seconds."
            )

            return result

        else:
            raise ValueError(f"Unknown type of container: {type(container)}")

    def __atoms2system(self, inp: Atoms | None) -> System:
        if inp is None:
            # parse system for first step
            try:
                self.catalyst: System = hydra_parse(self.config.system, System)
            except Exception:
                atoms = hydra_parse(self.config.system, Atoms)
                self.catalyst: System = System.from_ase(
                    atoms=atoms,
                    parse_bonds=self.config.bonds,  # type: ignore
                    parse_atoms_is_outer_or_not=True,
                )
            if self.config.system.get("attach_is_adsorbate", True):
                self.catalyst = self.catalyst.model_copy(
                    update=dict(
                        is_adsorbate=np.zeros_like(
                            self.catalyst.is_outer,
                            dtype=bool,
                        )
                    ),
                    deep=False,
                )
        elif isinstance(inp, Atoms):
            self.catalyst: System = System.from_ase(
                atoms=inp,
                parse_bonds=self.config.bonds,  # type: ignore
                parse_atoms_is_outer_or_not=True,
            )
        else:
            raise ValueError(f"Unknown type of input: {type(inp)}")

        assert isinstance(self.catalyst, System)
        assert self.catalyst.pair is not None
        assert self.catalyst.is_outer is not None
        assert self.catalyst.is_adsorbate is not None
        self.logger.info(f"Read the system: {self.catalyst}")
        return self.catalyst

    @property
    def gas_lst(self) -> list[Gas]:
        if len(self.network.metadata.basic.gas_info_lst) != 0:
            raise ValueError("First step does not support gas.")
            return self.__gas_lst
        else:
            return []

    @override
    def run(
        self,
        system: System | Atoms | None,
    ) -> dict[tuple[bool, int, str], Cluster]:
        """Return the dictionary of clusters.

        Keys:
            (is_surface, ncore, hash)
        Values:
            Cluster: the cluster of the core
        """
        if not isinstance(system, System):
            system = self.__atoms2system(system)

        oesc = bool(self.config.exploration.surface_only_explore_single_core)
        if oesc and len(self.gas_lst) == 0:
            max_ncore = 1
        else:
            max_ncore = int(self.config.exploration.max_ncore_for_surface)

        values: list[Cluster] = []
        keys: list[tuple[bool, int, str]] = []
        for core in system.get_site_core(max_ncore=max_ncore):
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
            assert system.is_outer is not None  # type: ignore
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
        self.logger.info(f"Find {len(values)} cluster for sys={system.hash}.")
        self.logger.info(f"Find {len(idxs)} unique cluster.")

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

        return result
