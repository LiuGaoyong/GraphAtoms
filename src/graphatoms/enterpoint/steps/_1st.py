import os
from typing import override

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

from concurrent.futures import Future
from time import perf_counter

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.parallel import get_executor
from graphatoms.system import Cluster, Gas, System  # type: ignore
from graphatoms.utils.asetools import call_optimization, call_vib
from graphatoms.utils.parser import hydra_parse

from ._0abc import BaseABC


class FirstStep(BaseABC):
    """The abstract base class for the runner."""

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        self.__gas_lst: list[Gas] = []
        for gas_info in self.network.metadata.basic.gas_info_lst:
            self.__gas_lst.append(
                Gas.from_name(
                    gas_info.name,
                    sticking=gas_info.sticking,
                    pressure=gas_info.pressure,
                    parse_bonds=self.config.bonds,  # type: ignore
                )
            )

        # optimize the gas in parallel mode
        with get_executor(
            name=self.pmode,
            max_workers=self.pworkers,
        ) as executor:
            futures: list[tuple[int, Future]] = []
            for i, gas in enumerate(self.__gas_lst):
                futures.append(
                    (
                        i,
                        executor.submit(
                            self.helper_cluster_optimization,
                            config=self.config,
                            cluster=gas,
                            allow_hash_change=False,
                            raise_on_failed=False,
                        ),
                    )
                )
            for i, f in futures:
                gas_or_msg, cost_time = f.result()
                if isinstance(gas_or_msg, Gas):
                    msg: str = f"Optimization (success): {gas_or_msg}."
                    self.__gas_lst[i] = gas_or_msg
                elif isinstance(gas_or_msg, str):
                    msg = str(gas_or_msg)
                    raise RuntimeError(msg)
                else:
                    msg = f"Unknown type of output: {type(gas_or_msg)}."
                    raise ValueError(msg)
                self.logger.info(f"CostTime={cost_time:.2f} for {msg}")

    def __atoms2system(self, inp: Atoms | None) -> System:
        if inp is None:
            # parse system for first step
            try:
                self.catalyst: System = hydra_parse(self.config.system, System)
            except Exception:
                self.catalyst: System = System.from_ase(
                    atoms=hydra_parse(self.config.system, Atoms),
                    parse_bonds=self.config.bonds,  # type: ignore
                    attach_is_adsorbate=True,
                    parse_bonds_outer=True,
                )
        elif isinstance(inp, Atoms):
            self.catalyst: System = System.from_ase(
                atoms=inp,
                parse_bonds=self.config.bonds,  # type: ignore
                attach_is_adsorbate=True,
                parse_bonds_outer=True,
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
            if system.move_fix_tag is None:  # type: ignore
                move_fix_tag = np.zeros_like(system.is_outer)  # type: ignore
            else:
                move_fix_tag = np.asarray(system.move_fix_tag)
                assert move_fix_tag.shape == system.is_outer.shape
            is_moved = move_fix_tag >= 0
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
        result: dict[tuple[bool, int, str], Cluster] = {}
        self.logger.info(f"Start to optimize the {len(idxs)} clusters.")
        start: float = perf_counter()
        with get_executor(
            name=self.pmode,
            max_workers=self.pworkers,
        ) as executor:
            futures: list[tuple[int, Future]] = []
            for i in idxs:
                cluster: Cluster = values[int(i)]
                if cluster not in self.network.db_minima:
                    futures.append(
                        (
                            i,
                            executor.submit(
                                self.helper_cluster_optimization,
                                config=self.config,
                                cluster=values[int(i)],
                                allow_hash_change=False,
                                raise_on_failed=False,
                            ),
                        )
                    )
            self.logger.info(
                f"Submit the optimization jobs by "
                f"{perf_counter() - start:.2f} seconds."
            )
            for i, f in futures:
                cluster_or_msg, cost_time = f.result()
                if isinstance(cluster_or_msg, Cluster):
                    msg: str = f"Optimization (success): {cluster_or_msg}."
                    result[keys[int(i)]] = cluster_or_msg
                elif isinstance(cluster_or_msg, str):
                    msg = str(cluster_or_msg)
                else:
                    msg = f"Unknown type of output: {type(cluster_or_msg)}."
                    raise ValueError(msg)
                self.logger.info(f"CostTime={cost_time:.2f} for {msg}")
            self.logger.info(
                "All optimization jobs are done in "
                f"{perf_counter() - start:.2f} seconds."
            )
        result.update({keys[int(i)]: values[int(i)] for i in idxs})
        return result

    @staticmethod
    def helper_cluster_optimization(
        config: Config,
        cluster: Cluster | Gas,
        *,
        allow_hash_change: bool = False,
        raise_on_failed: bool = False,
        **kwargs,
    ) -> tuple[Cluster | Gas | str, float]:
        """Optimize the cluster, analyze its vibrations and save it.

        Returns the optimized cluster and the time cost in seconds.
        """
        start = perf_counter()
        calc: Calculator = hydra_parse(
            config.calculator,  # type: ignore
            Calculator,
        )
        if isinstance(cluster, Gas):
            type = "gas"
        else:
            type = "minima"

        # call optimization
        lst, coveraged = call_optimization(
            atoms=cluster.to_ase().copy(),
            calc=calc,
            method=str(config.optimizer.method).upper(),
            max_steps=int(config.optimizer.steps),
            fmax=float(config.optimizer.fmax),
        )
        if not coveraged:
            msg = f"Optimization (failed): {cluster}."
            if raise_on_failed:
                raise BaseABC.OptimizationFailed(msg)
            else:
                return msg, perf_counter() - start

        # analyze vibrations & convert to Cluster/Gas
        new_atoms = lst[-1]
        freq, _ = call_vib(atoms=new_atoms, calc=calc)
        f = new_atoms.get_forces()
        if type == "minima":
            result = Cluster.from_ase(
                atoms=new_atoms,
                parse_bonds=config.bonds,  # type: ignore
                parse_bonds_distance=False,
                parse_bonds_order=False,
                energy=new_atoms.get_potential_energy(),
                fmax=np.linalg.norm(f, axis=1).max(),
                frequencies=freq,
                nadsorbate=0,
            )
        else:
            result = Gas.from_ase(
                atoms=new_atoms,
                sticking=cluster.sticking,  # type: ignore
                pressure=cluster.pressure,  # type: ignore
                parse_bonds=config.bonds,  # type: ignore
                energy=new_atoms.get_potential_energy(),
                fmax=np.linalg.norm(f, axis=1).max(),
                parse_bonds_distance=False,
                parse_bonds_order=False,
                frequencies=freq,
            )
        assert isinstance(result, (Cluster, Gas))
        if not result.check_minima(
            fmax=config.event.max_force,
            fqmin=config.event.min_frequency,
        ):
            msg = f"Check minima failed for {cluster}."
            if raise_on_failed:
                raise BaseABC.CheckVibrationFailed(msg)
            else:
                return msg, perf_counter() - start

        # check hash change or not
        if not allow_hash_change and result.hash != cluster.hash:  # type: ignore
            msg = "Cluster'hash changed after optimization."
            if raise_on_failed:
                raise ValueError(msg)
            else:
                return msg, perf_counter() - start
        return result, perf_counter() - start
