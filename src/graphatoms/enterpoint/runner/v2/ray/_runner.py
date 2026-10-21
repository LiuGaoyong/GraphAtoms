"""The Ray-native runner base.

This module intentionally does not inherit from :class:`RunnerABC` and does
not keep a ``self.network``/``self.executor``.  The runner owns one network
Actor and calls Ray directly; the shared pure helpers extracted from the old
runner package are reused for system parsing and cluster enumeration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ray
from ase import Atoms
from ase.units import _e as _electron_charge
from ase.units import kB as _boltzmann_constant
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.runner.common._base import (
    create_logger,
    find_clusters,
    parse_system,
)
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.bytestool import atomic_write_text

from ._actor import RayNetworkActorHandle, create_network_actor
from ._tasks import actor_call, submit_task, task_optimization

__all__ = ["RayRunnerBase"]

_kB_in_JperK = _boltzmann_constant * _electron_charge


class RayRunnerBase:
    """Base class for the Ray-native runners."""

    def __init__(self, *, config: Config) -> None:
        self.config: Config = config
        self.path = Path(config.outputs)
        self.path.mkdir(parents=True, exist_ok=True)
        self._log_length = 80
        self.logger = create_logger(config, self.path)

        self._init_ray()
        self._actor: RayNetworkActorHandle = create_network_actor(config)
        self._init_gases()
        self._init_gas_numbers()
        self._init_run_state()

        self.logger.info("=" * self._log_length)
        self.logger.info("The Configuration:\n" + OmegaConf.to_yaml(config))
        self.logger.info(f"self.path floder : {self.path}")
        self.logger.info("=" * self._log_length)

    # ------------------------------------------------------------------
    #                        initialization
    # ------------------------------------------------------------------
    def _init_ray(self) -> None:
        if ray.is_initialized():
            return
        pworkers = int(self.config.parallel_workers)
        ray.init(
            ignore_reinit_error=True,
            num_cpus=None if pworkers <= 0 else pworkers,
        )

    @property
    def actor(self) -> RayNetworkActorHandle:
        return self._actor

    def _gas_box_volume(self) -> float:
        length = self.config.gas_pressure_cache_box_length
        return (length * 1e-10) ** 3

    def _kBT_in_J(self) -> float:
        return float(self.config.event.temperature) * _kB_in_JperK

    def _init_gases(self) -> None:
        basic = actor_call(self._actor, "metadata_basic")
        gas_name_set = set(basic.gas_sticking.keys())
        gas_name_set.update(self.config.initial_gas_pressure.keys())

        new_gas_sticking_dct: dict[str, float] = {}
        self.gas_number_dct: dict[str, float] = {}
        self._gas_lst: list[Gas] = []
        for gas_name in gas_name_set:
            s = self.config.event.gas_sticking.get(gas_name, 1.0)
            gas = Gas.from_name(
                name_or_smiles=gas_name,
                sticking=float(s) if s is not None else 1.0,
                parse_bonds=self.config.bonds,  # type: ignore
            )
            gas_k = gas.get_key_for_metadata()
            if actor_call(self._actor, "contain_sysgraph", gas_k, "gas"):
                atoms = actor_call(self._actor, "read_atoms", gas_k, "gas")
                gas = Gas.from_ase(atoms)
            else:
                result = ray.get(
                    submit_task(
                        task_optimization,
                        actor=self._actor,
                        config=self.config,
                        graph=gas,
                        graph_label=gas_k,
                        allow_hash_change=False,
                        allow_not_connected=False,
                        run_vibration=True,
                        deep_copy=True,
                    )
                )
                if result.result is None:
                    msg = f"Failed to optimize gas({gas_name})."
                    self.logger.error(msg)
                    raise ValueError(msg)
                gas = result.result
                actor_call(
                    self._actor,
                    "write_sysgraph",
                    gas,
                    "gas",
                )
                self.logger.info(
                    f"Optimized gas({gas_name}) in "
                    f"{result.cost_time or 0.0:.2f} seconds."
                )
            self._gas_lst.append(gas)
            p = self.config.initial_gas_pressure.get(gas_name, 0.0)
            V, kBT = self._gas_box_volume(), self._kBT_in_J()
            self.gas_number_dct[gas_k] = p * V / kBT
            new_gas_sticking_dct[gas.smiles] = (
                float(s) if s is not None else 1.0
            )

        actor_call(self._actor, "set_gas_sticking", new_gas_sticking_dct)
        self.config.event.gas_sticking = new_gas_sticking_dct
        self._gas_sticking = dict(new_gas_sticking_dct)
        actor_call(self._actor, "persistence")
        self.logger.info(f"New gas sticking dict: {new_gas_sticking_dct}")

    def _init_gas_numbers(self) -> None:
        if str(self.config.run_type).lower() != "otfkmc":
            return
        gas_number_path = self.path.joinpath("gas_number.json")
        if self.config.restart and gas_number_path.exists():
            with gas_number_path.open() as f:
                self.gas_number_dct.update(json.load(f))
            self.logger.info(f"Load gas number dict from {gas_number_path}")
        else:
            self.persist_gas_number_dct()

    def _init_run_state(self) -> None:
        """Hook for subclasses to initialize run-specific state."""

    # ------------------------------------------------------------------
    #                          properties
    # ------------------------------------------------------------------
    @property
    def gas_lst(self) -> list[Gas]:
        if len(self._gas_sticking) != 0:
            return self._gas_lst
        return []

    @property
    def gas_pressure_dct(self) -> dict[str, float]:
        V, kBT = self._gas_box_volume(), self._kBT_in_J()
        result: dict[str, float] = {}
        for gas_key, N in self.gas_number_dct.items():
            N = float(0.0 if N <= 0 else N)
            result[gas_key] = N * kBT / V
        return result

    def persist_gas_number_dct(self) -> None:
        atomic_write_text(
            self.path.joinpath("gas_number.json"),
            json.dumps(self.gas_number_dct, indent=4),
        )

    # ------------------------------------------------------------------
    #                        system helpers
    # ------------------------------------------------------------------
    def get_system_for(self, inp: Atoms | None) -> System:
        return parse_system(self.config, inp, self.logger)

    def get_batch_system_for(self) -> list[System]:
        from ase.io import iread

        from graphatoms.enterpoint.config.atoms import AseReadAtomsConfig

        if self.config.run_type == "rxngen":
            if isinstance(self.config.atoms, AseReadAtomsConfig):
                return [
                    self.get_system_for(atoms)
                    for atoms in iread(self.config.atoms.filename)
                ]
        return [self.get_system_for(None)]

    def analysis(
        self,
        system: System | Atoms | None,
    ) -> tuple[dict[tuple[bool, int, str], Cluster], str, float]:
        if not isinstance(system, System):
            system = self.get_system_for(system)
        sys_k = system.get_key_for_metadata(False)
        if actor_call(self._actor, "contain_sysgraph", sys_k, "system"):
            atoms = actor_call(self._actor, "read_atoms", sys_k, "system")
            system = System.from_ase(atoms)
            self.logger.info(f"Load system({sys_k}) from database.")
        elif bool(self.config.exploration.optimization_for_system):
            result = ray.get(
                submit_task(
                    task_optimization,
                    actor=self._actor,
                    config=self.config,
                    graph=system,
                    graph_label=sys_k,
                    allow_hash_change=True,
                    allow_not_connected=False,
                    run_vibration=False,
                    deep_copy=True,
                )
            )
            if result.result is None:
                self.logger.warning(result.message)
                self.logger.info(
                    f"Optimization system({sys_k}) failed. skip it."
                )
                return {}, "", np.nan
            if not isinstance(result.result, System):
                msg = f"Unknown type of output: {type(result.result)}"
                self.logger.error(msg)
                raise ValueError(msg)
            system = result.result
            actor_call(self._actor, "write_sysgraph", system, "system")
            self.logger.info(
                f"Optimization system({sys_k}) successfully. E={system.energy}"
            )

        result, energy = find_clusters(
            self.config,
            system,
            self.gas_lst,
            self.logger,
        )
        return result, sys_k, energy

    # ------------------------------------------------------------------
    #                           lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Persist the network and stop the actor."""
        try:
            actor_call(self._actor, "persistence")
        finally:
            ray.kill(self._actor, no_restart=True)

    def run(self) -> None:
        raise NotImplementedError
