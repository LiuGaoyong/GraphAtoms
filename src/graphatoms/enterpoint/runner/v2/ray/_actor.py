"""The asynchronous Ray actor that owns the reaction network.

The actor is the single owner of :class:`ReactionNetwork` for a Ray run.
Every read and write of the network storage (recorder, scheduler, metadata
and the five databases) goes through this actor, so the workers never open
the ``event/`` storage themselves.

All methods are declared ``async`` to give callers an awaitable protocol,
but their bodies execute synchronously.  This is intentional: h5py/sqlite
handles and the pydantic state of ``ReactionNetwork`` are not designed for
concurrent access.  A method body without ``await`` runs atomically on the
actor's event loop, which keeps the single-writer invariant without setting
``max_concurrency``.  Do not add ``await`` inside these methods; an AST test
guards this invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import ray
from ase import Atoms
from ray.actor import ActorClass, ActorProxy

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.network._clsnetwork import NETWORK_PATH_PREFIX
from graphatoms.enterpoint.network._metadata import (
    _MetaDataBasic as MetaDataBasic,
)
from graphatoms.reaction import EventBase, EventInfo
from graphatoms.system import Cluster, Gas, SysGraph, System

__all__ = [
    "RayNetworkActor",
    "RayNetworkActorHandle",
    "create_network_actor",
]


class _RayNetworkActor:
    """The single owner of a :class:`ReactionNetwork` in a Ray run."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._network = ReactionNetwork(
            path=Path(config.outputs) / NETWORK_PATH_PREFIX,
            restart=bool(config.restart),
            format=config.event.db_format,
            remote_mode=False,
            config=config.event,
        )

    # ------------------------------------------------------------------
    #                          lifecycle
    # ------------------------------------------------------------------
    async def persistence(self) -> None:
        """Persist the recorder, scheduler and metadata to storage."""
        self._network.persistence()

    async def set_gas_sticking(self, dct: dict[str, float]) -> None:
        """Replace the gas sticking dictionary in the metadata."""
        self._network.metadata.basic.gas_sticking = dict(dct)

    # ------------------------------------------------------------------
    #                             reads
    # ------------------------------------------------------------------
    async def contain_sysgraph(
        self,
        key: str,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> bool:
        return self._network.contain_sysgraph(key, type)

    async def read_atoms(
        self,
        key: str,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> Atoms:
        return self._network.read_atoms(key, type)

    async def read_sysgraph(
        self,
        key: str,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
        return_sysgraph: bool = True,
    ) -> SysGraph | Cluster | Gas | System:
        return self._network.read_sysgraph(key, type, return_sysgraph)

    async def read_event_info(self, key: str) -> EventInfo:
        return self._network.read_event_info(key)

    async def read_event(self, key: str) -> tuple[EventInfo, EventBase]:
        return self._network.read_event(key)

    async def object_has_been_optimized(
        self,
        graph: Cluster | Gas | System,
    ) -> SysGraph | None:
        return self._network.object_has_been_optimized(graph)

    async def exploration_can_be_finished(
        self,
        confidence: float,
        cluster_key: str,
        gas_key: str | None = None,
    ) -> bool:
        return self._network.exploration_can_be_finished(
            confidence=confidence,
            cluster_key=cluster_key,
            gas_key=gas_key,
        )

    async def exploration_can_be_skipped(
        self,
        thetacutoff: float,
        diffpositions: np.ndarray,
        cluster_key: str,
    ) -> tuple[bool, float]:
        return self._network.exploration_can_be_skipped(
            thetacutoff=thetacutoff,
            diffpositions=diffpositions,
            cluster_key=cluster_key,
        )

    async def key_rxn(self) -> list[str]:
        return list(self._network.metadata.table.key_rxn)

    async def metadata_basic(self) -> MetaDataBasic:
        return self._network.metadata.basic

    async def db_system_items(self) -> dict[str, Atoms]:
        return dict(self._network.db_system.items())

    async def bkl_solver(
        self,
        matching_dct: dict[tuple[str, bool], np.ndarray],
        env_pressure: dict[str, float],
    ) -> tuple[pd.DataFrame, str, bool, float, float]:
        return self._network.metadata.bkl_solver(
            matching_dct=matching_dct,
            env_pressure=env_pressure,
        )

    # ------------------------------------------------------------------
    #                            writes
    # ------------------------------------------------------------------
    async def write_sysgraph(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> bool:
        return self._network.write_sysgraph(sysgraph, type)

    async def found(
        self,
        event: EventBase | str,
        for_gas: str | None,
        for_cluster: str,
        *,
        for_system: str | None = None,
        persist: bool = True,
    ) -> str:
        return self._network.found(
            event,
            for_gas=for_gas,
            for_cluster=for_cluster,
            for_system=for_system,
            persist=persist,
        )

    async def record_skip(self, cluster_key: str) -> None:
        self._network.recorder.cluster[cluster_key].found_skip()

    async def add_adsorption(self, graph_label: str) -> None:
        self._network.recorder.adsorption.add(graph_label)


RayNetworkActor: ActorClass[_RayNetworkActor] = ray.remote(
    num_cpus=0,
)(_RayNetworkActor)

RayNetworkActorHandle = ActorProxy[_RayNetworkActor]


def create_network_actor(config: Config) -> RayNetworkActorHandle:
    """Create the reaction-network actor for one Ray run."""
    return RayNetworkActor.remote(config)
