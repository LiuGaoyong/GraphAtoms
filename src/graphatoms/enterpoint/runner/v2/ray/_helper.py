from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import ray
from ray.actor import ActorClass, ActorProxy

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.network._clsnetwork import NETWORK_PATH_PREFIX
from graphatoms.enterpoint.runner.v2._helperv2 import (
    HelperAdsorptionResult,
    HelperDimerResult,
    HelperOptimizationResult,
)
from graphatoms.enterpoint.runner.v2._helperv2 import (
    helper_adsorption as _helper_adsorption,
)
from graphatoms.enterpoint.runner.v2._helperv2 import (
    helper_dimer as _helper_dimer,
)
from graphatoms.enterpoint.runner.v2._helperv2 import (
    helper_optimization as _helper_optimization,
)
from graphatoms.reaction import EventBase, EventInfo
from graphatoms.system import Cluster, Gas, SysGraph, System

__all__ = [
    "RayReactionNetwork",
    "RayNetworkProxy",
    "helper_optimization",
    "helper_dimer",
    "helper_adsorption",
]


# =============================================
#                Ray Actors
# ---------------------------------------------
class _RayReactionNetwork:
    """A :class:`ReactionNetwork` that lives in a single Ray actor.

    All reads and writes of the network (databases, recorder, scheduler and
    metadata) go through this actor, so concurrent exploration tasks no
    longer open the same storage from several workers at once.
    """

    def __init__(self, *, config: Config) -> None:
        self.__network = ReactionNetwork(
            path=Path(config.outputs) / NETWORK_PATH_PREFIX,
            restart=True,
            format=config.event.db_format,
            remote_mode=False,
            config=config.event,
        )

    def persistence(self) -> None:
        """Persist the network to its storage."""
        self.__network.persistence()

    def get_use_position_uuid(
        self,
        k: str | Cluster | Gas | System,
    ) -> bool:
        return self.__network.get_use_position_uuid(k)

    def object_has_been_optimized(
        self,
        graph: Cluster | Gas | System,
    ) -> SysGraph | None:
        return self.__network.object_has_been_optimized(graph)

    def exploration_can_be_finished(
        self,
        confidence: float,
        *args,
        cluster_key: str,
        gas_key: str | None = None,
        **kwargs,
    ) -> bool:
        return self.__network.exploration_can_be_finished(
            confidence,
            *args,
            cluster_key=cluster_key,
            gas_key=gas_key,
            **kwargs,
        )

    def exploration_can_be_skipped(
        self,
        thetacutoff: float,
        diffpositions: np.ndarray,
        *args,
        cluster_key: str | None,
        **kwargs,
    ) -> tuple[bool, float]:
        return self.__network.exploration_can_be_skipped(
            thetacutoff,
            diffpositions,
            *args,
            cluster_key=cluster_key,
            **kwargs,
        )

    def found(
        self,
        event: EventBase | str,
        for_gas: str | None,
        for_cluster: str,
        *,
        for_system: str | None = None,
        persist: bool = True,
        **kwargs,
    ) -> str:
        return self.__network.found(
            event,
            for_gas,
            for_cluster,
            for_system=for_system,
            persist=persist,
            **kwargs,
        )

    def write_sysgraph(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas", "cluster", "system"] = "minima",
    ) -> bool:
        return self.__network.write_sysgraph(sysgraph, type)

    def read_event(self, key: str) -> tuple[EventInfo, EventBase]:
        return self.__network.read_event(key)

    def reformat_message(self, msg: str) -> str:
        return ReactionNetwork.reformat_message(msg)

    def record_skip(self, cluster_key: str) -> None:
        """Mark the dimer search of ``cluster_key`` as skipped."""
        self.__network.recorder.cluster[cluster_key].found_skip()

    def add_adsorption(self, graph_label: str) -> None:
        """Mark the adsorption pair ``graph_label`` as explored."""
        self.__network.recorder.adsorption.add(graph_label)


RayReactionNetwork: ActorClass[_RayReactionNetwork] = ray.remote(
    _RayReactionNetwork
)
RayReactionNetworkHandle = ActorProxy[_RayReactionNetwork]


class RayNetworkProxy:
    """Synchronous view of a :class:`RayReactionNetwork` actor.

    The v2 helpers are ordinary functions that expect a ``ReactionNetwork``.
    This proxy gives them the small synchronous interface they use while
    forwarding every call to the actor with ``ray.get``.
    """

    def __init__(self, network: RayReactionNetworkHandle) -> None:
        self.__network = network

    def get_use_position_uuid(
        self,
        k: str | Cluster | Gas | System,
    ) -> bool:
        return cast(
            bool,
            ray.get(self.__network.get_use_position_uuid.remote(k)),  # type: ignore
        )

    def object_has_been_optimized(
        self,
        graph: Cluster | Gas | System,
    ) -> SysGraph | None:
        return cast(
            SysGraph | None,
            ray.get(self.__network.object_has_been_optimized.remote(graph)),  # type: ignore
        )

    def exploration_can_be_finished(
        self,
        confidence: float,
        *args,
        cluster_key: str,
        gas_key: str | None = None,
        **kwargs,
    ) -> bool:
        return cast(
            bool,
            ray.get(
                self.__network.exploration_can_be_finished.remote(  # type: ignore
                    confidence,
                    *args,
                    cluster_key=cluster_key,
                    gas_key=gas_key,
                    **kwargs,
                )
            ),
        )

    def exploration_can_be_skipped(
        self,
        thetacutoff: float,
        diffpositions: np.ndarray,
        *args,
        cluster_key: str | None,
        **kwargs,
    ) -> tuple[bool, float]:
        return cast(
            tuple[bool, float],
            ray.get(
                self.__network.exploration_can_be_skipped.remote(  # type: ignore
                    thetacutoff,
                    diffpositions,
                    *args,
                    cluster_key=cluster_key,
                    **kwargs,
                )
            ),
        )

    def reformat_message(self, msg: str) -> str:
        return ReactionNetwork.reformat_message(msg)


# =============================================
#                Ray Tasks
# ---------------------------------------------
@ray.remote
def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System,
    network: RayReactionNetworkHandle,
    **kwargs,
) -> HelperOptimizationResult:
    """Optimize ``graph`` after checking the shared network.

    This is the Ray-task counterpart of
    :func:`graphatoms.enterpoint.runner.v2._helperv2.helper_optimization`.
    The task talks to the network actor instead of opening the network
    storage itself.
    """
    return _helper_optimization(
        config=config,
        graph=graph,
        network=RayNetworkProxy(network),
        **kwargs,
    )


@ray.remote
def helper_dimer(
    config: Config,
    cluster: Cluster,
    network: RayReactionNetworkHandle,
    **kwargs,
) -> HelperDimerResult:
    """Run a dimer search using the shared network actor.

    This is the Ray-task counterpart of
    :func:`graphatoms.enterpoint.runner.v2._helperv2.helper_dimer`.
    """
    return _helper_dimer(
        config=config,
        cluster=cluster,
        network=RayNetworkProxy(network),
        **kwargs,
    )


@ray.remote
def helper_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System,
    network: RayReactionNetworkHandle,
    **kwargs,
) -> HelperAdsorptionResult:
    """Run an adsorption search using the shared network actor.

    This is the Ray-task counterpart of
    :func:`graphatoms.enterpoint.runner.v2._helperv2.helper_adsorption`.
    """
    return _helper_adsorption(
        gas=gas,
        config=config,
        graph=graph,
        network=RayNetworkProxy(network),
        **kwargs,
    )
