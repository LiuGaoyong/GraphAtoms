from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import ray
from ray.actor import ActorClass, ActorProxy

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.network._clsnetwork import NETWORK_PATH_PREFIX
from graphatoms.enterpoint.runner.common._base import (
    LOGURU_FORMAT,
    Core,
    Logger,
)
from graphatoms.reaction import (
    Adsorption,
    Desorption,
    EventBase,
    EventInfo,
    Reaction,
)
from graphatoms.system import Cluster, Gas, SysGraph, System


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
            ray.get(self.__network.get_use_position_uuid.remote(k)),
        )

    def object_has_been_optimized(
        self,
        graph: Cluster | Gas | System,
    ) -> SysGraph | None:
        return cast(
            SysGraph | None,
            ray.get(self.__network.object_has_been_optimized.remote(graph)),
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
                self.__network.exploration_can_be_finished.remote(
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
                self.__network.exploration_can_be_skipped.remote(
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
    *,
    graph_label: Any | None = None,
    allow_hash_change: bool = True,
    allow_not_connected: bool = False,
    raise_when_fail: bool = False,
    run_vibration: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Cluster | Gas | System | SysGraph | str, Any, float, str]:  # type: ignore
    """Optimize the input graph, analyze its vibrations and check it.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        graph: The graph to optimization.
        allow_hash_change: Whether to allow the hash change after optimization.
        raise_when_fail: Whether to raise exception after optimization failed.
        run_vibration: Whether to run the vibration check after optimization.
        deep_copy: Whether to deep copy the optimized graph.

    Raises:
        OptimizationFailed: If the optimization failed.
        CheckVibrationFailed: If the check of vibration failed.
        HelperException: If the graph hash changed after optimization.

    Returns:
        the optimized graph, graph_id, and the time cost in seconds.
    """
    log_container, logger = (
        StringIO(),
        Logger(
            core=Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        ),
    )
    logger.add(log_container, level="DEBUG", format=LOGURU_FORMAT)

    # for cls, db in [
    #                     (Gas, self.network.db_gas),
    #                     (Cluster, self.network.db_cluster),
    #                     (System, self.network.db_system),
    #                 ]:
    #                     if isinstance(sysgraph, cls) and sysgraph in db:
    #                         atoms: Atoms = db[key]
    #                         result[label] = v = cls.from_ase(atoms)
    #                         msg = f"Read '{key}' from the DB for {v}."
    #                         self.logger.info(self._reformat_message(msg))
    #                         has_been_explored = True
    #                         break
    #                 if not has_been_explored:

    # return log_container.getvalue().strip()


def helper_dimer(
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_fixed_bonds_change: bool = False,
    allow_desorption: bool | Literal["all", "input"] = "all",
    allow_metal_element_in_gas: bool = False,
    displacement: np.ndarray | None = None,
    network: ReactionNetwork | None = None,
    raise_when_fail: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Reaction | Desorption | str, Any, float, str]:  # type: ignore
    """Dimer search for transition state.

    Parameters:
        config: The configuration object.
        graph_label: The label of the graph.
        allow_fixed_bonds_change: Whether to
            allow the fixed bonds change after dimer.
        allow_desorption: Whether to
            allow the desorption of the gas after dimer search.
                If 'all', then allow the desorption
                    of all the gas molecules.
                If 'input', then allow the desorption of
                    the gas molecules only in the input file.
        displacement: The displacement vector for dimer.
        raise_when_fail: Whether to raise exception when failed.
        deep_copy: Whether to deep copy the optimized graph.

    Raises:
        OptimizationFailed: If the dimer failed.
        CheckVibrationFailed: If the check of vibration failed.
        HelperException: If the fixed bonds changed after dimer search.

    Returns:
        the optimized graph,
        graph_id,
        the time cost in seconds,
        and the result type.
    """


def helper_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System | SysGraph,
    *,
    graph_label: Any | None = None,
    allow_hash_change: bool = True,
    raise_when_fail: bool = True,
    irun: int | None = None,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Adsorption | str, Any, float, str]:  # type: ignore
    """Helper function for adsorption process.

    Returns:
        1. the adsorption object
        2. the graph label
        3. the cost time
        4. the result type
    """
