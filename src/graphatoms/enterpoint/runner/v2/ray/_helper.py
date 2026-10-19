from io import StringIO
from typing import Any, Literal, override

import numpy as np
import ray
from ray.actor import ActorProxy

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
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
class _RayReactionNetwork(ReactionNetwork):
    @ray.method()
    @override
    def persistence(self) -> None:  # type: ignore
        return super().persistence()

    @ray.method(num_returns=2)
    @override
    def read_event(self, key: str) -> tuple[EventInfo, EventBase]:  # type: ignore
        return super().read_event(key)

    @ray.method(num_returns=1)
    @override
    def object_has_been_optimized(self, graph: SysGraph) -> SysGraph | None:  # type: ignore
        return super().object_has_been_optimized(graph)


# RayReactionNetwork: ActorClass[_RayReactionNetwork] = ray.remote(
#     _RayReactionNetwork  # object: ActorProxy[_RayReactionNetwork]
# )


# =============================================
#                Ray Tasks
# ---------------------------------------------
@ray.remote
def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System,
    network: ActorProxy[_RayReactionNetwork],
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
