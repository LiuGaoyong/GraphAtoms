"""The extended helper functions for exploration and optimization.

These helper functions only run calculation, and check the result.
They also check whether the calculation is necessary and output logger.
"""

from io import StringIO
from typing import Literal

import numpy as np
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.runner.common import _helper as funcs
from graphatoms.enterpoint.runner.common._base import (
    LOGURU_FORMAT,
    LOGURU_FORMAT_LENGTH,
    Core,
    Logger,
)
from graphatoms.reaction import Desorption, Reaction
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.asetools import call_dimer_displace
from graphatoms.utils.parser import hydra_parse


def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System,
    *,
    allow_hash_change: bool = True,
    allow_not_connected: bool = False,
    run_vibration: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[Cluster | Gas | System | None, str]:
    """Optimize the input graph, analyze its vibrations and check it.

    Parameters:
        config(Config):  The configuration object.
        graph(Cluster | Gas | System):  The graph to optimization.
        allow_hash_change: Whether to allow the hash change after optimization.
        run_vibration: Whether to run the vibration check after optimization.
        deep_copy: Whether to deep copy the optimized graph.

    Returns:
        the first element is the optimized graph (None means failed).
        the second element is the logging message.
    """
    assert isinstance(graph, Cluster | Gas | System), (
        "The `graph` must be a Cluster, Gas, or System."
    )
    log_container, logger, network = (
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
        ReactionNetwork.from_config(config, restart=True, remote_mode=True),
    )
    logger.add(log_container, level="DEBUG", format=LOGURU_FORMAT)

    # --------------------------------------------
    #  check whether the graph has been optimized
    # --------------------------------------------
    use_pos_uuid = network.get_use_position_uuid(graph)
    graph_label = graph.get_key_for_metadata(use_pos_uuid)
    result = network.object_has_been_optimized(graph=graph)
    if result is not None:
        msg = f"Read '{graph_label}' from the DB for {result}."
        logger.info(network.reformat_message(msg))
        if not isinstance(result, Cluster | Gas | System):
            msg = f"Unknown type of result: {type(result)}. "
            msg += "Only Cluster, Gas, and System are allowed."
            logger.error(network.reformat_message(msg))
            return None, log_container.getvalue()[LOGURU_FORMAT_LENGTH:]
        else:
            return result, log_container.getvalue()[LOGURU_FORMAT_LENGTH:]

    # --------------------------------------------
    #               optimize the graph
    # --------------------------------------------
    result, label, cost_time = funcs.helper_optimization(
        graph=graph,
        config=config,
        graph_label=graph_label,
        allow_hash_change=allow_hash_change,
        allow_not_connected=allow_not_connected,
        run_vibration=run_vibration,
        raise_when_fail=False,
        deep_copy=deep_copy,
        **kwargs,
    )
    msg = f"CostTime={cost_time:.2f} for "
    if isinstance(result, Gas | Cluster | System):
        msg += f"Optimization (success) for {label}({result})."
        logger.info(network.reformat_message(msg))
        return result, log_container.getvalue()[LOGURU_FORMAT_LENGTH:]
    elif isinstance(result, str):
        logger.error(network.reformat_message(msg + result))
        return None, log_container.getvalue()[LOGURU_FORMAT_LENGTH:]
    else:
        msg += f"Unknown type of result for optimization: {type(result)}. "
        msg += "Only Cluster, Gas, and System are allowed."
        logger.error(network.reformat_message(msg))
        return None, log_container.getvalue()[LOGURU_FORMAT_LENGTH:]


def helper_dimer(
    config: Config,
    cluster: Cluster,
    *,
    allow_fixed_bonds_change: bool = False,
    allow_desorption: bool | Literal["all", "input"] = "all",
    allow_metal_element_in_gas: bool = False,
    displacement: np.ndarray | None = None,
    deep_copy: bool = True,
    **kwargs,
) -> tuple[
    Reaction | Desorption | None,
    str,
    bool,
    bool,
    bool,
    np.ndarray | None,
]:
    """Dimer search for transition state.

    Parameters:
        config(Config):  The configuration object.
        cluster(Cluster):  The cluster to dimer search.
        allow_fixed_bonds_change: Whether to
            allow the fixed bonds change after dimer.
        allow_desorption: Whether to
            allow the desorption of the gas after dimer search.
                If 'all', then allow the desorption
                    of all the gas molecules.
                If 'input', then allow the desorption of
                    the gas molecules only in the input file.
        displacement: The displacement vector for dimer.
        deep_copy: Whether to deep copy the optimized graph.

    Returns:
        the first element is the event object (None means failed).
        the second element is the logging message.
        the 3-5th element is the flag of whether this
            exploration has been finished, skiped, or failed.
        the 6th element is the displacement vector (trajectory positions).
    """
    assert isinstance(cluster, Cluster), "The `cluster` must be a Cluster."
    log_container, logger, network = (
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
        ReactionNetwork.from_config(config, restart=True, remote_mode=True),
    )
    logger.add(log_container, level="DEBUG", format=LOGURU_FORMAT)
    is_finished, is_skipped, is_failed = False, False, False

    # -----------------------------------------------------------
    # check whether the cluster has been explored
    # -----------------------------------------------------------
    use_pos_uuid = network.get_use_position_uuid(cluster)
    cluster_key = cluster.get_key_for_metadata(use_pos_uuid)
    if network.exploration_can_be_finished(
        confidence=config.exploration.maxconfidence,
        cluster_key=cluster_key,
    ):
        is_finished = True
        msg = f"the cluster {cluster_key} has been explored."
        # logger.info(network.reformat_message(msg))
        return None, msg, is_finished, is_skipped, is_failed, None

    # -----------------------------------------------------------
    # check if the cluster is at a minimum & prepare something
    # -----------------------------------------------------------
    if not cluster.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    ):
        msg = f"Cluster {cluster_key} is not at a minimum."
        # logger.error(network.reformat_message(msg))
        return None, msg, is_finished, is_skipped, is_failed, None

    # -----------------------------------------------------------
    # check whether this dimer search can be skipped
    # -----------------------------------------------------------
    thetacutoff = float(config.exploration.thetacutoff)
    if thetacutoff > 0:
        raise NotImplementedError("thetacutoff is not implemented yet.")
        displacement = call_dimer_displace(
            atoms=cluster.to_ase().copy(),
            parse_mask_from_atoms=True,
            calc=hydra_parse(
                config.calculator,  # type: ignore
                Calculator,
            ),
            mask=None,
        )
        can_be_skip, cosine = network.exploration_can_be_skipped(
            cluster_key=cluster_key,
            thetacutoff=thetacutoff,
            diffpositions=displacement,
        )
        if can_be_skip:
            is_skipped = True
            msg = f"Skip {cluster_key} exploration. Cosine={cosine:.4f}."
            return None, msg, is_finished, is_skipped, is_failed, None
    else:
        displacement = None

    # -----------------------------------------------------------
    # do dimer search
    # -----------------------------------------------------------
    rxn_or_msg, (_, lst), cost_time, _ = funcs.helper_dimer(
        config=config,
        graph_label=cluster_key,
        allow_fixed_bonds_change=allow_fixed_bonds_change,
        allow_metal_element_in_gas=allow_metal_element_in_gas,
        allow_desorption=allow_desorption,
        displacement=displacement,
        raise_when_fail=False,
        deep_copy=deep_copy,
        graph=cluster,
        **kwargs,
    )
    arr = np.array([a.positions for a in lst])
    if isinstance(rxn_or_msg, str):
        is_failed = True
        msg = f"Dimer search {cluster_key} by {cost_time:.2f}"
        msg += f" seconds because of {rxn_or_msg}"
        return None, rxn_or_msg, is_finished, is_skipped, is_failed, arr
    else:
        msg = "Dimer search successfully, and got "
        msg += f"{cluster_key} by {cost_time:.2f} seconds."
        return rxn_or_msg, msg, is_finished, is_skipped, is_failed, arr


# def helper_adsorption(
#     gas: Gas,
#     config: Config,
#     graph: Cluster | System | SysGraph,
#     *,
#     graph_label: Any | None = None,
#     allow_hash_change: bool = True,
#     raise_when_fail: bool = True,
#     irun: int | None = None,
#     deep_copy: bool = True,
#     **kwargs,
# ) -> tuple[Adsorption | str, Any, float, str]:
#     """Helper function for adsorption process.

#     Returns:
#         1. the adsorption object
#         2. the graph label
#         3. the cost time
#         4. the result type
#     """
