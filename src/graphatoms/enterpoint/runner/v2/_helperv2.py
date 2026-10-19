"""The extended helper functions for exploration and optimization.

These helper functions only run calculation, and check the result.
They also check whether the calculation is necessary and output logger.
"""

import dataclasses as dc
from io import StringIO
from typing import Any, Literal

import numpy as np
from ase import Atoms
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
from graphatoms.reaction import Adsorption, Desorption, Reaction
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.asetools import call_dimer_displace
from graphatoms.utils.parser import hydra_parse

__all__ = [
    "helper_dimer",
    "helper_adsorption",
    "helper_optimization",
    "HelperOptimizationResult",
    "HelperAdsorptionResult",
    "HelperDimerResult",
    "HelperResult",
]


@dc.dataclass
class HelperResult:
    result: Any
    message: str
    cost_time: float | None = None


@dc.dataclass
class HelperOptimizationResult(HelperResult):
    result: Cluster | Gas | System | None


def helper_optimization(
    config: Config,
    graph: Cluster | Gas | System,
    *,
    allow_hash_change: bool = True,
    allow_not_connected: bool = False,
    run_vibration: bool = True,
    deep_copy: bool = True,
    **kwargs,
) -> HelperOptimizationResult:
    """Optimize the input graph, analyze its vibrations and check it.

    Parameters:
        config(Config):  The configuration object.
        graph(Cluster | Gas | System):  The graph to optimization.
        allow_hash_change: Whether to allow the hash change after optimization.
        run_vibration: Whether to run the vibration check after optimization.
        deep_copy: Whether to deep copy the optimized graph.
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
            return HelperOptimizationResult(
                result=None,
                message=log_container.getvalue()[LOGURU_FORMAT_LENGTH:],
            )
        else:
            return HelperOptimizationResult(
                result=result,
                message=log_container.getvalue()[LOGURU_FORMAT_LENGTH:],
            )

    # --------------------------------------------
    #               optimize the graph
    # --------------------------------------------
    result, label, cost_time = funcs.helper_optimization(
        graph=graph,
        config=config,
        **(
            kwargs
            | dict(
                graph_label=graph_label,
                allow_hash_change=allow_hash_change,
                allow_not_connected=allow_not_connected,
                run_vibration=run_vibration,
                raise_when_fail=False,
                deep_copy=deep_copy,
            )
        ),
    )
    msg = f"CostTime={cost_time:.2f} for "
    if isinstance(result, Gas | Cluster | System):
        msg += f"Optimization (success) for {label}({result})."
        logger.info(network.reformat_message(msg))
        return HelperOptimizationResult(
            result=result,
            message=log_container.getvalue()[LOGURU_FORMAT_LENGTH:],
            cost_time=cost_time,
        )
    elif isinstance(result, str):
        logger.error(network.reformat_message(msg + result))
        return HelperOptimizationResult(
            result=None,
            message=log_container.getvalue()[LOGURU_FORMAT_LENGTH:],
            cost_time=cost_time,
        )
    else:
        msg += f"Unknown type of result for optimization: {type(result)}. "
        msg += "Only Cluster, Gas, and System are allowed."
        logger.error(network.reformat_message(msg))
        return HelperOptimizationResult(
            result=None,
            message=log_container.getvalue()[LOGURU_FORMAT_LENGTH:],
            cost_time=cost_time,
        )


@dc.dataclass
class HelperDimerResult(HelperResult):
    result: Reaction | Desorption | None
    displacement_trajectory: np.ndarray | None = None
    has_been_finished: bool = False
    is_skipped: bool = False
    is_failed: bool = False
    for_cluster: str = ""


def helper_dimer(
    config: Config,
    cluster: Cluster,
    *,
    allow_fixed_bonds_change: bool = False,
    allow_desorption: bool | Literal["all", "input"] = "all",
    allow_metal_element_in_gas: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> HelperDimerResult:
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
        deep_copy: Whether to deep copy the optimized graph.
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

    # -----------------------------------------------------------
    # check whether the cluster has been explored
    # -----------------------------------------------------------
    use_pos_uuid = network.get_use_position_uuid(cluster)
    cluster_key = cluster.get_key_for_metadata(use_pos_uuid)
    if network.exploration_can_be_finished(
        confidence=config.exploration.maxconfidence,
        cluster_key=cluster_key,
    ):
        msg = f"the cluster {cluster_key} has been explored."
        # logger.info(network.reformat_message(msg))
        return HelperDimerResult(
            result=None,
            message=msg,
            has_been_finished=True,
            for_cluster=cluster_key,
        )

    # -----------------------------------------------------------
    # check if the cluster is at a minimum & prepare something
    # -----------------------------------------------------------
    if not cluster.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    ):
        msg = f"Cluster {cluster_key} is not at a minimum."
        # logger.error(network.reformat_message(msg))
        return HelperDimerResult(
            result=None,
            message=msg,
            is_failed=True,
            for_cluster=cluster_key,
        )

    # -----------------------------------------------------------
    # check whether this dimer search can be skipped
    # -----------------------------------------------------------
    thetacutoff = float(config.exploration.thetacutoff)
    if thetacutoff > 0:
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
            msg = f"Skip {cluster_key} exploration. Cosine={cosine:.4f}."
            return HelperDimerResult(
                result=None,
                message=msg,
                is_skipped=True,
                for_cluster=cluster_key,
            )
    else:
        displacement = None

    # -----------------------------------------------------------
    # do dimer search
    # -----------------------------------------------------------
    rxn_or_msg, (_, lst), cost_time, _ = funcs.helper_dimer(
        config=config,
        graph=cluster,
        **(
            kwargs
            | dict(
                graph_label=cluster_key,
                allow_fixed_bonds_change=allow_fixed_bonds_change,
                allow_metal_element_in_gas=allow_metal_element_in_gas,
                allow_desorption=allow_desorption,
                displacement=displacement,
                raise_when_fail=False,
                deep_copy=deep_copy,
            )
        ),
    )
    arr = np.array([a.positions for a in lst])
    if isinstance(rxn_or_msg, str):
        msg = f"Dimer search for {cluster_key} failed by "
        msg += f"{cost_time:.2f} seconds because of {rxn_or_msg}"
        return HelperDimerResult(
            result=None,
            message=msg,
            is_failed=True,
            displacement_trajectory=arr,
            for_cluster=cluster_key,
            cost_time=cost_time,
        )
    else:
        msg = f"Dimer search for {cluster_key} successfully, and"
        msg += f" got {rxn_or_msg} by {cost_time:.2f} seconds."
        return HelperDimerResult(
            result=rxn_or_msg,
            message=msg,
            displacement_trajectory=arr,
            for_cluster=cluster_key,
            cost_time=cost_time,
        )


@dc.dataclass
class HelperAdsorptionResult(HelperResult):
    result: Adsorption | None
    for_cluster: str = ""
    for_gas: str = ""


def helper_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System,
    *,
    initial_atoms: Atoms | None = None,
    allow_fixed_atoms_adsorption: bool = False,
    deep_copy: bool = True,
    **kwargs,
) -> HelperAdsorptionResult:
    """Helper function for adsorption process."""
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
    cluster_key = graph.get_key_for_metadata(False)
    gas_key = gas.get_key_for_metadata(False)

    # ---------------------------------------------
    # check if the adsorption has been explored
    # ---------------------------------------------
    if network.exploration_can_be_finished(
        confidence=config.exploration.maxconfidence,
        cluster_key=cluster_key,
        gas_key=gas_key,
    ):
        msg = f"The adsorption for {cluster_key} with "
        msg += f"{gas_key} has been explored."
        logger.info(network.reformat_message(msg))
        return HelperAdsorptionResult(
            result=None,
            message=msg,
            for_cluster=cluster_key,
            for_gas=gas_key,
        )

    # ---------------------------------------------
    # check ncore for adsorption
    # ---------------------------------------------
    if not graph.ncore > 0:
        msg = f"Cluster {cluster_key} must have core."
        # logger.error(network.reformat_message(msg))
        return HelperAdsorptionResult(
            result=None,
            message=msg,
            for_cluster=cluster_key,
            for_gas=gas_key,
        )

    # ---------------------------------------------
    # do adsorption
    # ---------------------------------------------
    rxn_or_msg, _, cost_time, _ = funcs.helper_adsorption(
        gas=gas,
        graph=graph,
        config=config,
        **(
            kwargs
            | dict(
                initial_atoms=initial_atoms,
                graph_label=f"{cluster_key}_{gas_key}",
                allow_fixed_atoms_adsorption=allow_fixed_atoms_adsorption,
                allow_hash_change=True,
                raise_when_fail=False,
                deep_copy=deep_copy,
            )
        ),
    )
    msg = f"Adsorption search {cluster_key} with {gas_key} " + r"{}"
    msg += f" by {cost_time:.2f} seconds because of {rxn_or_msg}"
    if isinstance(rxn_or_msg, str):
        msg = msg.format("failed")
        logger.error(network.reformat_message(msg))
        return HelperAdsorptionResult(
            result=None,
            message=msg,
            for_cluster=cluster_key,
            for_gas=gas_key,
        )
    else:
        msg = msg.format("successfully")
        return HelperAdsorptionResult(
            result=rxn_or_msg,
            message=msg,
            for_cluster=cluster_key,
            for_gas=gas_key,
        )
