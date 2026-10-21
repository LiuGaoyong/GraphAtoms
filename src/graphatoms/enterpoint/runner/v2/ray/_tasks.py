"""Ray tasks for the Ray-native runner.

Each task receives the network actor handle and performs all reaction-network
reads/writes through it.  The expensive helper functions are reused from the
existing runner package; only the network I/O is moved into the actor.

Tasks are regular Ray tasks (not async tasks: Ray does not support
``@ray.remote`` on ``async def``).  They synchronously wait on the async actor
with :func:`ray.get`, which keeps the worker blocked only for the duration of
the network call.
"""

from __future__ import annotations

import numpy as np
import ray
from ase import Atoms
from ase.calculators.calculator import Calculator

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network._clsnetwork import ReactionNetwork
from graphatoms.enterpoint.runner.common._helper import (
    compute_apply,
    compute_match,
)
from graphatoms.enterpoint.runner.v2._helperv2 import (
    HelperAdsorptionResult,
    HelperDimerResult,
    HelperOptimizationResult,
    compute_adsorption,
    compute_dimer,
    compute_optimization,
)
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.asetools import call_dimer_displace
from graphatoms.utils.parser import hydra_parse

from ._actor import RayNetworkActorHandle

__all__ = [
    "submit_task",
    "wait_one",
    "actor_call",
    "task_optimization",
    "task_dimer",
    "task_adsorption",
    "task_match",
    "task_apply",
]


# ----------------------------------------------------------------------
#                        Ray helper functions
# ----------------------------------------------------------------------
def submit_task(
    task,
    *,
    actor: RayNetworkActorHandle,
    config: Config,
    **kwargs,
):
    """Submit one Ray task with the actor and the configured CPU count."""
    ncpus = max(1, int(config.parallel_ncpus_for_tasks))
    return task.options(num_cpus=ncpus).remote(
        actor=actor,
        config=config,
        **kwargs,
    )


def wait_one(refs: list) -> tuple[object, list]:
    """Return one finished result and the remaining ObjectRefs."""
    done, remaining = ray.wait(refs, num_returns=1)
    return ray.get(done[0]), list(remaining)


def actor_call(actor: RayNetworkActorHandle, method: str, *args, **kwargs):
    """Synchronously call one actor method from the driver or a task."""
    return ray.get(getattr(actor, method).remote(*args, **kwargs))


def _use_positions_uuid(graph) -> bool:
    return ReactionNetwork.DB_USE_POSITION_UUID_DICT.get(
        graph.__class__.__name__.lower(),
        False,
    )


# ----------------------------------------------------------------------
#                              tasks
# ----------------------------------------------------------------------
@ray.remote
def task_optimization(
    config: Config,
    graph: Cluster | Gas | System,
    graph_label: str,
    actor: RayNetworkActorHandle | None = None,
    **kwargs,
) -> HelperOptimizationResult:
    """Optimize one graph without touching the reaction network."""
    return compute_optimization(
        config=config,
        graph=graph,
        graph_label=graph_label,
        **kwargs,
    )


@ray.remote
def task_dimer(
    config: Config,
    cluster: Cluster,
    actor: RayNetworkActorHandle,
    **kwargs,
) -> HelperDimerResult:
    """Run a dimer search and write its result through the actor."""
    cluster_key = cluster.get_key_for_metadata(_use_positions_uuid(cluster))
    if actor_call(
        actor,
        "exploration_can_be_finished",
        float(config.exploration.maxconfidence),
        cluster_key,
    ):
        return HelperDimerResult(
            result=None,
            message=f"the cluster {cluster_key} has been explored.",
            has_been_finished=True,
            for_cluster=cluster_key,
        )

    if not cluster.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    ):
        msg = f"Cluster {cluster_key} is not at a minimum."
        actor_call(actor, "found", msg, None, cluster_key, persist=True)
        return HelperDimerResult(
            result=None,
            message=msg,
            is_failed=True,
            for_cluster=cluster_key,
        )

    thetacutoff = float(config.exploration.thetacutoff)
    displacement: np.ndarray | None = None
    if thetacutoff > 0:
        displacement = call_dimer_displace(
            atoms=cluster.to_ase().copy(),
            parse_mask_from_atoms=True,
            calc=hydra_parse(config.calculator, Calculator),  # type: ignore
            mask=None,
        )
        can_skip, cosine = actor_call(
            actor,
            "exploration_can_be_skipped",
            thetacutoff,
            displacement,
            cluster_key,
        )
        if can_skip:
            actor_call(actor, "record_skip", cluster_key)
            return HelperDimerResult(
                result=None,
                message=f"Skip {cluster_key} exploration. Cosine={cosine:.4f}.",
                is_skipped=True,
                for_cluster=cluster_key,
            )

    result = compute_dimer(
        config=config,
        cluster=cluster,
        cluster_key=cluster_key,
        displacement=displacement,
        **kwargs,
    )
    if result.result is not None:
        event_label = actor_call(
            actor,
            "found",
            result.result,
            None,
            cluster_key,
            persist=True,
        )
        if str(result.result) not in event_label:
            result.message += (
                " Simplify original event by threshold "
                f"{config.event.simplified_threshold:.2f}"
            )
    elif result.is_failed:
        actor_call(
            actor,
            "found",
            result.message,
            None,
            cluster_key,
            persist=True,
        )
    return result


@ray.remote
def task_adsorption(
    gas: Gas,
    config: Config,
    graph: Cluster | System,
    actor: RayNetworkActorHandle,
    **kwargs,
) -> HelperAdsorptionResult:
    """Run an adsorption search and write its result through the actor."""
    cluster_key = graph.get_key_for_metadata(False)
    gas_key = gas.get_key_for_metadata(False)
    if actor_call(
        actor,
        "exploration_can_be_finished",
        float(config.exploration.maxconfidence),
        cluster_key,
        gas_key,
    ):
        return HelperAdsorptionResult(
            result=None,
            message=(
                f"The adsorption for {cluster_key} with {gas_key} "
                "has been explored."
            ),
            for_cluster=cluster_key,
            for_gas=gas_key,
        )

    if not graph.ncore > 0:
        return HelperAdsorptionResult(
            result=None,
            message=f"Cluster {cluster_key} must have core.",
            for_cluster=cluster_key,
            for_gas=gas_key,
        )

    kwargs.pop("graph_label", None)
    kwargs.pop("raise_when_fail", None)
    result = compute_adsorption(
        gas=gas,
        config=config,
        graph=graph,
        cluster_key=cluster_key,
        gas_key=gas_key,
        **kwargs,
    )
    if result.result is not None:
        event_label = actor_call(
            actor,
            "found",
            result.result,
            gas_key,
            cluster_key,
            persist=True,
        )
        if str(result.result) not in event_label:
            result.message += (
                " Simplify original event by threshold "
                f"{config.event.simplified_threshold:.2f}"
            )
    return result


@ray.remote
def task_match(
    config: Config,
    system: System,
    rxn_is_forward: bool,
    rxn_key: str,
    actor: RayNetworkActorHandle,
    **kwargs,
) -> tuple[str, bool, np.ndarray | None]:
    """Match a system against one event, reading the event from the actor."""
    info = actor_call(actor, "read_event_info", rxn_key)
    key = info.key_r if rxn_is_forward else info.key_p
    pattern = actor_call(actor, "read_sysgraph", key, "minima", True)
    return compute_match(
        config=config,
        system=system,
        rxn_is_forward=rxn_is_forward,
        rxn_key=rxn_key,
        pattern=pattern,
    )


@ray.remote
def task_apply(
    rxn_key: str,
    config: Config,
    system: System | Atoms,
    match_mode: np.ndarray,
    actor: RayNetworkActorHandle,
    raise_when_fail: bool = True,
    forward: bool = True,
    **kwargs,
) -> tuple[str, bool, Atoms | str, float]:
    """Apply one event, reading the event from the actor."""
    _, rxn = actor_call(actor, "read_event", rxn_key)
    return compute_apply(
        config=config,
        system=system,
        rxn_key=rxn_key,
        forward=forward,
        match_mode=match_mode,
        rxn=rxn,
        raise_when_fail=raise_when_fail,
    )
