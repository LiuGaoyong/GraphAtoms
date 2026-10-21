"""Integration smoke tests for the Ray-native actor runner."""

from __future__ import annotations

import pytest

pytest.importorskip("ray")

import ray  # noqa: E402
from omegaconf import MISSING  # noqa: E402

from graphatoms.enterpoint.config import Config, EventConfig  # noqa: E402
from graphatoms.enterpoint.runner.v2.ray._actor import (  # noqa: E402
    create_network_actor,
)
from graphatoms.enterpoint.runner.v2.ray._tasks import actor_call  # noqa: E402


@pytest.fixture(scope="module")
def ray_cluster():
    started = not ray.is_initialized()
    if started:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            include_dashboard=False,
            logging_level="ERROR",
        )
    yield
    if started:
        ray.shutdown()


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        atoms=MISSING,
        bonds=MISSING,
        system=MISSING,
        calculator=MISSING,
        outputs=str(tmp_path / "run"),
        restart=False,
        event=EventConfig(),
    )


def test_actor_owns_network_and_gas_metadata(ray_cluster, config) -> None:
    actor = create_network_actor(config)
    try:
        basic = actor_call(actor, "metadata_basic")
        assert basic.gas_sticking == {}
        assert actor_call(actor, "key_rxn") == []

        actor_call(actor, "set_gas_sticking", {"CO": 1.0})
        basic = actor_call(actor, "metadata_basic")
        assert basic.gas_sticking == {"CO": 1.0}
        actor_call(actor, "persistence")
    finally:
        ray.kill(actor, no_restart=True)


def test_actor_can_be_reopened_from_persistence(ray_cluster, config) -> None:
    actor = create_network_actor(config)
    actor_call(actor, "set_gas_sticking", {"O2": 0.5})
    actor_call(actor, "persistence")
    ray.kill(actor, no_restart=True)

    config.restart = True
    config.event.gas_sticking = {"O2": 0.5}
    actor = create_network_actor(config)
    try:
        basic = actor_call(actor, "metadata_basic")
        assert basic.gas_sticking == {"O2": 0.5}
    finally:
        ray.kill(actor, no_restart=True)
