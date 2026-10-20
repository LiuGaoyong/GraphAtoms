"""Unit tests for the actor-backed Ray runner plumbing."""

from typing import Any

import pytest

pytest.importorskip("ray")

from graphatoms.enterpoint.runner.v2.ray import _helper  # noqa: E402
from graphatoms.enterpoint.runner.v2.ray._helper import (  # noqa: E402
    RayNetworkProxy,
    helper_adsorption,
    helper_dimer,
    helper_optimization,
)


class _FakeRemote:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def remote(self, *args, **kwargs) -> Any:
        self.calls.append((args, kwargs))
        return self.value


class _FakeActor:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, _FakeRemote(value))


@pytest.fixture
def proxy(monkeypatch) -> tuple[RayNetworkProxy, _FakeActor]:
    actor = _FakeActor(
        get_use_position_uuid=True,
        object_has_been_optimized="graph",
        exploration_can_be_finished=False,
        exploration_can_be_skipped=(True, 0.5),
    )
    monkeypatch.setattr(_helper.ray, "get", lambda ref: ref)
    return RayNetworkProxy(actor), actor  # type: ignore[arg-type]


def test_proxy_forwards_network_queries(proxy) -> None:
    net, actor = proxy
    graph = object()

    assert net.get_use_position_uuid("cluster") is True
    assert net.object_has_been_optimized(graph) == "graph"
    assert (
        net.exploration_can_be_finished(
            confidence=3.0,
            cluster_key="c",
            gas_key="g",
        )
        is False
    )
    assert net.exploration_can_be_skipped(
        thetacutoff=30.0,
        diffpositions=None,  # type: ignore[arg-type]
        cluster_key="c",
    ) == (True, 0.5)
    assert net.reformat_message("hello") == "hello"

    assert actor.exploration_can_be_finished.calls[0][1] == {
        "cluster_key": "c",
        "gas_key": "g",
    }


def test_proxy_reformat_message_does_not_touch_actor(proxy) -> None:
    net, actor = proxy
    assert net.reformat_message("msg") == "msg"
    assert not hasattr(actor, "reformat_message")


def test_ray_tasks_are_remote_functions() -> None:
    for task in (helper_optimization, helper_dimer, helper_adsorption):
        assert hasattr(task, "remote")


def test_runner_exports_ray_runners_lazily() -> None:
    import graphatoms.enterpoint.runner as runner

    assert runner.RayOnTheFlyKMC is runner.RayOTFKMC
    assert runner.RayReactionNetworkGenerator.__name__ == (
        "RayReactionNetworkGenerator"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
