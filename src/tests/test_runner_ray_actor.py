"""Unit tests for the Ray-native runner plumbing."""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("ray")

from graphatoms.enterpoint.runner.v2.ray import _actor, _tasks  # noqa: E402
from graphatoms.enterpoint.runner.v2.ray._runner import (  # noqa: E402
    RayRunnerBase,
)


def _async_methods(cls) -> list[ast.AsyncFunctionDef]:
    tree = ast.parse(inspect.getsource(cls))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
    ]


def test_actor_methods_have_no_internal_await() -> None:
    """The actor relies on method bodies being atomic (no await)."""
    methods = _async_methods(_actor._RayNetworkActor)
    assert methods
    for method in methods:
        for node in ast.walk(method):
            assert not isinstance(node, ast.Await), method.name
            assert not isinstance(node, ast.AsyncFor), method.name
            assert not isinstance(node, ast.AsyncWith), method.name


def test_ray_runner_does_not_keep_network_or_executor() -> None:
    assert "network" not in dir(RayRunnerBase)
    assert "executor" not in dir(RayRunnerBase)


def test_ray_tasks_do_not_construct_the_network() -> None:
    source = inspect.getsource(_tasks)
    assert "ReactionNetwork.from_config" not in source
    assert "ReactionNetwork(" not in source


def test_ray_tasks_are_remote_functions() -> None:
    for task in (
        _tasks.task_optimization,
        _tasks.task_dimer,
        _tasks.task_adsorption,
        _tasks.task_match,
        _tasks.task_apply,
    ):
        assert hasattr(task, "remote")


def test_submit_task_injects_actor_and_num_cpus() -> None:
    calls: list[tuple[tuple, dict]] = []

    class _FakeOptions:
        def options(self, **kwargs):
            calls.append(((), kwargs))
            return self

        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "ref"

    class _Config:
        parallel_ncpus_for_tasks = 2

    assert (
        _tasks.submit_task(
            _FakeOptions(),
            actor="the-actor",
            config=_Config(),
            value=1,
        )
        == "ref"
    )
    assert calls[0][1] == {"num_cpus": 2}
    forwarded = calls[1][1]
    assert forwarded["actor"] == "the-actor"
    assert isinstance(forwarded["config"], _Config)
    assert forwarded["value"] == 1


def test_runner_exports_ray_runners_lazily() -> None:
    import graphatoms.enterpoint.runner as runner

    assert runner.RayOnTheFlyKMC is runner.RayOTFKMC
    assert runner.RayReactionNetworkGenerator.__name__ == (
        "RayReactionNetworkGenerator"
    )
    assert callable(runner.run_ray)


def test_ray_optional_dependency_helper() -> None:
    from graphatoms.enterpoint.runner.v2.ray._optional import (
        RAY_INSTALL_HINT,
        RayNotInstalledError,
        require_ray,
    )

    assert issubclass(RayNotInstalledError, ImportError)
    assert "graphatoms[ray]" in RAY_INSTALL_HINT
    require_ray()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
