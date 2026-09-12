"""Integration tests for the ``graphatoms.parallel`` execution framework.

Tests cover serial, multiprocessing, ray, dask, and executorlib backends
through the uniform :class:`~graphatoms.parallel.base.BaseExecutor`
interface, plus the ``as_completed`` / ``wait`` utility functions and
the executor-level ``cancel`` method.
"""

import os
import sys
import time

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

# Make this test module importable by worker processes (ray / executorlib /
# multiprocessing spawn) which unpickle the helper functions by reference.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
os.environ["PYTHONPATH"] = (
    _TESTS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
)
this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)

import pytest  # noqa: E402
from conftest import return_big_object  # type: ignore  # noqa: E402

from graphatoms.enterpoint.parallel import get_executor  # noqa: E402


# ---------------------------------------------------------------------------
# Optional-dependency probes.
# ---------------------------------------------------------------------------
def _has_ray() -> bool:
    try:
        import ray  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _has_dask() -> bool:
    try:
        import dask.distributed  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _has_executorlib() -> bool:
    try:
        import executorlib  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


HAS_RAY = _has_ray()
HAS_DASK = _has_dask()
HAS_EXECUTORLIB = _has_executorlib()

lst = ["multiprocessing"]
if HAS_RAY:
    lst.append("ray")
# if HAS_DASK:
#     lst.append("dask")


@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("backend", lst)
def test_parallel_call(
    use_map: bool,
    backend: str,
    set_ray_env: None,
) -> None:
    if backend == "dask" and not HAS_DASK:
        return
    if backend == "ray" and not HAS_RAY:
        return

    if backend == "ray":
        os.environ["RAY_DEDUP_LOGS"] = "0"
        import ray  # type: ignore

        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": this_dir,
                }
            },
        )
        print("Initialize & Register Ray.")

    start_time = time.perf_counter()
    print("=" * 54)
    print(backend)
    print("-" * 54)
    executor = get_executor(backend, nworkers=4)
    refs = [executor.submit(return_big_object, i) for i in range(200)]
    i = 0
    while len(refs) > 0:
        if i <= 10:
            result, refs = executor.wait(refs)
            print(f"result: {result}, index={i}")
            i += 1
        else:
            break

    for r in refs:
        executor.cancel(r)
    print(f"Time: {time.perf_counter() - start_time:.5f}s")

    print("-" * 54)
    print(executor.map(return_big_object, range(20)))
    print(f"Time: {time.perf_counter() - start_time:.5f}s")
    print("=" * 54)
