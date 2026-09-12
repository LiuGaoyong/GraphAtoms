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

from graphatoms.enterpoint.parallel import (  # noqa: E402
    BaseFuture,
    get_executor,
    wait_one,
)

# # ---------------------------------------------------------------------------
# # Picklable helper functions (required by multiprocessing backend).
# # ---------------------------------------------------------------------------
# def _add_one(x: int) -> int:
#     return x + 1


# def _square(x: int) -> int:
#     return x * x


# def _raise_error() -> None:
#     raise ValueError("test error")


# def _slow_add_one(x: int) -> int:
#     time.sleep(0.1)
#     return x + 1


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

# # In sandboxed / container environments Ray's docker CPU-quota detection
# # reads /sys/fs/cgroup/cpu.max and may leak the file handle, producing an
# # unraisable exception during garbage collection.
# pytestmark = pytest.mark.filterwarnings(
#     "ignore::pytest.PytestUnraisableExceptionWarning"
# )

# # Backends that are always available.
# CORE_BACKENDS = ["serial", "multiprocessing"]


# @pytest.fixture(scope="session", autouse=True)
# def _init_ray_runtime_env() -> None:
#     """Initialise Ray with a runtime_env so workers can import tests."""
#     if not HAS_RAY:
#         return
#     import ray  # type: ignore  # noqa: F401

#     if not ray.is_initialized():
#         ray.init(
#             ignore_reinit_error=True,
#             runtime_env={"env_vars": {"PYTHONPATH": _TESTS_DIR}},
#         )


# # ---------------------------------------------------------------------------
# # 1. submit + result for every backend
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_submit_result_core(backend: str) -> None:
#     executor = get_executor(backend, max_workers=2)
#     try:
#         future = executor.submit(_add_one, 41)
#         assert future.result() == 42
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_submit_result_ray() -> None:
#     executor = get_executor("ray", max_workers=2)
#     try:
#         future = executor.submit(_add_one, 41)
#         assert future.result(timeout=30) == 42
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_DASK, reason="dask is not installed")
# def test_submit_result_dask() -> None:
#     executor = get_executor("dask", max_workers=2)
#     try:
#         future = executor.submit(_add_one, 41)
#         assert future.result(timeout=30) == 42
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_EXECUTORLIB, reason="executorlib not installed")
# def test_submit_result_executorlib() -> None:
#     executor = get_executor("executorlib", max_workers=2)
#     try:
#         future = executor.submit(_add_one, 41)
#         assert future.result(timeout=30) == 42
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 2. map method
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_map_core(backend: str) -> None:
#     executor = get_executor(backend, max_workers=2)
#     try:
#         result = list(executor.map(_square, [1, 2, 3, 4]))
#         assert result == [1, 4, 9, 16]
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_map_ray() -> None:
#     executor = get_executor("ray", max_workers=2)
#     try:
#         result = list(executor.map(_square, [1, 2, 3, 4], timeout=30))
#         assert result == [1, 4, 9, 16]
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_DASK, reason="dask is not installed")
# def test_map_dask() -> None:
#     executor = get_executor("dask", max_workers=2)
#     try:
#         result = list(executor.map(_square, [1, 2, 3, 4], timeout=30))
#         assert result == [1, 4, 9, 16]
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 3. Future state methods (done, running, cancelled)
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_future_state_after_result(backend: str) -> None:
#     executor = get_executor(backend, max_workers=2)
#     try:
#         future = executor.submit(_add_one, 1)
#         assert future.result() == 2
#         assert future.done() is True
#         assert future.running() is False
#         assert future.cancelled() is False
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 4. cancel method (Future-level and Executor-level)
# # ---------------------------------------------------------------------------
# def test_cancel_serial_returns_false() -> None:
#     """Serial futures: cancel returns False after completion."""
#     executor = SerialExecutor()
#     future = executor.submit(_add_one, 1)
#     assert future.result() == 2
#     assert future.cancel() is False
#     assert future.done() is True
#     executor.shutdown()


# def test_cancel_multiprocessing_pending() -> None:
#     """A pending multiprocessing task can be cancelled."""
#     executor = get_executor("multiprocessing", max_workers=1)
#     try:
#         blocker = executor.submit(_slow_add_one, 0)
#         pending = executor.submit(_add_one, 1)
#         pending.cancel()
#         assert pending.cancelled() is True
#         blocker.result()
#     finally:
#         executor.shutdown()


# def test_executor_cancel_default_delegates_to_future() -> None:
#     """BaseExecutor.cancel delegates to future.cancel() by default."""
#     executor = get_executor("multiprocessing", max_workers=1)
#     try:
#         future = executor.submit(_add_one, 1)
#         future.result()
#         assert future.cancelled() is False
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_executor_cancel_ray_with_force() -> None:
#     """RayExecutor.cancel supports force and recursive options."""
#     from graphatoms.enterpoint.parallel.ray import RayFuture
#     executor = get_executor("ray", max_workers=2)
#     try:
#         future = executor.submit(_slow_add_one, 0)
#         assert isinstance(future, RayFuture)
#         future.cancel(force=True, recursive=True)
#         assert future.cancelled() is True
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 5. add_done_callback
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_add_done_callback(backend: str) -> None:
#     called: list[BaseFuture] = []
#     executor = get_executor(backend, max_workers=2)
#     try:
#         future = executor.submit(_add_one, 1)
#         future.add_done_callback(called.append)
#         future.result()
#         for _ in range(100):
#             if called:
#                 break
#             time.sleep(0.01)
#         assert len(called) >= 1
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 6. Exception propagation
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_exception_propagation_core(backend: str) -> None:
#     executor = get_executor(backend, max_workers=2)
#     try:
#         future = executor.submit(_raise_error)
#         with pytest.raises(ValueError, match="test error"):
#             future.result()
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_exception_propagation_ray() -> None:
#     executor = get_executor("ray", max_workers=2)
#     try:
#         future = executor.submit(_raise_error)
#         with pytest.raises(ValueError, match="test error"):
#             future.result(timeout=30)
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 7. get_executor factory
# # ---------------------------------------------------------------------------
# def test_get_executor_serial() -> None:
#     ex = get_executor("serial")
#     assert isinstance(ex, SerialExecutor)
#     assert isinstance(ex, ProcessPoolExecutor)
#     ex.shutdown()


# def test_get_executor_multiprocessing() -> None:
#     ex = get_executor("multiprocessing", max_workers=2)
#     assert isinstance(ex, ProcessPoolExecutor)
#     ex.shutdown()


# def test_get_executor_unknown_raises() -> None:
#     with pytest.raises(ValueError, match="Unknown backend"):
#         get_executor("does-not-exist")


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_get_executor_ray() -> None:
#     from graphatoms.enterpoint.parallel.ray import RayExecutor

#     ex = get_executor("ray", max_workers=2)
#     assert isinstance(ex, RayExecutor)
#     ex.shutdown()


# @pytest.mark.skipif(not HAS_DASK, reason="dask is not installed")
# def test_get_executor_dask() -> None:
#     from graphatoms.enterpoint.parallel.dask import DaskExecutor

#     ex = get_executor("dask", max_workers=2)
#     assert isinstance(ex, DaskExecutor)
#     ex.shutdown()


# # ---------------------------------------------------------------------------
# # 8. Context manager
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_context_manager_core(backend: str) -> None:
#     with get_executor(backend, max_workers=2) as ex:
#         assert isinstance(ex, BaseExecutor)
#         future = ex.submit(_add_one, 1)
#         assert future.result() == 2


# # ---------------------------------------------------------------------------
# # 9. as_completed utility
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_as_completed_core(backend: str) -> None:
#     """as_completed yields futures in completion order."""
#     executor = get_executor(backend, max_workers=2)
#     try:
#         futures = [executor.submit(_square, i) for i in range(5)]
#         results = [f.result() for f in as_completed(futures)]
#         assert sorted(results) == [0, 1, 4, 9, 16]
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_as_completed_ray() -> None:
#     executor = get_executor("ray", max_workers=2)
#     try:
#         futures = [executor.submit(_square, i) for i in range(5)]
#         results = [f.result() for f in as_completed(futures)]
#         assert sorted(results) == [0, 1, 4, 9, 16]
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 10. wait utility
# # ---------------------------------------------------------------------------
# @pytest.mark.parametrize("backend", CORE_BACKENDS)
# def test_wait_core(backend: str) -> None:
#     """wait returns (done, not_done) lists."""
#     executor = get_executor(backend, max_workers=2)
#     try:
#         futures = [executor.submit(_square, i) for i in range(3)]
#         done, not_done = wait(futures, num_returns=1)
#         # At least one should be done after wait.
#         assert len(done) >= 1
#     finally:
#         executor.shutdown()


# @pytest.mark.skipif(not HAS_RAY, reason="ray is not installed")
# def test_wait_ray() -> None:
#     executor = get_executor("ray", max_workers=2)
#     try:
#         futures = [executor.submit(_square, i) for i in range(3)]
#         done, not_done = wait(futures, num_returns=1)
#         assert len(done) >= 1
#     finally:
#         executor.shutdown()


# # ---------------------------------------------------------------------------
# # 11. Friendly ImportError when optional dependencies are missing
# # ---------------------------------------------------------------------------
# def test_ray_import_error_when_missing() -> None:
#     from graphatoms.enterpoint.parallel import ray as ray_backend

#     with patch.object(ray_backend, "_RAY_AVAILABLE", False):
#         with pytest.raises(ImportError, match="ray is not installed"):
#             ray_backend.RayExecutor()


# def test_dask_import_error_when_missing() -> None:
#     from graphatoms.enterpoint.parallel import dask as dask_backend

#     with patch.object(dask_backend, "_DASK_AVAILABLE", False):
#         with pytest.raises(ImportError, match="dask is not installed"):
#             dask_backend.DaskExecutor()


# def test_executorlib_import_error_when_missing() -> None:
#     from graphatoms.enterpoint.parallel import executorlib as el_backend

#     with patch.object(el_backend, "_EXECUTORLIB_AVAILABLE", False):
#         with pytest.raises(ImportError, match="executorlib is not installed"):
#             el_backend.ExecutorLibExecutor()


# # ---------------------------------------------------------------------------
# # 12. Example function
# # ---------------------------------------------------------------------------
# def test_example_batch_consume_cancel() -> None:
#     from graphatoms.enterpoint.parallel._example import run_batch_consume_cancel

#     results = run_batch_consume_cancel(
#         "multiprocessing", n_tasks=10, n_results=3
#     )
#     assert len(results) == 3
#     assert all(r in range(0, 20, 2) for r in results)


# def test_example_batch_wait_cancel() -> None:
#     from graphatoms.enterpoint.parallel._example import run_batch_wait_cancel

#     results = run_batch_wait_cancel("multiprocessing", n_tasks=10, n_results=3)
#     assert len(results) == 3
#     assert all(r in range(0, 20, 2) for r in results)


# ---------------------------------------------------------------------------
# 13. test by LiuGaoyong
# ---------------------------------------------------------------------------
lst = ["serial", "multiprocessing"]
if HAS_RAY:
    lst.append("ray")
if HAS_DASK:
    lst.append("dask")


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
    with get_executor(backend, max_workers=8) as executor:
        if use_map:
            result = list(executor.map(return_big_object, range(20)))
            print(result)
        else:
            futures: list[BaseFuture] = [
                executor.submit(return_big_object, i) for i in range(200)
            ]
            i = 0
            while len(futures) > 0:
                if i <= 10:
                    result, futures = wait_one(futures)
                    print(result, i)
                    i += 1
                else:
                    break

            for f in futures:
                f.cancel()

    print(f"Time: {time.perf_counter() - start_time:.5f}s")
