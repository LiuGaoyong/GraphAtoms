"""Tests for RayExecutor.submit cache mechanism."""

# ruff: noqa: E501"""Tests for RayExecutor.submit cache mechanism."""
import gc
import weakref

import pytest

import graphatoms.enterpoint.parallel._ray as _ray_module
from graphatoms.enterpoint.parallel._ray import RayExecutor


class FakeObjectRef:
    def __init__(self, value=None):
        self._value = value


class FakeRemoteFn:
    """Stand-in for ray's remote handle; holds no reference to the original fn."""

    def __init__(self):
        self._options: dict = {}

    def options(self, **opts):
        self._options.update(opts)
        return self

    def remote(self, *args, **kwargs):
        return FakeObjectRef()


class FakeRay:
    """Minimal fake ray that does NOT retain strong refs to submitted functions."""

    def __init__(self):
        self.remote_calls: list[tuple[int, weakref.ref]] = []

    def is_initialized(self):
        return True

    def init(self, **kwargs):
        return None

    def remote(self, func):
        self.remote_calls.append((id(func), weakref.ref(func)))
        return FakeRemoteFn()

    def get(self, ref):
        return ref


@pytest.fixture
def fake_ray(monkeypatch):
    fake = FakeRay()
    monkeypatch.setattr(_ray_module, "ray", fake)
    monkeypatch.setattr(_ray_module, "_RAY_AVAILABLE", True)
    return fake


# --------------------------------------------------------------------------- #
# Core cache tests
# --------------------------------------------------------------------------- #
def test_submit_reuses_cached_remote_fn(fake_ray):
    executor = RayExecutor()

    def my_func(x):
        return x + 1

    executor.submit(my_func, 1)
    executor.submit(my_func, 2)
    executor.submit(my_func, 3)

    assert len(fake_ray.remote_calls) == 1
    assert fake_ray.remote_calls[0][0] == id(my_func)


def test_submit_different_functions_get_separate_cache_entries(fake_ray):
    executor = RayExecutor()

    def fn_a(x):
        return x

    def fn_b(x):
        return x

    executor.submit(fn_a, 1)
    executor.submit(fn_b, 1)
    executor.submit(fn_a, 2)

    assert len(fake_ray.remote_calls) == 2
    assert len(executor._remote_fn_cache) == 2


def test_submit_populates_cache(fake_ray):
    executor = RayExecutor()

    def my_func(x):
        return x

    assert executor._remote_fn_cache == {}

    executor.submit(my_func, 0)

    key = id(my_func)
    assert key in executor._remote_fn_cache
    func_ref, remote_fn = executor._remote_fn_cache[key]
    assert func_ref() is my_func
    assert isinstance(remote_fn, FakeRemoteFn)
    assert remote_fn._options == {"num_cpus": 1}


def test_submit_returns_fresh_object_ref_each_call(fake_ray):
    executor = RayExecutor()

    def my_func(x):
        return x

    ref1 = executor.submit(my_func, 1)
    ref2 = executor.submit(my_func, 2)

    assert ref1 is not ref2
    assert isinstance(ref1, FakeObjectRef)
    assert isinstance(ref2, FakeObjectRef)


def test_submit_invalidates_cache_when_function_is_garbage_collected(fake_ray):
    executor = RayExecutor()

    def make_fn():
        def inner(x):
            return x

        return inner

    # Use a container so no local name / frame keeps the function alive.
    holder = [make_fn()]
    executor.submit(holder[0], 1)
    first_key = id(holder[0])
    assert first_key in executor._remote_fn_cache
    assert len(fake_ray.remote_calls) == 1

    func_ref, _ = executor._remote_fn_cache[first_key]
    assert func_ref() is holder[0]

    holder.clear()
    gc.collect()

    func_ref, _ = executor._remote_fn_cache[first_key]
    assert func_ref() is None

    executor.submit(make_fn(), 2)
    assert len(fake_ray.remote_calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
