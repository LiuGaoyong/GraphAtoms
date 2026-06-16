import numpy as np
import pytest
from ase.geometry import geometry as ase_geometry
from ase.geometry import minkowski_reduce

from graphatoms.arrayapi import Array, ArrayNamespace
from graphatoms.geometry import mic as this_geometry


@pytest.mark.parametrize(
    "test_arr, function_name",
    [
        (np.random.randint(100, size=(5, 3)) / 50, "translate_pretty"),
        (np.random.rand(5, 3), "naive_find_mic"),
        (np.random.rand(5, 3), "find_mic"),
        (np.random.rand(3), "find_mic"),
    ],
)
def test_array_api(test_arr: np.ndarray, function_name: str) -> None:
    lst: list[ArrayNamespace] = [np]  # type: ignore
    try:
        import torch
    except ImportError:
        torch = None
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        jnp, jax = None, None
    if jax is not None:
        jax.config.update("jax_enable_x64", True)
    for xp in (jnp, torch):
        if xp is not None:
            lst.append(xp)  # type: ignore
    print()

    print("Random Array:\n", test_arr)
    ase_func = getattr(ase_geometry, function_name)
    this_func = getattr(this_geometry, function_name)
    if function_name == "translate_pretty":
        param2 = np.array([True, True, True])
    elif function_name == "naive_find_mic":
        param2 = np.diag(np.random.rand(3) + 0.2)
    elif function_name == "find_mic":
        param2 = np.random.rand(3, 3) + 0.2
        param2, _ = minkowski_reduce(param2)
    else:
        raise ValueError(f"Unknown function name {function_name}")

    y = ase_func(test_arr.copy(), param2)
    if function_name == "translate_pretty":
        pass
    else:
        y = y[1]
    for xp in lst:
        x0 = xp.asarray(test_arr, copy=True)
        if xp is torch and torch is not None:
            assert isinstance(x0, torch.Tensor)
            x0.requires_grad_(True)
        y0 = this_func(x0, xp.asarray(param2))  # type: ignore
        if function_name == "translate_pretty":
            pass
        else:
            y0 = y0[1]
        print(type(y0), isinstance(y0, Array))
        if xp is torch and torch is not None:
            assert isinstance(y0, torch.Tensor)
            y0_np = np.asarray(y0.clone().detach().numpy())
        else:
            y0_np = np.asarray(y0)
        print(f"Diff({xp.__name__}): \n", y0_np - y)  # type: ignore
        assert np.allclose(y, y0_np)

        if xp is torch and torch is not None:
            print(x0)
            print(x0.requires_grad)
            e = torch.sum(y0)  # type: ignore
            e.backward()
            print("TORCH grad: \n", x0.grad)
        elif xp is jnp and jnp is not None:
            assert jax is not None
            if function_name == "translate_pretty":
                grad_f = jax.grad(
                    lambda x, pbc: jnp.sum(
                        this_func(x, pbc),  # type: ignore
                    )
                )
            else:
                grad_f = jax.grad(
                    lambda x, pbc: jnp.sum(
                        this_func(x, pbc)[1],  # type: ignore
                    )
                )
            print("JAX grad:\n", grad_f(x0, xp.asarray(param2)))
        print("-" * 50)
    print("=" * 50)
