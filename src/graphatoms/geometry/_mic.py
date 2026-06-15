import jax
import jax.numpy as jnp
import torch

from graphatoms.utils import ArrayNamespace, get_namespace


# 通用函数
def frobenius_squared(x):
    np: ArrayNamespace = get_namespace(x)
    return np.sum(np.square(x))


# PyTorch 测试
def test_pytorch():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = frobenius_squared(x)
    loss.backward()
    print("PyTorch gradient:\n", x.grad)


# JAX 测试
def test_jax():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    grad_f = jax.grad(frobenius_squared)
    grad_val = grad_f(x)
    print("JAX gradient:\n", grad_val)


if __name__ == "__main__":
    test_pytorch()
    test_jax()
