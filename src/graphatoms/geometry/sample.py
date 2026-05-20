import numpy as np


def fibonacci_lattice(n: int) -> np.ndarray:
    """The Fibonacci Lattice grid.

    https://pubs.acs.org/doi/suppl/10.1021/acscatal.3c04964/suppl_file/cs3c04964_si_001.pdf
    """
    n = int(n)
    arange = np.arange(n)
    a = 2 * arange + 1
    theta = np.arccos(1 - a / n)
    psi = (1 + np.sqrt(5)) / 2 * a * np.pi
    return np.column_stack(
        [
            np.sin(theta) * np.cos(psi),
            np.sin(theta) * np.sin(psi),
            np.cos(theta),
        ]
    )


def inverse_3d_sphere_surface_sampling(n: int) -> np.ndarray:
    """The 3D Sphere Surface Sampling Based on Inverse Transform.

    Reference:
        https://en.wikipedia.org/wiki/Inverse_transform_sampling
        http://corysimon.github.io/articles/uniformdistn-on-sphere/
        https://niepp.github.io/2021/12/09/uniform-sampling-on-sphere.html

    Args:
        n (int): the number of points for sampling.

    Returns:
        np.ndarray: (2n, 3) shape float numpy.ndarray for half sphere
    """
    u = np.random.uniform(0, 1, size=n)
    v = np.random.uniform(0, 1, size=n)
    # inversion method
    phi = v * 2 * np.pi
    cos_theta = 1 - u
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z0, z1 = cos_theta, -cos_theta
    half0 = np.column_stack([x, y, z0])
    half1 = np.column_stack([x, y, z1])
    return np.vstack([half0, half1])


if __name__ == "__main__":
    from pathlib import Path

    from ase import Atoms
    from ase.visualize.plot import plot_atoms
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    N = 100

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=150)
    for ax, func, n, name in zip(
        axes,
        [
            fibonacci_lattice,
            inverse_3d_sphere_surface_sampling,
        ],
        [2 * N, N],
        ["Fibonacci Grid", "Inverse Transform Sampling"],
    ):
        assert isinstance(ax, Axes)
        atoms = Atoms([0] * 2 * N, func(n) * 2)
        plot_atoms(atoms, ax=ax)
        ax.set_title(f"{name} (N={N * 2})")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"))
