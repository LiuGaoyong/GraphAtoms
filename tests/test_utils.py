# ruff: noqa: D100, D101, D102, D103
import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.cluster import Octahedron
from ase.data import covalent_radii as COV_R
from ase.visualize import view
from scipy.spatial.distance import cdist

from GraphAtoms.utils.geometry import distance_factory as DIST_FAC
from GraphAtoms.utils.geometry import inverse_3d_sphere_surface_sampling
from GraphAtoms.utils.rotation import Rot, kabsch, rotate
from GraphAtoms.utils.string import compress as compress_string


def test_rotation() -> None:  # noqa: D103
    ben = molecule("C6H6")
    points = ben.positions + np.random.random(3)
    center = points.mean(axis=0)
    print(f"Points center: {center}")

    rotation = Rot.random()
    print(f"Random rotation:\n{rotation.as_quat()}")

    rotated = rotate(points, rotation, center=[0, 0, 0])
    print(f"Rotated points around zero point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) != center

    rotated = rotate(points, rotation, center)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center

    # Default center point as the geometry center:
    rotated = rotate(points, rotation)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center


def test_kabsch() -> None:  # noqa: D103
    B = molecule("C6H6").positions + np.random.random(3)
    R, T = Rot.random(), np.random.random(3)

    A = rotate(B, R) + T
    assert pytest.approx((A - T).mean(axis=0)) == B.mean(axis=0)

    R0, T0, rmsd = kabsch(A, B)
    assert isinstance(T0, np.ndarray)
    assert pytest.approx(T0) == T
    assert isinstance(R0, Rot)
    assert pytest.approx(R0.as_matrix()) == R.as_matrix()
    assert rmsd <= 1e-5
    assert pytest.approx(rotate(B, R0) + T0) == A


@pytest.mark.skip("Run once enough.")
@pytest.mark.parametrize("i_lst", [(0, 34)])
def test_is_inner(i_lst: list[int]):
    atoms = Octahedron("Cu", 8)
    for i in i_lst:
        mesh = inverse_3d_sphere_surface_sampling(1000)
        mesh = atoms.positions[i] + mesh * COV_R[atoms.numbers[i]]
        atoms = Atoms(
            numbers=np.append(atoms.numbers, [0] * len(mesh)),
            positions=np.vstack([atoms.positions, mesh]),
        )
    view(atoms)


@pytest.mark.skip("Run once enough.")
# Note: this test fail for some case ???
def test_get_distance_reduce_array() -> None:
    print()
    arr = np.random.rand(6, 3)
    dm = cdist(arr[:5, :], arr)
    dm[dm > 0.5] = np.inf
    dm = np.where(dm == 0, np.inf, dm)
    d0 = dm.min(axis=0)
    print(d0)
    d1 = DIST_FAC.get_distance_reduce_array(
        arr[:5, :],
        arr,
        max_distance=0.5,
        reduce_axis=0,
    )
    print(d1)
    np.testing.assert_array_compare(np.equal, d0, d1)


@pytest.mark.parametrize(
    "fmt, level",
    sorted(
        itertools.product(
            ["z", "gz", "bz2"],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    )
    + [
        ("snappy", 0),
        ("xz", 0),
        ("lzma", 0),
    ],
)
def test_compress(
    OctCu8_Cluster_ListofDict: list[dict],
    fmt: str,
    level: int,
) -> None:
    data: str = OctCu8_Cluster_ListofDict[0]["cluster_json"]
    compress_string(data, format=fmt, compresslevel=level)


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s", "--maxfail=1"])
