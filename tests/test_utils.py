# ruff: noqa: D100, D101, D102, D103
import numpy as np
import pytest
from ase import Atoms
from ase.cluster import Octahedron
from ase.data import covalent_radii as COV_R
from ase.visualize import view
from scipy.spatial.distance import cdist

from GraphAtoms.common.geometry import distance_factory as DIST_FAC
from GraphAtoms.common.geometry import inverse_3d_sphere_surface_sampling


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


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s", "--maxfail=1"])
