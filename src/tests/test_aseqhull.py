import numpy as np
from ase.cluster.octahedron import Octahedron

from graphatoms.utils.aseqhull import get_voronoi_candidates


def test_get_voronoi_candidates() -> None:
    atoms = Octahedron("Cu", 6)
    # the atom closest to the center has a bounded Voronoi cell
    index = int(np.argmin(np.linalg.norm(atoms.positions, axis=1)))
    candidates = get_voronoi_candidates(atoms, index)
    assert len(candidates) > 0

    keys = [k for k, _, _ in candidates]
    assert len(set(keys)) == len(keys)
    for key, initial, final in candidates:
        assert isinstance(key, str)
        assert len(initial) == len(atoms)
        assert len(final) == len(atoms)
        for image in (initial, final):
            moved = np.any(
                np.abs(image.positions - atoms.positions) > 1e-8,
                axis=1,
            )
            # only the `index`-th atom is displaced
            assert np.count_nonzero(moved) == 1
            assert moved[index]
        assert not np.allclose(initial.positions, final.positions)
