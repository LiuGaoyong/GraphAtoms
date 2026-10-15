import numpy as np
from ase import Atoms
from scipy.spatial import Voronoi

from graphatoms.utils.adsorption import _get_1order_nbr


def get_voronoi_lst(
    atoms: Atoms,
    index: int,
    *args,
    select: tuple[bool, bool, bool] = (True, True, True),
    **kwargs,
) -> list[tuple[Atoms, Atoms]]:
    """Get the voronoi list of the i-th atom in the atoms object."""
    vor = Voronoi(atoms.positions.copy()[:, select])
    nbr_1hop = _get_1order_nbr(atoms, [index])
    nbr_1hop_points = vor.points[np.append(nbr_1hop, [index])]
    nbr_1hop_min_point = np.min(nbr_1hop_points, axis=0)
    nbr_1hop_max_point = np.max(nbr_1hop_points, axis=0)
    first_lst: list[int] = [
        i
        for i in vor.regions[vor.point_region[index]]
        if (
            i >= 0
            and np.all(vor.vertices[i] >= nbr_1hop_min_point)
            and np.all(vor.vertices[i] <= nbr_1hop_max_point)
        )
    ]
    lst: list[tuple[int, int]] = []
    for region in vor.regions:
        if not np.any(np.isin(region, first_lst)):
            continue
        for idx, first in enumerate(region):
            second = region[idx + 1] if idx < len(region) - 1 else region[0]
            d = np.linalg.norm(vor.vertices[second] - vor.points[index])
            if d > 10.0:
                continue
            elif first < 0 or second < 0:
                continue
            elif first in first_lst and second not in first_lst:
                lst.append((first, second))
            elif first not in first_lst and second in first_lst:
                lst.append((second, first))
    result: list[tuple[Atoms, Atoms]] = []
    for first, second in sorted(set(lst)):
        item = [atoms.copy(), atoms.copy()]
        for idx, j in enumerate([first, second]):
            item[idx].positions[index, select] = vor.vertices[j]
        result.append(tuple(item))  # type: ignore
    return result


if __name__ == "__main__":
    from pathlib import Path

    from ase.cluster.octahedron import Octahedron
    from ase.io.animation import write_animation
    from ase.visualize import view  # type: ignore  # noqa: F401

    from graphatoms.system import SysGraph  # type: ignore  # noqa: F401

    atoms = Octahedron("Cu", 8)
    del atoms[atoms.positions[:, 2] > 0.1]
    del atoms[atoms.positions[:, 2] < -0.1]
    i, z_select = 28, False
    i, z_select = 22, False
    atoms.numbers[i] = 79

    this_dir = Path(__file__).parent
    fpath = this_dir / "../../tests-datasets/for-match/adsorption/Pd58CO.npz"
    sys = SysGraph.read_npz(fpath.as_posix())  # type: ignore
    atoms = sys.to_ase()
    i, z_select = 58, True

    lst, _lst = [], get_voronoi_lst(atoms, i, select=(True, True, z_select))
    for item in _lst:
        item = [atoms.copy(), *item]
        assert len(item) == 3
        for idx, image in enumerate(item):
            image.numbers[image.positions[:, 2] >= 0.1] = 2
            # image.numbers[i] = 1
            # if idx == 0:
            #     lst.append(image)
            print(len(lst), image.positions[i])
            lst.append(image)
    view(lst)
    fname = Path(__file__).with_suffix(".gif")
    write_animation(fname, lst, save_parameters={"fps": 3})
