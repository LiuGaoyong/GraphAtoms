import numpy as np
from ase import Atoms
from scipy.spatial import Voronoi


def get_voronoi_lst(
    atoms: Atoms,
    index: int,
    *args,
    select: tuple[bool, bool, bool] = (True, True, True),
    **kwargs,
) -> list[tuple[Atoms, Atoms]]:
    """Get the voronoi list of the i-th atom in the atoms object."""
    vor = Voronoi(atoms.positions.copy()[:, select])
    fisrt_lst: list[int] = vor.regions[vor.point_region[index]]
    lst: list[tuple[int, int]] = []
    for region in vor.regions:
        if not np.any(np.isin(region, fisrt_lst)):
            continue
        print(region, fisrt_lst)
        for idx, first in enumerate(region):
            second = region[idx + 1] if idx < len(region) - 1 else region[0]
            if first in fisrt_lst and second not in fisrt_lst:
                lst.append((first, second))
            elif first not in fisrt_lst and second in fisrt_lst:
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

    atoms = Octahedron("Cu", 8)
    del atoms[atoms.positions[:, 2] > 0.1]
    del atoms[atoms.positions[:, 2] < -0.1]
    i = 28
    atoms.numbers[i] = 79

    lst, _lst = [], get_voronoi_lst(atoms, i, select=(True, True, False))
    for item in _lst:
        item = [atoms.copy(), *item]
        assert len(item) == 3
        for idx, image in enumerate(item):
            if idx != 0:
                image.numbers[i] = 1
            else:
                lst.append(image)
            lst.append(image)
    fname = Path(__file__).with_suffix(".gif")
    write_animation(fname, lst, save_parameters={"fps": 3})
