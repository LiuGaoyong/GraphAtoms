from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import reduce
from pathlib import Path
from typing import Any, Literal, Self, override

import matplotlib
import numpy as np
import numpy.typing as npt
import pydantic
from ase import Atom, Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import chemical_symbols as SYMBOLS
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from graphatoms.geometry import neighbor_list
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.rdutils import rdmol2ase, smiles2rdmol

matplotlib.use("Agg")
from .asetool import call_optimization as optimize


#########################################
# The basic data types for adsorption.
class _XYZ(pydantic.BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, lst: list[float]) -> Self:
        return cls(x=lst[0], y=lst[1], z=lst[2])


class Point(_XYZ):
    """A point in 3D space."""


class Vector(_XYZ):
    """A vector in 3D space."""

    @property
    def length(self) -> float:
        v = [self.x, self.y, self.z]
        return float(np.linalg.norm(v))

    @property
    def normalize(self) -> Self:
        """The normalized vector."""
        t: float = self.length
        return self.__class__(
            x=self.x / t,
            y=self.y / t,
            z=self.z / t,
        )

    @classmethod
    def from_2points(cls, a: Point, b: Point) -> Self:
        return cls(
            x=b.x - a.x,
            y=b.y - a.y,
            z=b.z - a.z,
        )


class Site(pydantic.BaseModel):
    """The site for adsorption."""

    neighbor: list[Point]
    core: list[Point]

    @property
    def center(self) -> Point:
        """The center for adsoption."""
        core = np.asarray([p.to_list() for p in self.core])
        return Point.from_list(np.mean(core, axis=0))  # type: ignore

    @property
    def direction(self) -> Vector:
        """The direction vector for adsorption."""
        center = np.asarray(self.center.to_list())
        nbr = np.asarray([p.to_list() for p in self.neighbor])
        n2c = center - nbr  # the vector from the neighbor to the center
        n2c_norm = np.linalg.norm(n2c, axis=1)  # the norm of n2c
        n2c_eye = n2c / n2c_norm[:, None]  # the unit vector of n2c
        sorted_norm = n2c_norm[np.argsort(-n2c_norm)]  # sort by norm
        sorted_eye = n2c_eye[np.argsort(n2c_norm)]  # sort by norm
        _n2c = sorted_eye * sorted_norm[:, None]
        return Vector.from_list(np.mean(_n2c, axis=0))  # type: ignore

    @classmethod
    def from_numpy(cls, nbr: ArrayLike, core: ArrayLike) -> Self:
        """Create a site from numpy array."""
        nbr, core = np.array(nbr, dtype=float), np.array(core, dtype=float)  # type: ignore
        assert core.ndim == 2 and core.shape[1] == 3, "The core must be Nx3."  # type: ignore
        assert nbr.ndim == 2 and nbr.shape[1] == 3, "The neighbor must be Nx3."  # type: ignore
        return cls(
            core=[Point.from_list(c) for c in core],
            neighbor=[Point.from_list(n) for n in nbr],
        )


def quaternion_apply(quat, pos) -> np.ndarray:
    rot = Rotation.from_quat(quat)
    return rot.apply(pos)


#########################################
# The abstract class for adsorption.
class AdsorptionABC(ABC):
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.calculator = calculator
        self.max_steps_for_first_stage = int(max_steps_for_first_stage)
        self.max_steps_for_second_stage = int(max_steps_for_second_stage)
        self.max_force = float(max_force)
        self.debug = bool(debug)

    @abstractmethod
    def __call__(  # noqa: D417
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: ArrayLike | None = 0,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        pass

    def _opt_1st_stage(
        self,
        atoms: Atoms,
        natoms: int,
    ) -> tuple[list[Atoms], bool]:
        """Optimize the first stage of the adsorption."""
        assert self.calculator is not None, (
            "The calculator must be set before calling the method."
        )
        # first stage optimization
        atoms = atoms.copy()
        atoms.set_constraint(
            [
                FixAtoms(indices=list(range(natoms))),
                FixBondLengths(
                    np.column_stack(
                        np.triu_indices(len(atoms) - natoms, k=1),
                    )
                    + natoms
                ),
            ]
        )
        try:
            lst, coveraged = optimize(
                atoms,
                self.calculator,
                logfile="-" if self.debug else None,
                max_steps=self.max_steps_for_first_stage,
                fmax=self.max_force,
                trajectory=None,
            )
        except Exception:
            # Sometimes, FixBondLengths will cause an error:
            #     RuntimeError: Did not converge
            # TODO: use torch automatic differentiation instead.
            lst, coveraged = [atoms], False
        return lst, coveraged

    def _opt(self, atoms: Atoms, natoms: int) -> tuple[Atoms, Literal[0, 1, 2]]:
        if self.calculator is None:
            return atoms, 0
        else:
            lst_1, coveraged_1 = self._opt_1st_stage(
                atoms=atoms,
                natoms=natoms,
            )

            if len(lst_1) > 0:
                atoms_2 = lst_1[-1].copy()
            else:
                atoms_2 = atoms.copy()
            atoms_2.set_constraint(None)
            lst_2, coveraged_2 = optimize(
                atoms_2,
                self.calculator,
                logfile="-" if self.debug else None,
                max_steps=self.max_steps_for_second_stage,
                fmax=self.max_force,
                trajectory=None,
            )
            self._atoms_lst = result_lst = lst_1 + lst_2
            # assert coveraged_1 or coveraged_2, (
            #     "The coveraged of the first stage or "
            #     "the second stage must be True."
            # )
            coveraged = int(sum([coveraged_1, coveraged_2]))
            if coveraged == 0:
                return atoms, 0
            else:
                return result_lst[-1], coveraged  # type: ignore

    @staticmethod
    def _get_adsorbate(adsorbate: Atoms | Gas | Atom | str) -> Atoms:
        """Convert the adsorbate to an Atoms object."""
        if isinstance(adsorbate, Atoms):
            ads = adsorbate
        elif isinstance(adsorbate, Atom):
            ads = Atoms([adsorbate])
        elif isinstance(adsorbate, str):
            if adsorbate in SYMBOLS:
                ads = Atoms([Atom(adsorbate)])
            else:
                try:
                    ads = molecule(adsorbate)
                except Exception:
                    # convert SMILES into ase.Atoms.
                    ads = rdmol2ase(smiles2rdmol(adsorbate))
        elif isinstance(adsorbate, Gas):
            ads = adsorbate.to_ase(
                exclude_energetics=True,
                exclude_bond_attibutes=True,
            )
        else:
            raise KeyError(f"Invalid adsorbate type({type(adsorbate)}).")
        assert isinstance(ads, Atoms), (
            f"Invalid adsorbate type({type(adsorbate)}."
        )
        if len(ads) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        return ads


#########################################
# The raw adsorption class.
# ---------------------------------------
class RawAdsorption(AdsorptionABC):
    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: npt.ArrayLike | None = None,
        *,
        adsorbate_index: Literal["com"] | int | None = None,
        nbr1hop: npt.ArrayLike | None = None,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        """Run the adsorption calculation.

        Args:
            atoms (Atoms | System | Cluster): The surface or
                cluster onto which the adsorbate should be added.
            adsorbate (Atoms | Gas | Atom | str): The adsorbate.
                Must be one of the following three types:
                    1. An atoms object (for a molecular adsorbate).
                    2. An atom object.
                    3. A string:
                        the chemical symbol for a single atom.
                        the molecule string by `ase.build`.
                        the SMILES of the molecule.
            adsorbate_index (int | None, optional): The index of the adsorbate.
                Defaults to None. It means that the adsorbate's core
                is its COM. If it is interger, it means that the
                adsorbate's core is the atom.
            nbr1hop (npt.ArrayLike | list[int] |None, optional):
                The first hop neighbor of core atoms.
                If None, the code will generated automated.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
        """
        adsorbate = ads = self._get_adsorbate(adsorbate=adsorbate)
        if not isinstance(atoms, Atoms):
            atoms, _origin = atoms.to_ase(), atoms
        else:
            _origin: System | Cluster | None = None
        assert isinstance(_origin, (System, Cluster)) or _origin is None

        # A. get `adsorbate_index` & `ad_anchor`
        if adsorbate_index == "com":
            ad_anchor: np.ndarray = ads.get_center_of_mass()
        else:
            if adsorbate_index is None:
                ads_nonH_idx = np.where(ads.numbers != 1)[0]
                if len(ads) == 1:
                    adsorbate_index = 0
                elif len(ads_nonH_idx) == 1:
                    adsorbate_index = ads_nonH_idx.item()  # non H atom
                elif len(ads) == 2:
                    if ads.numbers[0] == ads.numbers[1]:
                        adsorbate_index = 0
                    elif 6 in ads.numbers and 8 in ads.numbers:
                        idx_C = np.where(ads.numbers == 6)[0]
                        adsorbate_index = idx_C.item()  # C atom for CO
                    else:
                        raise KeyError(
                            "Cannot determine the adsorbate index"
                            f" for {ads.get_chemical_formula()}."
                        )
                else:
                    raise KeyError(
                        "Please specify the adsorbate index"
                        f" for {ads.get_chemical_formula()}."
                    )
            else:
                adsorbate_index = int(adsorbate_index)
            assert isinstance(adsorbate_index, int), (
                "The adsorbate_index must be None or integer."
            )
            ad_anchor = ads.positions[adsorbate_index]
        assert isinstance(ad_anchor, np.ndarray) and ad_anchor.shape == (3,)
        assert isinstance(adsorbate_index, int) or adsorbate_index == "com"

        # B. Convert the core atoms to a list of integers (np.ndarray)
        core = np.asarray([core] if isinstance(core, int) else core, int)
        if len(core) > 6:
            raise ValueError(
                "The core size must be less than or equal"
                f" to 6. The value of core: {core}."
            )
        if nbr1hop is None:
            if _origin is not None:
                assert isinstance(_origin, (System, Cluster))
                lst = [_origin.get_neighbors(i) for i in core]
                nbr1hop = reduce(np.append, lst)
            else:
                nbr1hop = _get_1order_nbr(atoms, core)
        else:
            nbr1hop = np.asarray(nbr1hop, int).ravel()
        nbr1hop = np.setdiff1d(nbr1hop, core).ravel()
        assert len(nbr1hop) > 0, (
            f"No 1-hop neighbors found for the core of {core}."
        )
        site = Site.from_numpy(
            nbr=atoms.positions[nbr1hop],
            core=atoms.positions[core],
        )
        at_anchor: np.ndarray = np.asarray(site.center.to_list())
        direction: np.ndarray = np.asarray(site.direction.normalize.to_list())

        # C. get `distance_of_two_anchor`
        if isinstance(adsorbate_index, int):
            r1 = float(COV_R[adsorbate.numbers[adsorbate_index]])
        else:
            r1 = float(COV_R[adsorbate.numbers])
        r2 = float(np.mean(COV_R[atoms.numbers[core]]))
        if len(core) == 1:
            d2site = r1 + r2
        elif len(core) == 2:
            d2site = np.sqrt(r1**2 + 2 * r1 * r2)
        else:
            x2 = (r2 / np.sin(np.pi / len(core))) ** 2
            d2site = np.sqrt((r1 + r2) ** 2 - x2)

        ads = adsorbate.copy()
        if len(adsorbate) == 1:
            e = direction / np.linalg.norm(direction)
            ads.positions = at_anchor + (d2site) * e
        else:
            ref_pos = ad_anchor
            com_ads = ads.get_center_of_mass()
            if np.linalg.norm(com_ads - ref_pos) < 1e-5:
                # The COM is same as ref atom, high symmetry
                if np.linalg.matrix_rank(ads.positions) < 3:
                    ...
                else:
                    assert len(ads) > 4, (
                        "The length of adsorbate <=4, "
                        "and COM=REF, and it is not planar."
                    )
                    d = np.linalg.norm(ads.positions - com_ads, axis=1)
                    ref_pos = ads.positions[np.argsort(d)[-3:]].mean(axis=0)
            com_core = Atoms(atoms[core]).get_center_of_mass()
            d2com = float(np.linalg.norm(ref_pos - com_ads)) + d2site
            target_ref_pos = com_core + d2site * direction
            target_com_ads = com_core + d2com * direction
            ads.positions += target_ref_pos - ref_pos
            ads.rotate(
                a=ads.get_center_of_mass() - target_ref_pos,
                v=target_com_ads - target_ref_pos,
                center=target_ref_pos,
            )

        result = atoms.copy()
        result.extend(ads)
        return self._opt(
            natoms=len(atoms),
            atoms=result,
        )


def _get_1order_nbr(atoms: Atoms, core: np.ndarray | list[int]) -> np.ndarray:
    """Get the 1-order neighbors of the core atoms."""
    core, all = np.unique(core), np.arange(len(atoms))
    i = np.repeat(core, len(all))
    j = np.tile(all, len(core))
    d = atoms.get_distances(i, j, mic=True)
    d_ij = COV_R[atoms.numbers[i]] + COV_R[atoms.numbers[j]] + 0.3
    return j[d < d_ij]


#########################################
# The direct adsorption class.
# ---------------------------------------
class DirectAdsorption(AdsorptionABC):
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        super().__init__(
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
        )
        self.__nfibonacci = int(nfibonacci)

    def _combine(
        self,
        atoms: Atoms,
        adsorbate: Atoms,
        core: ArrayLike | None = None,
        *,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        anchor_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> Atoms:
        """Combine the substrate and adsorbate."""
        gas = adsorbate
        # A. get the direction of `adsorbate`
        if grid_ads is None:
            grid_ads, anchor_ads = self.__get_grids(adsorbate, None)
        else:
            grid_ads = np.asarray(grid_ads, dtype=float)
            anchor_ads = np.mean(adsorbate.positions, axis=0)
        assert grid_ads.ndim == 2 and grid_ads.shape[1] == 3
        assert len(grid_ads) == self.__nfibonacci
        if idx_grid_ads is None:
            idx_grid_ads = np.random.randint(len(grid_ads))
        idx_grid_ads = int(idx_grid_ads)
        # B. rotate adsorbate
        center = anchor_ads
        adsorbate.rotate(
            center + [0, 0, 1],
            grid_ads[idx_grid_ads],
            rotate_cell=False,
            center=center,
        )

        # C get the direction of core
        if grid_core is None:
            grid_core, anchor_core = self.__get_grids(atoms, core)
        else:
            assert anchor_core is not None, "anchor_core must be provided."
            anchor_core = np.asarray(anchor_core, dtype=float)
            grid_core = np.asarray(grid_core, dtype=float)
        assert grid_core.ndim == 2 and grid_core.shape[1] == 3
        if idx_grid_core is None:
            idx_grid_core = np.random.randint(len(grid_core))
        idx_grid_core = int(idx_grid_core)
        direction_core = grid_core[idx_grid_core] - anchor_core
        direction_core /= np.linalg.norm(direction_core)

        # place gas into
        if distance is None:
            d_gas = gas.positions - gas.positions.mean(axis=0)
            d_gas_max: float = np.max(np.linalg.norm(d_gas, axis=0))
            d_gas_min: float = np.max(COV_R[gas.numbers])
            v_core = grid_core - anchor_core
            _, d_core = find_mic(v_core, atoms.cell)
            distance = np.mean(d_core) + d_gas_min  # type: ignore
            distance += 0.5 * (d_gas_max - d_gas_min)  # type: ignore
        assert isinstance(distance, float)

        adsorbate.set_positions(
            adsorbate.positions
            - anchor_ads  # 1. move adsorbate to the zero position
            + anchor_core  # 2. move adsorbate to the core position
            + direction_core * distance  # 3. move adsorbate
        )

        # save some information
        self._adsorbate_pos = adsorbate.positions.copy()
        self._direction_core = direction_core
        self._anchor_core = anchor_core
        self._anchor_ads = anchor_ads
        self._distance = distance

        result = atoms.copy()
        result.extend(gas)
        return result

    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: ArrayLike | None = None,
        *,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        anchor_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        result = self._combine(
            atoms=atoms,
            adsorbate=self._get_adsorbate(adsorbate).copy(),
            core=core,
            idx_grid_core=idx_grid_core,
            grid_core=grid_core,
            anchor_core=anchor_core,
            grid_ads=grid_ads,
            idx_grid_ads=idx_grid_ads,
            distance=distance,
        )
        return self._opt(
            natoms=len(atoms),
            atoms=result,
        )

    def __get_grids(
        self,
        atoms: Atoms,
        core: ArrayLike | None = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(core, int):
            core = np.asarray([core])
        elif core is None:
            core = np.arange(len(atoms))
        core = np.asarray(core, dtype=int)
        core = np.unique(core.flatten())
        if len(core) == len(atoms):
            assert not atoms.pbc.any(), "PBC is not supported for 'COM'."
            anchor = np.mean(atoms.positions[core], axis=0)
            grid = fibonacci_lattice(self.__nfibonacci) + anchor
        else:
            grid, anchor = get_grid_and_anchor_of_core(
                atoms=atoms,
                select_core=core,
                nfibonacci=self.__nfibonacci,
            )
        return grid, anchor

    def grid_generation(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the grid of core and adsorbate.

        Returns:
            grid_core: The grid of core.
            grid_ads: The grid of adsorbate.
            anchor_core: The anchor of core.
        """
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        adsorbate = self._get_adsorbate(adsorbate)
        grid_ads, _ = self.__get_grids(adsorbate, None)
        grid_core, anchor_core = self.__get_grids(atoms, core)
        return grid_core, grid_ads, anchor_core


def get_grid_and_anchor_of_core(
    atoms: Atoms,
    select_core: int | list[int] | np.ndarray,
    nfibonacci: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(select_core, int):
        select_core = [select_core]
    core = np.asarray(select_core, dtype=int)
    core = np.unique(core)

    # move core atoms if mic
    _MIC_POS: np.ndarray = np.zeros(3)
    if any(atoms.get_pbc()) and len(core) > 1:
        _MIC_POS = atoms.cell.cartesian_positions([0.5, 0.5, 0.5])
        _MIC_POS -= atoms.positions[core[0]]
        atoms = atoms.copy()
        atoms.translate(_MIC_POS)
        atoms.wrap(pbc=True)
    assert _MIC_POS.shape == (3,)

    skin = 0.5
    scale = 1.5
    base_direction = None
    neighbors_exclude_core = False
    if len(core) == 1:
        neighbors_exclude_core = True
        skin, scale = 1.0, 1.2
    # elif len(core) == 2:
    #     skin, scale = 1.0, 1.6

    cov_core = COV_R[atoms.numbers[core]].max()
    grid = fibonacci_lattice(nfibonacci) * cov_core * float(scale)
    i, j = neighbor_list("ij", atoms, 5.0, self_interaction=False)
    cond = np.logical_and(np.isin(i, core), np.logical_not(np.isin(j, core)))
    nbrs = np.unique(np.append(np.append(i[cond], j[cond]), core).astype(int))
    if neighbors_exclude_core:
        nbrs = np.setdiff1d(nbrs, core.astype(int))
    cov_r = COV_R[atoms.numbers[nbrs]]
    pos = atoms.positions[nbrs]

    # calculate distance by minimum-image representation
    anchor = atoms.positions[core].mean(axis=0)
    grid: np.ndarray = anchor + grid
    if base_direction is not None:
        base_direction = np.asarray(base_direction)
        grid += base_direction.flatten()[:3]
    v = pos[:, np.newaxis, :] - grid[np.newaxis, :, :]  # (n_pos, n_grid, 3)
    _, vlen = find_mic(v.reshape(-1, 3), atoms.cell, True)
    d = vlen.reshape(v.shape[:2])

    matrix_cov_r = np.column_stack([cov_r] * len(grid))
    cond = np.all(matrix_cov_r + float(skin) < d, axis=0)
    grid = grid[cond] - _MIC_POS
    anchor = anchor - _MIC_POS
    return grid, anchor


class Helper:
    """The helper class for adsorption call."""

    def __init__(
        self,
        calculator: Calculator,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: ArrayLike | int = 0,
        use_direct: bool = True,
        use_raw: bool = True,
        *,
        use_direct_ad: bool = False,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        distance_lst: np.ndarray = np.arange(1.5, 5.0, 0.2),
        adsorbate_index: Literal["com"] | int | None = None,
        nbr1hop: ArrayLike | list[int] | None = None,
        bonds_cfg: Mapping[str, Any] = {},
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        """The helper function for adsorption call."""
        assert any([use_direct, use_raw]), (
            "At least one of `use_direct` and `use_raw` must be True."
        )
        # return the total number of runs
        self.__bonds_cfg = bonds_cfg
        self.__fmax = max_force
        self.__debug = debug
        self.nrun = 0
        if use_direct:
            if use_direct_ad:
                raise NotImplementedError(
                    "DirectAdsorptionAD is not implemented."
                )
                # _CLS = DirectAdsorptionAD
            else:
                _CLS = DirectAdsorption
            self.__obj_direct = obj = _CLS(
                calculator=calculator,
                max_steps_for_first_stage=max_steps_for_first_stage,
                max_steps_for_second_stage=max_steps_for_second_stage,
                nfibonacci=nfibonacci,
                max_force=max_force,
                debug=debug,
            )
            self.__grid_core, self.__grid_ads, self.__anchor_core = (
                obj.grid_generation(
                    adsorbate=adsorbate,
                    atoms=atoms,
                    core=core,
                )
            )
            self.__distance_lst = distance_lst
            self.nrun += (
                len(distance_lst)  #
                * len(self.__grid_core)  #
                * len(self.__grid_ads)
            )
        if use_raw:
            self.__obj_raw = RawAdsorption(
                calculator=calculator,
                max_steps_for_first_stage=max_steps_for_first_stage,
                max_steps_for_second_stage=max_steps_for_second_stage,
                max_force=max_force,
                debug=debug,
            )
            self.__adsorbate_index = adsorbate_index
            self.__nbr1hop = nbr1hop
            self.nrun += 1
        self.__use_raw = use_raw
        self.__use_direct = use_direct
        self.__adsorbate = adsorbate
        self.__atoms = atoms
        self.__core = core

    def __call__(  # noqa: D102
        self,
        irun: int = 0,
        outdir: Path = Path("."),
    ) -> dict[str, Any]:
        if self.__use_raw:
            irun -= 1

        if irun < 0:
            assert self.__use_raw, "use_raw must be True."
            result = self.__obj_raw.__call__(
                atoms=self.__atoms,
                adsorbate=self.__adsorbate,
                adsorbate_index=self.__adsorbate_index,  # type: ignore
                nbr1hop=self.__nbr1hop,
                core=self.__core,
            )
        else:
            assert self.__use_direct, "use_direct must be True."
            iother, idist = divmod(irun, len(self.__distance_lst))
            iad, icore = divmod(iother, len(self.__grid_core))
            result = self.__obj_direct.__call__(
                atoms=self.__atoms,
                adsorbate=self.__adsorbate,
                grid_ads=self.__grid_ads,
                idx_grid_ads=iad,
                core=self.__core,
                distance=self.__distance_lst[idist],
                idx_grid_core=icore,
                grid_core=self.__grid_core,
                anchor_core=self.__anchor_core,
            )
        result_atoms, nstage = result
        assert isinstance(result_atoms, Atoms)

        try:
            score = result_atoms.get_potential_energy(False, False)
            force = result_atoms.get_forces(False, False)
            fmax = np.linalg.norm(force, axis=1).max()
        except Exception:
            score = fmax = np.inf

        result = {"fmax": fmax, "score": score, "nstage": nstage}
        if not np.isinf(score) and fmax <= self.__fmax:
            sys = System.from_ase(result_atoms, parse_bonds=self.__bonds_cfg)
            if any(
                len(sys.get_neighbors(i)) > 0
                for i in range(len(self.__atoms), len(sys))
            ):
                if self.__debug:
                    key = [sys.symbols.get_chemical_formula("metal"), sys.hash]
                    # key.insert(0, f"E_{int(score * 1000):07d}meV")
                    key.append(f"stage_{nstage:d}")
                    s = "-".join(key)
                    result_atoms.write(
                        outdir.joinpath(f"{s}.xyz"), format="extxyz"
                    )
                    plot(result_atoms, pngfname=outdir.joinpath(f"{s}.png"))
                result["system"] = sys
        result["atoms"] = result_atoms
        return result


def plot(atoms: Atoms, pngfname: Path) -> None:
    """Plot the atoms."""
    atoms.wrap(pbc=any(atoms.pbc))
    fig, axes = plt.subplots(2, 2, dpi=150, figsize=(16, 12))
    for ax, rot in zip(
        axes.flatten(),
        [
            "0x, 0y, 0z",  # top
            "-90x, 0y, 0z",  #
            "0x, 90y, 0z",  #
            "45x, 45y, 45z",  #
        ],
    ):
        assert isinstance(ax, Axes)
        # atoms.write()
        plot_atoms(
            atoms,
            ax=ax,
            rotation=rot,
            radii=None,
            bbox=None,
            colors=None,
            scale=20,
            maxwidth=500,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(pngfname.with_suffix(".png"))
    plt.close(fig)
