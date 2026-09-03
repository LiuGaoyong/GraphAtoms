from functools import cached_property
from pathlib import Path
from typing import Self, override

import igraph
import numpy as np
from ase import Atoms
from ase.io.trajectory import TrajectoryReader
from pydantic import model_validator

from graphatoms.dataclasses import OurFrozenModel
from graphatoms.geometry.rotation import kabsch
from graphatoms.reaction.base.move import MoveABC
from graphatoms.system import (
    DEFAULT_WH_HASH_DEPTH,
    Cluster,
    Gas,
    SysGraph,
    System,
)
from graphatoms.utils.bytestool import hash_string

DEFAULT_CHECK_MINIMA_FMAX = 0.05  #    eV/Å
DEFAULT_CHECK_MINIMA_FQMIN = 30.0  #   cm^-1
DEFAULT_CHECK_TS_FQMIN = 20.0  #       cm^-1
DEFAULT_CHECK_TS_FMAX = 0.1  #         eV/Å


class RTGP(OurFrozenModel, MoveABC):
    R: SysGraph
    T: SysGraph | None = None
    G: Gas | None = None
    P: SysGraph

    @classmethod
    def from_ase_trajectory(cls, traj: list[Atoms] | str | Path) -> Self:
        raise NotImplementedError

    ########################################################################
    #                       Validation for the event.
    ########################################################################

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        self.__check_basic()
        self.__chech_gas()
        self.__check_ts()
        return self

    def __check_basic(self) -> None:
        assert isinstance(self.R, self.P.__class__), (
            "The `R` and `P` should be of the same class."
        )
        assert self.R.check_minima(
            fmax=DEFAULT_CHECK_MINIMA_FMAX,
            fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
        ), "The reactant should be a minima."
        assert self.P.check_minima(
            fmax=DEFAULT_CHECK_MINIMA_FMAX,
            fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
        ), "The product should be a minima."
        assert self.R.is_connected, "The `R` should be a connected graph."
        assert self.P.is_connected, "The `P` should be a connected graph."
        assert self.R.is_periodic == self.P.is_periodic, (
            "The `R` and `P` should be of the same periodicity."
        )
        assert self.R.is_orthorhombic == self.P.is_orthorhombic, (
            "The `R` and `P` should be of the same orthorhombicity."
        )
        assert np.allclose(self.R.ase_cell, self.P.ase_cell), (
            "The `R` and `P` should have the same cell."
        )
        assert self.R.hash != self.P.hash, (
            "The `R` and `P` should have different hash."
        )

        assert any(i is not None for i in [self.T, self.G]), (
            "At least one of `T` and `G` should be not None."
        )

    def __chech_gas(self) -> None:
        if self.G is not None:
            assert self.G.check_minima(
                fmax=DEFAULT_CHECK_MINIMA_FMAX,
                fqmin=DEFAULT_CHECK_MINIMA_FQMIN,
            ), "The gas should be a minima."
            n = abs(len(self.P) - len(self.R))
            assert len(self.G) == int(n), (
                "The number of gas atoms should match the difference in "
                "the number of atoms between the product and reactant."
            )

            small, big = sorted([self.R, self.P], key=lambda x: len(x))
            z: np.ndarray = np.append(small.numbers, self.G.numbers)
            assert np.array_equal(z, big.numbers), (
                "The combined numbers of the small `R/P` and "
                "gas should match the numbers of the big `R/P`."
            )

    def __check_ts(self) -> None:
        if self.T is not None:
            assert self.T.check_ts(
                fmax=DEFAULT_CHECK_TS_FMAX,
                fqmin=DEFAULT_CHECK_TS_FQMIN,
            ), "The `T` should be a transition state."
            assert isinstance(self.T, self.R.__class__), (
                "The `R` and `T` should be of the same class."
            )
            assert len(self.T) == max(len(self.R), len(self.P)), (
                "The number of atoms in the transition state "
                "should be equal to the maximum number of "
                "atoms in the reactant and product."
            )

    ########################################################################
    #                   the magic methods for the event.
    ########################################################################

    @override  # for __str__ method of the base class
    def _string(self) -> str:
        r_fml = self.R.symbols.get_chemical_formula("metal")
        p_fml = self.P.symbols.get_chemical_formula("metal")
        before, after = f"{r_fml}:{self.R.hash}", f"{p_fml}:{self.P.hash}"
        if self.G is not None:
            gas_fml = self.G.symbols.get_chemical_formula("metal")
            if len(self.R) < len(self.P):
                before = f"{gas_fml} + {before}"
            else:
                after = f"{gas_fml} + {after}"
        if self.T is None:
            ts = "none"
        else:
            ts = self.T.symbols.get_chemical_formula("metal")
            ts = f"{ts}:{self.T.hash}"
        return f"{before} --> {ts} --> {after}"

    @cached_property
    @override  # for __hash__ method of the base class
    def hash(self) -> str:
        t = self.T.hash if self.T is not None else ""
        g = self.G.hash if self.G is not None else ""
        v = ",".join([*sorted([self.R.hash, self.P.hash]), t, g])
        return hash_string(v, digest_size=DEFAULT_WH_HASH_DEPTH)

    def simplify(self, env_radius: float = 5.0) -> Self:
        """Simplify the event by removing the atoms that are not involved."""
        graph_node_moved: set[int] = set()
        g0: igraph.Graph = self.R.get_igraph()
        for g1 in [i.get_igraph() for i in [self.P, self.T] if i is not None]:
            for g2 in [g1.difference(g0), g0.difference(g1)]:
                for e in g2.es:
                    for vid in e.tuple:
                        graph_node_moved.add(vid)
        # print(f"graph_node_moved: {graph_node_moved}")

        # Got geometry-based simplification, but it is not used for now.
        # n = min(len(i) for i in [self.R, self.P, self.T] if i is not None)
        # r_geom, p_geom = self.R.positions[:n, :], self.P.positions[:n, :]
        # rp_var = np.linalg.norm(r_geom - p_geom, axis=1) < 0.05  # Angstrom
        # if self.T is not None:
        #     t_geom = self.T.positions[:n, :]
        #     rt_var = np.linalg.norm(r_geom - t_geom, axis=1) < 0.05
        #     tp_var = np.linalg.norm(t_geom - p_geom, axis=1) < 0.05
        #     rp_var = rp_var | rt_var | tp_var
        # geom_moved = set(np.argwhere(np.logical_not(rp_var)).flatten())
        # print(f"geom_moved: {geom_moved}")

        moved: np.ndarray = np.asarray(list(graph_node_moved), dtype=int)
        pos = self.R.positions[moved, :].reshape(-1, len(moved), 3)
        v = self.R.positions[:, np.newaxis, :] - pos
        d = np.linalg.norm(v, axis=-1).min(-1)
        sub = np.argwhere(d < env_radius).flatten()

        rtgp: list[SysGraph | None] = []
        for i in [self.R, self.T, self.G, self.P]:
            if i is None or isinstance(i, Gas):
                rtgp.append(i)
            elif isinstance(i, SysGraph):
                rtgp.append(
                    Cluster.select(
                        i,
                        sub_idxs=sub,
                        exclude_energetics=False,
                    )  # type: ignore
                )
            else:
                raise TypeError(f"Unknown type: {type(i)}")

        r, t, g, p = rtgp
        assert isinstance(r, SysGraph)
        assert isinstance(p, SysGraph)
        assert g is None or isinstance(g, Gas)
        assert t is None or isinstance(t, SysGraph)
        return self.__class__(R=r, T=t, G=g, P=p)

    def __reversed__(self) -> Self:  # type: ignore
        return self.__class__(R=self.P, G=self.G, T=self.T, P=self.R)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif self.hash == other.hash:
            return False
        else:
            return self.R.hash == other.R.hash


class Event(RTGP):
    """The base class for all KMC events in the reaction process.

    An event is a change of the system, which can be a reaction, a diffusion,
    or a desorption, etc. It is defined by the change of the system, which
    can be represented by the change of the graph.
    """

    @override
    @classmethod
    def from_ase_trajectory(cls, traj: list[Atoms] | str | Path) -> Self:
        if not isinstance(traj, list):
            traj = list(TrajectoryReader(traj))  # type: ignore
        assert isinstance(traj, list), "The trajectory must be a list."
        if any(not isinstance(t, Atoms) for t in traj):
            raise ValueError("The trajectory must be a list of ase.Atoms.")
        raise NotImplementedError("Adsorption is not implemented.")

    @model_validator(mode="after")
    def __check_something(self) -> Self:
        assert any(
            [
                self.is_reaction_LH,
                self.is_reaction_ER,
                self.is_adsorption,
                self.is_desorption,
            ]
        ), (
            "The event should be either a reaction based on "
            "Langmuir-Hinsher model, a reaction based on "
            "Eley-Rideal model, an adsorption or a desorption."
        )
        return self

    @override
    def apply(
        self,
        atoms: System | Atoms,
        *args,
        matched_indxs: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Atoms, float]:
        if matched_indxs is None:
            assert isinstance(atoms, System), (
                "The `atoms` should be a System "
                + "when `matched_indxs` is None."
            )
            matched_indxs = atoms.get_match_mode(self.R)  # type: ignore
        elif not isinstance(atoms, Atoms):
            atoms = atoms.to_ase(
                exclude_energetics=True,
                exclude_bond_attibutes=True,
            )

        matched_indxs = np.asarray(matched_indxs, dtype=int)

        if matched_indxs.ndim == 1:
            matched_indxs = matched_indxs.flatten()
            assert len(matched_indxs) == len(atoms)
            assert isinstance(atoms, Atoms)
            atoms.info = {}

            _i = np.vectorize(lambda x: np.argwhere(matched_indxs == x).item())(
                np.arange(len(self.R))
            )
            rot, t, rmsd = kabsch(
                A=self.R.positions,
                B=atoms.positions[_i, :],
            )  # A = rotate(B) + t
            rot_inv, t_inv = rot.inv(), -t

            # Old Usage: original atoms will be rotated.
            # # 1. geom --> geom reactant
            # geom = rot.apply(atoms.positions) + t
            # # 2. geom reactant --> geom product
            # geom[_i, :] += self.P.positions - self.R.positions
            # # 3. geom product --> result
            # geom = rot_inv.apply(geom) + t_inv

            # New Usage: original atoms will not be rotated.
            pos_r = rot_inv.apply(self.R.positions) + t_inv
            pos_p = rot_inv.apply(self.P.positions) + t_inv
            pos_diff = pos_p - pos_r
            geom = atoms.positions.copy()
            geom[_i, :] += pos_diff

            return Atoms(
                numbers=atoms.numbers,
                positions=geom,
                cell=atoms.cell,
                pbc=atoms.pbc,
            ), rmsd

        elif matched_indxs.ndim == 2:
            res_lst, rmsd_lst = [], []
            for i in range(len(matched_indxs)):
                res, rmsd = self.apply(
                    atoms=atoms,
                    matched_indxs=matched_indxs[i, :],
                )
                res_lst.append(res)
                rmsd_lst.append(rmsd)
            i = np.argmin(rmsd_lst)
            return res_lst[i], rmsd_lst[i]

        else:
            raise ValueError(
                "The `matched_indxs` should be either a 1D or 2D array."
            )

    ########################################################################
    #           Properties for checking the type of the event.
    ########################################################################
    @property
    def is_reaction(self) -> bool:
        """Whether the event is a reaction."""
        n = int(max(len(self.R), len(self.P)))
        return self.T is not None and len(self.T) == n

    @property
    def is_reaction_LH(self) -> bool:
        """Whether the event is a reaction based on Langmuir-Hinsher model."""
        return self.is_reaction and self.G is None

    @property
    def is_reaction_ER(self) -> bool:
        """Whether the event is a reaction based on Eley-Rideal model."""
        return (
            self.is_reaction
            and self.G is not None
            and (
                len(self.P) == len(self.R) + len(self.G)  #
                or len(self.R) == len(self.P) + len(self.G)
            )
        )

    @property
    def is_adsorption(self) -> bool:
        """Whether the event is an adsorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.P) == len(self.R) + len(self.G)
        )

    @property
    def is_desorption(self) -> bool:
        """Whether the event is a desorption."""
        return (
            self.T is None
            and self.G is not None
            and len(self.R) == len(self.P) + len(self.G)
        )
