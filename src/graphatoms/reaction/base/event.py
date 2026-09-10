from typing import Self, override

import numpy as np
from ase import Atoms
from pydantic import model_validator

from graphatoms.geometry.rotation import kabsch
from graphatoms.reaction.base.move import MoveABC
from graphatoms.reaction.base.rtgp import RTGP
from graphatoms.system import System

DEFAULT_CHECK_MINIMA_FMAX = 0.05  #    eV/Å
DEFAULT_CHECK_MINIMA_FQMIN = 30.0  #   cm^-1
DEFAULT_CHECK_TS_FQMIN = 20.0  #       cm^-1
DEFAULT_CHECK_TS_FMAX = 0.1  #         eV/Å


class EventABC(RTGP, MoveABC):
    """The base class for all KMC events in the reaction process.

    An event is a change of the system, which can be a reaction, a diffusion,
    or a desorption, etc. It is defined by the change of the system, which
    can be represented by the change of the graph.
    """

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
