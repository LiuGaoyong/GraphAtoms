import numpy as np
from ase.geometry.geometry import find_mic
from pydantic import model_validator
from scipy import sparse as sp
from typing_extensions import Self

from GraphAtoms.containner._aOther import OTHER_KEY
from GraphAtoms.containner._atomic import ATOM_KEY, AtomsWithBoxEng
from GraphAtoms.containner._gBonded import BondsWithComp
from GraphAtoms.containner._graph import BOND_KEY, Graph
from GraphAtoms.utils.geometry import distance_factory


class System(Graph):
    """The whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        msg = "The key of `{:s}` should be None for System."
        for k in (ATOM_KEY.MOVE_FIX_TAG, ATOM_KEY.COORDINATION):
            assert getattr(self, k) is None, msg.format(k)
        return self

    def local_update_geometry(
        self,
        new_index: np.ndarray | None,
        new_geometry: np.ndarray | None,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
    ) -> Self:
        if new_index is None and new_geometry is None:
            raise KeyError("new_index, new_geometry cannot be None")
        elif new_geometry is not None:
            geom = np.asarray(new_geometry)
            assert geom.shape == (self.natoms, 3)
            diff = find_mic(geom - self.positions, self.ase_cell)[1]
            index = np.asarray(np.where(diff > 1e-6)[0], int)
        else:
            index = np.asarray(new_index, dtype=int)
            geom = self.positions
        dct = self.model_dump(
            exclude_none=True,
            exclude={
                OTHER_KEY.ENERGY,
                OTHER_KEY.FREQS,
                OTHER_KEY.FMAXC,
                OTHER_KEY.FMAX,
                ATOM_KEY.IS_OUTER,
                ATOM_KEY.COORDINATION,
                ATOM_KEY.MOVE_FIX_TAG,
                ATOM_KEY.POSITION,
                BOND_KEY.ORDER,
            },
        ) | {ATOM_KEY.POSITION: geom}
        m = sp.csr_array(self.MATRIX, dtype=bool)
        m[new_index] = distance_factory.get_adjacency_sparse_matrix(
            numbers=self.Z,
            geometry=geom,
            batch=new_index,
            batch_other=None,
            cov_multiply_factor=multiply_factor,
            cov_plus_factor=plus_factor,
            cell=self.ase_cell,
        ).tocsr()
        m = sp.coo_array(sp.triu(m + m.T, k=1))
        dct[BOND_KEY.SOURCE] = m.row
        dct[BOND_KEY.TARGET] = m.col
        if self.order is not None:
            dct.update(
                BondsWithComp.infer_bond_as_dict(
                    AtomsWithBoxEng.model_validate(dct),
                    plus_factor=plus_factor,
                    multiply_factor=multiply_factor,
                    infer_order=True,
                    charge=0,
                )
            )
        assert self.is_outer is not None
        dct[ATOM_KEY.IS_OUTER] = [
            (
                distance_factory.get_is_inner(
                    i,
                    numbers=self.Z,
                    geometry=geom,
                    cell=self.ase_cell,
                    adjacency_matrix=m,
                )
                if i in index
                else self.is_outer[i]
            )
            for i in range(len(self))
        ]
        return self.model_validate(dct)
