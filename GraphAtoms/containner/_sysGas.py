from typing import override

import numpy as np
from ase import Atoms
from ase.build import molecule
from numpy.typing import ArrayLike
from pydantic import model_validator
from typing_extensions import Self

from GraphAtoms.containner._graph import GRAPH_KEY, GraphContainner


class Gas(GraphContainner):
    """The gas molecular system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        assert self.move_fix_tag is None
        assert self.coordination is None
        assert self.pressure is not None
        assert self.box is None
        assert self.is_nonmetal
        return self

    @classmethod
    @override
    def from_ase(
        cls,
        atoms: Atoms,
        infer_conn: bool = True,
        infer_order: bool = False,
        multiply_factor: float = 1,
        plus_factor: float = 0.5,
        charge: int = 0,
        energy: float = np.nan,
        pressure: float = 101325.0,
        frequencies: ArrayLike = np.array([]),
    ) -> Self:
        freqs = np.asarray(frequencies, float).flatten()
        obj = GraphContainner.from_ase(
            atoms,
            infer_conn,
            infer_order=infer_order,
            plus_factor=plus_factor,
            multiply_factor=multiply_factor,
            charge=charge,
        )
        dct = obj.model_dump(mode="python", exclude_none=True)
        dct[GRAPH_KEY.GRAPH.PRESSURE] = float(pressure)
        dct[GRAPH_KEY.GRAPH.ENERGY] = float(energy)
        dct[GRAPH_KEY.GRAPH.FREQS] = freqs
        return cls.model_validate(dct)

    @classmethod
    def from_molecule(
        cls,
        name: str,
        energy: float,
        frequencies: ArrayLike,
        pressure: float = 101325.0,
        infer_order: bool = False,
        infer_conn: bool = True,
    ) -> Self:
        return cls.from_ase(
            molecule(name),
            frequencies=frequencies,
            infer_order=infer_order,
            infer_conn=infer_conn,
            pressure=pressure,
            energy=energy,
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        energy: float,
        frequencies: ArrayLike,
        pressure: float = 101325.0,
    ) -> Self:
        raise NotImplementedError
