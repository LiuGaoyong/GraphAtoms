"""The Monte Carlo move for swap atomic pair."""

from typing import override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols, covalent_radii
from ase.geometry import find_mic
from vesin._ase import ase_neighbor_list

from ._base import CanonicalState
from ._base import MonteCarloRecord as Record
from ._moveSP import SPDispC


class NoSwap(RuntimeError):
    """No swap pair found."""


class SwapCutoff(SPDispC):
    def _get_pair_for_swap(self) -> tuple[int, int, float]:
        max_d = np.max(covalent_radii[self.atoms.numbers]) * 2 + 0.3
        i, j, d = ase_neighbor_list("ijd", self.atoms, cutoff=max_d)
        cond = d < (
            covalent_radii[self.atoms.numbers[i]]
            + covalent_radii[self.atoms.numbers[j]]
            + 0.3
        )
        cond_z_diff = self.atoms.numbers[i] != self.atoms.numbers[j]
        cond = np.logical_and(cond, cond_z_diff)
        i, j, d = i[cond], j[cond], d[cond]
        if len(i) == 0:
            raise NoSwap("No suitable pair found")
        else:
            select_pair_id = np.random.randint(len(i))
            select_i = int(i[select_pair_id])
            select_j = int(j[select_pair_id])
            select_d = float(d[select_pair_id])
            return select_i, select_j, select_d

    @override
    def _get_new_atoms(self, **kwargs) -> tuple[Atoms, str]:
        select_i, select_j, select_d = self._get_pair_for_swap()
        self._center = (
            self.atoms.positions[select_j]  #
            + self.atoms.positions[select_i]
        ) / 2

        new_atoms = self.atoms.copy()
        new_atoms.calc = self.calc
        (
            new_atoms.numbers[select_i],
            new_atoms.numbers[select_j],
        ) = (
            new_atoms.numbers[select_j],
            new_atoms.numbers[select_i],
        )
        sym_i = chemical_symbols[self.atoms.numbers[select_i]]
        sym_j = chemical_symbols[self.atoms.numbers[select_j]]
        fml = f"{sym_i}{select_i},{sym_j}{select_j}"
        str_d = f"d={select_d:.2f}\u212b"
        return (new_atoms, f"Swap({fml}),{str_d}")

    @classmethod
    def run(  # type: ignore
        cls,
        *,
        state: CanonicalState,
        calculator: Calculator | None = None,
        temperature: float = 300.0,
        cutoff: float = np.inf,
    ) -> tuple[CanonicalState, Record]:
        _state, record = super().run(
            state=state,
            calculator=calculator,
            temperature=temperature,
            cutoff=cutoff,
        )
        assert isinstance(_state, CanonicalState)
        return _state, record


class Swap(SwapCutoff):
    @override
    def _get_pair_for_swap(self) -> tuple[int, int, float]:
        if len(np.unique(self.atoms.numbers)) == 1:
            raise NoSwap("No suitable pair found")
        else:
            while True:
                pair = np.random.randint(len(self.atoms), size=2)
                select_i, select_j = int(pair[0]), int(pair[1])
                if select_i != select_j:
                    zi = self.atoms.numbers[select_i]
                    zj = self.atoms.numbers[select_j]
                    if zi != zj:
                        v = self.atoms.positions[select_i]
                        v -= self.atoms.positions[select_j]
                        _, d = find_mic(v, self.atoms.cell, pbc=True)
                        select_d = float(d)
                        break
            return select_i, select_j, select_d

    @classmethod
    def run(  # type: ignore
        cls,
        *,
        state: CanonicalState,
        calculator: Calculator | None = None,
        temperature: float = 300.0,
    ) -> tuple[CanonicalState, Record]:
        _state, record = super().run(
            state=state,
            calculator=calculator,
            temperature=temperature,
            cutoff=np.inf,
        )
        assert isinstance(_state, CanonicalState)
        return _state, record
