"""The Monte Carlo move for single particle displacement."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry.geometry import find_mic
from ase.units import kB
from typing_extensions import override

from ._base import CanonicalState
from ._base import MonteCarloMove as Move
from ._base import MonteCarloRecord as Record


class SPDispC(Move):
    """Single particle displacement with cutoff."""

    @override
    def _get_new_atoms(
        self,
        *,
        stepsize_max: float = 0.5,
        **kwargs,
    ) -> tuple[Atoms, str]:
        i = np.random.randint(len(self.atoms))
        dpos = np.zeros_like(self.atoms.positions)
        dpos_i = np.random.randn(3)
        dpos_i /= np.linalg.norm(dpos_i)
        stepsize = np.random.uniform(0, stepsize_max)
        dpos[i, :] += dpos_i * float(stepsize)
        self._center = self.atoms.positions[i]
        return (
            Atoms(
                self.atoms,
                self.atoms.positions + dpos,
                calculator=self.calc,
            ),
            f"Disp({i}),size={stepsize:.2f}\u212b",
        )

    @override
    def _check_accept(
        self,
        *,
        new_atoms: Atoms,
        cutoff: float = np.inf,
        temperature: float = 300.0,
        **kwargs,
    ) -> bool:
        ediff, fmax0, fmax1 = self.only_spe_around(
            atoms_old=self.atoms,
            atoms_new=new_atoms,
            calc=self.calc,
            center=self._center,
            cutoff=cutoff,
        )
        self.record.energy_change = ediff
        self.record.fmax_old = fmax0
        self.record.fmax_new = fmax1
        self.record.state_new = self.state.__class__(
            atoms=new_atoms,
            U=self.state.U + ediff,
        )
        kBT = kB * abs(float(temperature))
        p = min(1, np.exp(-ediff / kBT))
        self.record.accept_probability = p
        return self.record.random_number < p

    @classmethod
    def run(  # type: ignore
        cls,
        *,
        state: CanonicalState,
        calculator: Calculator | None = None,
        temperature: float = 300.0,
        stepsize_max: float = 0.5,
        cutoff: float = np.inf,
    ) -> tuple[CanonicalState, Record]:
        _state, record = super().run(
            state=state,
            calculator=calculator,
            stepsize_max=stepsize_max,
            temperature=temperature,
            cutoff=cutoff,
        )
        assert isinstance(_state, CanonicalState)
        return _state, record

    @staticmethod
    def only_spe_around(
        atoms_old: Atoms,
        atoms_new: Atoms,
        calc: Calculator,
        center: np.ndarray,
        cutoff: float = np.inf,
    ) -> tuple[float, float, float]:
        """Returns the energy difference between the two structures
        and the maximum force of the old & new structures."""
        if np.isinf(cutoff):
            e0 = atoms_old.get_potential_energy()
            fmax0 = np.max(np.linalg.norm(atoms_old.get_forces(), axis=1))
            fmax1 = np.max(np.linalg.norm(atoms_new.get_forces(), axis=1))
            e1 = atoms_new.get_potential_energy()
        else:
            v: np.ndarray = atoms_old.positions - center
            assert v.shape == (len(atoms_old), 3), "Invalid shape: {v.shape}"
            _, dmic = find_mic(v, atoms_old.cell, True)
            nbrs = np.where(dmic < float(cutoff))[0]
            atoms0: Atoms = atoms_old.__getitem__(nbrs)  # type: ignore
            atoms0.calc = calc
            atoms0.calc.reset()
            e0 = atoms0.get_potential_energy()
            fmax0 = np.max(np.linalg.norm(atoms0.get_forces(), axis=1))
            atoms1: Atoms = atoms_new.__getitem__(nbrs)  # type: ignore
            atoms1.calc = calc
            atoms1.calc.reset()
            e1 = atoms1.get_potential_energy()
            fmax1 = np.max(np.linalg.norm(atoms1.get_forces(), axis=1))
        return e1 - e0, fmax0, fmax1
