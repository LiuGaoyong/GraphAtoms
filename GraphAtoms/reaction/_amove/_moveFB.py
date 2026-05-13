"""The Monte Carlo moves by force-biased displacement.

See Details in the paper: https://doi.org/10.1063/1.2745293
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry.geometry import find_mic
from ase.units import kB
from typing_extensions import override

from ._base import CanonicalState
from ._base import MonteCarloMove as Move
from ._base import MonteCarloRecord as Record


class ForceBiasMixin:
    @staticmethod
    def get_displacement_probability(
        forces: np.ndarray,
        displacement: np.ndarray,
        temperature: float = 300.0,
        stepsize_max: float = 0.5,
        lambda_param: float = 0.5,  # 0.5 ~ 1.0
        old2new: bool = True,
    ) -> float:
        t = displacement if old2new else -displacement
        cond = np.logical_not(np.logical_or(np.isinf(t), t == 0))
        f = (lambda_param * forces[cond]) / (kB * temperature)
        f = np.clip(f, -500, 500)  # overflow encountered in sinh
        fenmu = 2 * np.sinh(f * stepsize_max)
        fenzi = f * np.exp(-f * t[cond])
        return np.prod(fenzi / fenmu).item()

    @classmethod
    def get_force_bias_displacement(
        cls,
        forces: np.ndarray,
        temperature: float = 300.0,
        stepsize_max: float = 0.5,
        lambda_param: float = 0.5,  # 0.5 ~ 1.0
    ) -> tuple[np.ndarray, float]:
        kBT = kB * abs(float(temperature))
        fenmu = lambda_param * forces / kBT
        fenzi0 = np.exp(-fenmu * stepsize_max)
        fenzi1 = np.sinh(fenmu * stepsize_max)
        u = np.random.uniform(size=forces.shape)
        disp: np.ndarray = np.log(fenzi0 + 2 * u * fenzi1) / fenmu
        disp[np.where(np.isnan(disp))] = 0
        disp[np.where(np.isinf(disp))] = 0
        return disp, cls.get_displacement_probability(
            forces=forces,
            displacement=disp,
            temperature=temperature,
            stepsize_max=stepsize_max,
            lambda_param=lambda_param,
            old2new=False,
        )


class FBDisp(Move):
    """Multi-particle displacement by force-biased method."""

    @override
    def _get_new_atoms(
        self,
        *,
        temperature: float = 300.0,
        stepsize_max: float = 0.5,
        lambda_param: float = 0.5,  # 0.5 ~ 1.0
        **kwargs,
    ) -> tuple[Atoms, str]:
        assert self.record.state_old is not None
        e0 = self.atoms.get_potential_energy()
        f0 = self.atoms.get_forces()
        fmax0 = np.max(np.linalg.norm(f0, axis=1))
        dpos, w_old2new = ForceBiasMixin.get_force_bias_displacement(
            forces=f0,
            temperature=temperature,
            stepsize_max=stepsize_max,
            lambda_param=lambda_param,
        )
        new_atoms = Atoms(
            self.atoms,
            self.atoms.positions + dpos,
            calculator=self.calc,
        )
        e1 = new_atoms.get_potential_energy()
        f1 = new_atoms.get_forces()
        w_new2old = ForceBiasMixin.get_displacement_probability(
            forces=f1,
            displacement=dpos,
            temperature=temperature,
            stepsize_max=stepsize_max,
            lambda_param=lambda_param,
            old2new=False,
        )
        self.record.fmax_new = np.max(np.linalg.norm(f1, axis=1))
        self.record.energy_change = ediff = e1 - e0
        self.record.state_new = CanonicalState(atoms=new_atoms, U=e1)
        kBT = kB * abs(float(temperature))
        print(w_new2old, w_old2new)
        p = min(1, np.exp(-ediff / kBT) * (w_new2old / w_old2new))
        self.record.accept_probability = p

        self.record.state_old.U = e0
        self.record.fmax_old = fmax0

        return (new_atoms, "Disp(all),ForceBias")

    @override
    def _check_accept(self, *args, **kwargs) -> bool:
        return self.record.random_number < self.record.accept_probability

    @classmethod
    def run(  # type: ignore
        cls,
        *,
        state: CanonicalState,
        calculator: Calculator | None = None,
        temperature: float = 300.0,
        stepsize_max: float = 0.5,
        lambda_param: float = 0.5,  # 0.5 ~ 1.0
        cutoff: float = np.inf,
    ) -> tuple[CanonicalState, Record]:
        _state, record = super().run(
            state=state,
            calculator=calculator,
            stepsize_max=stepsize_max,
            lambda_param=lambda_param,
            temperature=temperature,
            cutoff=cutoff,
        )
        assert isinstance(_state, CanonicalState)
        return _state, record


class FBDispC(FBDisp):
    @override
    def _get_new_atoms(
        self,
        *,
        temperature: float = 300,
        stepsize_max: float = 0.5,
        lambda_param: float = 0.5,
        cutoff: float = np.inf,
        **kwargs,
    ) -> tuple[Atoms, str]:
        if np.isinf(cutoff):
            return super()._get_new_atoms(
                temperature=temperature,
                stepsize_max=stepsize_max,
                lambda_param=lambda_param,
                **kwargs,
            )
        else:
            raise NotImplementedError
            i = np.random.randint(len(self.atoms))
            center: np.ndarray = self.atoms.positions[i]
            v: np.ndarray = self.atoms.positions - center
            _, dmic = find_mic(v, self.atoms.cell, True)
            nbrs = np.where(dmic < float(cutoff))[0]
            atoms0: Atoms = self.atoms.__getitem__(nbrs)  # type: ignore
            atoms0.calc = self.calc
            atoms0.calc.reset()
            e0 = atoms0.get_potential_energy()
            f0: np.ndarray = atoms0.get_forces()
            f = f0[nbrs == i].ravel()
            assert f.shape == (3,)
            displacement = np.zeros_like(self.atoms.positions)
            disp_i, self._w_old2new = self.get_force_bias_displacement(
                forces=f,
                temperature=temperature,
                stepsize_max=stepsize_max,
                lambda_param=lambda_param,
            )
            displacement[i, :] = disp_i
            how_to_change = f"Disp({i}),ForceBias"
