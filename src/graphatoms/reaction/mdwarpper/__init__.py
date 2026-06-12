# ruff: noqa: D104 D107
from collections.abc import Mapping
from typing import Any, override

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.md import MolecularDynamics

from ...utils.parser import DictConfig, hydra_parse
from ..base.move import MoveABC


class MDWrapper(MoveABC):
    """The wrapper class for MD."""

    def __init__(self, md_cfg: Mapping[str, Any]) -> None:
        dct = DictConfig({k: v for k, v in md_cfg.items()})
        dct.update({"logfile": None, "trajectory": None})
        self.__dct = dct

    @override
    def __call__(self, atoms: Atoms, *args, **kwargs) -> Atoms:
        assert isinstance(atoms, Atoms), (
            "The atoms must be a instance of ase.Atoms"
        )
        assert isinstance(atoms.calc, Calculator), (
            "The calculator must be a instance of "  #
            "`ase.calculators.calculator`"
        )
        self.__md: MolecularDynamics = hydra_parse(
            self.__dct,
            MolecularDynamics,
            atoms=Atoms(atoms, calculator=atoms.calc),  # Copy the atoms
        )
        self.__md.run(steps=1)
        return atoms
