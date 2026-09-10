from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class CalcConfig:
    _target_: str = MISSING


@dataclass
class EMTCalcConfig(CalcConfig):
    _target_: str = "ase.calculators.emt.EMT"


@dataclass
class _NequipCalcConfig(CalcConfig):
    _target_: str = "os.path.expanduser"
    path: str = MISSING


@dataclass
class NequipCalcConfig(CalcConfig):
    _target_: str = (
        "nequip.integrations.ase.NequIPCalculator.from_compiled_model"
    )
    chemical_species_to_atom_type_map: bool = True
    compile_path: _NequipCalcConfig = field(default_factory=_NequipCalcConfig)
    device: str = "cpu"
