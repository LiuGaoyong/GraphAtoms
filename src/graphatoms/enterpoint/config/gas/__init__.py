from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class GasConfig:
    pass


@dataclass
class AseReadGasConfig(GasConfig):
    _target_: str = "ase.io.read"
    filename: str = MISSING
    index: str = ":"
