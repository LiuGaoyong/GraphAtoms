from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class AtomsConfig:
    _target_: str = MISSING


@dataclass
class OctahedronAtomsConfig(AtomsConfig):
    _target_: str = "ase.cluster.Octahedron"
    symbol: str = MISSING
    length: int = MISSING
    cutoff: int = 0
    alloy: bool = False


@dataclass
class AseReadAtomsConfig(AtomsConfig):
    _target_: str = "ase.io.read"
    filename: str = MISSING
    index: int = -1
