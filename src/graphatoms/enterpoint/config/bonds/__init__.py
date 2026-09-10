from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BondsConfig:
    method: str = MISSING
    cfg: Any = MISSING


####### RAW ########
@dataclass
class _RawBondsConfig:
    multiply_factor: float = 1.0
    plus_factor: float = 0.5  # Angstrom


@dataclass
class RawBondsConfig(BondsConfig):
    method: str = "raw"
    cfg: _RawBondsConfig = field(default_factory=_RawBondsConfig)
