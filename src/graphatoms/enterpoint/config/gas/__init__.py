from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class GasConfig:
    name: str = MISSING
    sticking: float = 1.0
    pressure: float = 101325.0
