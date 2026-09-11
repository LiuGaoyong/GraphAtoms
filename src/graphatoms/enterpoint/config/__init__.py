import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ["HYDRA_FULL_ERROR"] = "1"
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from .atoms import AseReadAtomsConfig, AtomsConfig
from .atoms import OctahedronAtomsConfig as OctAtomsConfig
from .bonds import BondsConfig, RawBondsConfig
from .calculator import CalcConfig, EMTCalcConfig
from .calculator import NequipCalcConfig as NequipConfig
from .gas import GasConfig

CONFIG_DIR = Path(__file__).parent


@dataclass
class SiteConfig:
    method: str = "distance"  # or hop
    env_threshold: float = 15  # the environment threshold
    max_moved_threshold: float = 8  # the maximum moved threshold


@dataclass
class ExplConfig:
    maxtry: int = 1000
    maxconfidence: float = 10
    max_ncore_for_surface: int = 3  # 4, 5, 6
    surface_only_explore_single_core: bool = True
    allow_multiple_adsorption: bool = False
    allow_explore_bulk: bool = False
    thetacutoff: float = 30
    nfibonacci: int = 200

@dataclass
class EventConfig:
    db_format: str = "folder"
    check_frequency: bool = True  # whether to check the frequency
    max_force: float = 0.05  # eV / Angstrom
    min_frequency: float = 30.0  # cm^-1
    min_frequency_for_ts: float = 20.0  # cm^-1
    simplified_threshold: float = 7.0  # the simplified threshold for event
    gas_sticking: dict[str, float] = field(default_factory=dict)
    gas_pressure: dict[str, float] = field(default_factory=dict)
    default_pressure: float = 101325
    default_sticking: float = 1.0
    temperature: float = 300


@dataclass
class OptConfig:
    method: str = "lbfgs"
    fmax: float = 0.05
    steps: int = 200


@dataclass
class Config:
    atoms: AtomsConfig = MISSING
    bonds: BondsConfig = MISSING
    system: Any = MISSING

    calculator: CalcConfig = MISSING

    outputs: str = "./zzz"
    max_steps: int = 100000
    max_times: float = 1000  # seconds
    parallel_workers: int = 0  # 0 means use all cores
    parallel: str = "serial"  # serial, multiprocessing, ray, dask, executorlib
    loglevel: str = "info"  # Literal["debug", "info", "warning", "error"]
    logfile: str = "log.txt"
    restart: bool = True
    debug: bool = False

    site: SiteConfig = field(default_factory=SiteConfig)
    exploration: ExplConfig = field(default_factory=ExplConfig)
    event: EventConfig = field(default_factory=EventConfig)
    optimizer: OptConfig = field(default_factory=OptConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="atoms", name="atoms_aseread", node=AseReadAtomsConfig)
cs.store(group="atoms", name="atoms_octahedron", node=OctAtomsConfig)
cs.store(group="bonds", name="bonds_raw", node=RawBondsConfig)
cs.store(group="calculator", name="calc_emt", node=EMTCalcConfig)
cs.store(group="calculator", name="calc_nequip", node=NequipConfig)


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def print_config(cfg: Config) -> None:
    from shutil import rmtree

    print(OmegaConf.to_yaml(cfg))
    rmtree(Path(cfg.outputs))



