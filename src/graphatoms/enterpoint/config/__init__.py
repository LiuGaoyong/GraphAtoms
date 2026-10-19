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
    optimization_for_system: bool = True
    surface_only_explore_single_core: bool = True
    allow_multiple_adsorption: bool = False
    allow_explore_bulk: bool = False
    thetacutoff: float = -1  # 0-180, -1 means no cutoff
    step_distance: float = 0.4  # the step distance for adsorption
    min_distance: float = 1.0  # the minimum distance for adsorption
    max_distance: float = 3.0  # the maximum distance for adsorption
    nfibonacci: int = 10  # the number of fibonacci points for adsorption


@dataclass
class EventConfig:
    db_format: str = "folder"
    check_frequency: bool = True  # whether to check the frequency
    max_force: float = 0.05  # eV / Angstrom
    min_frequency: float = 10.0  # cm^-1
    min_frequency_for_ts: float = 20.0  # cm^-1
    simplified_threshold: float = 7.0  # the simplified threshold for event
    gas_sticking: dict[str, float] = field(default_factory=dict)  # MISSING
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

    outputs: str = "."
    max_steps: int = 100000
    max_times: float = 1000.0  # seconds
    run_type: str = "otfkmc"  # otfkmc, rxngen
    gas_pressure_cache_box_length: float = 1e4  # in Angstrom
    initial_gas_pressure: dict[str, float] = field(default_factory=dict)
    # the initial gas pressure only effect for `OTFKMC`, and the key of
    # `initial_gas_pressure` must be the same as the key of `gas_sticking`
    # 2. if the key of `initial_gas_pressure` is not in `gas_sticking`, then the initial gas pressure is 0.0    # noqa: E501
    # 1. if the key of `gas_sticking` is not in `initial_gas_pressure`, then the gas sticking is 1.0            # noqa: E501

    parallel_workers: int = 0  # 0 means use all cores
    parallel_ncpus_for_tasks: int = 1  # the number of cpus for tasks, only effect for `parallel=ray` # noqa: E501
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
    print(OmegaConf.to_yaml(cfg))
    print("""
hydra:
  run:
    dir: ${outputs}
  verbose: ${debug}
""")
