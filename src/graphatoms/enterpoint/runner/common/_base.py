import json
import os
import sys
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

os.environ["LOGURU_FORMAT"] = LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    + " | <level>{level: ^8}</level> | "
    + "<level>{message}</level>"
)

import hydra
import numpy as np
from ase import Atoms
from ase.io import iread
from ase.units import _e as _electron_charge
from ase.units import kB as _boltzmann_constant
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.config.atoms import AseReadAtomsConfig
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.parallel import get_executor
from graphatoms.system import Gas, System
from graphatoms.utils.parser import hydra_parse

from ._helper import helper_optimization

_kB_in_JperK = _boltzmann_constant * _electron_charge

class RunnerABC:
    """The base class for all classes.

    It provides:
        1. basic configuration (omegaconf.DictConfig)
        2. output directory (pathlib.Path)
    """

    __LOGURU_FORMAT_LENGTH: int = 37
    __LOGURU_TOTAL_LENGTH: int = 120

    @classmethod
    def _reformat_message(cls, msg: str) -> str:
        return msg
        len_msg = cls.__LOGURU_TOTAL_LENGTH - cls.__LOGURU_FORMAT_LENGTH
        lst = wrap_line(msg, width_min=len_msg, width_max=len_msg + 20)
        return ("\n" + " " * cls.__LOGURU_FORMAT_LENGTH).join(lst)

    def __init__(self, *, config: Config) -> None:
        assert isinstance(config, DictConfig | Config)
        self.config: Config = config
        self.path = Path(config.outputs)
        self.path.mkdir(parents=True, exist_ok=True)

        # Logger Configuration
        loglevel = str(config.loglevel).upper()
        try:
            hydracfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
            outlogfile = str(hydracfg.job_logging.handlers.file.filename)
        except Exception:
            outlogfile = hydracfg = None
        outlogfile = config.logfile
        assert outlogfile is not None
        outlogfile = str(outlogfile)

        # Logger Setting
        self.logger = log = Logger(
            core=Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        )
        log.add(sys.stderr, level=loglevel, format=LOGURU_FORMAT)
        logname = Path(outlogfile).name
        if logname != "-":
            logfile = self.path.joinpath(logname)
            log.add(logfile, level=loglevel, format=LOGURU_FORMAT)

        # logging directory configuration
        if hydracfg is not None:
            output_dir = hydracfg.runtime.output_dir
        else:
            output_dir = config.outputs
        output_dir = Path(output_dir).absolute()
        self._log_length = int(  # the maximum length of the line in loguru
            self.__LOGURU_TOTAL_LENGTH  # 80
            - self.__LOGURU_FORMAT_LENGTH
        )
        log.info("=" * self._log_length)
        log.info("The Configuration:\n" + OmegaConf.to_yaml(config))
        log.info(f"Working floder   : {os.getcwd()}")
        log.info(f"self.path floder : {self.path}")
        log.info(f"Output floder    : {output_dir}")
        log.info(f"Output logfile   : {outlogfile}")
        log.info(f"Output loglevel  : {loglevel.upper()}")
        log.info("=" * self._log_length)

        # restart/initialize configuration
        self.network: ReactionNetwork = ReactionNetwork(
            path=self.path / "event",
            config=config.event,
            format=config.event.db_format,
            restart=config.restart,
        )

        # check parallel mode
        parallel = str(config.parallel).lower()
        if hydracfg is not None and hydracfg.mode == "MULTIRUN":
            if parallel != "serial":
                msg = (
                    "Please delete '--multirun,-m' option "
                    "when running this script. The multirun "
                    "mode is not supported because this program "
                    f"will be parallelized by '{parallel}' innerly."
                )
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
        if parallel not in [
            "serial",
            "multiprocessing",
            "ray",
            "dask",
            "executorlib",
        ]:
            msg = (
                f"Invalid parallel mode: {parallel}. Please choose one from "
                "'serial', 'multiprocessing', 'ray', 'dask', 'executorlib'."
            )
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)
        pworkers = int(self.config.parallel_workers)
        pworkers: int | None = None if pworkers <= 0 else pworkers
        self.executor = get_executor(parallel, pworkers)

        # check run_type
        if self.config.run_type not in ["otfkmc", "rxngen"]:
            msg = f"Invalid run_type: {self.config.run_type}."
            msg += " Please choose one from 'otfkmc', 'rxngen'."
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

        T = self.config.event.temperature
        self.__kBT_in_J = T * _kB_in_JperK
        gas_box_length = self.config.gas_pressure_cache_box_length
        self.__gas_box_volume: float = (gas_box_length * 1e-10) ** 3  # m^3
        gas_name_set = set(self.network.metadata.basic.gas_sticking.keys())
        gas_name_set.update(self.config.initial_gas_pressure.keys())
        self.gas_number_dct: dict[str, float] = {}
        self.__gas_lst: list[Gas] = []
        for gas_name in gas_name_set:
            s = self.network.metadata.basic.gas_sticking.get(gas_name, 1.0)
            gas, _, cost_time = helper_optimization(
                config=self.config,
                graph=Gas.from_name(
                    name_or_smiles=gas_name,
                    sticking=float(s) if s is not None else 1.0,
                    parse_bonds=self.config.bonds,  # type: ignore
                ),
                raise_when_fail=True,
                allow_hash_change=False,
                allow_not_connected=False,
                run_vibration=True,
                deep_copy=True,
            )
            if isinstance(gas, Gas):
                msg = f"Optimized gas({gas_name}) in {cost_time:.2f} seconds."
                self.logger.info(self._reformat_message(msg))
                self.__gas_lst.append(gas)
                self.network.db_gas.add(gas)
                gas_positions_uuid = self.network.db_gas.use_positions_uuid
                gas_k = gas.get_key_for_metadata(gas_positions_uuid)
                p = self.config.initial_gas_pressure.get(gas_name, 0.0)
                V, kBT = self.__gas_box_volume, self.__kBT_in_J
                self.gas_number_dct[gas_k] = p * V / kBT
            else:
                msg = f"Failed to optimize gas({gas_name})."
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)

        msg = f"Gas number dict: {self.gas_number_dct}"
        self.logger.info(self._reformat_message(msg))
        self.__gas_number_path = self.path.joinpath("gas_number.json")
        if self.config.run_type.lower() == "otfkmc":
            if self.config.restart:
                self.load_gas_number_dct()
                msg1 = f"Load gas number dict from {self.__gas_number_path}"
                msg2 = f"Gas number dict(reload): {self.gas_number_dct}"
                self.logger.info(self._reformat_message(msg1))
                self.logger.info(self._reformat_message(msg2))
            else:
                self.persist_gas_number_dct()
                msg = f"Persist gas number dict to {self.__gas_number_path}"
                self.logger.info(self._reformat_message(msg))

        # persist the network for restart. [gas list]
        self.network.persistence()

    @property
    def gas_lst(self) -> list[Gas]:
        if len(self.network.metadata.basic.gas_sticking) != 0:
            return self.__gas_lst
        else:
            return []

    @property
    def gas_pressure_dct(self) -> dict[str, float]:
        """Get the gas pressure for each gas for this simulation.

        Returns:
            dict[str, float]: the gas pressure for each gas for this simulation.
        """
        V, kBT = self.__gas_box_volume, self.__kBT_in_J
        result: dict[str, float] = {}
        for gas_key, N in self.gas_number_dct.items():
            N = float(0.0 if N <= 0 else N)
            result[gas_key] = N * kBT / V
        return result

    def persist_gas_number_dct(self) -> None:
        """Persist the gas number dct to file."""
        with open(self.__gas_number_path, "w") as f:
            json.dump(self.gas_number_dct, f, indent=4)

    def load_gas_number_dct(self) -> None:
        """Load the gas number dct from file."""
        with open(self.__gas_number_path) as f:
            dct: dict[str, float] = json.load(f)
        self.gas_number_dct.update(dct)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the class."""

    def get_batch_system_for(self) -> list[System]:
        """Get the batch of systems for the Reaction Network."""
        if self.config.run_type == "rxngen":
            if isinstance(self.config.atoms, AseReadAtomsConfig):
                msg = "Read the system list from "
                msg += f"{self.config.atoms.filename}"
                self.logger.info(self._reformat_message(msg))
                lst: Iterable[Atoms] = iread(self.config.atoms.filename)
                return [self.get_system_for(inp=atoms) for atoms in lst]
        return [self.get_system_for(inp=None)]

    def get_system_for(self, inp: Atoms | None) -> System:
        """Get the system object for exploration.

        if inp is None:
            parse system for first step from config
        else:
            convert inp to System object

        Returns:
            System: the system object for exploration.
        """
        if inp is None:
            # parse system for first step
            try:
                result: System = hydra_parse(self.config.system, System)
            except Exception:
                atoms = hydra_parse(self.config.system, Atoms)
                result: System = System.from_ase(
                    atoms=atoms,
                    parse_bonds=self.config.bonds,  # type: ignore
                    parse_atoms_is_outer_or_not=True,
                )
            if self.config.system.get("attach_is_adsorbate", True):
                result = result.model_copy(
                    update=dict(
                        is_adsorbate=np.zeros_like(
                            result.is_outer,
                            dtype=bool,
                        )
                    ),
                    deep=False,
                )
        elif isinstance(inp, Atoms):
            result: System = System.from_ase(
                atoms=inp,
                parse_bonds=self.config.bonds,  # type: ignore
                parse_atoms_is_outer_or_not=True,
            )
        else:
            msg = f"Unknown type of input: {type(inp)}"
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

        assert isinstance(result, System)
        assert result.pair is not None
        assert result.is_outer is not None
        self.logger.info(self._reformat_message(f"Read the system: {result}"))
        return result


def wrap_line(
    text: str,
    width_min: int = 80,
    width_max: int = 100,
) -> list[str]:
    """Make a line of text fit in a given width.

    Args:
        text: The text to wrap.
        width_min: The minimum width of the line.
        width_max: The maximum width of the line.

    Returns:
        The wrapped lines.
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width_max:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    if len(lines) >= 2 and len(lines[-1]) < width_min:
        merged = lines[-2] + " " + lines[-1]
        if len(merged) <= width_max:
            lines[-2] = merged
            lines.pop()

    return lines
