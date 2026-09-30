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
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.config.atoms import AseReadAtomsConfig
from graphatoms.enterpoint.network import ReactionNetwork
from graphatoms.enterpoint.parallel import get_executor
from graphatoms.system import Gas, System
from graphatoms.utils.parser import hydra_parse

from ._helper import helper_optimization


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

        self.__gas_lst: list[Gas] = []
        for gas_info in self.network.metadata.basic.gas_info_lst:
            gas, _, cost_time = helper_optimization(
                config=self.config,
                graph=Gas.from_name(
                    gas_info.name,
                    sticking=gas_info.sticking,
                    pressure=gas_info.pressure,
                    parse_bonds=self.config.bonds,  # type: ignore
                ),
                raise_when_fail=True,
                allow_hash_change=False,
                allow_not_connected=False,
                run_vibration=True,
                deep_copy=True,
            )
            if isinstance(gas, Gas):
                self.logger.info(
                    self._reformat_message(
                        f"Optimized gas({gas_info.name}) "
                        f"in {cost_time:.2f} seconds."
                    )
                )
                self.__gas_lst.append(gas)
                self.network.db_gas.add(gas)
            else:
                msg = f"Failed to optimize gas({gas_info.name})."
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
        # persist the network for restart. [gas list]
        self.network.persistence()

    @property
    def gas_lst(self) -> list[Gas]:
        if len(self.network.metadata.basic.gas_info_lst) != 0:
            return self.__gas_lst
        else:
            return []

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
