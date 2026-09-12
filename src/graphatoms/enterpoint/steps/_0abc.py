import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any

os.environ["LOGURU_FORMAT"] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    + " | <level>{level: ^8}</level> | "
    + "<level>{message}</level>"
)
# LOGURU_FORMAT = env(
#     "LOGURU_FORMAT",
#     str,
#     "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>"
#     "{line}</cyan> - <level>{message}</level>",
# )

import hydra
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import RxNet
from graphatoms.enterpoint.parallel import get_executor


class BaseABC:
    """The base class for all classes.

    It provides:
        1. basic configuration (omegaconf.DictConfig)
        2. output directory (pathlib.Path)
    """

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
        self.logger
        log.add(sys.stderr, level=loglevel)
        logname = Path(outlogfile).name
        if logname != "-":
            logfile = self.path.joinpath(logname)
            log.add(logfile, level=loglevel)

        # logging directory configuration
        if hydracfg is not None:
            output_dir = hydracfg.runtime.output_dir
        else:
            output_dir = config.outputs
        output_dir = Path(output_dir).absolute()
        log.info("=" * 64)
        log.info("The Configuration:\n" + OmegaConf.to_yaml(config))
        log.info(f"Working floder   : {os.getcwd()}")
        log.info(f"self.path floder : {self.path}")
        log.info(f"Output floder    : {output_dir}")
        log.info(f"Output logfile   : {outlogfile}")
        log.info(f"Output loglevel  : {loglevel.upper()}")
        log.info("=" * 64)

        # restart/initialize configuration
        self.network: RxNet = RxNet(
            path=self.path / "event",
            config=config.event,
            format=config.event.db_format,
            restart=config.restart,
        )

        # check parallel mode
        parallel = str(config.parallel).lower()
        if hydracfg is not None and hydracfg.mode == "MULTIRUN":
            if parallel != "serial":
                raise ValueError(
                    "Please delete '--multirun,-m' option "
                    "when running this script. The multirun "
                    "mode is not supported because this program "
                    f"will be parallelized by '{parallel}' innerly."
                )
        assert parallel in [
            "serial",
            "multiprocessing",
            "ray",
            "dask",
            "executorlib",
        ], (
            f"Invalid parallel mode: {parallel}. Please choose one from "
            "'serial', 'multiprocessing', 'ray', 'dask', 'executorlib'."
        )
        pworkers = int(self.config.parallel_workers)
        pworkers: int | None = None if pworkers <= 0 else pworkers
        self.executor = get_executor(parallel, pworkers)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the class."""

    class OptimizationFailed(RuntimeError):
        """Optimization failed."""

    class CheckVibrationFailed(RuntimeError):
        """Check vibrations failed."""
