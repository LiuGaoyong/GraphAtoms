import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, Self

import hydra
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import RxNet


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
            path=self.path,
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
        self.pmode: Literal[
            "serial",
            "multiprocessing",
            "ray",
            "dask",
            "executorlib",
        ] = parallel  # type: ignore
        pworkers: int = int(self.config.parallel_workers)
        self.pworkers: int | None = None if pworkers <= 0 else pworkers

    def __enter__(self) -> Self:
        if self.pmode == "ray":
            import ray

            self._ray_module = ray
            ray.init(ignore_reinit_error=True)

        elif self.pmode == "dask":
            import dask.distributed as dds

            self._dask_client = dds.Client()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.pmode == "ray":
            self._ray_module.shutdown()
        elif self.pmode == "dask":
            self._dask_client.close()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the class."""

    class OptimizationFailed(RuntimeError):
        """Optimization failed."""

    class CheckVibrationFailed(RuntimeError):
        """Check vibrations failed."""
