import os
import sys
from abc import abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ["LOGURU_FORMAT"] = LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    + " | <level>{level: ^8}</level> | "
    + "<level>{message}</level>"
)  # the length of the line is 37 characters in loguru
# LOGURU_FORMAT = env(
#     "LOGURU_FORMAT",
#     str,
#     "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>"
#     "{line}</cyan> - <level>{message}</level>",
# )

import hydra
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from loguru._logger import Core, Logger
from omegaconf import DictConfig, OmegaConf

from graphatoms.enterpoint.config import Config
from graphatoms.enterpoint.network import RxNet
from graphatoms.enterpoint.parallel import get_executor
from graphatoms.system import (  # type: ignore  # type: ignore
    Cluster,
    Gas,
    SysGraph,
    System,
)
from graphatoms.utils import asetools
from graphatoms.utils.adsorption import Helper
from graphatoms.utils.parser import hydra_parse

from ._helper import helper_adsorption, helper_dimer, helper_optimization


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

    def get_system_for(self, inp: Atoms | None) -> System:
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


class ExplorationABC(RunnerABC):
    def _first_step(
        self,
        system: System | Atoms | None,
    ) -> tuple[dict[tuple[bool, int, str], Cluster], str]:
        """Return the dictionary of clusters & system key.

        Keys:
            (is_surface, ncore, hash)
        Values:
            Cluster: the cluster of the core
        """
        if not isinstance(system, System):
            system = self.get_system_for(system)
        if not isinstance(system, System):
            msg = f"Unknown type of input: {type(system)}"
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

        oesc = bool(self.config.exploration.surface_only_explore_single_core)
        if oesc and len(self.gas_lst) == 0:
            max_ncore = 1
        else:
            max_ncore = int(self.config.exploration.max_ncore_for_surface)

        values: list[Cluster] = []
        keys: list[tuple[bool, int, str]] = []
        for core in system.get_site_core(max_ncore=max_ncore):  # type: ignore
            idx_core = np.unique(np.where(core)).astype(int)
            values.append(
                Cluster.from_select(
                    system,
                    idx_core,
                    env_threshold=self.config.site.env_threshold,
                    max_moved_threshold=self.config.site.max_moved_threshold,
                    method=self.config.site.method,
                )
            )
            keys.append((True, len(idx_core), values[-1].hash))
        if bool(self.config.exploration.allow_explore_bulk):
            if system.is_outer is None:
                msg = "System.is_outer is None."
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            if system.is_fix is None:
                is_moved = np.ones_like(system.is_outer, dtype=bool)
            else:
                is_moved = np.logical_not(system.is_fix)
            is_inner = np.logical_not(system.is_outer)
            mask = np.logical_and(is_inner, is_moved)
            for idx in np.unique(np.where(mask)).astype(int):
                values.append(
                    Cluster.from_select(
                        system,
                        np.array([idx]),
                        env_threshold=self.config.site.env_threshold,
                        max_moved_threshold=self.config.site.max_moved_threshold,
                        method=self.config.site.method,
                    )
                )
                keys.append((False, 1, values[-1].hash))

        # remove duplicate cluster
        _, idxs = np.unique([i.hash for i in values], return_index=True)
        self.logger.info(
            self._reformat_message(
                f"Find {len(values)} cluster for system="
                f"{system.hash}. And {len(idxs)} unique cluster."
            )
        )

        # optimize the cluster in parallel mode
        result: dict[tuple[bool, int, str], Cluster] = (  # type: ignore
            self.__batch_optimization_parallel(
                container={keys[int(id)]: values[int(id)] for id in idxs},
                raise_on_failed=False,
                is_minima=True,
            )
        )

        # persist the network for restart. [minima list]
        self.network.persistence()
        return result, system.get_key_for_metadata()

    def __batch_optimization_parallel(
        self,
        container: list[SysGraph] | dict[Any, SysGraph],
        raise_on_failed: bool = True,
        is_minima: bool = True,
    ) -> list[SysGraph] | dict[Any, SysGraph]:
        """Optimize the container in parallel mode."""
        if isinstance(container, list):
            dct = self.__batch_optimization_parallel(
                container={
                    i: cluster  # type: ignore
                    for i, cluster in enumerate(container)
                },
                is_minima=is_minima,
                raise_on_failed=raise_on_failed,
            )
            if not isinstance(dct, dict):
                msg = f"Unknown type of output: {type(dct)}"
                self.logger.error(self._reformat_message(msg))
                raise ValueError(msg)
            return [dct[i] for i in sorted(dct.keys())]

        elif isinstance(container, dict):
            start, msg = perf_counter(), "cluster" if is_minima else "gas"
            msg = f"Start to optimize the {len(container)} {msg}."
            self.logger.info(self._reformat_message(msg))
            result: dict[Any, SysGraph] = {}
            futures: list = []

            # -------------------------------------------------
            # Submit the sysgraph optimization to the executor
            # -------------------------------------------------
            for label, sysgraph in container.items():
                key: str = sysgraph.get_key_for_metadata()
                if is_minima and sysgraph in self.network.db_minima:
                    atoms: Atoms = self.network.db_minima[key]
                    result[label] = v = Cluster.from_ase(atoms)
                    msg = f"Read '{key}' from the DB for {v}."
                    self.logger.info(self._reformat_message(msg))
                elif not is_minima and sysgraph in self.network.db_gas:
                    atoms: Atoms = self.network.db_gas[key]
                    result[label] = v = Gas.from_ase(atoms)
                    msg = f"Read '{key}' from the DB for {v}."
                    self.logger.info(self._reformat_message(msg))
                else:
                    futures.append(
                        self.executor.submit(
                            helper_optimization,
                            config=self.config,
                            graph_label=label,
                            graph=sysgraph,
                            allow_hash_change=False,
                            raise_when_fail=False,
                        )
                    )
            msg = f"Submit the optimization {len(futures)} jobs"
            msg += f" by {perf_counter() - start:.2f} seconds."
            self.logger.info(self._reformat_message(msg))
            # -------------------------------------------------
            # Wait for the sysgraph optimization to finish
            # -------------------------------------------------
            while len(futures) > 0:
                future_result, futures = self.executor.wait(futures)  # type: ignore
                sysgraph_or_msg, label, cost_time = future_result
                if isinstance(sysgraph_or_msg, Gas | Cluster):
                    msg: str = f"Optimization (success): {sysgraph_or_msg}."
                    if isinstance(sysgraph_or_msg, Cluster):
                        self.network.db_minima.add(sysgraph_or_msg)
                    else:
                        self.network.db_gas.add(sysgraph_or_msg)
                    result[label] = sysgraph_or_msg
                elif isinstance(sysgraph_or_msg, str):
                    msg = str(sysgraph_or_msg)
                    if raise_on_failed:
                        self.logger.error(self._reformat_message(msg))
                        raise RuntimeError(msg)
                else:
                    msg = f"Unknown type: {type(sysgraph_or_msg)}"
                    self.logger.error(self._reformat_message(msg))
                    raise ValueError(msg)
                msg = f"CostTime={cost_time:.2f} for {msg}"
                self.logger.info(self._reformat_message(msg))
            self.logger.info(
                self._reformat_message(
                    "All optimization jobs are done in "
                    f"{perf_counter() - start:.2f} seconds."
                )
            )
            return result
        else:
            msg = f"Unknown type of container: {type(container)}"
            self.logger.error(self._reformat_message(msg))
            raise ValueError(msg)

    def explore(self, system: System | Atoms | None = None) -> None:
        # 1 step: analyze the system
        dct, system_key = self._first_step(system)

        # 2 step: exploration
        ncore_4_adspt = int(self.config.exploration.max_ncore_for_surface)
        if bool(self.config.exploration.surface_only_explore_single_core):
            ncore_4_surface = 1
        else:
            ncore_4_surface = ncore_4_adspt
        # 2.1 group the clusters ... ...
        lst_4bulk: list[Cluster] = []
        lst_4surface: list[Cluster] = []
        lst_4adsorption: list[Cluster] = []
        for (is_surface, ncore, _), v in dct.items():  # type: ignore
            if is_surface:
                if ncore <= ncore_4_surface:
                    lst_4surface.append(v)
                elif ncore <= ncore_4_adspt:
                    lst_4adsorption.append(v)
            else:
                lst_4bulk.append(v)
        # 2.2 exploration for each cluster ... ...
        if len(lst_4bulk) > 0:
            self._second_step_bulk(
                lst_4bulk,
                system_key=system_key,
            )
        if len(lst_4adsorption) > 0:
            self._second_step_adsorption(
                lst_4adsorption,
                system_key=system_key,
            )
        if len(lst_4surface) > 0:
            for cluster in lst_4surface:
                self._second_step_surface(
                    cluster=cluster,
                    system_key=system_key,
                )
        self.network.recorder.system.add(system_key)
        self.network.persistence()

    def _second_step_surface(
        self,
        cluster: Cluster,
        system_key: str = "",
    ) -> None:
        # -----------------------------------------------------------
        # check if the cluster has been explored
        # -----------------------------------------------------------
        cluster_key = cluster.get_key_for_metadata()
        confidence = self.config.exploration.maxconfidence
        oldnew = self.network.recorder.cluster[cluster_key]
        if oldnew.exploration_can_be_finished(
            confidence=confidence,
            min_found=self.network.metadata.table.get_minconut_for(
                cluster_key=cluster_key,
            ),
        ):
            msg = f"the cluster {cluster_key} has been explored."
            self.logger.info(self._reformat_message(msg))
            return

        # -----------------------------------------------------------
        # check if the cluster is at a minimum & prepare something
        # -----------------------------------------------------------
        if not cluster.check_minima(
            fmax=float(self.config.event.max_force),
            fqmin=float(self.config.event.min_frequency),
        ):
            msg = f"Cluster {cluster_key} is not at a minimum."
            self.logger.error(self._reformat_message(msg))
            raise AssertionError(msg)
        calc: Calculator = hydra_parse(
            self.config.calculator,  # type: ignore
            Calculator,
        )
        start: float = perf_counter()
        futures: list = []

        # -----------------------------------------------------------
        # submit dimer tasks to executor
        # -----------------------------------------------------------
        for _ in range(int(self.config.exploration.maxtry)):
            thetacutoff = float(self.config.exploration.thetacutoff)
            if thetacutoff < 0:
                futures.append(
                    self.executor.submit(
                        helper_dimer,
                        config=self.config,
                        graph_label=cluster_key,
                        allow_fixed_bonds_change=False,
                        graph=cluster.model_copy(deep=True),
                        raise_when_fail=False,
                        displacement=None,
                    )
                )
            else:
                disp = asetools.call_dimer_displace(
                    atoms=cluster.to_ase().copy(),
                    calc=calc,
                    mask=None,
                    parse_mask_from_atoms=True,
                    start=start,
                )
                can_be_skip, cosine = self.network.scheduler.can_be_skip(
                    cluster_key,
                    diffpositions=disp,
                    thetacutoff=float(self.config.exploration.thetacutoff),
                )
                if can_be_skip:
                    oldnew.found_skip()
                    msg = "Skip to submit dimer task for "
                else:
                    msg = "Submit dimer task for "
                    futures.append(
                        self.executor.submit(
                            helper_dimer,
                            config=self.config,
                            graph_label=cluster_key,
                            allow_fixed_bonds_change=False,
                            graph=cluster.model_copy(deep=True),
                            raise_when_fail=False,
                            displacement=disp,
                        )
                    )
                msg = f"{msg}{cluster_key}, cosine={cosine:.2f}"
                self.logger.info(self._reformat_message(msg))
        self.logger.info(
            self._reformat_message(
                f"Submit {len(futures)} dimer tasks by "
                f"{perf_counter() - start:.2f} seconds"
            )
        )

        # -----------------------------------------------------------
        # wait for the dimer tasks to finish
        # -----------------------------------------------------------
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, _, cost_time, _ = future_result
            label = self.network.found(
                event,
                for_cluster=None,
                for_system=system_key,
                persist=True,
            )
            if label.startswith("fail"):
                msg = f"Dimer search {label} by {cost_time:.2f}"
                msg += f" seconds because of {event}"
                self.logger.info(self._reformat_message(msg))
                oldnew.found_fail()
            else:
                msg = "Dimer search successfully, and got "
                msg += f"{label} by {cost_time:.2f} seconds."
                if str(event) not in label:
                    msg += f" Simplify original {event} by threshold "
                    msg += f"{self.config.event.simplified_threshold:.2f}"
                self.logger.info(self._reformat_message(msg))
                if "new" in label:
                    oldnew.found_new()
                else:
                    oldnew.found_old()
            if oldnew.exploration_can_be_finished(
                confidence=confidence,
                min_found=self.network.metadata.table.get_minconut_for(
                    cluster_key=cluster_key,
                ),
            ):
                msg = f"Finish exploration for {cluster_key} with "
                msg += f"confidence {confidence:.2f}. {len(futures)}"
                msg += " dimer tasks left. They will be canceled."
                self.logger.info(self._reformat_message(msg))
                for future in futures:
                    self.executor.cancel(future)
                break

        self.network.persistence()

    def _second_step_adsorption(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> None:
        """The second step for the on-the-fly KMC simulation."""
        start: float = perf_counter()
        futures: list = []
        for cluster in lst:
            for gas in self.gas_lst:
                cluster_key = cluster.get_key_for_metadata(True)
                gas_key = gas.get_key_for_metadata(False)

                # ---------------------------------------------
                # check if the adsorption has been explored
                # ---------------------------------------------
                label = f"{cluster_key}_{gas_key}"
                if label in self.network.recorder.adsorption:
                    msg = f"The adsorption for {cluster_key} with "
                    msg += f"{gas_key} has been explored."
                    self.logger.info(self._reformat_message(msg))
                    continue
                # ---------------------------------------------
                # build helper for adsorption
                # ---------------------------------------------
                assert cluster.ncore > 0, "graph must have core"
                adsorption_helper = Helper(
                    atoms=cluster.to_ase(
                        exclude_bond_attibutes=True,
                        exclude_energetics=True,
                    ).copy(),
                    adsorbate=gas,
                    core=np.unique(cluster.idx_core),
                    nfibonacci=int(self.config.exploration.nfibonacci),
                    use_direct=True,
                    use_raw=True,
                )
                # ---------------------------------------------
                # submit adsorption tasks to executor
                # ---------------------------------------------
                self.network.recorder.adsorption.add(label)
                for irun in range(adsorption_helper.nrun):
                    futures.append(
                        self.executor.submit(
                            helper_adsorption,
                            config=self.config,
                            graph=cluster,
                            gas=gas,
                            irun=irun,
                            graph_label=None,
                            allow_hash_change=True,
                            raise_when_fail=False,
                            deep_copy=True,
                        )
                    )
        self.logger.info(
            self._reformat_message(
                f"Submit {len(futures)} adsorption tasks by "
                f"{perf_counter() - start:.2f} seconds"
            )
        )

        # -----------------------------------------------------------
        # wait for the adsorption tasks to finish
        # -----------------------------------------------------------
        while len(futures) > 0:
            future_result, futures = self.executor.wait(futures)  # type: ignore
            event, label, cost_time, _ = future_result
            label = self.network.found(
                event,
                for_cluster=None,
                for_system=system_key,
                persist=True,
            )
            if label.startswith("fail"):
                msg = f"Adsorption search {label} by {cost_time:.2f}"
                msg += f" seconds because of {event}"
                self.logger.info(self._reformat_message(msg))
            else:
                msg = "Adsorption search successfully, and got "
                msg += f"{label} by {cost_time:.2f} seconds."
                if str(event) not in label:
                    msg += f" Simplify original {event} by threshold "
                    msg += f"{self.config.event.simplified_threshold:.2f}"
                self.logger.info(self._reformat_message(msg))
        self.network.persistence()

    def _second_step_bulk(
        self,
        lst: list[Cluster],
        system_key: str = "",
    ) -> None:
        """The second step for the on-the-fly KMC simulation."""
        raise NotImplementedError


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
