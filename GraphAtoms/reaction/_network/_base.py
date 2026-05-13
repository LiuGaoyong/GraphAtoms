from abc import ABC
from io import StringIO
from pathlib import Path
from typing import Literal

from igraph import Graph
from omegaconf import DictConfig, OmegaConf

from GraphAtoms.system import SysGraph

from ...utils.logger import LoggerBase
from ._asedb import AseSqliteDB


class RxNetABC(ABC):
    def __init__(
        self,
        workdir: Path,
        config: DictConfig,
        logger: LoggerBase,
        prefix: str = "event",
        restart: bool = False,
    ) -> None:
        p_db_ts = workdir / f"{prefix}-ts.db"
        p_db_gas = workdir / f"{prefix}-gas.db"
        p_db_minima = workdir / f"{prefix}-minima.db"
        p_graph = workdir / f"{prefix}.picklez"
        if restart:
            self.db_ts = AseSqliteDB(p_db_ts, append=True)
            self.db_gas = AseSqliteDB(p_db_gas, append=True)
            self.db_minima = AseSqliteDB(p_db_minima, append=True)
            self.graph = Graph.Read_Picklez(p_graph)
            assert "config" in self.graph, "The graph should have a `config`."
            assert isinstance(self.graph["config"], str), (
                "The `config` must be a string."
            )
            cfg0 = OmegaConf.load(StringIO(self.graph["config"]))
            assert cfg0 == config, "The config should be the same."
        else:
            self.db_ts = AseSqliteDB(p_db_ts, append=False)
            self.db_minima = AseSqliteDB(p_db_minima, append=False)
            self.db_gas = AseSqliteDB(p_db_gas, append=False)
            self.graph = Graph(n=0, directed=True)
            self.graph["config"] = OmegaConf.to_yaml(config)
        assert isinstance(logger, LoggerBase)
        self._logger = logger

    def write(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas"] | str,
    ) -> bool:
        """Return True if the value is new, False otherwise."""
        if type == "minima":
            return self.db_minima.add(sysgraph)
        elif type == "ts":
            return self.db_ts.add(sysgraph)
        elif type == "gas":
            return self.db_gas.add(sysgraph)
        else:
            raise ValueError(f"Unknown type: {type}")
