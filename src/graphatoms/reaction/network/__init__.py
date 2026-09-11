from pathlib import Path
from typing import Literal

from graphatoms.system import SysGraph
from graphatoms.system.database import DatabaseABC, get_db

from ._metadata import MetaData
from ._recorder import Recorder
from ._scheduler import Scheduler

__all__ = [
    "RxNet",
    "MetaData",
    "Recorder",
    "Scheduler",
]

class RxNet:
    def __init__(
        self,
        path: Path | str,
        restart: bool = False,
        format: str | Literal["dir", "h5", "sqlite"] = "ASE",
    ) -> None:
        self.__path = path = Path(path)
        if restart:
            raise NotImplementedError("Restart is not implemented.")
            assert format in (
                "sqlite",
                "directory",
                "dir",
                "folder",
                "h5",
                "hdf5",
            ), f"The format of `{format}` is not supported for restart."
            self.scheduler = Scheduler.read_npz(path / "scheduler.npz")
            self.recorder = Recorder.read_json(path / "recorder.json")
        else:
            self.recorder: Recorder = Recorder()
            self.scheduler: Scheduler = Scheduler()
            self.metadata: MetaData = MetaData()

            # initialize the databases
        lst: list[DatabaseABC] = [
            get_db(
                path=path,
                format=format,
                prefix=prefix,
                append=restart,
            )
            for prefix in ["ts", "gas", "minima"]
        ]
        self.db_ts, self.db_gas, self.db_minima = lst

    def persistence(self) -> None:
        """Persist the data to the database."""
        self.recorder.write_json(self.__path / "recorder.json")
        self.scheduler.write_npz(self.__path / "scheduler.npz")

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
