import dataclasses as dc
from pathlib import Path
from typing import Literal

from graphatoms.enterpoint.config import EventConfig
from graphatoms.system import SysGraph
from graphatoms.system.database import DatabaseABC, get_db

from ._metadata import MetaData, MetaDataBasic
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
        config: EventConfig,
        *args,
        restart: bool = False,
        format: str | Literal["dir", "h5", "sqlite"] = "ASE",
        **kwargs,
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
            self.metadata = MetaData.from_storage(path / "metadata")
            # check equality
            assert self.metadata.basic == config
        else:
            self.recorder: Recorder = Recorder()
            self.scheduler: Scheduler = Scheduler()
            self.metadata: MetaData = MetaData(
                basic=MetaDataBasic(
                    **(
                        dc.asdict(config)  #
                        if dc.is_dataclass(config)
                        else config
                    )
                ),
            )

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
        self.metadata.persistence(self.__path)

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
