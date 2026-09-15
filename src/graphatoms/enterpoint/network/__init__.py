import dataclasses as dc
from pathlib import Path
from re import M
from typing import Literal

from graphatoms.enterpoint.config import EventConfig
from graphatoms.reaction import EventBase, Reaction
from graphatoms.system import Cluster, SysGraph
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
        metadata_basic = MetaDataBasic(
            **(
                dc.asdict(config)  #
                if dc.is_dataclass(config)
                else config
            )
        )

        if restart:
            assert format in (
                "sqlite",
                "directory",
                "dir",
                "folder",
                "h5",
                "hdf5",
            ), f"The format of `{format}` is not supported for restart."
            assert self.__path.exists(), (
                f"The path `{self.__path}` does not exist."
                + " Please create it or use `restart=False`."
            )
            self.scheduler = Scheduler.read_npz(path / "scheduler.npz")
            self.recorder = Recorder.read_json(path / "recorder.json")
            self.metadata = MetaData.from_storage(path)
            # check equality between the basic metadata and the config
            assert self.metadata.basic == metadata_basic, (
                f"Metadata basic({self.metadata.basic}) is not"
                + f" equal to the config({metadata_basic})."
            )
        else:
            assert not self.__path.exists(), (
                f"The path `{self.__path}` does exist."
                + " Please delete it or use `restart=True`."
            )
            self.recorder: Recorder = Recorder()
            self.scheduler: Scheduler = Scheduler()
            self.metadata: MetaData = MetaData(basic=metadata_basic)

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

    def read(self, key: str) -> EventBase:
        if key in self.metadata.table.key_rxn:
            idx = self.metadata.table.key_rxn.index(key)
            kp = self.metadata.table.key_p[idx]
            kr = self.metadata.table.key_r[idx]
            kg = self.metadata.table.key_g[idx]
            kt = self.metadata.table.key_t[idx]
            if kg is not None:
                raise NotImplementedError("Ads/Des is not supported.")
            else:
                assert kt is not None
                ts = Cluster.from_ase(self.db_ts[kt])
                r = Cluster.from_ase(self.db_minima[kr])
                p = Cluster.from_ase(self.db_minima[kp])
                return Reaction(T=ts, R=r, P=p)
        else:
            raise ValueError(f"Event {key} is not in the database.")

    def write(self, event: EventBase) -> bool:
        """Write the event to the database.

        Returns:
            bool: True if the event is new, False otherwise.
        """
        if self.metadata.table.write(event):  # event is new
            self.persistence()
            try:
                self.__write(event.R, "minima")
                self.__write(event.P, "minima")
                if event.G is not None:
                    self.__write(event.G, "gas")
                if event.T is not None:
                    self.__write(event.T, "ts")
                return True  # event is new
            except Exception as e:
                print(self.metadata.table.dataframe)
                # pop metadata from the table
                for k in self.metadata.table.__pydantic_fields__:
                    v = getattr(self.metadata.table, k)
                    if isinstance(v, list):
                        v.pop(-1)
                # Save some files for debug.
                print(self.metadata.table.dataframe)
                print(f"Save {event._string()} failed !!!!!!!!!!")
                k = event._string().replace(" ", "")
                k = k.replace(">", "").replace(":", "_")
                event.R.write_npz(f"{k}-R.npz")  # type: ignore
                event.P.write_npz(f"{k}-P.npz")  # type: ignore
                if event.G is not None:
                    event.G.write_npz(f"{k}-G.npz")  # type: ignore
                if event.T is not None:
                    event.T.write_npz(f"{k}-T.npz")  # type: ignore
                print(self.metadata.table.dataframe)
                self.persistence()
                raise e
        else:
            return False  # event is already in the database

    def __write(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas"] | str,
    ) -> bool:
        """Return True if the value is new, False otherwise."""
        if type == "minima":
            return self.db_minima.add(sysgraph, check_positions=True)
        elif type == "ts":
            return self.db_ts.add(sysgraph, check_positions=True)
        elif type == "gas":
            return self.db_gas.add(sysgraph, check_positions=False)
        else:
            raise ValueError(f"Unknown type: {type}")
