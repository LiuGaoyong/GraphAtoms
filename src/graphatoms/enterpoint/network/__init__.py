import dataclasses as dc
import pickle
from pathlib import Path
from typing import Literal

from graphatoms.enterpoint.config import EventConfig
from graphatoms.reaction import EventBase, EventInfo, Reaction
from graphatoms.system import Cluster, SysGraph
from graphatoms.system.database import DatabaseABC, get_db

from ._metadata import MetaData
from ._metadata import _MetaDataBasic as MetaDataBasic
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
        *args,
        restart: bool = False,
        config: EventConfig | None = None,
        format: str | Literal["dir", "h5", "sqlite"] = "dir",
        **kwargs,
    ) -> None:
        self.__path = path = Path(path)
        if config is not None:
            metadata_basic = MetaDataBasic(
                **(
                    dc.asdict(config)  # type: ignore
                    if dc.is_dataclass(config)
                    else config
                )
            )
        else:
            metadata_basic = None

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
            if metadata_basic is not None:
                assert self.metadata.basic == metadata_basic, (
                    f"Metadata basic({self.metadata.basic}) is not"
                    + f" equal to the config({metadata_basic})."
                )
            else:
                pass
                # raise ValueError("The config is not provided.")
        else:
            assert not self.__path.exists(), (
                f"The path `{self.__path}` does exist."
                + " Please delete it or use `restart=True`."
            )
            self.recorder: Recorder = Recorder()
            self.scheduler: Scheduler = Scheduler()
            if metadata_basic is not None:
                self.metadata: MetaData = MetaData(basic=metadata_basic)
            else:
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

    def summary(self) -> str:
        """Return the summary of the network."""
        return "fasdasdgase"

    def persistence(self) -> None:
        """Persist the data to the database."""
        self.recorder.write_json(self.__path / "recorder.json")
        self.scheduler.write_npz(self.__path / "scheduler.npz")
        self.metadata.persistence(self.__path)

    def read(self, key: str) -> tuple[EventInfo, EventBase]:
        if self.metadata.has(key):
            info = self.metadata.read(key)
            if info.key_g is not None:
                raise NotImplementedError("Ads/Des is not supported.")
            else:
                assert info.key_t is not None
                ts = Cluster.from_ase(self.db_ts[info.key_t])
                r = Cluster.from_ase(self.db_minima[info.key_r])
                p = Cluster.from_ase(self.db_minima[info.key_p])
                return info, Reaction(T=ts, R=r, P=p)
        else:
            raise ValueError(f"Event {key} is not in the database.")

    def found(
        self,
        event: EventBase | str,
        *,
        for_cluster: str | None = None,
        for_system: str | None = None,
        persist: bool = True,
        **kwargs,
    ) -> Literal["new", "old", "fail"] | str:
        if isinstance(event, str):
            return "fail"
        elif isinstance(event, EventBase):
            simplified_threshold = self.metadata.basic.simplified_threshold
            if for_system is None:
                for_system = ""
            if for_cluster is None:
                for_cluster = event.R.get_key_for_metadata()
            if simplified_threshold > 0:
                try:
                    event = event.simplify(simplified_threshold)
                except Exception as e:
                    msg = f"Event {event._string()} failed "
                    msg += f"to simplify. because of {e}"
                    fname = self.__path / "event-simplify-fail.pkl"
                    fname.write_bytes(pickle.dumps(event))
                    fname.with_suffix(".err").write_text(msg)
                    raise ValueError(msg)
            is_new = self._write(
                event,
                for_cluster,
                for_system,
                persist=persist,
                **kwargs,
            )
            return f"{'new' if is_new else 'old'} {event}"
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

    def _write(
        self,
        event: EventBase,
        for_cluster: str,
        for_system: str,
        *,
        persist: bool = True,
        **kwargs,
    ) -> bool:
        """Write the event to the database."""
        if self.metadata.has(event):
            self.metadata.rxn_count_add_one(event)
            return False
        else:
            self.metadata.write(event, for_cluster, for_system)
            try:
                self.__write(event.R, "minima")
                self.__write(event.P, "minima")
                if event.G is not None:
                    self.__write(event.G, "gas")
                if event.T is not None:
                    self.__write(event.T, "ts")
                if persist:
                    self.persistence()
                return True
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
