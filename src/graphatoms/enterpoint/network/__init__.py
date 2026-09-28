import dataclasses as dc
import pickle
from pathlib import Path
from typing import Literal

from graphatoms.enterpoint.config import EventConfig
from graphatoms.reaction import (
    Adsorption,
    Desorption,
    EventBase,
    EventInfo,
    Reaction,
)
from graphatoms.system import Gas, SysGraph
from graphatoms.system.database import DatabaseABC, get_db

from ._metadata import MetaData
from ._metadata import _MetaDataBasic as MetaDataBasic
from ._recorder import Recorder, RecorderInfo
from ._scheduler import Scheduler

__all__ = [
    "MetaData",
    "Recorder",
    "Scheduler",
    "ReactionNetwork",
]


class ReactionNetwork:
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
        self.__path_exploration_recorder = path / "exploration.csv"
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
            assert self.__path_exploration_recorder.exists(), (
                f"The path `{self.__path_exploration_recorder}` does not exist."
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
            self.__path.mkdir(parents=True, exist_ok=True)
            self.__path_exploration_recorder.write_text(
                RecorderInfo.get_csv_title() + "\n"
            )
            self.recorder: Recorder = Recorder()
            self.scheduler: Scheduler = Scheduler()
            if metadata_basic is not None:
                self.metadata: MetaData = MetaData(basic=metadata_basic)
            else:
                self.metadata: MetaData = MetaData()
        self.__format = format

        # initialize the databases
        (
            self.db_ts,
            self.db_gas,
            self.db_minima,
            self.db_cluster,
            self.db_system,
        ) = [
            get_db(
                path=path,
                format=format,
                prefix=prefix,
                append=restart,
            )
            for prefix in ["ts", "gas", "minima", "cluster", "system"]
        ]
        assert isinstance(self.db_ts, DatabaseABC)
        assert isinstance(self.db_gas, DatabaseABC)
        assert isinstance(self.db_minima, DatabaseABC)
        assert isinstance(self.db_cluster, DatabaseABC)
        assert isinstance(self.db_system, DatabaseABC)

    def summary(self) -> str:
        """Return the summary of the network."""
        msg = f"ReactionNetwork: {self.__path} (format={self.__format})\n"
        msg += f"    #System    :{len(self.db_system)}\n"
        msg += f"    #Cluster   :{len(self.db_cluster)}\n"
        msg += f"    #Minima    :{len(self.db_minima)}\n"
        msg += f"    #Gas       :{len(self.db_gas)}\n"
        msg += f"    #TS        :{len(self.db_ts)}\n"
        msg += f"#Event     :{len(self.metadata.table)}\n"
        return msg

    def persistence(self) -> None:
        """Persist the data to the database."""
        self.recorder.write_json(self.__path / "recorder.json")
        self.scheduler.write_npz(self.__path / "scheduler.npz")
        self.metadata.persistence(self.__path)

    def read_event(self, key: str) -> tuple[EventInfo, EventBase]:
        """Read the event from the database."""
        if self.metadata.has(key):
            info = self.metadata.read(key)
            if info.key_g is not None:
                assert info.key_t is None
                g = Gas.from_ase(self.db_gas[info.key_g])
                r = SysGraph.from_ase(self.db_minima[info.key_r])
                p = SysGraph.from_ase(self.db_minima[info.key_p])
                if r.natoms < p.natoms:
                    return info, Adsorption(G=g, R=r, P=p)
                else:
                    return info, Desorption(G=g, R=r, P=p)
            else:
                assert info.key_t is not None
                ts = SysGraph.from_ase(self.db_ts[info.key_t])
                r = SysGraph.from_ase(self.db_minima[info.key_r])
                p = SysGraph.from_ase(self.db_minima[info.key_p])
                return info, Reaction(T=ts, R=r, P=p)
        else:
            raise ValueError(f"Event {key} is not in the database.")

    def found(
        self,
        event: EventBase | str,
        for_gas: str | None,
        for_cluster: str,
        *,
        for_system: str | None = None,
        persist: bool = True,
        **kwargs,
    ) -> Literal["new", "old", "fail"] | str:
        if for_system is None:
            for_system = ""
        if for_gas is None:
            for_gas = ""
            old_new = self.recorder.cluster[for_cluster]
            min_found = self.metadata.table.get_minconut_for(
                cluster_key=for_cluster,
                system_key=for_system,
                gas_key=None,
            )
        else:
            old_new = self.recorder.adsorption[f"{for_cluster}_{for_gas}"]
            min_found = self.metadata.table.get_minconut_for(
                cluster_key=for_cluster,
                system_key=for_system,
                gas_key=for_gas,
            )

        if isinstance(event, str):
            old_new.found_fail()
            result = "fail"
        elif isinstance(event, EventBase):
            simplified_threshold = self.metadata.basic.simplified_threshold
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
            if self.write_event(
                event,
                for_cluster,
                for_system,
                persist=persist,
                **kwargs,
            ):
                old_new.found_new()
                result = f"new {event}"
            else:
                old_new.found_old()
                result = f"old {event}"
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

        info = RecorderInfo.from_oldnew(
            oldnew=old_new,
            min_found=min_found,
            for_cluster=for_cluster,
            for_system=for_system,
            for_gas=for_gas,
        )
        with self.__path_exploration_recorder.open("a") as f:
            f.write(info.to_csv_line() + "\n")
        return result

    def write_event(
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
                self.write_sysgraph(event.R, "minima")
                self.write_sysgraph(event.P, "minima")
                if event.G is not None:
                    self.write_sysgraph(event.G, "gas")
                if event.T is not None:
                    self.write_sysgraph(event.T, "ts")
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

    def write_sysgraph(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> bool:
        """Return True if the value is new, False otherwise."""
        if type.lower() == "minima":
            return self.db_minima.add(sysgraph, check_positions=True)
        elif type.lower() == "ts":
            return self.db_ts.add(sysgraph, check_positions=True)
        elif type.lower() == "gas":
            return self.db_gas.add(sysgraph, check_positions=False)
        elif type.lower() == "system":
            return self.db_system.add(sysgraph, check_positions=True)
        elif type.lower() == "cluster":
            return self.db_cluster.add(sysgraph, check_positions=True)
        else:
            raise ValueError(f"Unknown type: {type}")
