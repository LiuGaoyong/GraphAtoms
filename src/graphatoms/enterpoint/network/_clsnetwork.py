import pickle
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Literal, Self, override

import numpy as np
from ase import Atoms

from graphatoms.enterpoint.config import Config, EventConfig
from graphatoms.enterpoint.network._metadata import MetaData
from graphatoms.enterpoint.network._metadata import (
    _MetaDataBasic as MetaDataBasic,
)
from graphatoms.enterpoint.network._recorder import (
    OldNewRecorder4Confidence,
    Recorder,
    RecorderInfo,
)
from graphatoms.enterpoint.network._scheduler import Scheduler
from graphatoms.reaction import (
    Adsorption,
    Desorption,
    EventBase,
    EventInfo,
    Reaction,
)
from graphatoms.system import Cluster, Gas, SysGraph, System
from graphatoms.system.database import DatabaseABC, get_db
from graphatoms.utils.bytestool import atomic_write, atomic_write_text

__all__ = [
    "MetaData",
    "Recorder",
    "Scheduler",
    "ReactionNetwork",
]
NETWORK_PATH_PREFIX = "event"


class ReactionNetwork(Mapping[str, tuple[EventInfo, EventBase]]):
    DB_USE_POSITION_UUID_DICT: dict[str, bool] = {
        "ts": True,
        "minima": True,
        "cluster": False,
        "system": False,
        "gas": False,
    }

    def __init__(
        self,
        path: Path | str,
        *args,
        restart: bool = False,
        config: EventConfig | None = None,
        format: str | Literal["dir", "h5", "sqlite"] = "dir",
        remote_mode: bool = False,
        **kwargs,
    ) -> None:
        self.__path = path = Path(path)
        self.__remote_mode = remote_mode
        self.__path_exploration_recorder = path / "exploration.csv"
        if restart:
            assert format in (
                "sqlite",
                "directory",
                "folder",
                "dir",
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
            self.__scheduler = Scheduler.read_npz(path / "scheduler.npz")
            self.__recorder = Recorder.read_json(path / "recorder.json")
            self.__metadata = MetaData.from_storage(path)
            # check equality between the basic metadata and the config
            if config is not None:
                self.__metadata.basic.check_consistency(config)
        else:
            assert not self.__path.exists(), (
                f"The path `{self.__path}` does exist."
                + " Please delete it or use `restart=True`."
            )
            self.__path.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.__path_exploration_recorder,
                RecorderInfo.get_csv_title() + "\n",
            )
            self.__recorder = Recorder()
            self.__scheduler = Scheduler()
            if config is not None:
                self.__metadata = MetaData(
                    basic=MetaDataBasic.from_config(config)
                )
            else:
                self.__metadata = MetaData()
        self.__format = format

        # initialize the databases
        (
            self.__db_ts,
            self.__db_minima,
            self.__db_cluster,
            self.__db_system,
            self.__db_gas,
        ) = [
            get_db(
                path=path,
                format=format,
                prefix=prefix,
                append=restart,
                use_positions_uuid=use_pos_uuid,
            )
            for prefix, use_pos_uuid in self.DB_USE_POSITION_UUID_DICT.items()
        ]
        assert isinstance(self.__db_ts, DatabaseABC)
        assert isinstance(self.__db_gas, DatabaseABC)
        assert isinstance(self.__db_minima, DatabaseABC)
        assert isinstance(self.__db_cluster, DatabaseABC)
        assert isinstance(self.__db_system, DatabaseABC)

    @classmethod
    def from_config(
        cls,
        config: Config,
        *args,
        restart: bool | None = None,
        remote_retry_times: float = 2.0,
        remote_mode: bool = False,
        **kwargs,
    ) -> Self:  # type: ignore
        """Create a network from a config file."""
        if remote_mode is True:
            # When remote_mode is True, some files may not exist or in write.
            # We need to try to create the network again.
            err = None
            for _ in range(10):
                try:
                    return cls(
                        path=Path(config.outputs) / NETWORK_PATH_PREFIX,
                        restart=restart
                        if restart is not None
                        else config.restart,
                        format=config.event.db_format,
                        remote_mode=remote_mode,
                        config=config.event,
                        **kwargs,
                    )
                    break
                except Exception as e:
                    time.sleep(abs(remote_retry_times))
                    err = e
            if err is not None:
                raise RuntimeError(
                    f"Failed to create the network in remote mode. {err}"
                )
        else:
            return cls(
                path=Path(config.outputs) / NETWORK_PATH_PREFIX,
                restart=restart if restart is not None else config.restart,
                format=config.event.db_format,
                remote_mode=remote_mode,
                config=config.event,
                **kwargs,
            )

    @classmethod
    def merge(
        cls,
        *networks: Self,
        new_path: Path | str,
        new_format: str | Literal["dir", "h5", "sqlite"] = "dir",
        **kwargs,
    ) -> Self:
        """Merge the networks.

        Note: the recorder will be set empty.
        """
        if len(networks) <= 1:
            raise ValueError("At least two network is required.")
        result = cls(
            path=Path(new_path),
            restart=False,
            config=networks[0].metadata.basic,
            format=new_format,
        )
        result.metadata.basic.db_format = new_format
        for network in networks[1:]:
            try:
                network.metadata.basic.check_consistency(
                    result.metadata.basic,
                    exclude_keys={"db_format", "gas_sticking"},
                )
                for k, v in network.metadata.basic.gas_sticking.items():
                    if k not in result.metadata.basic.gas_sticking:
                        result.metadata.basic.gas_sticking[k] = v
                    else:
                        v0 = result.metadata.basic.gas_sticking[k]
                        assert v0 == v, (
                            f"Failed to merge the networks because the "
                            f"gas sticking of {k} is not the same as "
                            f"input. self={v0}, input={v}"
                        )
                    result.metadata.basic.gas_sticking[k] = v
            except Exception as e:
                raise ValueError(
                    f"Failed to merge the networks because "
                    f"of check consistency failed: {e}"
                )
            for k, use_pos_uuid in cls.DB_USE_POSITION_UUID_DICT.items():
                db_res: DatabaseABC = getattr(result, f"db_{k}")
                db: DatabaseABC = getattr(network, f"db_{k}")
                for _, v in db.items():
                    graph = SysGraph.from_ase(v)
                    graph_key = graph.get_key_for_metadata(use_pos_uuid)
                    if graph_key not in db_res:
                        db_res._save(graph_key, graph)
            for i, rxn in enumerate(network.metadata.table.key_rxn):
                if rxn not in result.metadata.table.key_rxn:
                    key_r = network.metadata.table.key_r[i]
                    key_p = network.metadata.table.key_p[i]
                    key_g = network.metadata.table.key_g[i]
                    key_ts = network.metadata.table.key_t[i]
                    if key_r not in result.db_minima:
                        print(
                            f"Find invalid key: {rxn}\n"
                            f"  Missing key of reactant: {key_r}\n",
                            flush=True,
                        )
                        continue
                    elif key_g is not None and key_g not in result.db_gas:
                        print(
                            f"Find invalid key: {rxn}\n"
                            f"  Missing key of gas: {key_g}\n",
                            flush=True,
                        )
                        continue
                    elif key_ts is not None and key_ts not in result.db_ts:
                        print(
                            f"Find invalid key: {rxn}\n"
                            f"  Missing key of transition state: {key_ts}\n",
                            flush=True,
                        )
                        continue
                    elif key_p is not None and key_p not in result.db_minima:
                        print(
                            f"Find invalid key: {rxn}\n"
                            f"  Missing key of product: {key_p}\n",
                            flush=True,
                        )
                        continue
                result.metadata.table.merge(  # Output duplicate info
                    network.metadata.table, idxs=[i]
                )
        n = len(result.metadata.table.key_rxn)
        print(f"Merge networks({n} events) done.", flush=True)
        result.scheduler.clear()
        result.recorder.clear()
        result.persistence()
        return result

    @classmethod
    def reformat_message(cls, msg: str) -> str:
        return msg
        __LOGURU_FORMAT_LENGTH: int = 37
        __LOGURU_TOTAL_LENGTH: int = 120
        len_msg = __LOGURU_TOTAL_LENGTH - __LOGURU_FORMAT_LENGTH
        lst = wrap_line(msg, width_min=len_msg, width_max=len_msg + 20)
        return ("\n" + " " * __LOGURU_FORMAT_LENGTH).join(lst)

    @override
    def __len__(self) -> int:
        return len(self.metadata.table.key_rxn)

    @override
    def __contains__(self, key: str) -> bool:  # type: ignore
        return key in self.metadata.table.key_rxn

    @override
    def __getitem__(self, key: str) -> tuple[EventInfo, EventBase]:
        return self.read_event(key)

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.metadata.table.key_rxn

    @property
    def path(self) -> Path:
        return self.__path

    @property
    def metadata(self) -> MetaData:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `metadata` property is not supported in remote mode."
            )
        return self.__metadata

    @property
    def scheduler(self) -> Scheduler:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `scheduler` property is not supported in remote mode."
            )
        return self.__scheduler

    @property
    def recorder(self) -> Recorder:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `recorder` property is not supported in remote mode."
            )
        return self.__recorder

    @property
    def db_system(self) -> DatabaseABC:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `db_system` property is not supported in remote mode."
            )
        return self.__db_system

    @property
    def db_cluster(self) -> DatabaseABC:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `db_cluster` property is not supported in remote mode."
            )
        return self.__db_cluster

    @property
    def db_minima(self) -> DatabaseABC:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `db_minima` property is not supported in remote mode."
            )
        return self.__db_minima

    @property
    def db_gas(self) -> DatabaseABC:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `db_gas` property is not supported in remote mode."
            )
        return self.__db_gas

    @property
    def db_ts(self) -> DatabaseABC:
        if self.__remote_mode is True:
            raise RuntimeError(
                "the `db_ts` property is not supported in remote mode."
            )
        return self.__db_ts

    def exploration_can_be_finished(
        self,
        confidence: float,
        *args,
        cluster_key: str,
        gas_key: str | None = None,
        **kwargs,
    ) -> bool:
        """Check whether the exploration (dimer) can be finished."""
        if gas_key is not None:
            graph_label = f"{cluster_key}_{gas_key}"
            return graph_label in self.__recorder.adsorption
        else:
            if cluster_key not in self.__recorder.cluster:
                self.__recorder.cluster[cluster_key] = (
                    OldNewRecorder4Confidence()
                )
            oldnew = self.__recorder.cluster[cluster_key]
            return oldnew.exploration_can_be_finished(
                confidence=confidence,
                min_found=self.__metadata.table.get_minconut_for(
                    cluster_key=cluster_key,
                    gas_key=gas_key,
                    **kwargs,
                ),
            )

    def exploration_can_be_skipped(
        self,
        thetacutoff: float,
        diffpositions: np.ndarray,
        *args,
        cluster_key: str | None,
        **kwargs,
    ) -> tuple[bool, float]:
        """Check whether the exploration (dimer) can be skipped."""
        if cluster_key is not None:
            return self.__scheduler.can_be_skip(
                cluster_key,
                diffpositions=diffpositions,
                thetacutoff=thetacutoff,
            )
        else:
            raise NotImplementedError(
                "`cluster_key=None` is not supported now."
            )

    def summary(self) -> str:
        """Return the summary of the network."""
        msg = f"ReactionNetwork: {self.__path} (format={self.__format})\n"
        msg += f"    #System    :{len(self.__db_system)}\n"
        msg += f"    #Cluster   :{len(self.__db_cluster)}\n"
        msg += f"    #Minima    :{len(self.__db_minima)}\n"
        msg += f"    #Gas       :{len(self.__db_gas)}\n"
        msg += f"    #TS        :{len(self.__db_ts)}\n"
        msg += f"#Event         :{len(self.__metadata.table)}\n"
        msg += f"Configuration  :{self.__metadata.basic.to_str(indent=4)}\n"
        return msg

    def persistence(self) -> None:
        """Persist the reaction network to local storage."""
        if self.__remote_mode is True:
            raise RuntimeError("this function is not supported in remote mode.")
        self.__recorder.write_json(self.__path / "recorder.json")
        self.__scheduler.write_npz(self.__path / "scheduler.npz")
        self.__metadata.persistence(self.__path)

    def get_use_position_uuid(self, k: str | System | Gas | Cluster) -> bool:
        """Return the `use_positions_uuid` of the database."""
        if not isinstance(k, str):
            k = k.__class__.__name__
        k = str(k).lower()
        if k in self.DB_USE_POSITION_UUID_DICT:
            return self.DB_USE_POSITION_UUID_DICT[k]
        else:
            raise ValueError(f"{k} is not in the database.")

    def object_has_been_optimized(self, graph: SysGraph) -> SysGraph | None:
        """Check if the object has been optimized.

        If the object has been optimized, return the optimized object.
        Otherwise, return None.
        """
        v = None
        for cls, db in [
            (Gas, self.__db_gas),
            (Cluster, self.__db_cluster),
            (System, self.__db_system),
        ]:
            key = graph.get_key_for_metadata(db.use_positions_uuid)
            if isinstance(graph, cls) and graph in db:
                v = cls.from_ase(db[key])
                break
        return v

    def read_event_info(self, key: str) -> EventInfo:
        """Read the event infomation from the database metadata.

        Raises:
            ValueError: If the event is not in the database.
        """
        if self.__metadata.has(key):
            return self.__metadata.read(key)
        else:
            raise ValueError(f"Event {key} is not in the database.")

    def read_event(self, key: str) -> tuple[EventInfo, EventBase]:
        """Read the event from the database.

        Args:
            key: The key of the event.
        Returns:
            The event information(EventInfo) and the event(EventBase) object.
            The event object is of type Adsorption, Desorption, or Reaction.
        Raises:
            ValueError: If the event is not in the database.
        """
        info = self.read_event_info(key)
        if info.key_g is not None:
            assert info.key_t is None
            g = Gas.from_ase(self.__db_gas[info.key_g])
            r = SysGraph.from_ase(self.__db_minima[info.key_r])
            p = SysGraph.from_ase(self.__db_minima[info.key_p])
            if r.natoms < p.natoms:
                return info, Adsorption(G=g, R=r, P=p)
            else:
                return info, Desorption(G=g, R=r, P=p)
        else:
            assert info.key_t is not None
            ts = SysGraph.from_ase(self.__db_ts[info.key_t])
            r = SysGraph.from_ase(self.__db_minima[info.key_r])
            p = SysGraph.from_ase(self.__db_minima[info.key_p])
            return info, Reaction(T=ts, R=r, P=p)

    def found(
        self,
        event: EventBase | str,
        for_gas: str | None,
        for_cluster: str,
        *,
        for_system: str | None = None,
        persist: bool = True,
        env_hop: int = 1,  # TODO: make this into whole configuration
        **kwargs,
    ) -> Literal["new", "old", "fail"] | str:
        """Record the event as found.

        If the event is a string, record it as a fail.
        If the event is an EventBase, record it as a new/old event.
        This function will write the event to the database if new event
        is found. Otherwise, it will update the event information.
        """
        if self.__remote_mode is True:
            raise RuntimeError("this function is not supported in remote mode.")
        if for_system is None:
            for_system = ""
        if for_gas is None:
            for_gas = ""
        old_new = self.__recorder.cluster[for_cluster]
        min_found = self.__metadata.table.get_minconut_for(
            cluster_key=for_cluster,
            system_key=for_system,
            gas_key=None,
        )
        if isinstance(event, str):
            old_new.found_fail()
            result = "fail"
        elif isinstance(event, EventBase):
            simplified_threshold = self.__metadata.basic.simplified_threshold
            if simplified_threshold > 0:
                try:
                    event = event.simplify(
                        env_radius=simplified_threshold,
                        env_hop=env_hop,
                    )
                except Exception as e:
                    msg = f"Event {event._string()} failed "
                    msg += f"to simplify. because of {e}"
                    fname = self.__path / "event-simplify-fail.pkl"
                    atomic_write_text(fname.with_suffix(".err"), msg)
                    atomic_write(
                        fname,
                        writer=lambda p: p.write_bytes(pickle.dumps(event)),
                    )
                    event.R.write_npz(fname.with_suffix(".R.npz"))
                    event.P.write_npz(fname.with_suffix(".P.npz"))
                    if event.G is not None:
                        event.G.write_npz(fname.with_suffix(".G.npz"))
                    if event.T is not None:
                        event.T.write_npz(fname.with_suffix(".T.npz"))
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
        if self.__remote_mode is True:
            raise RuntimeError("this function is not supported in remote mode.")
        if isinstance(event, Desorption):
            event = event.reversed
        assert isinstance(event, Adsorption | Reaction), (
            f"Unknown event type: {type(event)}. Only "  #
            "Adsorption and Reaction are supported, "  #
            "and the Desorption will be reversed."
        )
        if event.G is not None:
            assert event.R.natoms < event.P.natoms, (
                f"Event {event._string()} is not Adsorption. "
                "The R must have fewer atoms than P."
            )

        if self.__metadata.has(event):
            self.__metadata.rxn_count_add_one(event)
            return False
        else:
            self.__metadata.write(event, for_cluster, for_system)
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
                # ===========================================
                # the following code is for debug
                # ===========================================
                print(self.__metadata.table.dataframe)
                # pop metadata from the table
                for k in self.__metadata.table.__pydantic_fields__:
                    v = getattr(self.__metadata.table, k)
                    if isinstance(v, list):
                        v.pop(-1)
                # Save some files for debug.
                print(self.__metadata.table.dataframe)
                print(f"Save {event._string()} failed !!!!!!!!!!")
                k = event._string().replace(" ", "")
                k = k.replace(">", "").replace(":", "_")
                event.R.write_npz(f"{k}-R.npz")  # type: ignore
                event.P.write_npz(f"{k}-P.npz")  # type: ignore
                if event.G is not None:
                    event.G.write_npz(f"{k}-G.npz")  # type: ignore
                if event.T is not None:
                    event.T.write_npz(f"{k}-T.npz")  # type: ignore
                print(self.__metadata.table.dataframe)
                self.persistence()
                raise e

    def __get_database(self, type: str) -> DatabaseABC:
        if type.lower() == "minima":
            return self.__db_minima
        elif type.lower() == "ts":
            return self.__db_ts
        elif type.lower() == "gas":
            return self.__db_gas
        elif type.lower() == "system":
            return self.__db_system
        elif type.lower() == "cluster":
            return self.__db_cluster
        else:
            raise ValueError(f"Unknown type: {type}")

    def contain_sysgraph(
        self,
        key: str | SysGraph,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> bool:
        """Return True if the sysgraph is in the database, False otherwise.

        Raises:
            ValueError: If the type is unknown (i.e. not
            "minima", "ts", "gas", "cluster", "system").
        """
        db = self.__get_database(type)
        return db.__contains__(key)

    def read_atoms(
        self,
        key: str,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> Atoms:
        db = self.__get_database(type)
        return db.__getitem__(key)

    def read_sysgraph(
        self,
        key: str,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
        return_sysgraph_for_minima_or_ts: bool = True,
    ) -> SysGraph:
        """Read the sysgraph from the database.

        If the type is "gas", the result will be a Gas object.
        If the type is "system", the result will be a System object.
        If the type is "cluster", the result will be a Cluster object.
        If the type is "minima" or "ts", the result will be a SysGraph object
            if `return_sysgraph_for_minima_or_ts` is True,
            otherwise it will be a Cluster object.

        Raises:
            ValueError: If the sysgraph is not in the database.
        """
        if not self.contain_sysgraph(key, type):
            raise ValueError(f"Sysgraph {key} is not in the network.")
        if type.lower() == "minima":
            if return_sysgraph_for_minima_or_ts:
                return SysGraph.from_ase(self.__db_minima[key])
            else:
                return Cluster.from_ase(self.__db_minima[key])
        elif type.lower() == "ts":
            if return_sysgraph_for_minima_or_ts:
                return SysGraph.from_ase(self.__db_ts[key])
            else:
                return Cluster.from_ase(self.__db_ts[key])
        elif type.lower() == "gas":
            return Gas.from_ase(self.__db_gas[key])
        elif type.lower() == "system":
            return System.from_ase(self.__db_system[key])
        elif type.lower() == "cluster":
            return Cluster.from_ase(self.__db_cluster[key])
        else:
            raise ValueError(f"Unknown type: {type}")

    def write_sysgraph(
        self,
        sysgraph: SysGraph,
        type: Literal["minima", "ts", "gas", "cluster", "system"] | str,
    ) -> bool:
        """Return True if the value is new, False otherwise."""
        if self.__remote_mode is True:
            raise RuntimeError("this function is not supported in remote mode.")
        if type.lower() == "minima":
            return self.__db_minima.add(sysgraph, check_positions=True)
        elif type.lower() == "ts":
            return self.__db_ts.add(sysgraph, check_positions=True)
        elif type.lower() == "gas":
            return self.__db_gas.add(sysgraph, check_positions=False)
        elif type.lower() == "system":
            return self.__db_system.add(sysgraph, check_positions=False)
        elif type.lower() == "cluster":
            return self.__db_cluster.add(sysgraph, check_positions=False)
        else:
            raise ValueError(f"Unknown type: {type}")


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


if __name__ == "__main__":
    import json
    import shutil
    from pathlib import Path

    this_dir = Path(__file__).parent
    root_path = this_dir.parent.parent.parent.parent
    print(root_path)

    new_path = root_path / "zzzz-merged" / "event"
    shutil.rmtree(new_path.parent, ignore_errors=True)

    network_lst: list[ReactionNetwork] = []
    for zzzz_path in root_path.glob("zzzz-*"):
        if "merge" not in zzzz_path.name:
            p = zzzz_path / "event"
            (p / "recorder.json").unlink(missing_ok=True)
            atomic_write_text(
                p / "recorder.json",
                json.dumps({}, indent=4),
            )
            network_lst.append(
                ReactionNetwork(
                    zzzz_path / "event",
                    restart=True,
                )
            )
    new_network = ReactionNetwork.merge(*network_lst, new_path=new_path)
