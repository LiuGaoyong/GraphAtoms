from typing import Literal, Self, override

import numpy as np
from pydantic import model_validator

from graphatoms.geometry import distance_pairs
from graphatoms.system.atoms import Energetics
from graphatoms.system.graph import AtomTag, GasMixin, SysGraph
from graphatoms.system.system import System
from graphatoms.utils.subgraph import subgraph


class Cluster(SysGraph):
    """Cluster extracted from a system with fixed/moved atom tags."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        msg = "The key of `{:s}` should be not None for Cluster."
        assert self.move_fix_tag is not None, msg.format("move_fix_tag")
        assert self.coordination is not None, msg.format("coordination")
        assert not self.is_gas, "The `is_gas` should be False for System."
        object.__setattr__(self, "is_outer", None)
        return self

    @override
    def _string(self) -> str:
        return SysGraph._string(self)

    @classmethod
    def from_select(
        cls,
        system: System,
        core: np.ndarray,
        method: str | Literal["hop", "distance"] = "distance",
        max_moved_threshold: float | int = 8.0,
        env_threshold: float | int = 15.0,
    ) -> Self:
        """Get Cluster by select method.

        Args:
            system (System): the given system.
            core (np.ndarray): the core atoms' index of site.
            method (str | Literal["hop", "distance"], optional):
                The select method. Defaults to "distance".
            max_moved_threshold (float | int, optional): The maximum distance
                atoms which can be moved in selected Cluster. Defaults to 8.0.
            env_threshold (float | int, optional): The environment distance
                which is fixed atoms. Defaults to 15.0.

        Note: (for move_fix_tag)
            if negative, then it is fixed atoms.
            if zero, then it is core atoms.
            if positive, then it is moved atoms.
            The absolute value of it is the distance or hop to the core atoms.
        """
        if method == "hop":
            return cls.__select_by_hop(
                system,
                core,
                env_hop=int(env_threshold) - int(max_moved_threshold),
                max_moved_hop=int(max_moved_threshold),
            )
        elif method == "distance":
            return cls.__select_by_distance(
                system,
                core,
                env_distance=float(env_threshold),
                max_moved_distance=float(max_moved_threshold),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    @classmethod
    def __select_by_hop(
        cls,
        system: System,
        core: np.ndarray,
        env_hop: int = 1,
        max_moved_hop: int = 2,
    ) -> Self:
        """Get Cluster by hop infomation.

        Note: The `max_moved_hop` + `env_hop` atoms will be fixed.
        For `movefixtag`:

        Args:
            sys (System): the given system.
            hop (np.ndarray): the site hop infomation.
            env_hop (int, optional): The environment hop layer
                which is fixed atoms. Defaults to 1.
            max_moved_hop (int, optional): The maximum hop atoms which
                can be moved in selected Cluster for. Defaults to 2.
        """
        kidxs = cls.get_index(core, system.numbers.shape[0])
        hop: np.ndarray = system.get_hop_distance(kidxs)
        assert len(hop) == system.natoms, "hop != natoms"
        assert np.all(hop[kidxs] == 0), "hop[kidxs] != 0"
        idxs = np.where(hop <= int(max_moved_hop + env_hop))[0]
        hop = hop[idxs].astype(int)
        tag = np.where(hop > max_moved_hop, -hop, hop)
        return cls.__select(system, sub_idxs=idxs, movefixtag=tag)

    @classmethod
    def __select_by_distance(
        cls,
        system: System,
        core: np.ndarray,
        env_distance: float = 12.0,
        max_moved_distance: float = 8.0,
    ) -> Self:
        """Get Cluster by hop infomation.

        Note: The `max_moved_distance` must less than `env_distance`.
            And all of them have to be positive float number.

        Args:
            system (System): the given system.
            core (np.ndarray): the core atoms' index of site.
            env_distance (float, optional): The environment layer by distance
                which is fixed atoms. Defaults to 12.0.
            max_moved_distance (float, optional): The maximum distance of atoms
                which can be moved in selected Cluster for. Defaults to 8.0.
        """
        kidxs = cls.get_index(core, system.numbers.shape[0])
        i, j, d = distance_pairs(
            "ijd",
            p1=system.positions[kidxs],
            p2=system.positions,
            cell=system.ase_cell,
            cutoff=float(env_distance),
        )
        d[d < 1e-5] = np.inf
        d2 = np.full([len(kidxs), len(system.positions)], np.inf)
        d2[i, j] = d
        d1 = np.min(d2, axis=0)
        d1[kidxs] = 0
        assert d1.shape == (system.natoms,), "len(d1) != natoms"
        assert np.allclose(d1[kidxs], 0), "d1[kidxs] != 0"
        idxs = np.unique(np.where(d1 <= float(env_distance)))
        hop = np.round(d1[idxs]).astype(int)
        tag = np.where(d1[idxs] > max_moved_distance, -hop, hop)
        return cls.__select(system, sub_idxs=idxs, movefixtag=tag)

    @classmethod
    def __select(
        cls,
        sys: System,
        sub_idxs: np.ndarray,
        movefixtag: np.ndarray,
    ) -> Self:
        """Select a Cluster object from a System object.

        The `sub_idxs` is induced subgraph index of select cluster.
        """
        idxs = cls.get_index(sub_idxs, sys.natoms)
        dct: dict[str, np.ndarray] = sys.to_dict(
            exclude_none=True,
            exclude=(
                set(Energetics.__pydantic_fields__.keys())
                | set(GasMixin.__pydantic_fields__.keys())
                | set(AtomTag.__pydantic_fields__.keys())
                | {"coordination", "hashes"}
            ),
        ) | {"coordination": sys.CN}
        dct["pair"], _, pair_mask = subgraph(  # type: ignore
            subset=idxs,  # type: ignore
            edge_index=dct["pair"],  # type: ignore
            edge_attr=None,
            num_nodes=sys.natoms,
            relabel_nodes=True,
            return_edge_mask=True,
        )
        for k, v in dct.items():
            if v is not None and k != "pair":
                assert isinstance(v, np.ndarray)
                if len(v) == sys.natoms:
                    dct[k] = v[idxs]
                elif len(v) == sys.nbonds:
                    dct[k] = v[pair_mask]
        return super().from_dict(dct | dict(move_fix_tag=movefixtag))
