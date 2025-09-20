import numpy as np
from pydantic import model_validator
from typing_extensions import Self, override

from GraphAtoms.containner._aOther import OTHER_KEY
from GraphAtoms.containner._atomic import ATOM_KEY
from GraphAtoms.containner._graph import Graph
from GraphAtoms.containner._system import System
from GraphAtoms.utils.geometry import distance_factory


class Cluster(Graph):
    """The select cluster from the whole system."""

    @model_validator(mode="after")
    def __some_keys_should_xxx(self) -> Self:
        msg = "The key of `{:s}` should be not None for Cluster."
        for k in (ATOM_KEY.MOVE_FIX_TAG, ATOM_KEY.COORDINATION):
            assert getattr(self, k) is not None, msg.format(k)
        return self

    @override
    def _string(self) -> str:
        result = f"{super()._string()}"
        result += f",NCORE={np.sum(self.ncore)}"
        result += f",NFIX={np.sum(self.nfix)}"
        return result

    @classmethod
    def __select(
        cls,
        system: System,
        sub_idxs: np.ndarray,
        movefixtag: np.ndarray,
    ) -> Self:
        """Select a Cluster object from a System object.

        The `sub_idxs` is induced subgraph index of select cluster.

        For `movefixtag`:
            0  -->  fix atoms
            1  -->  core atoms
            2  -->  first layer of moved atoms
            x  -->  x-th layer of moved atoms
        """
        dct = system.get_induced_subgraph(sub_idxs).model_dump(
            mode="python",
            exclude_none=True,
            exclude=set(OTHER_KEY._DICT.values()),
        ) | {
            ATOM_KEY.MOVE_FIX_TAG: movefixtag,
            ATOM_KEY.COORDINATION: system.CN_MATRIX[sub_idxs],
        }
        return cls.model_validate(dct)

    @classmethod
    def select_by_hop(
        cls,
        system: System,
        hop: np.ndarray,
        env_hop: int = 1,
        max_moved_hop: int = 2,
    ) -> Self:
        """Get Cluster by hop infomation.

        Args:
            system (System): the given system.
            hop (np.ndarray): the site hop infomation.
            env_hop (int, optional): The environment hop layer
                which is fixed atoms. Defaults to 1.
            max_moved_hop (int, optional): The maximum hop atoms which
                can be moved in selected Cluster for. Defaults to 2.
        """
        assert not system.is_sub, "Only can be run for total system."
        hop = np.asarray(hop, dtype=float).flatten()
        assert len(hop) == system.natoms, "hop != natoms"
        idxs = np.where(hop <= int(max_moved_hop + env_hop))[0]
        tag = np.where(hop > max_moved_hop, -hop, hop)[idxs]
        return cls.__select(system=system, sub_idxs=idxs, movefixtag=tag)

    @classmethod
    def select_by_distance(
        cls,
        system: System,
        core: np.ndarray,
        env_distance: float = 12.0,
        max_moved_distance: float = 8.0,
    ) -> Self:
        """Get Cluster by hop infomation.

        Args:
            system (System): the given system.
            core (np.ndarray): the core atoms' index of site.
            env_distance (float, optional): The environment layer by distance
                which is fixed atoms. Defaults to 12.0.
            max_moved_distance (float, optional): The maximum distance of atoms
                which can be moved in selected Cluster for. Defaults to 8.0.

        Note: The `max_moved_distance` must less than `env_distance`.
            And all of them have to be positive float number.
        """
        assert not system.is_sub, "Only can be run for total system."
        core = np.asarray(core, dtype=int).flatten()
        kidxs = cls.get_index(core, system.Z.shape[0])
        hop = np.asarray(system.IGRAPH.distances(kidxs)).min(axis=0)
        assert hop.shape == (system.natoms,), "hop != natoms"
        d = distance_factory.get_distance_reduce_array(
            p1=system.positions[kidxs],
            p2=system.positions,
            cell=system.ase_cell,
            max_distance=float(env_distance),
            reduce_axis=0,
        )
        d[core] = 0
        assert d.shape == (system.natoms,), "d != natoms"
        idxs = np.where(d <= float(env_distance))[0]
        tag = np.where(d > float(max_moved_distance), -hop, hop)[idxs]
        return cls.__select(system=system, sub_idxs=idxs, movefixtag=tag)
