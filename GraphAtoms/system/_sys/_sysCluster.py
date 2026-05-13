from typing import Any

import numpy as np
from pydantic import model_validator
from typing_extensions import Self, override

from ...geometry import get_distance_factory
from .._base._atomsAttr import MoveFixTag
from .._base._boxMixin import BoxMixin
from .._base._engMixin import EnergeticsMixin
from ._sysAllThing import SysGraph, System


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
        dct: dict[str, Any] = {
            "move_fix_tag": movefixtag,
            "coordination": system.CN[sub_idxs],
        } | SysGraph.get_induced_subgraph(system, sub_idxs).to_dict(
            exclude_none=True,
            exclude_computed_fields=True,
            exclude=(
                set(EnergeticsMixin.__pydantic_fields__.keys())
                | set(MoveFixTag.__pydantic_fields__.keys())
                | set(BoxMixin.__pydantic_fields__.keys())
                | {"coordination"}
            ),
            numpy_ndarray_compatible=True,
            numpy_convert_to_list=False,
        )
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

        Note: The `max_moved_hop` + `env_hop` atoms will be fixed.

        Args:
            system (System): the given system.
            hop (np.ndarray): the site hop infomation.
            env_hop (int, optional): The environment hop layer
                which is fixed atoms. Defaults to 1.
            max_moved_hop (int, optional): The maximum hop atoms which
                can be moved in selected Cluster for. Defaults to 2.
        """
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
        core = np.asarray(core, dtype=int).flatten()
        kidxs = cls.get_index(core, system.numbers.shape[0])
        hop: np.ndarray = system.get_hop_distance(kidxs)
        assert hop.shape == (system.natoms,), "hop != natoms"
        dist_fac = get_distance_factory("pymatgen")
        d = dist_fac.get_distance_reduce_array(
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
