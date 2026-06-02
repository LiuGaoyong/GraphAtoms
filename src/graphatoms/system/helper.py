from collections.abc import Mapping
from typing import Any

from ase import Atoms

from ._graph import SysGraph


def get_hash_of_atoms(
    atoms: Atoms,
    parse_bonds: Mapping[str, Any] | None = {"method": "raw"},
    **kwargs,
) -> str:
    """A helper function for hash calculation for `ase.Atoms`."""
    return SysGraph.from_ase(
        atoms=atoms,
        parse_bonds=parse_bonds,
        **kwargs,
    ).hash
