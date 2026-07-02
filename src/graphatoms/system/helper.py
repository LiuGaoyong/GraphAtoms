from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms

from graphatoms.geometry._inner_outer import check_atom_is_inner
from graphatoms.utils.bytestool import hash_string

from ._sysGas import System
from .graph._bonds import DEFAULT_WH_HASH_SIZE
from .graph._sysGraph import SysGraph


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


def analysis(
    atoms: Atoms,
    bonds_cfg: Mapping[str, Any] = {},
    clear_info: bool | Sequence[str] = True,
) -> System:
    """A helper function for analysis of `ase.Atoms`."""
    if clear_info is False:
        pass
    else:
        atoms = atoms.copy()
        if clear_info is True:
            clear_info = list(atoms.info.keys())
            atoms.info.clear()
        atoms.info = {
            k: atoms.info[k]
            for k in atoms.info  #
            if k not in clear_info
        }

    sys = System.from_ase(
        atoms,
        parse_bonds=bonds_cfg.get("parser", {"method": "raw"}),
        parse_bonds_distance=bonds_cfg.get("parse_distance", False),
        parse_bonds_order=bonds_cfg.get("parse_order", False),
    )
    is_inner = [
        check_atom_is_inner(
            i,
            numbers=sys.numbers,
            geometry=sys.positions,
            adjacency_array=sys.MATRIX[i : i + 1, :],
            cell=atoms.cell,
        )
        for i in range(len(atoms))
    ]
    is_outer = np.logical_not(is_inner)
    result = sys.from_dict(sys.to_dict(), is_outer=is_outer)
    result.hash  # call hash calculation to set hash & hashes
    return result


def analysis_site(
    sys: System,
    active: np.ndarray | None = None,
    max_ncore: int = 3,
) -> np.ndarray:
    """A helper function for analysis of `graphatoms.system.System`."""
    sys.hash  # call hash calculation to set hash & hashes
    assert sys.pair is not None, "Please run analysis first on the System."
    assert sys.hashes is not None, "Please run analysis first on the System."
    assert sys.is_outer is not None, "Please run analysis first on the System."
    if active is None:
        active = sys.is_outer
    else:
        active0 = sys.is_outer
        assert active.shape == active0.shape
        active = active & active0
    arange = np.arange(len(sys))

    result_1 = np.asarray(
        [
            np.isin(arange, [i])
            for i in arange  #
            if active[i]
        ]
        + [
            np.isin(arange, [i, j])
            for i, j in sys.pair  #
            if active[i] and active[j]
        ]
    )
    result_2 = sys.get_chordless_cycles(
        batch_nbr_order=0,
        max_ncore=max_ncore,  # type: ignore
        batch=active,
    )
    result = np.vstack([result_1, result_2])
    hashes = [
        hash_string(
            "-".join([sys.hashes[i] for i in range(len(sys)) if site[i]]),
            digest_size=DEFAULT_WH_HASH_SIZE,
        )
        for site in result
    ]
    _, idx = np.unique(hashes, return_index=True)
    return result[idx]
