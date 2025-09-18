# ruff: noqa: D100, D101, D102, D103, D104

import numpy as np
import pytest
from rdkit import Chem

from GraphAtoms.utils.rdtool import (
    _get_adjacency,
    _get_structure,
    get_smiles,
    smiles2rdmol,
)


def test_get_adjacency() -> None:
    print("CO2", _get_adjacency(smiles2rdmol("O=C=O")))
    print("H2O", _get_adjacency(smiles2rdmol("O")))
    print("C6H6", _get_adjacency(smiles2rdmol("c1ccccc1")))


@pytest.mark.parametrize(
    "bonds",
    [
        [(0, 2), (0, 1)],
        [(0, 2), (1, 0)],
        [(0, 2), (1, 0), (0, 1)],
    ],
)
def test_get_smiles(bonds):  # noqa: D103
    numbers = np.array([8, 1, 1])
    source = np.array([i[0] for i in bonds])
    target = np.array([i[1] for i in bonds])
    order = np.ones(len(bonds)) * 1.0
    result = get_smiles(
        numbers=numbers, source=source, target=target, order=order
    )
    print(result)

    assert result == Chem.CanonSmiles("[H]-O-[H]")


def test_smiles_benzene():  # noqa: D103
    numbers = np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1])
    source = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    target = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11])
    order = np.ones_like(source) * 1.5001
    smi = get_smiles(
        numbers=numbers,
        order=order,
        source=source,
        target=target,
    )
    assert smi == Chem.CanonSmiles("c1ccccc1")
    rdmol = smiles2rdmol(smi)
    print(_get_structure(rdmol)[0])
