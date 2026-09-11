from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase import Atoms
from ase.collections import g2

from graphatoms.system import SysGraph
from graphatoms.system.database import get_db


@pytest.mark.parametrize("fmt", ["dict", "sqlite", "h5", "dir"])
def test_systemdb(fmt: str) -> None:
    # Create a temporary file
    with TemporaryDirectory() as tmp:
        if fmt == "sqlite":
            temp_file_path = Path(tmp) / "test.db"
        elif fmt == "h5":
            temp_file_path = Path(tmp) / "test.h5"
        elif fmt == "dir":
            temp_file_path = Path(tmp) / "test"
        else:
            temp_file_path = None

        db = get_db(path=temp_file_path, format=fmt)
        print(type(db), tmp)
        for k in g2.names[:5]:
            v = SysGraph.from_ase(
                g2[k],
                energy=2.5,
                fmax=0.05,
                frequencies=np.array([1.0, 2.0, 3.0]) + 50.0,
            )
            print(v)
            print(v.hash, k)
            db.add(v)

        print(db.keys())
        for k in db:
            atoms: Atoms = db[k]
            g = SysGraph.from_ase(atoms, parse_bonds=None)
            print(atoms.info)
            print(g, g.hash)
            print(k, k in db)
            print("-----------------")

        print(type(db), tmp)
