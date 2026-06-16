from pathlib import Path

import numpy as np
import pytest

from graphatoms.system import SysGraph

this_dir = Path(__file__).parent
data_dir = this_dir.parent / "tests-data"


@pytest.mark.parametrize("p", list(data_dir.glob("*.npz")))
def test_smarts(p: Path) -> None:
    print(p)
    sys = SysGraph.read_npz(p)
    sys = SysGraph.read_npz(
        p,
        parse_bonds={"method": "raw"},
        is_adsorbate=np.arange(len(sys)) >= 80,
    )
    print(sys)
    print(sys.smarts)
    print("-" * 32)
