from pathlib import Path

import numpy as np

from graphatoms.system import SysGraph

this_dir = Path(__file__).parent
for p in this_dir.glob("*.npz"):
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
