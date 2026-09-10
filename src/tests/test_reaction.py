from pathlib import Path

import pytest
from ase.cluster import Octahedron

from graphatoms.reaction import EventBase, Reaction
from graphatoms.system import SysGraph, System

this_dir = Path(__file__).parent
data_dir = this_dir.parent / "tests-data-for-match"


@pytest.fixture(scope="module")
def rxn() -> EventBase:
    p = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "515f12.npz")
    r = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "1824a8.npz")
    ts = SysGraph.read_npz(data_dir / "ts" / "Pd236" / "1824a8.npz")
    return Reaction(R=r, T=ts, P=p)

    # def test_simplify(rxn: Event) -> None:
    #     rxn.simplify()


@pytest.mark.parametrize("n", [8, 9, 10])
@pytest.mark.parametrize("simplify", [True, False])
def test_apply(rxn: EventBase, n: int, simplify: bool) -> None:
    sys = System.from_ase(Octahedron("Pd", n))
    if simplify:
        rxn = rxn.simplify()
    matched = sys.get_match_mode(rxn.R)

    print()
    print("-" * 32)
    print(n, simplify)
    if matched is None and not simplify:
        print(f"n={n}: No match found.")
        return

    res, rmsd = rxn.apply(sys)
    print(f"n={n}: RMSD={rmsd:.4f}")

    # from ase.io import write
    # write(f"test_apply_{n}_{simplify}.xyz", sys.to_ase(), append=False)
    # write(f"test_apply_{n}_{simplify}.xyz", res, append=True)
