from pathlib import Path

import pytest
from ase.visualize import view

from graphatoms.reaction.event import Reaction
from graphatoms.system.graph import SysGraph

this_dir = Path(__file__).parent
data_dir = this_dir.parent / "tests-data-for-match"


@pytest.fixture(scope="module")
def rxn() -> Reaction:
    r = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "515f12.npz")
    p = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "1824a8.npz")
    ts = SysGraph.read_npz(data_dir / "ts" / "Pd236" / "1824a8.npz")
    return Reaction(R=r, T=ts, P=p)


def test_apply(rxn: tuple[SysGraph, SysGraph]) -> None:
    view([i.to_ase() for i in rxn])
    assert False

    p = data_dir.joinpath("minima").joinpath(k)
    p_lst = list(p.glob("*.npz"))
    if len(p_lst) < 2:
        return

    r, p = [SysGraph.read_npz(pp) for pp in p_lst[:2]]
    view([r.to_ase(), p.to_ase()])

    print(p)
    # sub = SysGraph.read_npz(p)
    # print(sys.CN)
    # print(sub.CN)
    # print(sys.get_match_mode(sub))
    # print("-" * 32)
