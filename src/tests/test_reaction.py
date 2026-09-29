from pathlib import Path

import pytest
from ase import Atoms
from ase.cluster import Octahedron
from ase.io import read

from graphatoms.reaction import Adsorption, Desorption, EventBase, Reaction
from graphatoms.system import Cluster, Gas, SysGraph, System

this_dir = Path(__file__).parent

data_dir = this_dir.parent / "tests-datasets" / "for-match"
data_dir_4simplify = data_dir.parent / "for-simplify"
assert data_dir_4simplify.exists()


def _rxn() -> EventBase:
    p = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "515f12.npz")
    r = SysGraph.read_npz(data_dir / "minima" / "Pd236" / "1824a8.npz")
    ts = SysGraph.read_npz(data_dir / "ts" / "Pd236" / "1824a8.npz")
    return Reaction(R=r, T=ts, P=p)


def _adsorption() -> EventBase:
    p = SysGraph.read_npz(data_dir / "adsorption" / "Pd58CO.npz")
    r = SysGraph.read_npz(data_dir / "adsorption" / "Pd58.npz")
    gas = Gas.read_npz(data_dir / "adsorption" / "CO.npz")
    return Adsorption(R=r, G=gas, P=p)


@pytest.mark.parametrize(
    "event",
    [_rxn()] + sorted(data_dir_4simplify.glob("event-*")),
)
def test_simplify(event: EventBase | Path) -> None:
    if isinstance(event, Path):
        dct: dict[str, SysGraph | None] = {}
        for k in "RTGP":
            fname = event / f"{k}.npz"
            if not fname.exists():
                dct[k] = None
            elif k == "G":
                dct[k] = Gas.read_npz(fname)
            else:
                dct[k] = Cluster.read_npz(fname)

        if dct.get("G", None) is not None:
            try:
                event = Adsorption(**dct)  # type: ignore
            except Exception:
                event = Desorption(**dct)  # type: ignore
        else:
            event = Reaction(**dct)  # type: ignore

    assert isinstance(event, EventBase)
    event.simplify()
    print(event)
    print(event.reversed)
    event.reversed.simplify()
    event.reversed.reversed.simplify()


@pytest.mark.parametrize("n", [8, 9, 10])
@pytest.mark.parametrize("simplify", [True, False])
def test_apply(n: int, simplify: bool) -> None:
    rxn = _rxn()
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

@pytest.mark.parametrize("n", [8, 9, 10])
@pytest.mark.parametrize("simplify", [True, False])
def test_apply_adsorption(n: int, simplify: bool) -> None:
    rxn = _adsorption()  # CO Adsorption Reaction
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
    print(len(res))
    print(f"n={n}: RMSD={rmsd:.4f}")
    print(res.info.keys())
    assert len(res) == len(sys) + 2
    assert "is_adsorbate" in res.info

    # from ase.io import write
    # write(f"test_apply_{n}_{simplify}.xyz", sys.to_ase(), append=False)
    # write(f"test_apply_{n}_{simplify}.xyz", res, append=True)


def test_apply_desorption() -> None:
    r = Cluster.read_npz(data_dir / "desorption" / "Pd49O2.npz")
    p = Cluster.read_npz(data_dir / "desorption" / "Pd49.npz")
    gas = Gas.read_npz(data_dir / "desorption" / "O2.npz")
    rxn = Desorption(R=r, G=gas, P=p)  # O2 Desorption Reaction

    atoms = read(data_dir / "desorption" / "system.xyz")
    assert isinstance(atoms, Atoms), "The atoms must be ase.Atoms."
    sys = System.from_ase(atoms)
    matched = sys.get_match_mode(rxn.R)

    print()
    print("-" * 32)
    if matched is None:
        print("No match found.")
        return

    res, rmsd = rxn.apply(sys)
    print(len(res))
    print(f"RMSD={rmsd:.4f}")
    print(res.info.keys())
    assert len(res) == len(sys) - 2
    assert "is_adsorbate" in res.info

    # from ase.io import write
    # write(f"test_apply_{n}_{simplify}.xyz", sys.to_ase(), append=False)
    # write(f"test_apply_{n}_{simplify}.xyz", res, append=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
