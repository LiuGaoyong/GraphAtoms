import os
import subprocess
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.optimize import LBFGS


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    atoms = Octahedron("Cu", 10)
    atoms.positions -= atoms.get_center_of_mass()
    atoms.numbers[atoms.positions[:, 2] > 0] = 47
    atoms.calc = EMT()
    opt = LBFGS(atoms)
    opt.run()
    return atoms


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent.parent.parent  # the GraphAtoms dir
    p = p.parent.parent / ".cache.cli.test"
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*")
    return p


# @pytest.mark.skip("Run once enough.")
@pytest.mark.parametrize(
    "cli",
    [
        "mmc nvt --max-steps=100",  # one particle, no cutoff
        "mmc nvt --max-steps=100 --lambda-for-force-bias=0.5",
        "mmc nvt --max-steps=100 --swap-interval=2",
    ],
)
def test_cli_mmc(  # noqa: D103
    cli: str,
    atoms: Atoms,
    result_dir: Path,
) -> None:  # noqa: D103
    for p in result_dir.glob("*"):
        print(p)
        p.unlink()
    atoms.write(result_dir.joinpath("structure.xyz"))
    os.chdir(result_dir)
    subprocess.run(["GraphAtoms", *cli.split(), f"--workdir={result_dir}"])
    print(result_dir.joinpath("record.table").read_text())


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
