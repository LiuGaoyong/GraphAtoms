from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner.common import helper_dimer
from graphatoms.system import Cluster

this_dir = Path(__file__).parent
pattern = f"{Path(__file__).stem}-?.npz"


@pytest.mark.parametrize("fname", list(this_dir.rglob(pattern)))
def test_debug_dimer_des(fname: Path) -> None:
    with TemporaryDirectory(dir=this_dir) as tmp:
        Path(tmp).mkdir(exist_ok=True, parents=True)
        print(f"Test in the temporary folder: '{tmp}'")
        job_name, config_path = "config", CONFIG_DIR
        overrides = ["calculator=emt", "atoms=octahedron"]

        with initialize(
            config_path=Path(config_path)
            .relative_to(
                this_dir,
                walk_up=True,
            )
            .as_posix(),
            job_name=job_name,
            version_base=None,
        ):
            cfg: Config = compose(  # type: ignore
                config_name="run",
                overrides=overrides,
            )

        for _ in range(100):
            rxn, _, _, _ = helper_dimer(
                graph=Cluster.read_npz(fname),
                config=cfg,
            )
            print(rxn)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--lf"])
