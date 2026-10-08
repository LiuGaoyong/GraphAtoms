import shutil
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner import ReactionNetworkGenerator
from graphatoms.enterpoint.runner.common._base import RunnerABC

this_dir = Path(__file__).parent


# @pytest.mark.skip()
@pytest.mark.parametrize(
    "parallel",
    [
        # "serial",
        # "multiprocessing",
        "ray",
    ],
)
@pytest.mark.parametrize(
    "Mock",
    [
        # OnTheFlyKMC,
        ReactionNetworkGenerator,
    ],
)
def test_run_step(parallel: str, Mock: type[RunnerABC]) -> None:
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

            maxtry = 3
            cfg.restart = False
            cfg.logfile = f"run-{maxtry:d}.log"
            cfg.parallel = parallel
            cfg.parallel_workers = 4
            cfg.exploration.maxtry = maxtry
            cfg.run_type = "rxngen"
            cfg.max_steps = 2
            cfg.max_times = float("inf")
            cfg.event.gas_pressure = {"O2": 1.0, "CO": 1.0}
            print(list(Path(tmp).rglob("*")))
            print(OmegaConf.to_yaml(cfg))
            if Path(cfg.outputs).exists():
                shutil.rmtree(Path(cfg.outputs))

            print("-----------------")
            print(f"Test {Mock.__name__}")
            print("-----------------")
            obj = Mock(config=cfg)  # type: ignore
            obj.run()
            pprint(list(Path(tmp).rglob("*")))

            print("-----------------")
            print("Test restart")
            cfg.restart = True
            obj2 = Mock(config=cfg)  # type: ignore
            obj2.run()
            pprint(list(Path(tmp).rglob("*")))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--lf"])
