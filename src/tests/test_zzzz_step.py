from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner.otfkmc import OTFKMC as Mock

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
def test_run_step(parallel) -> None:
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

            maxtry = 300
            cfg.restart = False
            cfg.logfile = f"run-{maxtry:d}.log"
            cfg.parallel = parallel
            cfg.parallel_workers = 4
            cfg.exploration.maxtry = maxtry
            cfg.max_steps = 1000
            cfg.max_times = float("inf")
            # cfg.outputs = Path(tmp).as_posix()
            # cfg.event.min_frequency_for_ts = 10.0
            # cfg.event.min_frequency = 10.0
            # cfg.event.max_force = 0.05
            print(list(Path(tmp).rglob("*")))
            print(OmegaConf.to_yaml(cfg))

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
