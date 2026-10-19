import shutil
from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner import (  # noqa: F401
    OnTheFlyKMC,
    ReactionNetworkGenerator,
)
from graphatoms.enterpoint.runner.common._base import RunnerABC
from graphatoms.enterpoint.runner.v2.general.otfkmc import (
    OTFKMC as OnTheFlyKMC_v2,  # noqa: F401
)
from graphatoms.enterpoint.runner.v2.general.rxngen import (
    ReactionNetworkGenerator as ReactionNetworkGenerator_v2,  # noqa: F401
)

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
        # ReactionNetworkGenerator_v2,
        OnTheFlyKMC_v2,
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

            maxtry = 1000
            cfg.restart = False
            cfg.logfile = f"run-{maxtry:d}.log"
            cfg.parallel = parallel
            if parallel == "ray":
                cfg.parallel_ncpus_for_tasks = 2
            cfg.exploration.maxtry = maxtry
            cfg.run_type = "rxngen"
            cfg.max_steps = 1
            cfg.max_times = float("inf")
            cfg.event.simplified_threshold = 0.5
            cfg.exploration.surface_only_explore_single_core = True
            cfg.exploration.max_ncore_for_surface = 1
            # cfg.initial_gas_pressure = {"O2": 1e4, "CO": 1e3}
            # cfg.event.gas_sticking = {"O2": 1.0, "CO": 1.0}
            print(list(Path(tmp).rglob("*")))
            print(OmegaConf.to_yaml(cfg))
            if Path(cfg.outputs).exists():
                shutil.rmtree(Path(cfg.outputs))

            print("-----------------")
            print(f"Test {Mock.__name__}")
            print("-----------------")
            obj = Mock(config=cfg)  # type: ignore
            obj.run()  # type: ignore
            pprint(list(Path(tmp).rglob("*")))

            print("-----------------")
            print("Test restart")
            cfg.restart = True
            obj2 = Mock(config=cfg)  # type: ignore
            obj2.run()  # type: ignore
            pprint(list(Path(tmp).rglob("*")))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--lf"])
