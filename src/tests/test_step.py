from pathlib import Path
from pprint import pprint
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.steps import FirstStep

this_dir = Path(__file__).parent


class Mock(FirstStep):
    pass


@pytest.mark.parametrize(
    "parallel",
    [
        # "serial",
        "multiprocessing",
        # "ray",
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
            cfg.restart = False
            cfg.parallel = parallel
            cfg.outputs = Path(tmp).as_posix()
            print(list(Path(tmp).rglob("*")))
            print(OmegaConf.to_yaml(cfg))

            obj = Mock(config=cfg)  # type: ignore
            for k, cluster in obj.run(None).items():
                obj.logger.info(f"{k} {cluster.hash} {cluster}")  # type: ignore
            pprint(list(Path(tmp).rglob("*")))

            print("-----------------")
            print("Test restart")
            cfg.restart = True
            obj2 = Mock(config=cfg)  # type: ignore
            for k, cluster in obj2.run(None).items():
                obj2.logger.info(f"{k} {cluster.hash} {cluster}")  # type:
            pprint(list(Path(tmp).rglob("*")))
