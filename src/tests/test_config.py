from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from graphatoms.enterpoint.config import CONFIG_DIR, Config

this_dir = Path(__file__).parent


def test_first_step(config: str) -> None:
    run_1st_step(config=config)


def test_config() -> None:
    run_1st_step(config=None)


def run_1st_step(config: str | None = None) -> None:
    with TemporaryDirectory(dir=this_dir) as tmp:
        Path(tmp).mkdir(exist_ok=True, parents=True)
        print(f"Test in the temporary folder: '{tmp}'")
        if config is not None:  # test for first_step
            job_name, config_path, overrides = "first_step", tmp, []
            Path(tmp).joinpath("run.yaml").write_text(config)
        else:
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
            cfg.outputs = Path(tmp).as_posix()
            print(list(Path(tmp).rglob("*")))
            print(OmegaConf.to_yaml(cfg))
            # obj = FirstStep(config=cfg)  # type: ignore
            # sys = obj.atoms2system(None)
            # for k, cluster in obj.system2cluster(sys).items():
            #     obj.logger.info(f"{k} {cluster.hash} {cluster}")  # type: ignore
            # print(list(Path(tmp).rglob("*")))


@pytest.fixture(scope="session")
def config() -> str:
    return CONFIG_DIR.joinpath("example.yaml").read_text()
