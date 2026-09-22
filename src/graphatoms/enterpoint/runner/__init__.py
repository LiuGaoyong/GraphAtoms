import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra

from graphatoms.enterpoint.config import CONFIG_DIR, Config

from .otfkmc import OTFKMC
from .rxngen import ReactionNetworkGenerator


@hydra.main(
    config_path=CONFIG_DIR.as_posix(),
    config_name="run",
    version_base=None,
)
def run(cfg: Config) -> None:
    if cfg.run_type == "otfkmc":
        OTFKMC(config=cfg).run()
    elif cfg.run_type == "rxngen":
        ReactionNetworkGenerator(config=cfg).run()
    else:
        raise ValueError(f" run_type `{cfg.run_type}` is not supported.")
