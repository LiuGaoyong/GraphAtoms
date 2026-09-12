from pathlib import Path

from hydra import compose, initialize

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.steps import Surf
from graphatoms.reaction import Reaction
from graphatoms.system import Cluster

this_dir = Path(__file__).parent

if __name__ == "__main__":
    overrides = ["calculator=emt", "atoms=octahedron"]
    with initialize(
        config_path=Path(CONFIG_DIR)
        .relative_to(
            this_dir,
            walk_up=True,
        )
        .as_posix(),
        job_name="run",
        version_base=None,
    ):
        cfg: Config = compose(  # type: ignore
            config_name="run",
            overrides=overrides,
        )
        cfg.restart = False
        cfg.parallel = "serial"
        cfg.exploration.maxtry = 100

    lst: list[Cluster] = [
        Cluster.read_npz(p) for p in Path(this_dir / "minima").glob("*.npz")
    ]
    lst[0]

    rxn, cot = Surf.helper_dimer(cfg, cluster=lst[0])
    print(rxn)
    if isinstance(rxn, Reaction):
        for k in rxn.__pydantic_fields__:
            v: Cluster | None = getattr(rxn, k)
            if v is not None:
                v.write_npz(this_dir / f"{k}.npz")
    print(cot)
