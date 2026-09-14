from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase.cluster import Octahedron
from hydra import compose, initialize

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.steps import (
    HelperException,
    helper_dimer,
    helper_optimization,
)
from graphatoms.system import Cluster, SysGraph, System

this_dir = Path(__file__).parent


@pytest.fixture(scope="module")
def config() -> Config:
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
            cfg.exploration.maxtry = 100
            cfg.outputs = Path(tmp).as_posix()
    return cfg


@pytest.mark.parametrize(
    "graph",
    [Cluster.read_npz(p) for p in Path(this_dir / "nominima").glob("*.npz")]
    + [System.from_ase(Octahedron("Pd", 9), parse_bonds={"method": "raw"})],
)
@pytest.mark.skip(reason="Skip for now")
def test_opt(graph: SysGraph | Cluster | System, config: Config) -> None:
    result, label, cost = helper_optimization(
        graph=graph,
        config=config,
        allow_hash_change=False,
    )
    print(f"optimization cost: {cost:.2f}")
    for i in range(1, 3):
        floder = f"minima-{i}"
        p = this_dir / floder
        if p.joinpath(f"{label}.npz").exists():
            ref = Cluster.read_npz(p.joinpath(f"{label}.npz"))
            assert isinstance(ref, result.__class__)
            for k in ref.__pydantic_fields__:
                if "is" in k:
                    k = k.replace("is", "idx")
                print("-----------------")
                v0, v1 = getattr(ref, k), getattr(result, k)
                print(k, type(v0), type(v1))
                if v0 is None and v1 is None:
                    continue
                elif type(v0) is not type(v1):
                    continue

                if isinstance(v0, np.ndarray):
                    if "idx" in k:
                        if v0.shape != v1.shape:
                            continue

                    if not np.allclose(v0, v1):
                        print(k)
                        print(k, v0 - v1)
                    else:
                        try:
                            print(k, np.max(np.abs(v0 - v1)))
                        except Exception:
                            print(k, np.abs(v0 - v1))
                elif isinstance(v0, list):
                    assert isinstance(v1, list)
                    assert k == "hashes"
                    assert all(v == v1[i] for i, v in enumerate(v0))
                else:
                    print(k, v0 - v1)


@pytest.mark.parametrize(
    "graph",
    [Cluster.read_npz(p) for p in Path(this_dir / "minima-2").glob("*.npz")],
)
def test_dimer(graph: SysGraph | Cluster | System, config: Config) -> None:
    for _ in range(10):
        print("#" * 50)
        try:
            result, label, cost = helper_dimer(
                graph=graph,
                config=config,
                allow_fixed_bonds_change=False,
            )
            print(f"dimer cost: {cost:.2f} for {label}")
            print(result)
        except HelperException as e:
            print(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
