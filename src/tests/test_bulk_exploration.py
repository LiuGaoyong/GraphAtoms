from pathlib import Path

import numpy as np
import pytest
from ase.cluster.octahedron import Octahedron
from hydra import compose, initialize

import graphatoms.enterpoint.runner.v2.general._expl as v2expl
from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner.v2._helperv2 import HelperNebResult
from graphatoms.enterpoint.runner.v2.general.otfkmc import OTFKMC
from graphatoms.system import Cluster, System

this_dir = Path(__file__).parent


class EagerExecutor:
    """Run the submitted function in-process (no worker pool)."""

    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def wait(self, futures):
        return futures.pop(0), futures


def fake_helper_neb(config, cluster, intermediate_atoms, final_atoms, **kwargs):
    """Stand-in for the NEB helper: always report a failed search."""
    return HelperNebResult(
        result=None,
        message=f"fake NEB failure for {kwargs.get('candidate_key')}",
        candidate_key=str(kwargs.get("candidate_key", "")),
        for_cluster=cluster.get_key_for_metadata(False),
        for_system=str(kwargs.get("for_system", "")),
        is_failed=True,
    )


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("bulk-exploration")
    with initialize(
        config_path=Path(CONFIG_DIR)
        .relative_to(this_dir, walk_up=True)
        .as_posix(),
        job_name="test-bulk-exploration",
        version_base=None,
    ):
        cfg: Config = compose(  # type: ignore
            config_name="run",
            overrides=["calculator=emt", "atoms=octahedron"],
        )
    cfg.outputs = tmp.as_posix()
    cfg.restart = False
    cfg.parallel = "serial"
    return cfg


@pytest.fixture(scope="module")
def cluster() -> Cluster:
    system = System.from_ase(
        Octahedron("Cu", 6),
        parse_atoms_is_outer_or_not=True,
    )
    index = int(np.argmin(np.linalg.norm(system.positions, axis=1)))
    return Cluster.from_select(
        system,
        np.asarray([index]),
        method="distance",
    )


def test_bulk_exploration_flow(
    config: Config,
    cluster: Cluster,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = OTFKMC(config=config)
    candidate_runner = runner._ExplorationBase__second_step_bulk
    monkeypatch.setattr(v2expl, "helper_neb", fake_helper_neb)
    monkeypatch.setattr(runner, "executor", EagerExecutor())

    cluster_key = cluster.get_key_for_metadata(False)
    candidates = runner._2step_bulk_candidates(cluster)
    assert len(candidates) > 0

    # 1. all the candidates are submitted in one go
    futures = candidate_runner([cluster], system_key="")
    assert len(futures) == len(candidates)

    # 2. the results are collected and every candidate is recorded as done
    runner.collect(futures)
    for key, _, _ in candidates:
        assert f"{cluster_key}@{key}" in runner.network.recorder.bulk

    # 3. the exploration of the cluster is finished when the list is empty
    assert runner._2step_bulk_candidates(cluster) == []
    assert candidate_runner([cluster], system_key="") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
