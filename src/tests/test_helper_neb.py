from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from hydra import compose, initialize

from graphatoms.enterpoint.config import CONFIG_DIR, Config
from graphatoms.enterpoint.runner.common._helper import (
    helper_neb,
    helper_optimization,
)
from graphatoms.reaction import Reaction
from graphatoms.system import Cluster, System
from graphatoms.utils.aseqhull import get_voronoi_candidates

this_dir = Path(__file__).parent


@pytest.fixture(scope="module")
def config() -> Config:
    with initialize(
        config_path=Path(CONFIG_DIR)
        .relative_to(this_dir, walk_up=True)
        .as_posix(),
        job_name="test-helper-neb",
        version_base=None,
    ):
        cfg: Config = compose(  # type: ignore
            config_name="run",
            overrides=["calculator=emt", "atoms=octahedron"],
        )
    return cfg


@pytest.fixture(scope="module")
def bulk_cluster(config: Config) -> Cluster:
    """A Cu(3x3x3) bulk cluster with one vacancy, optimized."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    vacancy = atoms.positions[13].copy()
    atoms = atoms[[i for i in range(len(atoms)) if i != 13]]
    system = System.from_ase(
        atoms,
        parse_bonds=config.bonds,
        parse_atoms_is_outer_or_not=True,
    )
    idx_inner = np.unique(np.where(np.logical_not(system.is_outer)))
    distances = np.linalg.norm(atoms.positions[idx_inner] - vacancy, axis=1)
    core = np.asarray([idx_inner[int(np.argmin(distances))]])
    cluster = Cluster.from_select(
        system,
        core,
        env_threshold=float(config.site.env_threshold),
        max_moved_threshold=float(config.site.max_moved_threshold),
        method=str(config.site.method),
    )
    graph, _, _ = helper_optimization(
        config=config,
        graph=cluster,
        raise_when_fail=True,
        run_vibration=True,
    )
    assert isinstance(graph, Cluster)
    return graph


def test_helper_neb_bulk(config: Config, bulk_cluster: Cluster) -> None:
    atoms = bulk_cluster.to_ase(
        exclude_bond_attibutes=True,
        exclude_energetics=True,
    ).copy()
    index = int(np.unique(bulk_cluster.idx_core)[0])
    candidates = get_voronoi_candidates(atoms, index)
    assert len(candidates) > 0

    reaction: Reaction | None = None
    for key, initial, final in candidates:
        result, (label, trajectory), cost_time, result_type = helper_neb(
            config=config,
            graph=bulk_cluster,
            initial_atoms=initial,
            final_atoms=final,
            graph_label=key,
            nimages=5,
        )
        assert label == key
        assert result_type == "neb"
        assert cost_time >= 0
        if isinstance(result, Reaction):
            reaction = result
            print(f"Found a reaction by the candidate {key}: {result}")
            break
    assert reaction is not None, "No reaction found by NEB."
    assert reaction.T is not None
    assert reaction.R.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    )
    assert reaction.P.check_minima(
        fmax=float(config.event.max_force),
        fqmin=float(config.event.min_frequency),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
