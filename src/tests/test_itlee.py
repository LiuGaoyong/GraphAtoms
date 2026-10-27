# ruff: noqa: D100, D101, D102, D103, D104
import numpy as np
import pytest
from ase.cluster import Octahedron
from scipy.spatial.transform import Rotation as Rot

from graphatoms.geometry.itlee import (
    delta_from_energy_error,
    itlee_check_permutation,
    itlee_equiv,
    max_safe_delta,
)
from graphatoms.system import Cluster, System
from graphatoms.system.bonds import matchmode2nmatch


@pytest.fixture(scope="module")
def system() -> System:
    return System.from_ase(
        Octahedron("Pd", 8),
        parse_atoms_is_outer_or_not=True,
    )


@pytest.fixture(scope="module")
def cluster(system: System) -> Cluster:
    return Cluster.from_select(
        system,
        np.array([0]),
        method="hop",
        max_moved_threshold=1,
        env_threshold=2,
    )


def _random_cloud(
    n: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(n, 3))
    cur = ref + rng.normal(scale=0.002, size=ref.shape)
    rotation = Rot.random(random_state=seed).as_matrix()
    cur = cur @ rotation.T + np.array([1.0, 2.0, 3.0])
    return ref, cur, rotation


class Test_ITLEECore:
    def test_rotation_translation_permutation(self) -> None:
        ref, cur, _ = _random_cloud()
        rng = np.random.default_rng(1)
        permutation = rng.permutation(len(ref))
        result = itlee_equiv(ref, cur[permutation], 0.05)
        assert result is not None
        assert result.l2 <= 0.05
        assert np.array_equal(result.permutation, np.argsort(permutation))

    def test_noise_threshold(self) -> None:
        ref, cur, _ = _random_cloud()
        assert itlee_equiv(ref, cur, 0.05) is not None
        rng = np.random.default_rng(2)
        bad = cur + rng.normal(scale=0.5, size=cur.shape)
        assert itlee_equiv(ref, bad, 0.02) is None

    def test_colors_reject(self) -> None:
        ref, cur, _ = _random_cloud()
        colors = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        bad = colors.copy()
        bad[0] = 1
        result = itlee_equiv(
            ref,
            cur,
            0.05,
            ref_colors=colors,
            cur_colors=bad,
        )
        assert result is None

    def test_reflection(self) -> None:
        ref, cur, _ = _random_cloud()
        mirrored = cur.copy()
        mirrored[:, 0] *= -1.0
        assert itlee_equiv(ref, mirrored, 0.05) is None
        assert (
            itlee_equiv(
                ref,
                mirrored,
                0.05,
                allow_reflection=True,
            )
            is not None
        )

    def test_check_permutation(self) -> None:
        ref, cur, _ = _random_cloud()
        result = itlee_check_permutation(
            ref,
            cur,
            np.arange(len(ref)),
            0.05,
        )
        assert result is not None
        assert result.l2 <= 0.05
        assert result.transform(cur).shape == ref.shape

    def test_bounds_and_energy_error(self) -> None:
        ref, _, _ = _random_cloud()
        assert max_safe_delta(ref) > 0
        assert delta_from_energy_error(10.0, 0.05) == pytest.approx(
            np.sqrt(2 * 0.05 / 10.0)
        )
        with pytest.raises(ValueError):
            delta_from_energy_error(0.0, 0.05)

    def test_unequal_sizes(self) -> None:
        ref, cur, _ = _random_cloud()
        assert itlee_equiv(ref, cur[:-1], 0.05) is None

    def test_ambiguous_delta(self) -> None:
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = itlee_equiv(ref, ref, 2.0)
        assert result is not None
        assert result.ambiguous is True


class Test_SysGraphEquivalence:
    def test_graph_and_itlee(self, cluster: Cluster) -> None:
        rotation = Rot.random(random_state=3).as_matrix()
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        dct["positions"] = cluster.positions @ rotation.T + np.array([1, 2, 3])
        other = Cluster.model_validate(dct)

        assert cluster.is_equivalent(other, method="graph")
        assert cluster.is_equivalent(
            other,
            method="itlee",
            tolerance=0.01,
            use_hash_bucket=False,
        )

    def test_hash_veto_and_geometry(self, cluster: Cluster) -> None:
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        dct["coordination"] = (cluster.CN_MATRIX + 1).astype("uint8")
        other = Cluster.model_validate(dct)
        assert other.hash != cluster.hash

        # L0 hash veto reports a false-new environment.
        assert not cluster.is_equivalent(
            other,
            method="itlee",
            tolerance=0.05,
            use_hash_bucket=True,
        )
        # Bypassing the veto recovers the geometric equivalence.
        assert cluster.is_equivalent(
            other,
            method="itlee",
            tolerance=0.05,
            use_hash_bucket=False,
        )
        # The graphs themselves differ.
        assert not cluster.is_equivalent(
            other,
            method="graph",
            use_hash_bucket=False,
        )

    def test_tolerance_min(self, cluster: Cluster) -> None:
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        rng = np.random.default_rng(4)
        dct["positions"] = cluster.positions + rng.normal(
            scale=0.01, size=cluster.positions.shape
        )
        other = Cluster.model_validate(dct)
        result = cluster.get_itlee_equiv(
            other,
            0.2,
            other_tolerance=0.001,
        )
        assert result is None

    def test_unequal_graph(self, cluster: Cluster) -> None:
        assert not cluster.is_equivalent(
            System.from_ase(Octahedron("Pd", 2)),
            method="graph",
        )


class Test_MatchTolerance:
    def test_exact_unchanged(self, system: System, cluster: Cluster) -> None:
        matching = system.get_match_mode(
            cluster, algorithm="lad", use_itlee=True
        )
        assert isinstance(matching, np.ndarray)
        assert matching.shape == (48, len(system))
        matching0 = np.asarray(
            [
                np.vectorize(lambda x: np.argwhere(row == x).item())(
                    np.arange(len(cluster))
                )
                for row in matching
            ]
        )
        assert matchmode2nmatch(matching0) == 6

    def test_stale_coordination(self, system: System, cluster: Cluster) -> None:
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        dct["coordination"] = (cluster.CN_MATRIX + 1).astype("uint8")
        pattern = Cluster.model_validate(dct)
        assert system.get_match_mode(pattern, use_itlee=True) is None
        matching = system.get_match_mode(
            pattern,
            tolerance=0.05,
            return_metrics=True,
            use_itlee=True,
        )
        assert isinstance(matching, list)
        assert len(matching) > 0
        assert all(np.isfinite(rmsd) for _, rmsd in matching)
        assert min(rmsd for _, rmsd in matching) < 0.05

    def test_geometry_reject(self, system: System, cluster: Cluster) -> None:
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        positions = cluster.positions.copy()
        positions[0] += np.array([5.0, 0.0, 0.0])
        dct["positions"] = positions
        pattern = Cluster.model_validate(dct)
        assert (
            system.get_match_mode(
                pattern,
                tolerance=0.05,
                return_metrics=True,
                use_itlee=True,
            )
            is None
        )

    def test_negative_tolerance(self, system: System, cluster: Cluster) -> None:
        with pytest.raises(ValueError):
            system.get_match_mode(cluster, tolerance=-0.1, use_itlee=True)

    def test_tolerant_return_options(
        self, system: System, cluster: Cluster
    ) -> None:
        dct = cluster.to_dict()
        dct.pop("hashes", None)
        dct["coordination"] = (cluster.CN_MATRIX + 1).astype("uint8")
        pattern = Cluster.model_validate(dct)

        count = system.get_match_mode(
            pattern,
            tolerance=0.05,
            only_count=True,
            return_match_target=False,
            use_itlee=True,
        )
        assert isinstance(count, int) and count >= 1

        mapping = system.get_match_mode(
            pattern,
            tolerance=0.05,
            return_match_target=False,
            use_itlee=True,
        )
        assert isinstance(mapping, np.ndarray)
        assert mapping.shape[1] == len(pattern)

        vf2 = system.get_match_mode(
            pattern,
            tolerance=0.05,
            algorithm="vf2",
            return_metrics=True,
            use_itlee=True,
        )
        assert isinstance(vf2, list) and len(vf2) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
