"""Invariant and Tolerant Local Environment Equivalence (ITLEE).

This module implements the geometric equivalence criterion proposed by
Williams (2024), *Off-lattice kinetic Monte Carlo methods and the Fe-H
system*, PhD thesis, University of Cambridge, Section 3.1.2, pp. 33-43.

Two equally sized, coloured point clouds ``P`` and ``Q`` are equivalent
when there exist an orthogonal matrix ``O`` and a permutation ``pi`` such
that

.. math::

    \\sum_i \\|p_i - O q_{\\pi(i)}\\|^2 \\le \\delta^2

with ``O O^T = I`` and matching colours.  The residual used here is the
Euclidean (L2) norm of the whole displacement vector, following Eq. (3.5)
of the thesis.  The corresponding RMSD is ``rmsd = l2 / sqrt(n)``.

The implementation is intentionally dependency-light (NumPy + SciPy) and
does not import :mod:`graphatoms.system`, so it can be used by both the
system layer and the runner layer without circular imports.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

__all__ = [
    "ITLIEEResult",
    "delta_from_energy_error",
    "itlee_check_permutation",
    "itlee_equiv",
    "max_safe_delta",
]


@dataclass(frozen=True)
class ITLIEEResult:
    """Result of an ITLEE comparison.

    Attributes:
        delta: The L2 tolerance used for the comparison.
        permutation: ``permutation[i]`` is the index in ``cur`` that is
            matched to index ``i`` in ``ref``.
        rotation: Orthogonal matrix ``O`` with ``ref ~= cur[perm] @ O.T + t``.
        translation: Translation vector ``t``.
        distances: Per-atom residual distances.
        l2: Euclidean norm of ``distances``.
        rmsd: Root-mean-square of ``distances``.
        max_distance: Maximum per-atom residual distance.
        n: Number of matched atoms.
        ambiguous: Whether ``delta >= r_min``.  In that case the mapping is
            still one-to-one by construction, but the tolerance is larger
            than the shortest interatomic distance and the result should be
            treated with care.
    """

    delta: float
    permutation: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    distances: np.ndarray
    l2: float
    rmsd: float
    max_distance: float
    n: int
    ambiguous: bool = False

    @property
    def is_valid(self) -> bool:
        """Whether the L2 residual is within the requested tolerance."""
        return self.l2 <= self.delta + 1e-12

    def transform(self, points: ArrayLike) -> np.ndarray:
        """Apply the fitted transform to the given points."""
        p = np.asarray(points, dtype=float)
        return p @ self.rotation.T + self.translation

    def permutation_matrix(self) -> np.ndarray:
        """Return the permutation as an ``(n, n)`` boolean matrix."""
        result = np.zeros((self.n, self.n), dtype=bool)
        result[np.arange(self.n), self.permutation] = True
        return result


def _as_positions(points: ArrayLike, name: str = "points") -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {p.shape}.")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return p


def _as_colors(
    colors: ArrayLike | None,
    n: int,
    name: str = "colors",
) -> np.ndarray:
    if colors is None:
        return np.zeros(n, dtype=int)
    c = np.asarray(colors)
    if c.ndim != 1 or len(c) != n:
        raise ValueError(f"{name} must have length {n}, got {c.shape}.")
    return c


def _same_color_multiset(ref: np.ndarray, cur: np.ndarray) -> bool:
    ref_unique, ref_counts = np.unique(ref, return_counts=True)
    cur_unique, cur_counts = np.unique(cur, return_counts=True)
    if len(ref_unique) != len(cur_unique):
        return False
    if not np.array_equal(ref_unique, cur_unique):
        return False
    return bool(np.array_equal(ref_counts, cur_counts))


def _min_positive_pair_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    result = float(np.min(d))
    return result if result > 1e-12 else 0.0


def max_safe_delta(positions: ArrayLike) -> float:
    """Return the recommended upper bound ``(2/5) * r_min`` for ``delta``.

    Williams (2024) Eq. (3.19) requires ``delta <= (2/5) r_min`` to keep the
    number of candidate matches bounded.  The function returns ``inf`` when
    fewer than two atoms are provided.
    """
    p = _as_positions(positions)
    r_min = _min_positive_pair_distance(p)
    if not np.isfinite(r_min):
        return float("inf")
    return 0.4 * r_min


def delta_from_energy_error(lambda_max: float, energy_error: float) -> float:
    """Convert an energy-error bound to an ITLEE tolerance.

    From Williams (2024) Eq. (3.10), ``Delta U <= lambda_max * delta**2 / 2``,
    hence ``delta = sqrt(2 * Delta U / lambda_max)``.

    Args:
        lambda_max: Largest Hessian eigenvalue, in eV / Angstrom^2.
        energy_error: Acceptable energy error, in eV.

    Returns:
        The corresponding tolerance in Angstrom.
    """
    if lambda_max <= 0:
        raise ValueError("lambda_max must be positive.")
    if energy_error < 0:
        raise ValueError("energy_error must be non-negative.")
    return sqrt(2.0 * float(energy_error) / float(lambda_max))


def _procrustes(
    ref: np.ndarray,
    cur: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``ref ~= cur @ O.T + t`` and return ``(O, t, distances)``."""
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    ref_c = ref - ref_centroid
    cur_c = cur - cur_centroid

    if len(ref) == 0:
        return np.eye(3), np.zeros(3), np.zeros(0, dtype=float)

    cross = cur_c.T @ ref_c
    u, _, vt = np.linalg.svd(cross)
    rotation = vt.T @ u.T
    if not allow_reflection and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = ref_centroid - rotation @ cur_centroid
    residual = ref - (cur @ rotation.T + translation)
    distances = np.linalg.norm(residual, axis=1)
    return rotation, translation, distances


def _result_from_permutation(
    ref: np.ndarray,
    cur: np.ndarray,
    permutation: np.ndarray,
    delta: float,
    *,
    ambiguous: bool,
    allow_reflection: bool,
) -> ITLIEEResult:
    rotation, translation, distances = _procrustes(
        ref,
        cur[permutation],
        allow_reflection=allow_reflection,
    )
    l2 = float(np.linalg.norm(distances))
    rmsd = float(np.sqrt(np.mean(distances**2))) if len(distances) else 0.0
    max_distance = float(np.max(distances)) if len(distances) else 0.0
    return ITLIEEResult(
        delta=float(delta),
        permutation=np.asarray(permutation, dtype=int),
        rotation=rotation,
        translation=translation,
        distances=distances,
        l2=l2,
        rmsd=rmsd,
        max_distance=max_distance,
        n=len(ref),
        ambiguous=ambiguous,
    )


def _fingerprint(
    positions: np.ndarray,
    colors: np.ndarray,
) -> dict[tuple[Any, Any], np.ndarray]:
    """Return sorted inner distances grouped by colour pairs."""
    result: dict[tuple[Any, Any], np.ndarray] = {}
    d = cdist(positions, positions)
    unique = list(np.unique(colors))
    for i, c1 in enumerate(unique):
        for c2 in unique[i:]:
            idx1 = np.flatnonzero(colors == c1)
            idx2 = np.flatnonzero(colors == c2)
            if c1 == c2:
                rows, cols = np.triu_indices(len(idx1), k=1)
                values = d[np.ix_(idx1, idx2)][rows, cols]
            else:
                values = d[np.ix_(idx1, idx2)].ravel()
            result[(c1, c2)] = np.sort(values)
    return result


def _fingerprint_ok(
    ref: np.ndarray,
    cur: np.ndarray,
    ref_colors: np.ndarray,
    cur_colors: np.ndarray,
    delta: float,
) -> bool:
    """Cheap necessary condition based on sorted inner distances."""
    fp_ref = _fingerprint(ref, ref_colors)
    fp_cur = _fingerprint(cur, cur_colors)
    if set(fp_ref) != set(fp_cur):
        return False
    threshold = sqrt(2.0) * delta + 1e-12
    for key, values_ref in fp_ref.items():
        values_cur = fp_cur[key]
        if len(values_ref) != len(values_cur):
            return False
        if len(values_ref) == 0:
            continue
        diff = np.abs(np.sort(values_ref) - np.sort(values_cur)).max()
        if float(diff) > threshold:
            return False
    return True


def _anchor_index(ref: np.ndarray, ref_colors: np.ndarray) -> int:
    """Choose a deterministic, symmetry-breaking anchor atom."""
    unique, counts = np.unique(ref_colors, return_counts=True)
    rarest = unique[int(np.argmin(counts))]
    candidates = np.flatnonzero(ref_colors == rarest)
    if len(candidates) == 1:
        return int(candidates[0])
    centroid = ref.mean(axis=0)
    distances = np.linalg.norm(ref[candidates] - centroid, axis=1)
    return int(candidates[int(np.argmax(distances))])


def _match_permutation(
    ref: np.ndarray,
    cur: np.ndarray,
    ref_colors: np.ndarray,
    cur_colors: np.ndarray,
    delta: float,
    *,
    allow_reflection: bool,
    max_branches: int,
) -> np.ndarray | None:
    """Search for a valid permutation using anchors and inner distances."""
    n = len(ref)
    if n != len(cur):
        return None
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    d_ref = cdist(ref, ref)
    d_cur = cdist(cur, cur)
    threshold = sqrt(2.0) * delta + 1e-12
    anchor = _anchor_index(ref, ref_colors)

    # Candidate anchors in the current cloud.
    anchor_candidates = np.flatnonzero(cur_colors == ref_colors[anchor])
    if len(anchor_candidates) == 0:
        return None

    branches = [0]
    exhausted = [False]

    def search(
        assigned_ref: list[int],
        assigned_cur: list[int],
        used: np.ndarray,
        candidates: list[list[int]],
        order: list[int],
        depth: int,
    ) -> np.ndarray | None:
        if exhausted[0]:
            return None
        if depth == len(order):
            permutation = np.empty(n, dtype=int)
            for i, j in zip(assigned_ref, assigned_cur, strict=True):
                permutation[i] = j
            result = _result_from_permutation(
                ref,
                cur,
                permutation,
                delta,
                ambiguous=False,
                allow_reflection=allow_reflection,
            )
            return permutation if result.is_valid else None

        i = order[depth]
        for j in candidates[depth]:
            if used[j]:
                continue
            ok = True
            for ii, jj in zip(assigned_ref, assigned_cur, strict=True):
                if abs(d_ref[i, ii] - d_cur[j, jj]) > threshold:
                    ok = False
                    break
            if not ok:
                continue

            branches[0] += 1
            if branches[0] > max_branches:
                exhausted[0] = True
                return None

            assigned_ref.append(i)
            assigned_cur.append(j)
            used[j] = True
            # Partial Procrustes pruning once we have four non-coplanar points.
            if len(assigned_ref) >= 4:
                partial = _result_from_permutation(
                    ref[assigned_ref],
                    cur[assigned_cur],
                    np.arange(len(assigned_ref), dtype=int),
                    delta,
                    ambiguous=False,
                    allow_reflection=allow_reflection,
                )
                if partial.l2 > delta + 1e-12:
                    assigned_ref.pop()
                    assigned_cur.pop()
                    used[j] = False
                    continue
            found = search(
                assigned_ref,
                assigned_cur,
                used,
                candidates,
                order,
                depth + 1,
            )
            assigned_ref.pop()
            assigned_cur.pop()
            used[j] = False
            if found is not None:
                return found
        return None

    for anchor_cur in anchor_candidates:
        candidates: list[list[int]] = []
        for i in range(n):
            if i == anchor:
                candidates.append([int(anchor_cur)])
                continue
            mask = (cur_colors == ref_colors[i]) & (
                np.abs(d_cur[anchor_cur] - d_ref[anchor, i]) <= threshold
            )
            cand = np.flatnonzero(mask).tolist()
            if not cand:
                candidates.append([])
                break
            candidates.append(cand)
        if len(candidates) != n:
            continue
        order = [anchor]
        order.extend(
            sorted(
                (k for k in range(n) if k != anchor),
                key=lambda k: (len(candidates[k]), k),
            )
        )
        # Precompute candidates in search order.
        ordered_candidates = [candidates[k] for k in order]
        used = np.zeros(n, dtype=bool)
        used[anchor_cur] = True
        result = search(
            [anchor],
            [int(anchor_cur)],
            used,
            ordered_candidates,
            order,
            1,
        )
        if result is not None:
            return result
        if exhausted[0]:
            break

    if exhausted[0]:
        warnings.warn(
            "ITLEE search exceeded max_branches; returning no match.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def itlee_check_permutation(
    ref_positions: ArrayLike,
    cur_positions: ArrayLike,
    permutation: ArrayLike,
    delta: float,
    *,
    ref_colors: ArrayLike | None = None,
    cur_colors: ArrayLike | None = None,
    allow_reflection: bool = False,
) -> ITLIEEResult | None:
    """Validate a known one-to-one mapping with the ITLEE L2 criterion.

    Args:
        ref_positions: Reference point cloud, shape ``(n, 3)``.
        cur_positions: Current point cloud, shape ``(n, 3)``.
        permutation: For each reference atom ``i``, the matching current
            index ``permutation[i]``.
        delta: L2 tolerance (Angstrom).
        ref_colors: Optional colors of the reference cloud.
        cur_colors: Optional colors of the current cloud.
        allow_reflection: Allow improper rotations (O(3) instead of SO(3)).

    Returns:
        An :class:`ITLIEEResult` when the mapping is valid, otherwise
        ``None``.  Only mappings with an L2 residual within ``delta`` are
        returned.
    """
    ref = _as_positions(ref_positions, "ref_positions")
    cur = _as_positions(cur_positions, "cur_positions")
    if len(ref) != len(cur):
        return None
    if delta <= 0:
        raise ValueError("delta must be positive.")
    perm = np.asarray(permutation, dtype=int).reshape(-1)
    if len(perm) != len(ref):
        raise ValueError("permutation must have the same length as ref.")
    if len(np.unique(perm)) != len(perm):
        return None
    if np.any(perm < 0) or np.any(perm >= len(cur)):
        return None

    ref_colors_arr = _as_colors(ref_colors, len(ref), "ref_colors")
    cur_colors_arr = _as_colors(cur_colors, len(cur), "cur_colors")
    if not np.array_equal(ref_colors_arr, cur_colors_arr[perm]):
        return None

    r_min = min(
        _min_positive_pair_distance(ref),
        _min_positive_pair_distance(cur),
    )
    ambiguous = bool(np.isfinite(r_min) and delta >= r_min)
    if ambiguous:
        warnings.warn(
            f"delta={delta:.3f} >= r_min={r_min:.3f}; "
            "the ITLEE result is ambiguous.",
            RuntimeWarning,
            stacklevel=2,
        )
    result = _result_from_permutation(
        ref,
        cur,
        perm,
        delta,
        ambiguous=ambiguous,
        allow_reflection=allow_reflection,
    )
    return result if result.is_valid else None


def itlee_equiv(
    ref_positions: ArrayLike,
    cur_positions: ArrayLike,
    delta: float,
    *,
    ref_colors: ArrayLike | None = None,
    cur_colors: ArrayLike | None = None,
    allow_reflection: bool = False,
    max_branches: int = 200_000,
) -> ITLIEEResult | None:
    """Return the first ITLEE-equivalent mapping between two point clouds.

    The two clouds must have the same number of atoms and the same colour
    multiset.  The search uses a sorted-inner-distance fingerprint as a
    necessary prefilter, then anchor/distance-guided backtracking and
    orthogonal Procrustes.

    Args:
        ref_positions: Reference point cloud, shape ``(n, 3)``.
        cur_positions: Current point cloud, shape ``(n, 3)``.
        delta: L2 tolerance (Angstrom), following Williams (2024) Eq. (3.5).
        ref_colors: Optional colors of the reference cloud.
        cur_colors: Optional colors of the current cloud.
        allow_reflection: Allow improper rotations (O(3)).
        max_branches: Maximum number of search branches before failing
            closed.

    Returns:
        An :class:`ITLIEEResult` or ``None``.
    """
    ref = _as_positions(ref_positions, "ref_positions")
    cur = _as_positions(cur_positions, "cur_positions")
    if len(ref) != len(cur):
        return None
    if delta <= 0:
        raise ValueError("delta must be positive.")

    ref_colors_arr = _as_colors(ref_colors, len(ref), "ref_colors")
    cur_colors_arr = _as_colors(cur_colors, len(cur), "cur_colors")
    if not _same_color_multiset(ref_colors_arr, cur_colors_arr):
        return None

    if len(ref) == 0:
        return _result_from_permutation(
            ref,
            cur,
            np.array([], dtype=int),
            delta,
            ambiguous=False,
            allow_reflection=allow_reflection,
        )

    if not _fingerprint_ok(
        ref,
        cur,
        ref_colors_arr,
        cur_colors_arr,
        delta,
    ):
        return None

    permutation = _match_permutation(
        ref,
        cur,
        ref_colors_arr,
        cur_colors_arr,
        delta,
        allow_reflection=allow_reflection,
        max_branches=max_branches,
    )
    if permutation is None:
        return None

    r_min = min(
        _min_positive_pair_distance(ref),
        _min_positive_pair_distance(cur),
    )
    ambiguous = bool(np.isfinite(r_min) and delta >= r_min)
    if ambiguous:
        warnings.warn(
            f"delta={delta:.3f} >= r_min={r_min:.3f}; "
            "the ITLEE result is ambiguous.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _result_from_permutation(
        ref,
        cur,
        permutation,
        delta,
        ambiguous=ambiguous,
        allow_reflection=allow_reflection,
    )
