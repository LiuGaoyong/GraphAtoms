import itertools

import array_api_extra as xpx

from graphatoms.utils import Array, ArrayNamespace, get_namespace
from graphatoms.utils._array_api_typing import LinalgNamespace


def pbc2pbc(pbc: bool, np: ArrayNamespace) -> Array:
    """Convert a boolean pbc to an Array pbc."""
    # assert isinstance(np, ArrayNamespace), type(np)
    return np.full(3, fill_value=bool(pbc))


def translate_pretty(fractional: Array, pbc: Array) -> Array:
    """Translates atoms such that fractional positions are minimized."""
    np: ArrayNamespace = get_namespace(fractional)

    frac_T = fractional.T
    indices = np.argsort(frac_T, axis=1)  # (3, N)
    sp = np.take_along_axis(frac_T, indices, axis=1)  # (3, N)
    rolled = np.concat([sp[:, -1:], sp[:, :-1]], axis=1)  # (3, N)
    widths = (rolled - sp) % 1.0  # (3, N)
    min_idx = np.argmin(widths, axis=1)  # (3,)
    shifts = sp[np.arange(3), min_idx]  # (3,)
    new_frac_T = (frac_T - shifts[:, np.newaxis]) % 1.0  # (3, N)
    return np.where(pbc[:, np.newaxis], new_frac_T, frac_T).T


def wrap_positions(
    positions: Array,
    complete_cell: Array,
    pbc: bool | Array = True,
    center: float | tuple[float, ...] = (0.5, 0.5, 0.5),
    pretty_translation: bool = False,
    eps: float = 1e-7,
) -> Array:
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.  See also the
    :meth:`ase.Atoms.wrap` method.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: three float
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    pretty_translation: bool
        Translates atoms such that fractional coordinates are minimized.
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    Example:

    >>> from ase.geometry import wrap_positions
    >>> wrap_positions([[-0.1, 1.01, -0.5]],
    ...                [[1, 0, 0], [0, 1, 0], [0, 0, 4]],
    ...                pbc=[1, 1, 0])
    array([[ 0.9 ,  0.01, -0.5 ]])
    """
    np: ArrayNamespace = get_namespace(complete_cell, positions)  # type: ignore
    linalg: LinalgNamespace | None = getattr(np, "linalg", None)
    assert linalg is not None and isinstance(linalg, LinalgNamespace), (
        f"ArrayNamespace({np}) does not have linalg attribute."
    )
    cell = complete_cell

    if not hasattr(center, "__len__"):
        center = (center,) * 3  # type: ignore
    center = np.asarray(center)  # type: ignore
    if isinstance(pbc, bool):
        pbc = pbc2pbc(pbc, np)  # type: ignore
    shift = np.asarray(center) - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift = xpx.at(shift, np.logical_not(pbc)).set(0.0)

    # assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)
    fractional = linalg.solve(cell.T, positions.T).T - shift

    if pretty_translation:
        fractional = translate_pretty(fractional, pbc)
        shift = np.asarray(center) - 0.5
        shift = xpx.at(shift, np.logical_not(pbc)).set(0.0)
        fractional = fractional + shift  # type: ignore
    else:
        fractional = np.where(
            np.reshape(pbc, (1, -1)),
            (fractional % 1.0) + shift,
            fractional,
        )
    return np.matmul(fractional, cell)


# 通用函数
def naive_find_mic(v: Array, complete_cell: Array) -> tuple[Array, Array]:
    """Finds the minimum-image representation of vector(s) v.
    Safe to use for (pbc.all() and (norm(v_mic) < 0.5 * min(cell.lengths()))).
    Can otherwise fail for non-orthorhombic cells.
    Described in:
    W. Smith, "The Minimum Image Convention in Non-Cubic MD Cells", 1989,
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696."""

    np: ArrayNamespace = get_namespace(complete_cell, v)  # type: ignore
    linalg: LinalgNamespace | None = getattr(np, "linalg", None)
    assert linalg is not None and isinstance(linalg, LinalgNamespace), (
        f"ArrayNamespace({np}) does not have linalg attribute."
    )

    f = linalg.solve(complete_cell.T, v.T).T
    vmin = (f - np.floor(f + 0.5)) @ complete_cell
    vlen = linalg.vector_norm(vmin, axis=1, ord=2)  # L2 norm
    return vmin, vlen


def general_find_mic(
    v: Array,
    minkowski_rcell: Array,
    pbc: bool | Array = True,
) -> tuple[Array, Array]:
    """Finds the minimum-image representation of vector(s) v. Using the
    Minkowski reduction the algorithm is relatively slow but safe for any cell.
    """
    np: ArrayNamespace = get_namespace(minkowski_rcell, v)
    linalg: LinalgNamespace | None = getattr(np, "linalg", None)
    assert linalg is not None and isinstance(linalg, LinalgNamespace), (
        f"ArrayNamespace({np}) does not have linalg attribute."
    )
    if isinstance(pbc, bool):
        pbc = pbc2pbc(pbc, np)
    rcell = minkowski_rcell
    positions = wrap_positions(v, rcell, pbc=pbc, eps=0)
    npos = int(v.shape[0])  # type: ignore

    # In a Minkowski-reduced cell we only need to test nearest neighbors,
    # or "Voronoi-relevant" vectors. These are a subset of combinations of
    # [-1, 0, 1] of the reduced cell vectors.

    # Define ranges [-1, 0, 1] for periodic directions and [0] for aperiodic
    # directions.
    ranges = [np.arange(-1 * p, p + 1) for p in pbc]  # type: ignore

    # Get Voronoi-relevant vectors.
    # Pre-pend (0, 0, 0) to resolve issue #772
    hkls = np.asarray(
        [(0, 0, 0)] + list(itertools.product(*ranges)), dtype=rcell.dtype
    )
    vrvecs = np.matmul(hkls, rcell)

    # Map positions into neighbouring cells.
    x = positions + vrvecs[:, None]

    # Find minimum images
    lengths = linalg.vector_norm(x, axis=2, ord=2)  # L2 norm
    indices = np.argmin(lengths, axis=0)
    vmin = x[indices, np.arange(npos), :]
    vlen = lengths[indices, np.arange(npos)]
    return vmin, vlen


def find_mic(
    v: Array,
    minkowski_rcell: Array,
    pbc: bool | Array = True,
) -> tuple[Array, Array]:
    """Finds the minimum-image representation of vector(s) v using either one
    of two find mic algorithms depending on the given cell, v and pbc."""
    np: ArrayNamespace = get_namespace(minkowski_rcell, v)
    linalg: LinalgNamespace | None = getattr(np, "linalg", None)
    assert linalg is not None and isinstance(linalg, LinalgNamespace), (
        f"ArrayNamespace({np}) does not have linalg attribute."
    )
    assert v.ndim == 2 and v.shape[1] == 3, f"Invalid vector shape: {v.shape}."

    cell = minkowski_rcell
    if isinstance(pbc, bool):
        pbc = pbc2pbc(pbc, np)  # type: ignore
    pbc = np.any(cell, axis=1) & pbc
    dim = np.sum(pbc)
    if dim > 0:
        naive_find_mic_is_safe = False
        if dim == 3:
            vmin, vlen = naive_find_mic(v, cell)
            # naive find mic is safe only for the following condition
            cell_lengths = linalg.vector_norm(cell, axis=1, ord=2)  # L2 norm
            if np.all(vlen < 0.5 * min(cell_lengths)):
                naive_find_mic_is_safe = True  # hence skip Minkowski reduction
        if not naive_find_mic_is_safe:
            vmin, vlen = general_find_mic(v, cell, pbc=pbc)
    else:
        vmin = np.asarray(v, copy=True)
        vlen = linalg.vector_norm(vmin, axis=1, ord=2)  # L2 norm
    return vmin, vlen  # type: ignore
