"""Optional-Ray import handling.

``ray`` is an optional dependency (``pip install 'graphatoms[ray]'``).  This
module mirrors the pattern used by
:mod:`graphatoms.enterpoint.parallel._ray`: it imports Ray when available and
otherwise exposes ``ray = None`` plus a clear error helper, so the runner
package can be imported on installations without Ray.
"""

from __future__ import annotations

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when ray absent
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False

RAY_INSTALL_HINT = (
    "ray is not installed. Install the optional Ray dependency with: "
    "pip install 'graphatoms[ray]'"
)


class RayNotInstalledError(ImportError):
    """Raised when a Ray-only runner is requested without Ray installed."""


def require_ray() -> None:
    """Raise a descriptive error if Ray is not importable."""
    if not _RAY_AVAILABLE:
        raise RayNotInstalledError(RAY_INSTALL_HINT)


__all__ = [
    "RAY_INSTALL_HINT",
    "RayNotInstalledError",
    "require_ray",
]
