"""Per-local-environment tolerance registry.

The registry stores the ITLEE tolerance ``delta`` for each local
environment (cluster key) together with simple reuse statistics.  It is a
small JSON sidecar next to the reaction-network metadata; the event and
metadata schemas are intentionally left unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
)
from pydantic import Field

from graphatoms.utils.bytestool import atomic_write_text

__all__ = ["EquivalenceEntry", "EquivalenceRegistry"]


class EquivalenceEntry(BaseModel):
    """Tolerance and counters for one catalogue entry."""

    model_config = ConfigDict(extra="ignore")

    delta: PositiveFloat
    attempts: NonNegativeInt = 0
    failures: NonNegativeInt = 0


class EquivalenceRegistry(BaseModel):
    """Per-key tolerance registry persisted as ``equivalence.json``."""

    model_config = ConfigDict(extra="ignore")

    clusters: dict[str, EquivalenceEntry] = Field(default_factory=dict)
    events: dict[str, EquivalenceEntry] = Field(default_factory=dict)

    @classmethod
    def read_json(cls, path: Path | str) -> EquivalenceRegistry:
        """Load a registry, returning an empty one when the file is absent."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls.model_validate(data)

    def write_json(self, path: Path | str) -> None:
        """Persist the registry atomically."""
        atomic_write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    def _get(self, key: str) -> EquivalenceEntry | None:
        return self.clusters.get(key) or self.events.get(key)

    def get_delta(self, key: str, default: float) -> float:
        """Return the stored tolerance for ``key`` or ``default``."""
        entry = self._get(key)
        return float(entry.delta) if entry is not None else float(default)

    def set_delta(self, key: str, delta: float) -> EquivalenceEntry:
        """Create or update the tolerance for ``key``."""
        entry = self.clusters.get(key)
        if entry is None:
            entry = EquivalenceEntry(delta=float(delta))
            self.clusters[key] = entry
        else:
            entry = EquivalenceEntry(
                delta=float(delta),
                attempts=entry.attempts,
                failures=entry.failures,
            )
            self.clusters[key] = entry
        return entry

    def halve(self, key: str, *, default: float = 0.1) -> float:
        """Halve the tolerance for ``key`` and return the new value."""
        entry = self.clusters.get(key)
        current = float(entry.delta) if entry is not None else float(default)
        new_delta = max(current / 2.0, 1e-6)
        self.set_delta(key, new_delta)
        return new_delta

    def record_attempt(self, key: str, *, default: float = 0.1) -> None:
        """Record one reuse attempt."""
        entry = self.clusters.get(key)
        if entry is None:
            entry = self.set_delta(key, default)
        entry = EquivalenceEntry(
            delta=entry.delta,
            attempts=entry.attempts + 1,
            failures=entry.failures,
        )
        self.clusters[key] = entry

    def record_failure(self, key: str, *, default: float = 0.1) -> None:
        """Record one failed reuse attempt."""
        entry = self.clusters.get(key)
        if entry is None:
            entry = self.set_delta(key, default)
        entry = EquivalenceEntry(
            delta=entry.delta,
            attempts=entry.attempts + 1,
            failures=entry.failures + 1,
        )
        self.clusters[key] = entry

    def false_reuse_rate(self, key: str) -> float:
        """Return ``failures / attempts`` for ``key`` (0 when unknown)."""
        entry = self._get(key)
        if entry is None or entry.attempts == 0:
            return 0.0
        return float(entry.failures) / float(entry.attempts)
