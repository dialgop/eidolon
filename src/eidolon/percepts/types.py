"""Shared perception input contract: what an embodiment/adapter hands to
Eidolon. Eidolon never sees pixels or sensors, only these structures.

Extracted from `eidolon.labs.visual_attention` once a second consumer
(`eidolon.labs.world_model`) needed the same types — the Rule of Three,
not a speculative abstraction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class PerceivedObject:
    """One object as reported by a perception adapter at one moment.

    Equality and hashing use `track_id` only: the same tracked object seen
    again (new position, new confidence, even a corrected `label`) is still
    *the same object*. Everything else is volatile observation data.

    `features` is flat name -> number, produced by the adapter (hue, size,
    motion, ...). `position` is carried but unused by v0 attention; when
    given, `frame` must name the coordinate frame it's expressed in.
    """

    track_id: str
    observed_at: int = field(compare=False)
    label: str | None = field(default=None, compare=False)
    confidence: float = field(default=1.0, compare=False)
    features: Mapping[str, float] = field(default_factory=dict, compare=False)
    position: tuple[float, float, float] | None = field(default=None, compare=False)
    frame: str | None = field(default=None, compare=False)
    source: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.position is not None and self.frame is None:
            raise ValueError("frame is required when position is given")
        for name, value in self.features.items():
            if not math.isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))


@dataclass(frozen=True)
class CoverageRegion:
    """A region of space a perception adapter claims to have observed,
    expressed as a sphere: `center` + `radius` in `frame`.

    Used for negative evidence: a consumer that trusts an adapter's
    coverage claim can treat "no object here, but this region was covered"
    as "this object is gone", not just "unobserved". `frame` follows the
    same rule as `PerceivedObject.position`/`frame` — a consumer checking a
    region must be working in that same frame, or raise.
    """

    center: tuple[float, float, float]
    radius: float
    frame: str

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must be a non-empty string")


@dataclass(frozen=True)
class Scene:
    """One frame's worth of perceived objects, plus what was observed.

    `objects` and `coverage` must be tuples (not lists) so a caller can't
    keep a mutable handle on either, and `track_id`s must be unique within
    the scene. `coverage` is optional (default `()`): with no coverage
    claimed, a consumer has no negative evidence and cannot conclude an
    object is gone versus merely unobserved — a conscious limitation, not a
    default to fill in casually.
    """

    observed_at: int
    objects: tuple[PerceivedObject, ...]
    coverage: tuple[CoverageRegion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be a tuple")
        if not isinstance(self.coverage, tuple):
            raise TypeError("coverage must be a tuple")
        ids = [o.track_id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("track_id must be unique within a scene")
