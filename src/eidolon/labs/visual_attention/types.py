"""Input contract for visual attention: what an embodiment/adapter hands to
Eidolon. Eidolon never sees pixels or sensors, only these structures."""

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
class Scene:
    """One frame's worth of perceived objects. `objects` must be a tuple (not
    a list) so a caller can't keep a mutable handle on it, and `track_id`s
    must be unique within the scene."""

    observed_at: int
    objects: tuple[PerceivedObject, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be a tuple")
        ids = [o.track_id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("track_id must be unique within a scene")
