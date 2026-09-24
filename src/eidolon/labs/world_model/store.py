"""A belief store: persists beliefs about objects' existence and last known
state, independent of whether they're currently attended.

Scene provides existence; working memory (and, later, semantic memory)
provide focus and interpretation, not existence. `WorldModel` reads
`Scene`s directly, never `WorkingMemory`, so what's believed to exist never
shrinks to what's currently attended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from eidolon.percepts import CoverageRegion, PerceivedObject, Scene


@dataclass(frozen=True)
class Belief:
    """A snapshot belief about one tracked object.

    Frozen like `Episode`, not mutated in place like working memory's
    `Item`: a refresh replaces the store's entry with a new `Belief`
    rather than mutating fields of the old one.
    """

    percept: PerceivedObject
    confidence: float
    last_observed_at: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

    @property
    def track_id(self) -> str:
        return self.percept.track_id

    @property
    def label(self) -> str | None:
        return self.percept.label

    def confidence_at(self, observed_at: int, decay_rate: float) -> float:
        """Confidence as of `observed_at`, decayed from this belief's own
        `confidence`/`last_observed_at` — this does *not* read or change
        anything stored: it is a pure function of this snapshot and the
        time asked about. `stored_confidence` (this object's `.confidence`)
        is never itself decayed in place; see the module docstring."""
        elapsed = observed_at - self.last_observed_at
        return self.confidence * (1.0 - decay_rate) ** elapsed


@dataclass(frozen=True)
class UpdateReport:
    """What one `WorldModel.update()` call changed, grouped like
    `WorkingMemory.tick()`'s return rather than left for the caller to diff
    two snapshots."""

    new: tuple[Belief, ...] = ()
    refreshed: tuple[Belief, ...] = ()
    contradicted: tuple[Belief, ...] = ()
    forgotten: tuple[Belief, ...] = ()


@dataclass
class WorldModel:
    """`frame` is the single coordinate frame this store holds beliefs in;
    it does no transforms itself (see the lab README). `decay_rate` and
    `forget_threshold` govern how an unobserved belief's confidence fades;
    see `update()` (added in this lab's dynamics round) for how."""

    frame: str
    decay_rate: float = 0.1
    forget_threshold: float = 0.05

    _beliefs: dict[str, Belief] = field(default_factory=dict, init=False)
    _last_observed_at: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must be a non-empty string")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if not 0.0 < self.forget_threshold <= 1.0:
            raise ValueError("forget_threshold must be within (0, 1]")

    @property
    def beliefs(self) -> list[Belief]:
        return list(self._beliefs.values())

    def get(self, track_id: str) -> Belief | None:
        return self._beliefs.get(track_id)

    def __len__(self) -> int:
        return len(self._beliefs)

    def update(self, scene: Scene) -> UpdateReport:
        """Fold one `Scene` into the store.

        Every object in `scene.objects` creates or refreshes a belief
        (`new`/`refreshed`), replacing confidence and position outright —
        the same "add replaces, doesn't combine" rule working memory uses.

        Every *other* existing belief either:
        - is **contradicted** (removed, reported in `contradicted`): its
          position falls inside a region in `scene.coverage` but it wasn't
          observed — the adapter claims to have looked there and found
          nothing, real negative evidence; or
        - **decays** (survives silently, or is removed and reported in
          `forgotten` once its confidence, decayed for the elapsed time
          since `last_observed_at`, drops below `forget_threshold`) —
          absence of evidence, not evidence of absence.
        Contradiction is checked first and wins: a belief that is both
        covered-and-absent and would also have decayed past the threshold
        is reported as contradicted, not forgotten — explicit negative
        evidence is stronger than passive decay.

        A belief with no `position` can never be contradicted (nothing to
        check against `coverage`), only decayed. `scene.observed_at` must
        not decrease between calls, and a rejected call changes nothing.
        Any object or coverage region whose `frame` doesn't match this
        model's `frame` raises, before anything is mutated.
        """
        if self._last_observed_at is not None and scene.observed_at < self._last_observed_at:
            raise ValueError("scene.observed_at must not decrease between calls")
        for percept in scene.objects:
            if percept.position is not None and percept.frame != self.frame:
                raise ValueError(f"object frame {percept.frame!r} does not match world model frame {self.frame!r}")
        for region in scene.coverage:
            if region.frame != self.frame:
                raise ValueError(f"coverage frame {region.frame!r} does not match world model frame {self.frame!r}")

        self._last_observed_at = scene.observed_at
        seen_ids = {percept.track_id for percept in scene.objects}

        new: list[Belief] = []
        refreshed: list[Belief] = []
        for percept in scene.objects:
            belief = Belief(percept=percept, confidence=percept.confidence, last_observed_at=scene.observed_at)
            (refreshed if percept.track_id in self._beliefs else new).append(belief)
            self._beliefs[percept.track_id] = belief

        contradicted: list[Belief] = []
        forgotten: list[Belief] = []
        for track_id, belief in list(self._beliefs.items()):
            if track_id in seen_ids:
                continue
            if _is_contradicted(belief, scene.coverage):
                contradicted.append(belief)
                del self._beliefs[track_id]
            elif belief.confidence_at(scene.observed_at, self.decay_rate) < self.forget_threshold:
                forgotten.append(belief)
                del self._beliefs[track_id]

        return UpdateReport(
            new=tuple(new), refreshed=tuple(refreshed),
            contradicted=tuple(contradicted), forgotten=tuple(forgotten),
        )


def _is_contradicted(belief: Belief, coverage: tuple[CoverageRegion, ...]) -> bool:
    position = belief.percept.position
    if position is None:
        return False
    return any(_distance(position, region.center) <= region.radius for region in coverage)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
