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

from eidolon.labs.representation import most_similar
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
class Reidentification:
    """One re-identification event: a previously unseen `new_track_id` was
    folded into the belief originally created under `canonical_track_id`,
    at the given appearance `similarity`. Kept separate from a plain
    refresh — this is a different claim (inferred identity across a track
    id change, not an exact id match)."""

    new_track_id: str
    canonical_track_id: str
    similarity: float


@dataclass(frozen=True)
class UpdateReport:
    """What one `WorldModel.update()` call changed, grouped like
    `WorkingMemory.tick()`'s return rather than left for the caller to diff
    two snapshots."""

    new: tuple[Belief, ...] = ()
    refreshed: tuple[Belief, ...] = ()
    reidentified: tuple[Reidentification, ...] = ()
    contradicted: tuple[Belief, ...] = ()
    forgotten: tuple[Belief, ...] = ()


@dataclass
class WorldModel:
    """`frame` is the single coordinate frame this store holds beliefs in;
    it does no transforms itself (see the lab README). `decay_rate` and
    `forget_threshold` govern how an unobserved belief's confidence fades;
    see `update()` for how.

    `reidentify_threshold`/`reidentify_max_distance` enable re-identifying
    a new `track_id` as a previously tracked, currently unobserved object —
    off by default (both `None`). Enabling one without the other raises:
    appearance alone is not sufficient evidence (see the lab README);
    position must corroborate it.
    """

    frame: str
    decay_rate: float = 0.1
    forget_threshold: float = 0.05
    reidentify_threshold: float | None = None
    reidentify_max_distance: float | None = None

    _beliefs: dict[str, Belief] = field(default_factory=dict, init=False)
    _aliases: dict[str, str] = field(default_factory=dict, init=False)
    _last_observed_at: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must be a non-empty string")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")
        if not 0.0 < self.forget_threshold <= 1.0:
            raise ValueError("forget_threshold must be within (0, 1]")
        if (self.reidentify_threshold is None) != (self.reidentify_max_distance is None):
            raise ValueError("reidentify_threshold and reidentify_max_distance must be given together")
        if self.reidentify_threshold is not None and not -1.0 <= self.reidentify_threshold <= 1.0:
            raise ValueError("reidentify_threshold must be within [-1, 1]")
        if self.reidentify_max_distance is not None and self.reidentify_max_distance <= 0:
            raise ValueError("reidentify_max_distance must be > 0")

    @property
    def beliefs(self) -> list[Belief]:
        return list(self._beliefs.values())

    def get(self, track_id: str) -> Belief | None:
        """Resolves through re-identification aliases: both the id a belief
        was first created under and any later id folded into it via
        re-identification return the same `Belief`."""
        return self._beliefs.get(self._aliases.get(track_id, track_id))

    def __len__(self) -> int:
        return len(self._beliefs)

    def update(self, scene: Scene) -> UpdateReport:
        """Fold one `Scene` into the store.

        Processing happens in three passes, in this order, and the order is
        load-bearing (see the "contradiction/forgetting runs before
        re-identification" note below) — not an implementation detail:

        1. Every object in `scene.objects` whose `track_id` (or a
           previously established alias of it) is already a known belief
           **refreshes** it (`refreshed`): confidence and position are
           replaced outright, the same "add replaces, doesn't combine" rule
           working memory uses.
        2. Every *other* existing belief (not just refreshed) either:
           - is **contradicted** (removed, reported in `contradicted`): its
             position falls inside a region in `scene.coverage` but it
             wasn't observed — the adapter claims to have looked there and
             found nothing, real negative evidence; or
           - **decays** (survives silently, or is removed and reported in
             `forgotten` once its confidence, decayed for the elapsed time
             since `last_observed_at`, drops below `forget_threshold`) —
             absence of evidence, not evidence of absence.
           Contradiction is checked first and wins: a belief that is both
           covered-and-absent and would also have decayed past the
           threshold is reported as contradicted, not forgotten — explicit
           negative evidence is stronger than passive decay.
        3. Every remaining, still-unrecognized object is either
           **re-identified** (`reidentified`) as one of the beliefs that
           survived pass 2, by appearance and position together (see
           below), folding its `track_id` in as an alias of that belief
           rather than creating a duplicate; or becomes a **new** belief
           (`new`), if re-identification is off, found no candidate, or had
           nothing (no embedding/position) to compare.

        **Re-identification runs after contradiction/forgetting, not
        before — the order matters.** A belief that's gone, by either kind
        of evidence, must not be offered as a re-identification candidate:
        contradiction already outranks passive decay, and it outranks
        re-identification the same way (an object explicitly confirmed
        absent from a location can't simultaneously be "the same thing" as
        something freshly spotted nearby). Running pass 2 first removes
        such beliefs from the store outright, so pass 3 simply can't see
        them — not a separate check duplicating pass 2's logic.

        Among candidates that did survive pass 2 and aren't already claimed
        this call, `reidentify_threshold`/`reidentify_max_distance` (both
        required together, off by default) find the single most
        appearance-similar one (via
        `eidolon.labs.representation.most_similar`) with a `position`
        within `reidentify_max_distance` and an `embedding`, at or above
        `reidentify_threshold`. Ties — e.g. two indistinguishable objects —
        go to whichever belief was created first (known limitation, not
        resolved: appearance here is "same kind of thing", not "same
        individual thing"; see the lab README). A belief matched this way
        cannot also be matched by another object in the same call. `get()`
        resolves either `track_id` to the same `Belief`, whose own
        `track_id` becomes the new one — the dict key this belief was first
        created under stays stable, but is not itself exposed; see the
        README for this distinction.

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

        refreshed: list[Belief] = []
        claimed: set[str] = set()
        unresolved: list[PerceivedObject] = []

        for percept in scene.objects:
            canonical = self._aliases.get(percept.track_id, percept.track_id)
            if canonical in self._beliefs:
                belief = Belief(percept=percept, confidence=percept.confidence, last_observed_at=scene.observed_at)
                self._beliefs[canonical] = belief
                refreshed.append(belief)
                claimed.add(canonical)
            else:
                unresolved.append(percept)

        # Contradiction/forgetting runs before re-identification, not after:
        # a belief that is gone — by either kind of evidence — must not be
        # offered as a re-id candidate. Running this pass first removes it
        # from `_beliefs` outright, so re-identification needs no separate
        # "is this candidate still alive" check of its own; it simply can't
        # see what no longer exists.
        contradicted: list[Belief] = []
        forgotten: list[Belief] = []
        for track_id, belief in list(self._beliefs.items()):
            if track_id in claimed:
                continue
            if _is_contradicted(belief, scene.coverage):
                contradicted.append(belief)
                del self._beliefs[track_id]
            elif belief.confidence_at(scene.observed_at, self.decay_rate) < self.forget_threshold:
                forgotten.append(belief)
                del self._beliefs[track_id]

        new: list[Belief] = []
        reidentified: list[Reidentification] = []
        for percept in unresolved:
            match = self._find_reidentification_match(percept, claimed)
            belief = Belief(percept=percept, confidence=percept.confidence, last_observed_at=scene.observed_at)
            if match is not None:
                canonical, similarity = match
                self._beliefs[canonical] = belief
                self._aliases[percept.track_id] = canonical
                reidentified.append(Reidentification(percept.track_id, canonical, similarity))
                claimed.add(canonical)
            else:
                self._beliefs[percept.track_id] = belief
                new.append(belief)
                claimed.add(percept.track_id)

        return UpdateReport(
            new=tuple(new), refreshed=tuple(refreshed), reidentified=tuple(reidentified),
            contradicted=tuple(contradicted), forgotten=tuple(forgotten),
        )

    def _find_reidentification_match(
        self, percept: PerceivedObject, claimed: set[str]
    ) -> tuple[str, float] | None:
        """Candidates are drawn from `self._beliefs` *after* the
        contradiction/forgetting pass has already run and removed whatever
        didn't survive it — so no separate staleness check is needed here:
        anything still in `_beliefs` is, by construction, still live."""
        if self.reidentify_threshold is None or percept.embedding is None or percept.position is None:
            return None
        candidates = []
        for track_id, belief in self._beliefs.items():
            b_percept = belief.percept
            if (
                track_id in claimed
                or b_percept.embedding is None
                or b_percept.position is None
                or _distance(percept.position, b_percept.position) > self.reidentify_max_distance
            ):
                continue
            candidates.append((track_id, b_percept.embedding))
        matches = most_similar(percept.embedding, candidates, k=1, min_similarity=self.reidentify_threshold)
        return (matches[0].key, matches[0].similarity) if matches else None


def _is_contradicted(belief: Belief, coverage: tuple[CoverageRegion, ...]) -> bool:
    position = belief.percept.position
    if position is None:
        return False
    return any(_distance(position, region.center) <= region.radius for region in coverage)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
