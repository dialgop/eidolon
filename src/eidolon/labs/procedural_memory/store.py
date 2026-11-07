"""Skill learning and context-sensitive action selection.

Two pillars, kept honest about which is sourced and which isn't:

- The power law of practice (Newell & Rosenbloom, 1981): execution speed
  improves as a power function of practice count. `curve()` implements the
  *shape* of this, not the full four-parameter model (see its docstring).
- A Luce choice rule (Luce, 1959) over accumulated, context- and time-
  weighted "strength" selects among competing skills — the same move
  `semantic_memory` makes with Nosofsky's Eq. 5, but the strength
  mechanism itself (not the choice rule) is our own design, not sourced:
  see the module-level "Design notes" in the README for why.

No internal clock: `practice()`'s `timestamp` and `strength()`/`select()`'s
`now` are caller-supplied, like `world_model`/`visual_attention`, not an
internal counter like `working_memory`/`episodic_memory`. Unit:
milliseconds, the same as `world_model`'s and `self_model`'s — a single,
coherent convention across every caller-supplied-timestamp lab rather than
three separately-decided ones, since a real adapter (one robot, one
Webots) is expected to feed all three. Source: caller-supplied; the
adapter and every consumer must share one monotonic clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from eidolon.labs.representation import similarity
from eidolon.percepts import Embedding


@dataclass(frozen=True)
class Practice:
    """One practice event. `context` is optional: a context-free practice
    (drilled abstractly, not tied to a situation) contributes to every
    contextual strength query without being weighted by context similarity
    — see `ProceduralMemory.strength`.

    `timestamp`: unit is milliseconds, on a monotonic clock shared with
    the adapter and whoever calls `strength`/`select` (see module
    docstring)."""

    context: Embedding | None
    timestamp: int


@dataclass(frozen=True)
class SkillMatch:
    skill: str
    strength: float
    probability: float


@dataclass
class ProceduralMemory:
    """All four hyperparameters are required, with no derivable default —
    same reasoning as `semantic_memory`'s `sensitivity`: none of them has a
    value the source papers supply, so a default would invite using it
    unexamined.

    `space`: the single embedding space `context`s must share, fixed per
    instance and validated against every call, like `WorldModel.frame`.
    `alpha`: the power-law exponent for `curve()`.
    `sensitivity`: the Shepard-style exponential scale for context
    similarity (same role as `semantic_memory.sensitivity`, recomputed
    locally here rather than imported — the two labs are peers, not a
    consolidation chain, so neither imports the other's internals).
    `decay_rate`: time decay, in the same `(1 - decay_rate) ** elapsed`
    form `world_model` uses, so `decay_rate` means the same thing in both
    labs rather than two different conventions under one name — and since
    `timestamp`/`now` are also in milliseconds like `world_model`'s
    `observed_at`, `decay_rate` is calibrated against the same real unit
    in both, not just the same formula shape.
    """

    space: str
    alpha: float
    sensitivity: float
    decay_rate: float

    _practices: dict[str, list[Practice]] = field(default_factory=dict, init=False)
    _dim: int | None = field(default=None, init=False)
    _last_timestamp: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("space must be a non-empty string")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be >= 0")

    @property
    def skills(self) -> list[str]:
        return list(self._practices.keys())

    def practice_count(self, skill: str) -> int:
        return len(self._practices.get(skill, ()))

    def __len__(self) -> int:
        """Number of distinct skills known, not total practice events —
        same convention as `semantic_memory.__len__` counting labels, not
        exemplars. See `practice_count` for a skill's own event count."""
        return len(self._practices)

    def practice(self, skill: str, context: Embedding | None, timestamp: int) -> None:
        """Record one practice event: an unseen `skill` name creates it (no
        separate registration step), an existing one just adds to its
        history — mirroring `semantic_memory.learn`'s new-label-vs-
        existing-label behavior exactly, down to there being no `Skill`
        type to register first.

        `timestamp` must not decrease across calls to `practice()` (the
        only mutating method here; `strength`/`select` are pure functions
        of a given `now` and enforce no ordering of their own, the same
        way `Belief.confidence_at` doesn't). A rejected call stores
        nothing. `context`'s space/dimension are validated against every
        other `context` ever given to this instance (across *all* skills,
        not just this one) if `context` is not `None` — `select()` compares
        contexts across skills, so a mismatch anywhere would break it.
        """
        if not isinstance(skill, str) or not skill:
            raise ValueError("skill must be a non-empty string")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp must not decrease between calls")
        if context is not None:
            self._check_context(context, establish=True)
        self._last_timestamp = timestamp
        self._practices.setdefault(skill, []).append(Practice(context=context, timestamp=timestamp))

    def strength(self, skill: str, context: Embedding | None, now: int) -> float:
        """Summed, time- and context-weighted evidence for `skill` — not
        nearest-practice, the full history each time, same "keep every
        instance, sum over all of them" shape `semantic_memory` uses for
        concepts. `0.0` for a skill that's never been practiced (not an
        error): absence of practice is just absence of evidence, the same
        convention `semantic_memory.prototype` uses for an unknown label
        (there it's `None`; here the return type is a plain `float`, so
        `0.0` is the natural "nothing" value instead).

        Each stored practice contributes `(1 - decay_rate) ** (now -
        practice.timestamp)` (time weight; the same form
        `Belief.confidence_at` uses, so `decay_rate` means the same thing
        in both labs) times a context weight: `1.0` if either `context` or
        the practice's own context is `None` (a context-free practice, or a
        context-free query, carries no context-specificity information, so
        it neither helps nor hurts — it contributes fully regardless of
        context), otherwise `exp(-sensitivity * (1 - similarity(context,
        practice.context)))` — Shepard's exponential gradient over cosine
        similarity, the same construction `semantic_memory._eta` uses for
        concepts, written locally rather than imported for the same
        peer-labs reason noted on the class.
        """
        if context is not None:
            self._check_context(context, establish=False)
        total = 0.0
        for p in self._practices.get(skill, ()):
            time_weight = (1.0 - self.decay_rate) ** (now - p.timestamp)
            if context is None or p.context is None:
                context_weight = 1.0
            else:
                context_weight = self._eta(context, p.context)
            total += time_weight * context_weight
        return total

    def select(
        self, context: Embedding, now: int, min_probability: float | None = None
    ) -> SkillMatch | None:
        """The single most probable skill for `context` right now, via a
        Luce choice rule (Luce, 1959) over every known skill's `strength`:
        competition between similar, comparably-practiced skills is a
        direct consequence of this normalization, not a separate
        interference mechanism bolted on. Ties keep skill-insertion order
        (first practiced), the same convention used throughout this
        project.

        `None` if no skill has ever been practiced, or if every skill's
        strength is zero for this `context`/`now` (nothing to normalize
        against). `min_probability` is a rejection *threshold* on the
        winner already computed, not a candidate filter — exactly
        `semantic_memory.classify`'s contract: it can never change which
        skill would have won, only whether it's reported.
        """
        self._check_context(context, establish=False)
        raw = {skill: self.strength(skill, context, now) for skill in self._practices}
        total = sum(raw.values())
        if total <= 0.0:
            return None
        best_skill = max(raw, key=raw.get)
        probability = raw[best_skill] / total
        if min_probability is not None and probability < min_probability:
            return None
        return SkillMatch(skill=best_skill, strength=raw[best_skill], probability=probability)

    def curve(self, skill: str) -> list[tuple[int, float]]:
        """`(N, relative_speed)` pairs for `N` = 0 .. this skill's actual
        practice count, where `relative_speed = (N + 1) ** (-alpha)`: 1.0
        at N=0, decreasing toward 0 as N grows. This is the *shape* of
        Newell & Rosenbloom's power law of practice, not their full model
        — see the README for exactly what's dropped and why `(N+1)` is our
        own choice, not theirs. A nonexistent or unpracticed skill gets
        just `[(0, 1.0)]`.
        """
        return [(n, (n + 1) ** (-self.alpha)) for n in range(self.practice_count(skill) + 1)]

    def _eta(self, a: Embedding, b: Embedding) -> float:
        return math.exp(-self.sensitivity * (1.0 - similarity(a, b)))

    def _check_context(self, context: Embedding, *, establish: bool) -> None:
        """`establish=True` (only from `practice()`) lets this be the call
        that fixes `_dim` from real stored data, if nothing has yet. A
        pure query (`establish=False`) only validates against a dimension
        already on record — it must never set one itself, or querying
        `strength`/`select` before any `practice()` call would silently
        fix the dimension from a throwaway query context instead of from
        actual data, which is exactly the kind of implicit, order-
        dependent state `world_model`'s decay bug taught us to avoid."""
        if context.space != self.space:
            raise ValueError(f"context space {context.space!r} does not match {self.space!r}")
        if self._dim is None:
            if establish:
                self._dim = context.dim
        elif context.dim != self._dim:
            raise ValueError(f"context dim {context.dim} does not match existing dim {self._dim}")
