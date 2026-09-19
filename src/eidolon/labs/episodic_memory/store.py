"""A minimal episodic memory store: unbounded, unfiltered encoding with
exact-match cued recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Episode:
    """`occurred_at` is v0's encoding time (when `encode()` ran), not event
    time — consolidation happens after an item already left working memory,
    so this timestamp is consistently later than when the thing actually
    happened. Once working memory can supply an event timestamp, this will
    need to split into `occurred_at` (event) and `encoded_at`
    (consolidation); until then, "when it occurred" and "when it was
    archived" are conflated on purpose, not by oversight.

    `context`/`provenance` are shallow-copied at `encode()` time: mutating a
    nested dict/list inside a `context`/`provenance` value you passed in
    *will* still change what's stored (top-level keys are protected, deeper
    structure isn't). Known v0 limitation — deep-copying is deferred until
    a lab actually needs nested context; see `EpisodicMemory.encode`.
    """

    content: Any
    occurred_at: int
    context: MappingProxyType
    provenance: MappingProxyType


@dataclass
class EpisodicMemory:
    _episodes: list[Episode] = field(default_factory=list, init=False)
    _clock: int = field(default=0, init=False)

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def encode(
        self,
        content: Any,
        context: dict[str, Any],
        provenance: dict[str, Any],
    ) -> Episode:
        """Store an episode.

        `occurred_at` is stamped from episodic memory's own logical clock
        (advanced once per `encode()` call) — not wall-clock time, and not
        borrowed from `working_memory`'s clock; the two labs share neither.

        `context` and `provenance` are both required, with no default:
        `context` is the episodic context (what else was going on), while
        `provenance` is *why* this episode exists (e.g. how it got
        consolidated) — kept separate so "why we stored this" doesn't leak
        into "what was happening when it occurred". In v0, with no
        attention/goals to generate real situational context, callers
        typically pass `context={}` and put what they have into
        `provenance`.

        Both dicts are shallow-copied and stored as read-only
        (`MappingProxyType`), so later mutating the caller's original dict's
        top-level keys, or attempting to mutate the returned `Episode`'s
        `context`/`provenance` directly, cannot change what's stored — a
        memory isn't modifiable in place. This protection is shallow,
        though: mutating a nested dict/list *value* inside `context`/
        `provenance` after the call still reaches the stored episode. See
        `Episode`'s docstring.

        TODO: v0 consolidates everything unconditionally — there is no
        selectivity in encoding. Selectivity in *encoding* will come once
        attention/goals labs exist to decide what's worth remembering;
        for now, selectivity only exists in *retrieval*, via recall cues.
        """
        self._clock += 1
        episode = Episode(
            content=content,
            occurred_at=self._clock,
            context=MappingProxyType(dict(context)),
            provenance=MappingProxyType(dict(provenance)),
        )
        self._episodes.append(episode)
        return episode

    def recall_by_content(self, cue: Any, *, newest_first: bool = False) -> list[Episode]:
        """Exact-match recall (v0): episodes whose content equals `cue`.

        Results are in encoding order (oldest first) unless `newest_first`
        reverses it. That's encoding order, not `occurred_at` order — today
        the two coincide because `occurred_at` *is* the encoding clock, but
        once `occurred_at`/`encoded_at` split (see `Episode`'s docstring),
        `newest_first` will still mean "most recently encoded", not "most
        recent event"; event-ordered recall would need its own parameter.
        """
        results = [e for e in self._episodes if e.content == cue]
        return list(reversed(results)) if newest_first else results

    def recall_by_context(self, cue: dict[str, Any], *, newest_first: bool = False) -> list[Episode]:
        """Episodes whose `context` *contains* `cue` (`cue` is a subset of
        it, not an exact match) — so `cue={}` matches every episode.

        Result order: see `recall_by_content` — encoding order (oldest
        first) unless `newest_first`, not `occurred_at` order.
        """
        return self._recall_by_subset(cue, "context", newest_first)

    def recall_by_provenance(self, cue: dict[str, Any], *, newest_first: bool = False) -> list[Episode]:
        """Episodes whose `provenance` *contains* `cue` (subset match, same
        semantics as `recall_by_context`, applied to provenance instead —
        including result order; see `recall_by_content`)."""
        return self._recall_by_subset(cue, "provenance", newest_first)

    def _recall_by_subset(self, cue: dict[str, Any], field_name: str, newest_first: bool) -> list[Episode]:
        results = [e for e in self._episodes if cue.items() <= getattr(e, field_name).items()]
        return list(reversed(results)) if newest_first else results
