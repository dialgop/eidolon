# Episodic Memory Lab

## What this implements

An unbounded store of discrete episodes — `content` + an `occurred_at`
timestamp (episodic memory's own logical clock) + required `context` and
`provenance` dicts + an optional `embedding` — with cued recall by content
(exact-match), by context/provenance (subset-match), or by similarity
(`recall_by_similarity`, ranked, via `eidolon.labs.representation`).

Unlike `working_memory`, this lab is not self-sufficient: it doesn't decide
what's worth remembering. It only stores what it's told to, via `encode()`.
The interesting design decision here is *what* calls `encode()` and *when*.

## Ingestion path: consolidation from working memory

v0 consolidates from `working_memory`: whenever an item leaves the working
memory buffer — evicted for capacity (`WorkingMemory.add`'s return value) or
forgotten by decay (`WorkingMemory.tick`'s return value) — it's a candidate
for `EpisodicMemory.encode()`. **Both** paths are consolidated in v0, not
just decay-based forgetting; see `demo.py`.

`EpisodicMemory.encode()` never receives a working_memory `Item` directly.
It takes plain fields so the two labs don't share a type or a clock — the
caller doing the consolidating (here, `demo.py`) pulls `item.content` out of
a `working_memory.Item` and supplies `context`/`provenance` itself;
`occurred_at` is stamped by episodic memory, not passed in.

## Design notes

- **v0 consolidates everything, unconditionally — no selectivity in
  encoding.** Every item that leaves working memory gets encoded, whether
  it was meaningful or not. This is deliberate, not an oversight: deciding
  *what's worth remembering* is attention's and goals' job, and neither
  exists yet. v0's only selectivity is in *retrieval* (recall cues), not in
  what gets stored. See the `TODO` in `EpisodicMemory.encode()`.
- **`context` and `provenance` are separate, both required with no
  default.** `context` is meant to be the episodic context — what else was
  going on when the episode occurred (place, other perceptions, ...).
  `provenance` is *why the episode exists* — consolidation bookkeeping like
  `{"reason": "decay_forgotten"}`. Conflating the two would make `context`
  useless for anything but consolidation metadata. Since v0 has no
  attention/perception to generate real situational context yet, callers
  currently pass `context={}` and put what they actually have into
  `provenance` (see `demo.py`) — but the fields stay separate so `context`
  is ready to carry real content once something produces it, without a
  later rename/reshape.
- **`occurred_at` comes from episodic memory's own logical clock**, bumped
  once per `encode()` call — not wall-clock time, and not borrowed from
  `working_memory`'s clock (the two labs share neither a type nor a clock).
  This keeps recall order deterministic and tests reproducible. If the
  agent later has one global clock, this may get replaced by a caller-
  supplied agent timestamp instead — deferred until that clock exists.
- **`occurred_at` is encoding time, not event time.** Consolidation happens
  after an item has already left working memory, so this timestamp is
  consistently later than when the thing actually happened — that's fine
  for v0, but not "when it occurred" in a strict sense. Once working memory
  can supply an event timestamp, this should split into `occurred_at`
  (event) and `encoded_at` (consolidation). Relatedly: `newest_first` on
  the `recall_by_*` methods sorts by *encoding* order, not by `occurred_at`
  — the two happen to coincide today only because `occurred_at` *is* the
  encoding clock.
- **`context`/`provenance` copying is shallow, not deep** — a known v0
  limitation, not a fix pending. `encode()` copies the top-level dict, so a
  caller mutating a *nested* dict/list value after the call still reaches
  the stored episode (top-level key reassignment doesn't, but nested
  mutation does). Deferred: deep-copying is real cost for no v0 benefit,
  since `context` is currently always `{}` anyway. Revisit once a lab
  actually needs nested context.
- **`Episode` is frozen, and `context`/`provenance` are read-only.**
  `encode()` copies both dicts and wraps them in `MappingProxyType` before
  storing, so reassigning a top-level key in the caller's original dict
  afterward, or trying to mutate `episode.context`/`.provenance` directly,
  cannot change what's stored (`TypeError` on the latter). A memory
  shouldn't be modifiable in place — modulo the shallow-copy limitation on
  nested values noted above.
- **`embedding: Embedding | None = None` is optional, unlike `context`/
  `provenance`.** Those two are required with no default to force callers
  to be explicit about why an episode exists; `embedding` is just data that
  may or may not be available (not every episode is consolidated from
  something perceived), so a default makes sense here where it didn't
  there. It's stored as given — `Embedding` is already immutable, nothing
  to copy — and it's excluded from `Episode`'s equality (`compare=False`,
  same reasoning as `PerceivedObject`: an episode's identity doesn't depend
  on its embedding). One side effect worth knowing: `Episode` was already
  unhashable (`context`/`provenance` are `MappingProxyType`, which isn't
  hashable), so this changes nothing there — nothing needs to hash an
  `Episode`; `recall_by_similarity`'s `Match.key` is opaque and never
  hashed.
- **`recall_by_context`/`recall_by_provenance` are subset matches, not
  equality.** An episode matches a cue if the cue's key/value pairs are a
  subset of the episode's — so a cue with one field matches episodes whose
  context/provenance has that field plus others, and `cue={}` matches
  everything. Equality-only matching would make these fields nearly
  useless for any context with more than one field.
- **Recall order:** the exact/subset `recall_by_*` methods (`content`,
  `context`, `provenance`) return results in encoding order (oldest first)
  by default; pass `newest_first=True` to reverse it. `recall_by_similarity`
  is different — see below.
- **v1 (similarity-based recall) is implemented: `recall_by_similarity`,
  via `eidolon.labs.representation`.** This completes the two-phase roadmap
  from `/docs/architecture.md` (v0 exact/subset-match, v1 similarity,
  added alongside the existing methods, not replacing them). `episodic_memory`
  now depends on `representation`, alongside its existing dependency on
  `working_memory` (consolidation).
  - `recall_by_similarity(cue: Embedding, k, min_similarity=None, *,
    newest_first=False) -> list[Match]` — imports `Match`/`most_similar`
    from `representation` rather than reimplementing ranking, `k`,
    `min_similarity` or cross-space validation. `cue` is an `Embedding`
    directly, not an `Episode` — simpler and keeps the two labs decoupled.
    `Match.key` is the matched `Episode`.
  - Episodes with no `embedding` are skipped, not an error — not every
    episode has one. A candidate from a different space (or dimension)
    raises `ValueError`, same as `representation`.
  - **This method ranks; the others filter.** Its default order is
    therefore similarity (most similar first), not encoding time — applying
    "oldest first" as the primary order would defeat the reason it exists.
    Ties (equally similar episodes) are broken by encoding order, and
    `newest_first` reverses *only that tie-break*, not the overall ranking:
    a more similar, older episode still outranks a less similar, newer one
    regardless of `newest_first`. See `demo.py` for both tie-break
    directions side by side.
- **Deliberately out of scope for v0:** `action`, `outcome`, `keys`
  (multi-field cue indexing). These aren't stubbed out as unused fields —
  they're not part of `Episode` at all yet, added only when a concrete lab
  needs them.

## Question this experiment asks

If encoding is unconditional (everything that leaves working memory gets
stored) and retrieval is cued (exact-match on content, subset-match on
context/provenance), is that enough to demonstrate a working two-store
model — short-term items that get bumped or decay, and a long-term store
that never forgets what it was handed — before any selectivity or
generalization exists?

## Run it

```bash
python -m eidolon.labs.episodic_memory.demo
```

## Files

- `store.py` — `EpisodicMemory` and `Episode`; imports `Match`/`most_similar`
  from `eidolon.labs.representation` for `recall_by_similarity`.
- `demo.py` — consolidates both capacity-evicted and decay-forgotten items
  from a `WorkingMemory`, then recalls by content and by provenance, then a
  `recall_by_similarity` scenario with a tie broken both ways.
- `tests/labs/episodic_memory/` — unit tests for the same behaviors,
  including `test_recall_by_similarity.py`.
