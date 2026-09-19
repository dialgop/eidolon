# Episodic Memory Lab

## What this implements

An unbounded store of discrete episodes — `content` + an `occurred_at`
timestamp (episodic memory's own logical clock) + required `context` and
`provenance` dicts — with cued recall by content (exact-match) or by
context/provenance (subset-match).

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
- **`recall_by_context`/`recall_by_provenance` are subset matches, not
  equality.** An episode matches a cue if the cue's key/value pairs are a
  subset of the episode's — so a cue with one field matches episodes whose
  context/provenance has that field plus others, and `cue={}` matches
  everything. Equality-only matching would make these fields nearly
  useless for any context with more than one field.
- **Recall order:** all `recall_by_*` methods return results in encoding
  order (oldest first) by default; pass `newest_first=True` to reverse it.
- **Retrieval is exact-match (for content) / subset-match (for
  context/provenance) in v0**, per the two-phase roadmap in
  `/docs/architecture.md`. Similarity-based recall (v1) is deferred until a
  representation lab exists, and would likely be a `recall_by_similarity`
  method added alongside these, not a replacement for them.
- **Deliberately out of scope for v0:** `action`, `outcome`, `keys`
  (multi-field cue indexing), `embedding`. These aren't stubbed out as
  unused fields — they're not part of `Episode` at all yet, added only when
  a concrete lab needs them.

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

- `store.py` — `EpisodicMemory` and `Episode`.
- `demo.py` — consolidates both capacity-evicted and decay-forgotten items
  from a `WorkingMemory`, then recalls by content and by provenance.
- `tests/labs/episodic_memory/` — unit tests for the same behaviors.
