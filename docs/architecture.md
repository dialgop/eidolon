# Architecture

## Layers

The project develops progressively through several layers, from pure theory
down to an embodied agent:

```text
Theory
  ↓
Cognitive Micro-Laboratories   (Python, src/eidolon/labs/)
  ↓
Cognitive Core                 (Python, src/eidolon/core/ — not yet started)
  ↓
ROS 2 Embodiment Interface     (C++, embodiment/ — reused/adapted from spin_the_bottle)
  ↓
NAO Humanoid Robot
  ↓
Webots Environment
  ↓
Embodied Experiments
```

Early-stage experiments implement individual mechanisms in isolation, as
**cognitive micro-labs** (see `src/eidolon/labs/README.md`):

- Episodic memory
- Working memory
- Attention
- Perception
- World models
- Self-models
- Goals and motivation
- Planning
- Metacognition
- Uncertainty
- Global Workspace
- Predictive processing
- Semantic memory *(added after the original roadmap; implemented — see
  note below)*
- Procedural memory *(added after the original roadmap; implemented — see
  note below)*
- Representation *(added after the original roadmap; implemented — see
  note below)*

Semantic memory (conceptual/categorical knowledge — "what a cup is", not
"the cup I saw") wasn't in the original roadmap above but belongs on it: a
belief store gives an object identity and persistence, not meaning. It was
deliberately deferred, not forgotten, until a representation layer
(embeddings) existed.

**A note on an earlier version of this paragraph:** it said semantic memory
"needs temporal continuity (the world model) to have raw material to
generalize from," which reads like a claim that semantic memory feeds from
`world_model`. It doesn't (see `semantic_memory`'s contract below) — that
line was about a *prerequisite*, not a feed path: re-identification is what
makes repeated sightings of the same real object count reliably as repeated
evidence, not a data source for category learning. `episodic_memory` is the
only store that accumulates many distinct exemplars over time, which is
what Nosofsky's exemplar-sum classification and Rosch's prototypes actually
need — so that's what it's fed from, the same `working_memory →
episodic_memory` pattern one level further along.

Representation (embeddings and similarity over them) was that missing
prerequisite, and is now built. It consumes embeddings; it does not produce
them (the adapter does). All three follow-ups it unlocked are now also
built: `episodic_memory.recall_by_similarity`, `world_model`'s
re-identification, and `semantic_memory` (see their contracts below).

Procedural memory (skill learning and action selection — "how to do it",
not "what it is" or "what happened") wasn't in the original roadmap either,
and wasn't one of representation's three promised follow-ups — it's a new
addition, built once `semantic_memory` had already established the
`episodic_memory → (peer lab)` consolidation pattern and the Luce-choice
machinery this lab reuses via `representation.similarity`. It is a peer of
`semantic_memory`, not downstream of it: both are fed from `episodic_memory`
directly, and neither imports the other's internals.

## Inter-lab interface contracts

Labs are meant to be built in isolation but eventually read from and write
into each other (e.g. episodic memory consolidating what falls out of
working memory; attention selecting what working memory holds). This
section tracks what each lab actually promises to the others, so that
"isolated experiment" doesn't quietly turn into "nobody knows what depends
on what." Update it whenever a lab's public surface changes.

### `eidolon.labs.working_memory`

- **Exposes:** `WorkingMemory(capacity, decay_rate, forget_threshold,
  rehearsal_activation)` and `Item(content, activation, created_at,
  last_accessed)`.
- **Operations:** `add(content) -> (Item, evicted: Item | None)`,
  `rehearse(content) -> Item | None`, `tick() -> list[Item]` (forgotten
  items), `.items -> list[Item]` (read-only snapshot), `len(wm)`.
- **Retrieval semantics:** exact-match only (`Item.content == query`) in v0.
  This is an explicit, reversible scoping decision, not a permanent
  contract — see the "Design notes" in the lab's own README. Consumers
  should not assume `_find`'s matching strategy stays `==` forever, only
  that `add`/`rehearse` take arbitrary `content: Any`.
- **Refresh semantics:** `add` on content that matches an existing item
  refreshes it in place (no duplicate, no eviction) **and replaces its
  `content` with the newly added object**, even if it compares equal: `==`
  may mean identity (e.g. a `PerceivedObject` is equal by `track_id`), and
  `add` carries a new observation. The `Item` object, its buffer position
  and `created_at` are kept. `rehearse` only refreshes activation and keeps
  the stored content. Consumers therefore see the latest content in
  `.items` and in the evicted/forgotten items that `add`/`tick` return.
- **Time semantics:** `tick()` is a logical step, not wall-clock time.
  `decay_rate`/`forget_threshold` are only meaningful relative to how often
  the caller ticks — a consumer driving this from a real control loop (e.g.
  ROS 2 at ~10 Hz) must recalibrate these constants against that tick rate,
  not reuse the lab's demo defaults.
- **Resolved:** `working_memory` stays exact-match-only in v0 and does not
  grow a `get_by_similarity` operation. `episodic_memory` (below) is what
  consolidates working memory's output and does its own retrieval on its
  own store — similarity, if it ever gets added, lands there (v1), not
  here.

### `eidolon.labs.episodic_memory`

- **Exposes:** `EpisodicMemory()` and `Episode(content, occurred_at,
  context, provenance, embedding)` (frozen; `context`/`provenance` are
  `MappingProxyType`, read-only; `embedding` is `Embedding | None`,
  excluded from equality). `Episode` is not hashable (`context`/`provenance`
  are `MappingProxyType`) — not a problem, nothing needs to hash one;
  `recall_by_similarity`'s `Match.key` is opaque and never hashed.
- **Operations:** `encode(content, context, provenance, embedding=None) ->
  Episode`, `recall_by_content(cue, *, newest_first=False) ->
  list[Episode]`, `recall_by_context(cue, *, newest_first=False) ->
  list[Episode]`, `recall_by_provenance(cue, *, newest_first=False) ->
  list[Episode]`, `recall_by_similarity(cue: Embedding, k,
  min_similarity=None, *, newest_first=False) -> list[Match]`,
  `.episodes -> list[Episode]` (read-only snapshot), `len(em)`.
- **New dependency: `episodic_memory → representation`.**
  `recall_by_similarity` imports `Match`/`most_similar` from
  `eidolon.labs.representation` rather than reimplementing ranking, `k`,
  `min_similarity` or cross-space validation — alongside the lab's existing
  `working_memory → episodic_memory` dependency (consolidation).
- **Ingestion path: consolidates from `working_memory`, unconditionally.**
  `EpisodicMemory` never pulls from working memory itself — the caller
  (currently the lab's own `demo.py`) is the consolidation point: it calls
  `encode()` for every item working memory reports as gone, whether via
  `WorkingMemory.add`'s `evicted` return (capacity) or `WorkingMemory.tick`'s
  `forgotten` list (decay). **Both** paths are consolidated, not just
  decay. v0 has no selectivity in encoding — everything that leaves working
  memory gets stored; selectivity is deferred until attention/goals labs
  exist to decide what's worth remembering (see the `TODO` in
  `EpisodicMemory.encode`). v0's only selectivity is in retrieval, via
  recall cues.
- **Type decoupling:** `encode()` takes plain fields, never a
  `working_memory.Item` — the two labs share no type and no clock.
  `occurred_at` is stamped by episodic memory's own logical clock
  (incremented per `encode()` call), not wall-clock time and not working
  memory's tick counter (see working_memory's own "Time semantics" above
  for why the latter wouldn't transfer meaningfully anyway). If the agent
  ever gets one shared/global clock, a caller-supplied agent timestamp may
  replace this — deferred until that clock exists.
- **`occurred_at` is encoding time, not event time**, since consolidation
  happens after an item already left working memory — consistently later
  than the actual event. Will split into `occurred_at` (event) /
  `encoded_at` (consolidation) once working memory can supply an event
  timestamp. `newest_first` on the `recall_by_*` methods sorts by encoding
  order, not `occurred_at` — they only coincide today because `occurred_at`
  *is* the encoding clock.
- **`context`/`provenance` copying is shallow, not deep — known v0
  limitation.** `encode()` copies the top-level dict only; mutating a
  *nested* dict/list value in what the caller passed in still reaches the
  stored episode. Deferred (not fixed) since `context` is always `{}` in
  v0 anyway; revisit once a lab needs nested context.
- **`context` vs. `provenance`, both required, no default.** `context` is
  the episodic context (what else was going on); `provenance` is why the
  episode was stored (consolidation bookkeeping, e.g.
  `{"reason": "decay_forgotten"}`). Kept as separate fields so consolidation
  metadata doesn't crowd out `context`, which v0 currently leaves `{}` (no
  attention/perception yet to populate it) but which should stay usable for
  real situational data once something produces it.
- **`context`/`provenance` are immutable once stored, at the top level.**
  `encode()` copies both and wraps them in `MappingProxyType`; reassigning
  a top-level key in the caller's original dict afterward, or mutating the
  returned `Episode`'s fields directly, cannot change what's stored —
  modulo the shallow-copy limitation on nested values noted above.
- **Retrieval semantics:** `recall_by_content` is exact-match (`==`).
  `recall_by_context`/`recall_by_provenance` are **subset**-match — a cue
  matches if its key/value pairs are a subset of the episode's field, so
  `cue={}` matches everything and multi-field contexts aren't reduced to
  exact-equality-or-nothing. **`recall_by_similarity` (v1, implemented)**
  completes the two-phase roadmap decided for this lab: `cue` is an
  `Embedding` directly (not an `Episode` — simpler, keeps the labs
  decoupled); episodes with no `embedding` are skipped, not an error; a
  candidate from a different space or dimension raises `ValueError` (via
  `most_similar`); `k` is a maximum and `min_similarity` is inclusive, both
  passed straight through.
- **Recall order — one exception.** The exact/subset `recall_by_*` methods
  return oldest-first (encoding order) by default; `newest_first=True`
  reverses it. **`recall_by_similarity` ranks instead of filtering**, so its
  default order is similarity (most similar first), not encoding time —
  "oldest first" as the primary order would defeat the method's purpose.
  Ties (equal similarity) are broken by encoding order; `newest_first`
  there reverses only that tie-break, never the ranking itself (a more
  similar, older episode still outranks a less similar, newer one either
  way).
- **Deliberately out of scope for v0:** `action`, `outcome`, `keys`
  (multi-field cue indexing). Not stubbed as unused fields — simply not
  part of `Episode` yet.

### `eidolon.percepts`

The perception input contract: what an embodiment/adapter hands to
Eidolon. Eidolon stays a pure cognition library — it never sees pixels,
sensors or a robot; an adapter (scripted scenes today, Webots/NAO later)
reports what it perceives as these structures, and it does not know where
objects come from.

Originally lived inside `visual_attention` (`types.py`); extracted to this
top-level package once `world_model` needed the same types — the Rule of
Three, not a speculative abstraction — with **no compatibility shim**:
every import site was updated directly (`visual_attention`'s code, tests
and demo), and the full test suite was re-run as the check for hidden
coupling.

- **Exposes:** `PerceivedObject`, `Scene`, `CoverageRegion`, `Embedding`.
- **`PerceivedObject`** (frozen): `track_id` (required, non-empty),
  `observed_at: int`, and optional `label`, `confidence` in `[0, 1]`, flat
  `features: Mapping[str, float]` (read-only), `position` + `frame`
  (`frame` required when `position` is given), `source`, and `embedding`.
  **Equality and hashing use `track_id` only** (the embedding takes no part).
  `position` is unused by v0 attention.
- **`Embedding`** (frozen): `vector: tuple[float, ...]` and `space: str`, plus
  a `dim` property. A point in a learned space produced by the adapter (e.g.
  CLIP on a GPU); Eidolon consumes embeddings and never produces them. `space`
  names the model and version, and embeddings are only comparable within one
  space, so a consumer comparing across spaces must raise. A list raises
  `TypeError`; an empty vector, non-finite components, an all-zero vector
  (cosine is undefined) or an empty `space` raise `ValueError`. It is a
  separate `PerceivedObject` field, not a feature, because it has its own
  semantics; a bare tuple there raises `TypeError`.
- **`CoverageRegion`** (frozen): `center` (3-tuple), `radius` (`> 0`),
  `frame` (required, non-empty). A region of space an adapter claims to
  have observed — the basis for negative evidence (see `world_model`
  below). Its `frame` follows the same rule as `PerceivedObject.frame`: a
  consumer working in a different frame must raise, not guess.
- **`Scene`** (frozen): `observed_at: int`, `objects: tuple[...]`, and
  `coverage: tuple[CoverageRegion, ...] = ()`. Both tuple fields raise
  `TypeError` on a list — no mutable handle survives construction.
  `track_id`s must be unique within a scene. `coverage` defaults to
  `()`: a `Scene` with no coverage claimed carries no negative evidence,
  which is a conscious limitation for whichever lab reads it, not a
  default to fill in casually.

### `eidolon.labs.visual_attention`

The first consumer of `eidolon.percepts`, deciding which perceived objects
get selected.

```text
adapter (outside Eidolon) → Scene → VisualAttention.select() → caller adds selected to WM
```

`select` returns its choice; it never writes to working memory. The caller
does, the same pattern as episodic consolidation, so labs stay decoupled.

- **Exposes:** `VisualAttention`, `Selection`, `learn_weights` (not
  `PerceivedObject`/`Scene` — import those from `eidolon.percepts`).
- **Operations:**
  `VisualAttention(suppression_duration=4, peak_ratio=0.5).select(scene, k,
  t=0.0, weights=None) -> list[Selection]` and
  `learn_weights(training_scene, target_track_id) -> Mapping[str, float]`
  (raises `ValueError` for an unknown target, a scene with no other object,
  or a target with no features). `Selection` is `(percept, salience)`.
- **Selection semantics:** salience is `(1 - t)·bottom_up + t·top_down`,
  `t` in `[0, 1]` (`ValueError` outside), each map normalized by its
  maximum. `t=0` is pure bottom-up (goal-free exploration) and `t=1` pure
  top-down (search with learned `weights`, required when `t > 0`). `k` is a
  maximum: up to `k` objects, most salient first, ties in scene order.
- **Inhibition of return, two parts.** Within a scene it is just the ranked
  top-`k` (select, inhibit, repeat). Across frames it is our own extension:
  everything `select` returns is excluded from later calls for
  `suppression_duration` units of `Scene.observed_at`. Suppressed objects
  still count as context for the others' contrast.
- **Time semantics:** `VisualAttention` has no clock; it uses the caller's
  `Scene.observed_at`, and `select` never advances one. Durations are in the
  adapter's `observed_at` units. `observed_at` must not decrease between
  calls (`ValueError`, and the rejected call changes nothing).
- **Sources:** "inspired by" Frintrop, Backer & Rome (DAGM 2005) for the
  two-mode framework, and Frintrop, Werner & García (CVPR 2015) for
  center-surround contrast only. The object-level adaptation and the parts
  that are ours (normalization, peak definition, weight clamping, cross-frame
  inhibition of return) are listed in the lab's README.
- **Working memory keeps the latest observation.** Adding a selected
  `PerceivedObject` again (same `track_id`, newer position/confidence)
  replaces the content held in working memory rather than keeping the stale
  one; see working_memory's "Refresh semantics" above.
- **Deferred:** several training scenes (geometric mean of weights),
  search by label, use of `position`, a salience threshold.

### `eidolon.labs.world_model`

A belief store: persists beliefs about objects' existence and last known
state, whether or not they're currently attended.

```text
adapter → Scene ─┬→ VisualAttention.select() → caller → WM → (consolidation) → EM
                 └→ WorldModel.update(scene)
```

**Scene provides existence; working memory (and, later, semantic memory)
provide focus and interpretation, not existence.** `WorldModel.update()`
takes a `Scene` directly, never `working_memory`: working memory holds
~4 attended items and forgets on its own rehearsal-driven schedule, so
feeding beliefs from it would conflate "not currently attended" with "not
believed to exist" and shrink object permanence to attention span. When
focus/interpretation signals from working memory or `semantic_memory`
(exists now, but no such integration is built — still deferred) are
integrated, they modulate an *existing* belief's confidence or
interpretation — never create or delete one. `WorldModel` does not write to
`episodic_memory` either; consolidation
stays `working_memory → episodic_memory`, unchanged. This makes
`world_model` a third, independent store with its own dynamics, not a
derivative of the other two.

v0 is a **belief store**, not a **predictor** (state + action → next
state, à la Ha & Schmidhuber); a predictor needs actions, which don't exist
yet, and is deferred.

- **Exposes:** `WorldModel`, `Belief`, `UpdateReport`, `Reidentification`.
- **`Belief`** (frozen, like `Episode` — not mutated in place like working
  memory's `Item`): `percept: PerceivedObject`, `confidence` in `[0, 1]`,
  `last_observed_at: int`, plus `track_id`/`label` properties reading
  through to `percept`, and `confidence_at(observed_at, decay_rate)` — a
  pure computation, reads nothing stored, mutates nothing.
- **`UpdateReport`** (frozen): `new`, `refreshed`, `reidentified`,
  `contradicted`, `forgotten` — `tuple[Belief, ...]` except `reidentified`,
  which is `tuple[Reidentification, ...]` — grouped like
  `WorkingMemory.tick()`'s return rather than left for the caller to diff
  two snapshots.
- **`Reidentification`** (frozen): `new_track_id`, `canonical_track_id`,
  `similarity`. Kept as its own `UpdateReport` field, not folded into
  `refreshed` — a different claim (inferred identity across a `track_id`
  change), the same reasoning that keeps `contradicted` separate from
  `forgotten`.
- **`WorldModel(frame, decay_rate=0.1, forget_threshold=0.05,
  reidentify_threshold=None, reidentify_max_distance=None)`:**
  `update(scene) -> UpdateReport`, `beliefs`, `get(track_id)`, `len()`.
  `get()` resolves re-identification aliases: any `track_id` ever folded
  into a belief, original or re-identified, returns that same `Belief`.
- **Observation semantics: replace, not combine** — the same rule as
  `working_memory.add()`. An object present in `scene.objects` creates a
  belief (`new`) or replaces an existing one's `percept`/`confidence`/
  `last_observed_at` outright (`refreshed`).
- **Decay semantics — the storage invariant that avoids double-counting.**
  A belief's stored `confidence`/`last_observed_at` are *never* mutated by
  decay, only by a real observation or removal. `confidence` therefore
  always means "confidence at `last_observed_at`", so
  `Belief.confidence_at(t, decay_rate) = confidence * (1 - decay_rate) **
  (t - last_observed_at)` can be computed fresh from that fixed origin on
  any call, with no compounding risk. (The alternative — overwriting stored
  confidence with a decayed snapshot each `update()` call while leaving
  `last_observed_at` fixed — would decay an already-decayed number again
  next call, double-counting elapsed time.) A consequence: `.get(...).confidence`
  is the as-last-observed value, not a live number; callers wanting current
  confidence call `.confidence_at(now, wm.decay_rate)`.
- **Contradicted vs. forgotten are different claims, both removals, kept as
  separate `UpdateReport` fields.** `forgotten`: passive, decayed past
  `forget_threshold` with no evidence either way (absence of evidence).
  `contradicted`: active, the belief's position falls inside a
  `Scene.coverage` region but the object wasn't observed there (evidence of
  absence). Contradiction is checked first and wins when both would apply:
  explicit negative evidence outranks passive decay.
- **`Scene.coverage` is optional and defaults to `()`** (see
  `eidolon.percepts` above); with none claimed, this lab has no negative
  evidence for that call and can only decay beliefs, never contradict them
  — a conscious limitation. A belief with no `position` can likewise only
  decay, never be contradicted (nothing to check).
- **Frame and clock rules match `visual_attention`'s**, not new ones:
  `frame` is configured once, no transforms happen here, and any object's
  or coverage region's `frame` mismatching it raises — validated in a pass
  before anything is mutated, so a rejected `update()` call changes
  nothing. `observed_at` must not decrease between calls.
- **Re-identification: one specific failure mode, not general identity
  resolution.** Still "a new `track_id` is a new belief" by default.
  `reidentify_threshold`/`reidentify_max_distance` (required together, off
  unless both given) let an otherwise-unrecognized object be folded into a
  currently unobserved belief instead, when a single best match clears
  both an appearance gate (`most_similar`, imported from
  `representation` — the lab's second consumer, after
  `episodic_memory.recall_by_similarity`) and a position gate
  (`distance <= reidentify_max_distance`). Ties among equally similar
  candidates go to whichever belief was created first (same insertion-order
  convention as elsewhere) — a documented limitation, not detected: two
  indistinguishable, simultaneously unobserved objects can be
  misattributed. A belief claimed by one re-identified object in a call is
  removed from the pool for the rest of that call, and a belief already
  refreshed by an exact `track_id` match can't also be re-identified onto.
  `get(track_id)` resolves any id ever folded into a belief (original or
  re-identified) to the same `Belief`, always in one hop — `_beliefs` keys
  never change, so an alias recorded for a third id points straight at the
  root, never at an intermediate alias. The dict key a belief is first
  created under is stable and never itself exposed, but `Belief.track_id`
  reads the most recently observed percept, so it can differ from whichever
  id was used to look the belief up — deliberate, not patched with a
  separate canonical-id field.
- **`update()`'s three-pass order (refresh → contradiction/forgetting →
  re-identification) is load-bearing, not an implementation detail — both
  the time gate and a real bug fix depend on it.** A belief that's gone
  this call, by either kind of evidence, must not be offered as a
  re-identification candidate:
  - *Time:* rather than a separate staleness parameter or a per-candidate
    check, re-identification simply runs after the contradiction/
    forgetting pass has already deleted anything with
    `confidence_at(observed_at, decay_rate) < forget_threshold` — even
    within this same call.
  - *Contradiction outranks re-identification*, the same way it already
    outranks passive forgetting: a belief with explicit negative evidence
    against it can't be re-identified onto by a new, appearance-matching
    object in the same scene. This was a real bug caught mid-round (a
    contradicted belief could still be silently re-identified onto) when a
    reviewer asked whether the staleness regression test would actually
    fail under pass reordering — it wouldn't have, which is what surfaced
    the gap.
  - **Why there's no separate confidence check in the re-identification
    code, and the risk if that changes:** the contradiction/forgetting pass
    runs to completion before re-identification starts, using the one
    `scene.observed_at` value for the whole call and the same decay
    formula, so a per-candidate check could never fire against anything
    re-identification would see — provable from the code, not just
    empirically true. **If a future refactor removes, skips, or reorders
    that pass relative to re-identification, this guarantee breaks
    silently** and the explicit check would need to come back; two
    regression tests fail under such a reordering today (see the lab's
    README), but the invariant itself lives in pass order, not in a line of
    code that states it.
- **Sources (re-identification): framing inspired by** Wojke, Bewley &
  Paulus (2017, DeepSORT — appearance + motion association across
  occlusion; we use a plain distance gate, not a motion model) and Fields
  (2013, verified against the full text — an earlier draft of this note
  attributed a "causal continuator" phrase, a "BIC model" and a
  hippocampal-reactivation claim to this paper that are not actually in it;
  what it actually argues is that re-identification is solved by heuristic
  best guesses combining featural similarity with the plausibility of an
  object's causal history, not a principled three-way split). Fields
  (2012a) is noted as related, unread, work in the lab's README — no claims
  attributed to it. Full detail in the lab's own "Sources" section.
- **Deferred:** motion model for unobserved beliefs (they don't move), a
  predictor, and any read from working memory or semantic memory.

### `eidolon.labs.semantic_memory`

Category learning and classification over adapter-supplied embeddings:
Nosofsky's (1986) Generalized Context Model, not nearest-neighbor or
nearest-centroid. Where earlier labs cited Shepard/Nosofsky/Rosch as
"framing inspired by," this lab implements their actual equations
(Equations 2–5 for Nosofsky, verified against the extracted PDF text), with
explicit, documented adaptations. Full derivations and verbatim equations
are in the lab's own README; this entry covers only the contract.

```text
working_memory → episodic_memory → semantic_memory
```

The same consolidation pattern one level further along: the caller
orchestrates it (currently `semantic_memory`'s own `demo.py`), calling
`learn(label, embedding)` per episode that has an embedding.

- **Exposes:** `SemanticMemory`, `ConceptMatch`.
- **`ConceptMatch`** (frozen): `label: str`, `probability: float` — **not**
  `representation.Match`: a GCM response is a probability normalized across
  every known label, not a raw cosine similarity, so reusing `Match` would
  conflate two different quantities under one field name.
- **`SemanticMemory(space, sensitivity)`:** both required, no defaults.
  `space` is fixed per instance and validated against every call, like
  `WorldModel.frame`. `sensitivity` has no value derivable from either
  source paper — it doesn't appear in their equations; it's needed only
  because we lack Nosofsky's multidimensional-scaling fit (see below) — so
  a default would invite using it unexamined, the same reasoning that kept
  `episodic_memory`'s `context`/`provenance` required.
- **Operations:** `learn(label, embedding) -> None` (new label creates a
  category; existing label adds an exemplar; unbounded — every exemplar is
  kept, nothing is averaged away during learning), `classify(embedding,
  min_probability=None) -> str | None`, `similar_concepts(embedding, k) ->
  list[ConceptMatch]`, `prototype(label) -> Embedding | None`, `labels`,
  `exemplar_count(label)`, `len()` (distinct labels, not total exemplars —
  exemplars are a concept's internal structure, not a separate top-level
  unit).
- **Classification is Nosofsky's Equation 5, a Luce choice rule**, not
  nearest-anything: `P(Rⱼ|Sᵢ) = [bⱼ·Σ_{k∈Cⱼ}η_ik] / [Σ_K bₖ·Σ_{k∈Cₖ}η_ik]` —
  summed similarity to *every* exemplar of a category, normalized against
  the same sum across *every known category*. v0 fixes the category bias
  bⱼ uniform (cancels in the ratio) rather than expose it as a tunable.
- **η comes from Shepard (Equations 2, 4b), expressed via cosine, not
  implemented as literally stated.** Nosofsky's η = e^(−d²) uses a distance
  `d` from a Minkowski metric over multidimensional-scaling-fitted
  coordinates calibrated to his specific stimuli; we have raw, opaque
  embedding vectors with no such fit. Adaptation, reusing `d² = 2(1−cosine)`
  for unit vectors (the same identity `representation`'s README uses to
  justify dropping Euclidean as redundant with cosine):
  `η(a,b) = exp(-sensitivity * (1 - similarity(a, b)))`, with `similarity`
  imported from `eidolon.labs.representation`, not reimplemented.
- **`prototype` (Rosch) and `classify`/`similar_concepts` (Nosofsky) are
  deliberately different mechanisms, not the same computation under two
  names.** Prototype theory and exemplar theory are historically competing
  accounts of categorization; `prototype(label)` returns the flat,
  re-normalized centroid of a label's exemplars and plays no role in
  classification (tested: calling it never changes what `classify` returns).
- **`min_probability` is a rejection threshold on `classify`'s winner, not
  a candidate filter.** The winner is computed exactly as
  `similar_concepts` would rank it; the floor only decides whether to
  report it or return `None` — it can never change which label would have
  won. This is also why `similar_concepts` has no equivalent filter
  parameter of its own: filtering individual matches on an
  already-cross-label-normalized quantity doesn't have as clean a meaning
  as filtering raw cosine similarity does elsewhere.
- **Unbounded exemplar storage is faithful to Nosofsky, not a convenience
  default — and it has a real cost.** `classify`/`similar_concepts` are
  O(total exemplars stored) per call, every exemplar of every label
  compared against the query every time. Known v0 limitation; v1 would need
  incremental or approximate structures to scale past a small pool.
- **v0 is supervised** (labels from `PerceivedObject.label`, via the glue
  code — `SemanticMemory` itself imports neither `PerceivedObject` nor
  `Episode`); unsupervised clustering is v1, not started.

### `eidolon.labs.procedural_memory`

Skill learning and context-sensitive action selection: a peer of
`semantic_memory` (both fed from `episodic_memory` directly, neither
importing the other), reusing the same Luce-choice-over-accumulated-
evidence shape with its own strength mechanism instead of Nosofsky's.

```text
working_memory → episodic_memory → semantic_memory
                                 └→ procedural_memory
```

- **Exposes:** `ProceduralMemory`, `Practice`, `SkillMatch`.
- **`Practice`** (frozen): `context: Embedding | None`, `timestamp: int`.
  **`SkillMatch`** (frozen): `skill: str`, `strength: float`, `probability:
  float`.
- **`ProceduralMemory(space, alpha, sensitivity, decay_rate)`:** all four
  required, no defaults — same reasoning as `semantic_memory.sensitivity`:
  none has a value either source paper supplies.
- **Operations:** `practice(skill, context, timestamp) -> None` (unseen
  name creates the skill; existing name adds to its history — no `Skill`
  type, no separate registration, mirroring `semantic_memory.learn`
  exactly), `strength(skill, context, now) -> float`, `select(context, now,
  min_probability=None) -> SkillMatch | None`, `curve(skill) -> list[tuple[int,
  float]]`, `skills`, `practice_count(skill)`, `len()` (distinct skills, not
  total practice events).
- **No internal clock; only `practice()` enforces ordering.**
  `practice()`'s `timestamp` must not decrease across calls (the one
  mutating method). `strength`/`select` take a caller-supplied `now` and
  compute fresh each time, enforcing no ordering of their own — exactly
  `Belief.confidence_at`'s contract: a pure function of stored data and the
  time asked about, nothing mutated, no decay step to fall out of sync with
  anything.
- **`strength` sums every practice, time- and context-weighted — the same
  "keep every instance" shape `semantic_memory` uses, not nearest-practice
  or a running average.** Time weight is `(1 - decay_rate) **
  (now - timestamp)`, the identical form `world_model` uses, so
  `decay_rate` means the same thing in both labs. Context weight is
  `exp(-sensitivity * (1 - similarity(context, practice.context)))`
  (Shepard's gradient over cosine, the same construction
  `semantic_memory._eta` uses, written locally since the two labs are peers
  rather than a consolidation chain). A `None` context, on either side,
  contributes at full weight (1.0) rather than being excluded or
  penalized — it's uninformative about context-specificity, not negative
  evidence. A never-practiced skill's strength is `0.0`, not an error.
- **`select` is a Luce choice rule (Luce, 1959) over every known skill's
  strength**, the same move `semantic_memory` makes via Nosofsky's
  Equation 5. `None` if nothing has been practiced, or if every skill
  scores zero. Ties go to whichever skill was practiced first. Interference
  between context-similar skills is a free consequence of this
  normalization, not a separate mechanism — tested explicitly with two
  equally-practiced, context-identical skills splitting probability 50/50.
  `min_probability` is a rejection threshold on the already-computed
  winner, not a candidate filter (`semantic_memory.classify`'s contract,
  reused).
- **`curve` implements only the *shape* of Newell & Rosenbloom's (1981)
  power law, not their model.** Their actual "general power function" is
  four-parameter, `E(RT_N) = A_P + B_P·(N+E)^(−β)` (verified against the
  equation as quoted by Heathcote, Brown & Mewhort, since the original
  chapter itself wasn't freely available), with an asymptote A_P and a
  *fitted* prior-practice parameter E — not a fixed shift, and "E fixed at
  zero" (the common simplification) does **not** resolve N=0 either.
  `curve()` drops the asymptote and uses `(N+1)` as the N=0 fix — **our own
  choice, confirmed only after checking theirs wasn't simply "+1"** — and
  returns a dimensionless relative-speed multiplier, not a calibrated
  duration.
- **Sources, one corrected mid-design.** Newell & Rosenbloom (1981) for the
  power law (verified); Luce (1959) for the choice rule in `select`
  (verified, same grounding `semantic_memory` relies on). The
  accumulated-strength mechanism itself is **not** sourced: an earlier
  proposal cited Logan (1988) for a "habit strength" idea, but the actual
  paper (read in full, grepped for "Luce," "habit strength," "choice rule")
  explicitly positions instance theory **against** strength theories — a
  race among stored traces, not an accumulated scalar. Same category of
  error as the earlier Fields (2013) misattribution in `world_model`'s
  README, caught the same way: read the source before citing it. No
  replacement citation is given; a real one would need its own verification
  pass. Full detail in the lab's own README.
- **No `compile_from_episodes`.** Same discipline as every other cross-lab
  ingestion in this project: plain fields on the receiving method, with the
  sending lab's shape assumed only in external glue code, not imported.

### `eidolon.labs.representation`

Similarity over adapter-supplied embeddings. It consumes `Embedding`s (from
`eidolon.percepts`) and never produces them, so it is model-agnostic and
needs no torch, CUDA or GPU; tests use synthetic vectors.

- **Exposes:** `similarity`, `most_similar`, `Match`, `recognition_rates`,
  `RecognitionRates`.
- **`similarity(a, b)`:** cosine, in `[-1, 1]`. Raises `ValueError` across
  spaces or across dimensions. The clamp to the range only absorbs
  floating-point error (a unit-vector dot product can reach
  `1.0000000000000002`); it is not a modelling choice. Cosine only in v0: on
  L2-normalized vectors Euclidean ranks identically, so it is redundant.
- **`most_similar(query, candidates, k, min_similarity=None)`:** `candidates`
  are `(key, Embedding)` pairs (any iterable, read once); returns
  `list[Match]`, most similar first. `k` is a maximum (int `>= 1`),
  `min_similarity` is inclusive and within `[-1, 1]`, ties keep candidate
  order. Every candidate is validated against the query's space and
  dimension, so a bad one raises even outside the top `k`; nothing is
  skipped silently. **`Match.key` is opaque**: the caller decides what it is
  (a `track_id`, an index, an `Episode`), and this lab never hashes,
  compares or inspects it. Callers filter out objects with no embedding.
- **`recognition_rates(same_pairs, different_pairs, threshold)`:** pairs are
  `(Embedding, Embedding)`, judged "same" at or above `threshold`
  (inclusive, via `similarity`); returns `RecognitionRates(hit_rate,
  false_alarm_rate)`. Both pair sets must be non-empty.
- **The phenomenon:** recognition is a hit / false-alarm trade-off. With
  well-separated categories some threshold is perfect; with overlapping ones
  none is. Thresholds are per space and per modality, not universal, so none
  is hardcoded; a real model's threshold is calibrated against its own
  rates.
- **Similarity is not identity.** Two different objects can share an
  embedding (two identical mugs). Re-identification needs appearance,
  position and time together; this lab supplies the appearance evidence
  only — `world_model`'s re-identification is the consumer that combines
  it with position and time (see that lab's contract above).
- **Sources:** framing inspired by Shepard (1987), Nosofsky (1986), Rosch &
  Mervis (1975) and, for the hit / false-alarm vocabulary, Green & Swets
  (1966) — none of which *this lab itself* implements (it stays at raw
  cosine thresholds). `semantic_memory` is what actually implements
  Shepard's and Nosofsky's equations, built on this lab's `similarity()`;
  see that lab's contract above and its own README for the exact formulas
  and adaptations.
- **Deferred:** a real-CLIP fixture recorded once on the GPU machine and
  committed; Euclidean distance.

### `eidolon.labs.self_model`

The agent's own body state — pose, battery, per-limb engine temperature,
task status — kept independently of `world_model`'s beliefs about
everything else, for `goals`/`planning` to consult. Fed incrementally, not
as one atomic `Scene`-like snapshot: each field remembers its own
last-updated time, so one field can be fresh while another is stale.

```text
adapter → SelfState → SelfModel.update() → (goals/planning: query()+freshness())
                                          └→ (logging/demos: state())
```

- **Exposes:** `SelfModel`, `SelfState`, `SelfSnapshot`, `Pose`, `Battery`,
  `Field`, `JOINT_GROUPS`.
- **`Pose`** (frozen): `position` (3-tuple), `orientation` (quaternion
  `(x, y, z, w)`, must be unit-norm within `1e-3`), `frame` — validated
  against `SelfModel.frame` the same way `WorldModel.frame` is. **`Battery`**
  (frozen): `level` in `[0, 1]`, `charging: bool` — kept together, not
  split, since both come from one sensor read and share one timestamp.
  **`engine_temperature`** is a closed set of five anatomical groups,
  `JOINT_GROUPS = ("head", "l_arm", "r_arm", "l_leg", "r_leg")` — our own
  grouping for this lab, not verified against any specific robot's SDK.
  All floats across `Pose`/`Battery`/`engine_temperature` must be finite.
- **`SelfState` (input) vs. `SelfSnapshot` (output) are two different
  types, deliberately, not one type serving both roles.** `SelfState` is
  what `update()` takes: `observed_at` plus whichever fields the adapter
  has something new to report (`None` = "nothing new in this message," not
  "unknown"). `state()` returns a `SelfSnapshot`, which has no
  `observed_at` — so `sm.update(sm.state())` doesn't just violate a
  documented warning, it raises `AttributeError` immediately. This replaced
  an earlier one-type design that had a real round-trip footgun: feeding
  `state()`'s output back into `update()` would have collapsed every
  field's independent timestamp onto one shared value. `SelfSnapshot.as_of`
  is the max of the reported fields' timestamps — a rough "anything fresh
  at all" signal, not a per-field value. **`state()` is for logging, demos
  and visualization; a consumer making a decision uses `query()` +
  `freshness()`, never `state()`.**
- **`query(field)`/`freshness(field, now) -> int | None`:** `field` is
  `Literal["pose", "battery", "engine_temperature", "task_status"]`, with
  `@overload`s on `query` narrowing its return type per field; both also
  validate `field` at runtime and raise `ValueError` for anything else.
  `query` returns unconditionally regardless of staleness — `WorldModel.get`'s
  contract, reused. `freshness` is `now − last_updated_at` for that field,
  or `None` if never reported — **a raw elapsed time, not a decayed
  confidence**: see "Decay semantics differ from `world_model`'s" below.
  Only `freshness` has a `confidence`-shaped future extension (natural once
  a field has more than one source); not built now.
- **Decay semantics differ from `world_model`'s — the reason this lab
  reuses the decay *pattern*, not the *mechanism*.** `WorldModel.confidence_at`
  decays because the *world* may have changed while unobserved. `self_model`
  has no `decay_rate`: a stale field here means *the sensor stopped
  reporting*, not *the agent is less sure of its own state*, so there's no
  quantity to discount. `goals`/`planning` must treat `self_model` staleness
  as a reporting gap, `world_model` staleness as an eroding belief — the two
  are not the same claim.
- **The pose/`task_status` staleness ambiguity is documented, not
  resolved, and has no `is_moving` field.** A stale `pose` alone can't
  distinguish "idle, sensor quiet" from "moved, sensor quiet" — a consumer
  needing that distinction cross-checks `task_status` itself.
- **`task_status` is an open string by design**, not a closed `Literal` —
  `goals`/`planning` don't exist yet to dictate the real vocabulary;
  closing it prematurely risks guessing wrong. To be closed once a real
  consumer exists.
- **Units and clock-sharing are a documented precondition, not an
  abstraction.** `observed_at`/`as_of`/`now` are milliseconds, on one
  monotonic clock the adapter and every consumer must share — mixing a
  Python monotonic clock with ROS 2 time across a process boundary breaks
  `freshness` silently (it still returns a number; the number stops
  meaning what's expected). No clock abstraction is added to prevent this.
- **Single-threaded by contract, like every other lab's mutating method —
  stated explicitly here because ROS 2 callbacks are concurrent by
  default.** `update()` has no internal locking; a ROS 2 deployment must
  serialize calls into one `SelfModel` itself.
- **Not extracted to `eidolon.percepts`.** Proprioception isn't perception
  of the external world, so `SelfState`/`SelfSnapshot` live in
  `eidolon.labs.self_model`. If a second lab ever needs this type, the same
  Rule of Three that moved `PerceivedObject`/`Scene` out of
  `visual_attention` would apply here too — not before.
- **Sources:** none. Unlike `semantic_memory`/`procedural_memory`, this lab
  implements no cited equation — it reuses `Belief.confidence_at`'s *shape*
  (pure function of stored `(value, timestamp)` and a query time) without
  adopting its decay *mechanism*. See the lab's own README for the full
  reasoning.

### Clocks

Five separate clocks exist today, by conscious choice rather than
oversight: `working_memory`'s logical tick (advanced by `tick()`),
`episodic_memory`'s encode counter (advanced by `encode()`), the
caller-supplied `Scene.observed_at` that `visual_attention` and
`world_model` both read (the same clock, not a separate one each —
`world_model` is a second consumer of it, same monotonicity rule, no clock
of its own), `procedural_memory`'s caller-supplied `timestamp`/`now` — a
genuinely separate namespace from `Scene.observed_at`, not a third consumer
of it, since this lab never sees a `Scene` at all, only `Embedding`
contexts and plain `int`s the caller supplies directly — and
`self_model`'s caller-supplied `observed_at`/`now`, a fifth, independent
namespace with its own unit (milliseconds, documented, since `freshness()`
is meant to be calibrated against a real threshold, unlike the other labs'
more abstract tick counts) and its own per-field granularity (each field
remembers its own last-updated time, not one shared value per call, unlike
every other lab here). None is wall-clock time and none is shared across
labs, so each stays testable on its own and deterministic. The consequence
is that "3 ticks" means something different in each lab. Unifying them
behind one agent clock is deferred until embodiment provides one; when it
does, `PerceivedObject.observed_at` is the natural event time for the
`occurred_at`/`encoded_at` split in episodic memory, and `self_model`'s
documented "adapter and consumer must share one clock" precondition is the
first thing such a unification would actually resolve, rather than merely
document.

### Shared base interface (e.g. a `CognitiveModule` protocol)

**Decided: not yet. Reason: the clocks differ structurally.** This was
originally "revisit when the shape is visible"; with four labs built it is
now an explicit decision, not a deferral by default.

The labs don't just use different clock *values*, they relate to time in
different ways: `working_memory` advances an internal logical clock by
exactly one per explicit `tick()`; `episodic_memory` has an internal counter
that advances as a side effect of `encode()` and has no notion of elapsed
time at all; `visual_attention` and `world_model` have no clock — time is an
input carried by the data (`Scene.observed_at`) and can jump by any amount
between calls, which is why `world_model` decays per elapsed unit while
`working_memory` decays per tick. A shared `tick()`-shaped protocol would
have to be false for at least two of the three, and the overlap that
remains (a read-only view of state) is too thin to be worth abstracting.

Nothing is implemented. What would reopen it: a single agent clock that
every lab consumes (making a `step(now)`-shaped protocol truthful), or a
cognitive core (`eidolon.core`) that needs to drive labs uniformly.

## Language split

- **Cognitive core and labs: Python.** This is where the actual thinking
  happens — fast iteration on models of memory, attention, world models,
  etc., with the numerical/ML ecosystem (numpy, and later whatever's needed
  for a given lab).
- **Embodiment layer: C++.** Reused/adapted from the existing
  [`spin_the_bottle`](https://github.com/dialgop/spin_the_bottle) project,
  which already has a working ROS 2 + Webots + simulated NAO setup,
  `ros2_control` joint control, and a C++17 vision pipeline. The game-specific
  logic from that project is not carried over — only the robot/simulation
  infrastructure.
- **Bridge: ROS 2.** ROS 2 supports both `rclpy` and `rclcpp` natively, so the
  Python cognitive core can drive the C++ embodiment layer over ROS 2
  topics/services without needing a single-language rewrite on either side.

## Target cognitive architecture (once embodied)

```text
                    COGNITIVE CORE
        Perception → Attention → Global Workspace
             │             │              │
             ▼             ▼              ▼
          Memory       World Model    Reasoning
             │             │              │
             └─────────────┼──────────────┘
                           ▼
                       Self Model
                           │
                           ▼
                         Goals
                           │
                           ▼
                       Planning
                           │
                           ▼
                         Action
                           │
                           ▼
                    NAO / Webots
                           │
                           ▼
                      Environment
                           │
                           └──────→ Perception
```

This forms a continuous perception → cognition → action loop: the agent has a
physical representation in the simulated world, receives sensory information,
maintains internal state, makes decisions, moves its body, and observes the
consequences of its actions.

## Experimental methodology

The project stays experimental rather than claiming the resulting system is
genuinely conscious. Architectures are compared progressively by adding one
mechanism at a time and studying how behavior changes:

```text
Memory
  ↓
Memory + Attention
  ↓
Memory + Attention + World Model
  ↓
+ Self Model
  ↓
+ Goals and Planning
  ↓
+ Metacognition
  ↓
+ Global Workspace
  ↓
Integrated Cognitive Architecture
```

The same environments/tasks are reused across these stages so that adding or
removing a mechanism is the only variable.

## Reusing `spin_the_bottle`

```text
Existing Repository
spin_the_bottle
        │
        │  reuse/adapt infrastructure
        ▼
ROS 2 / Webots / NAO interface
        │
        ▼
New Repository
eidolon
        │
        └── Cognitive Architecture
```

`spin_the_bottle` remains an independent robotics/computer-vision project.
Only the infrastructure needed to talk to the simulated NAO and Webots
(ROS 2 setup, `ros2_control`, NAO head/arm control, the Webots robot
environment, the vision pipeline) gets selectively pulled into
`embodiment/` here — none of the game-specific logic.

## Embodiment plan (documented, not started)

**Decided: not starting yet.** The cognitive pipeline isn't complete
enough to close a perception–cognition–action loop: `self_model` is now
built, but goals and planning don't exist yet. Embodiment also deserves its
own cycle, not a tail on a lab round. This section records the plan so it
isn't re-derived later; it is not a spec.

**What the embodiment/adapter side owns** (everything Eidolon's contracts
already assume it does, and Eidolon never does):
- detection, tracking (a stable `track_id`), and feature extraction;
- **embeddings**, e.g. CLIP run on a GPU on the adapter side. Eidolon
  consumes embeddings and never produces them, so it has no torch, CUDA or
  model dependency; the model is swappable behind the same interface;
- transforming positions into the `world_model`'s single configured frame
  (`tf` in ROS 2); Eidolon does no transforms;
- honest `Scene.coverage` claims (what was actually looked at), which is
  what makes contradiction possible;
- one consistent `observed_at` clock across the stream.

**What Eidolon expects:** a stream of `eidolon.percepts.Scene`. No ROS 2,
Webots, torch or CUDA import ever enters `src/eidolon/`.

**Bridge:** ROS 2, with the Python cognitive core as `rclpy` nodes and the
C++ vision/control from `spin_the_bottle` on the other side; message ↔
dataclass conversion happens at that boundary. `Scene`s should stay
serializable (e.g. JSON lines) so a real Webots run can be recorded and
replayed against Eidolon without the simulator, which is also how real
streams become test fixtures.

**Entry criteria** (as decided, revisit when reached): a cognitive pipeline
complete enough to close the loop — goals and planning, now that
`self_model` and the representation lab both already exist. Actions and a
predictor (state + action → next state) belong to that later cycle, not
this document.

## Long-term vision

Not to claim NAO becomes conscious, but to construct an increasingly
sophisticated artificial agent whose architecture contains functional
mechanisms inspired by theories of consciousness, place those mechanisms in
an embodied system, and experimentally study the resulting behavior.

> From cognitive concepts, to computational mechanisms, to an integrated
> cognitive architecture, to an embodied artificial agent.
