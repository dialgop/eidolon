# Semantic Memory Lab

## What this implements

Category learning and classification over adapter-supplied embeddings:
`SemanticMemory.learn(label, embedding)` to add an exemplar,
`classify`/`similar_concepts` to categorize a new embedding, and
`prototype(label)` to get a label's central tendency.

Unlike every classifier built from "nearest something," this one follows
Nosofsky's (1986) Generalized Context Model: a label's "model" of the
category is the full set of exemplars ever learned under it — nothing is
averaged away during learning — and classification is a **Luce choice
rule** over summed similarity, not nearest-neighbor and not
nearest-centroid.

v0 is supervised: labels come from the adapter (via `PerceivedObject.label`,
through the glue code — see "Ingestion"), not discovered. Unsupervised
clustering is deferred to v1.

## Sources — this round actually implements the formulas, not just the framing

Earlier labs cited these three papers as "framing inspired by." This one
implements their actual equations (verified against the extracted PDF
text, not just abstracts), with explicit, documented adaptations where our
situation (opaque model embeddings, no multidimensional-scaling fit) didn't
match the original experimental setup.

- **R. M. Nosofsky, *Attention, similarity, and the identification–
  categorization relationship*, JEP: General 115(1), 39–57 (1986).**
  **Equation 5**, the actual category response rule:

  ```
  P(Rⱼ|Sᵢ) = [bⱼ · Σ_{k∈Cⱼ} η_ik] / [Σ_K bₖ · Σ_{k∈Cₖ} η_ik]
  ```

  The probability of classifying stimulus *i* into category *j* is the
  **summed similarity to every exemplar in category *j***, weighted by a
  category bias bⱼ, normalized against the same sum across *every known
  category*. This is a Luce choice rule, not nearest-neighbor and not
  nearest-centroid — classifying by "which single exemplar is closest" or
  "which centroid is closest" would both be different models. **v0 fixes
  bⱼ uniform** (so it cancels in the ratio) rather than expose it as a
  per-category tunable — a documented simplification, not a claim that
  response bias doesn't matter in general.
- **R. N. Shepard, *Toward a universal law of generalization for
  psychological science*, Science 237, 1317–1323 (1987).** Supplies η
  itself: Equation 2 says η is *some* monotonically decreasing function of
  distance; Nosofsky's own Equations 4a/4b test two specific forms,
  exponential (η = e^(−d^r)) and Gaussian (η = e^(−d²)), with distance
  computed from a Minkowski r-metric (Equation 3) over coordinates from a
  multidimensional-scaling fit to the specific stimulus set.
- **E. Rosch & C. B. Mervis, *Family resemblances*, Cognitive Psychology
  7(4), 573–605 (1975).** `prototype(label)` — the central tendency
  (centroid) of a category's members. Deliberately **not** what
  `classify`/`similar_concepts` use: prototype theory and GCM's exemplar
  theory are two different, historically competing accounts of
  categorization, not interchangeable framings of the same mechanism. See
  "Design notes" below for why `prototype` plays no role in classification
  here.

## The adaptation: expressing η (Shepard/Nosofsky) from cosine similarity

We have no MDS-fitted coordinates — just raw, opaque embedding vectors from
whatever model the adapter used. We also don't need to invent a new
distance machinery: for unit vectors, `d² = 2(1 − cosine)`, which
`eidolon.labs.representation`'s own README already uses to justify why
Euclidean distance is redundant with cosine. That identity gives a direct,
reuse-everything route to Nosofsky's Gaussian form (4b):

```
η(a, b) = exp(-sensitivity * (1 - similarity(a, b)))
```

where `similarity` is `eidolon.labs.representation.similarity` (cosine),
reused rather than reimplemented. **`sensitivity` does not come from either
paper.** Nosofsky's d values were pre-scaled by an MDS fit calibrated to
his specific experiment; we have no equivalent fitting step, so some scale
constant is unavoidable to make η behave reasonably on arbitrary embedding
geometry. This is our own necessary addition, not a value either source
specifies — which is exactly why it has no default (see "Design notes").

## Ingestion: consolidation from episodic memory

Fed from `episodic_memory`, the same pattern as `working_memory →
episodic_memory`: the caller orchestrates it (`demo.py`'s
`consolidate_into_semantic_memory`), calling `learn(label, embedding)` once
per episode that has an embedding. `SemanticMemory.learn()` takes plain
fields, never an `Episode` — matching `EpisodicMemory.encode()`'s own
precedent of not taking a `working_memory.Item`. The assumption that
`episode.content` is a `PerceivedObject` (so the glue can reach
`.content.label`) lives in the glue code, not inside either lab.

This resolves an apparent tension in `/docs/architecture.md`'s older
roadmap note, which said semantic memory "needs temporal continuity (the
world model) to have raw material to generalize from." That's about
*prerequisites*, not a literal feed path: re-identification is what makes
repeated sightings of the same real object count reliably as repeated
evidence, not a claim that `SemanticMemory` reads `WorldModel` directly. It
doesn't; `episodic_memory` is the only store that *accumulates* many
distinct exemplars over time, which is what both Rosch's prototypes and
Nosofsky's exemplar sums actually need.

## Design notes

- **`sensitivity` is required, with no default.** There is no value
  derivable from the source papers — it doesn't exist in their equations at
  all, it's purely our adaptation (see above) — so a default would invite
  using it unexamined. Same reasoning that kept `context`/`provenance`
  required with no default in `episodic_memory`: the caller calibrates it
  by looking at the actual probability trade-off it produces on real data,
  the same way a cosine recognition threshold gets calibrated in
  `representation`.
- **`space` is fixed per instance, like `WorldModel.frame`.** Required at
  construction, validated against every `learn()`/query call — not inferred
  from whatever embedding happens to arrive first.
- **`ConceptMatch`, not `representation.Match`.** `Match.similarity` means
  raw cosine everywhere else it's used (`recall_by_similarity`,
  re-identification). A GCM response is a *probability normalized across
  every known label* — a different quantity, not a similarity score.
  Reusing `Match` would conflate the two under one field name.
- **`prototype` plays no role in `classify`/`similar_concepts`.** They're
  exemplar-based (Nosofsky); `prototype` is prototype-based (Rosch). These
  are different, documented-as-competing accounts in the categorization
  literature, kept as genuinely separate mechanisms here, not the same
  computation exposed under two names (tested explicitly — calling
  `prototype()` never changes what `classify()` returns).
- **`learn()` is unbounded — faithful to Nosofsky, not a convenience
  default.** Every exemplar is kept forever; nothing is averaged into a
  running centroid during learning (that's what `prototype()` computes
  separately, on demand). The direct cost: `classify`/`similar_concepts`
  are **O(total exemplars stored)** per call — every exemplar of every
  label is compared against the query, every time. This is a known v0
  limitation, not an oversight: v1 would need incremental/approximate
  structures (e.g. running per-label sufficient statistics, or approximate
  nearest-neighbor indexing) to scale past a small exemplar pool.
- **`min_probability` is a rejection threshold on `classify`'s winner, not
  a candidate filter.** The winning label is computed exactly as
  `similar_concepts` would rank it; `min_probability` only decides whether
  to report it or return `None`. It can never change *which* label would
  have won — tested explicitly (`classify` at the exact winning probability
  still accepts; above it, rejects, while `similar_concepts`'s own ranking
  is unaffected either way). This is why `similar_concepts` itself has no
  equivalent filter parameter: filtering individual matches on a quantity
  that's already normalized across every label doesn't have as clean a
  meaning as filtering raw, unnormalized cosine similarity does elsewhere.
- **`__len__` counts labels (concepts), not exemplars.** What this store
  holds is concepts; exemplars are a concept's internal structure, not a
  separate top-level unit — `exemplar_count(label)` is the latter.
- **New label vs. existing label in `learn()`:** a label not seen before
  creates a category (its first exemplar); an existing label just grows
  its exemplar pool. Both are tested explicitly, including that the
  unbounded pool really does keep every exemplar (`exemplar_count` keeps
  growing, not a capped/averaged number).
- **Dimension is validated globally, across every label, not just within
  one.** Classification sums across *all* labels together, and the
  centroid in `prototype()` needs consistent dimensionality within a
  label — a dimension mismatch anywhere would break both, so `learn()`
  raises `ValueError` before storing anything rather than letting the
  mismatch surface later inside `similarity()`.
- **v0 is supervised; v1 is deferred, not started.** Labels are given by
  the adapter via `PerceivedObject.label`; no clustering/unsupervised
  category discovery exists yet.

## Question this experiment asks

Does Nosofsky's exemplar-sum Luce-choice rule, fed by whatever episodic
memory happens to accumulate (no capacity limit, no pruning), reproduce the
basic categorization phenomenon it was built to explain — clear cases
classified confidently, ambiguous ones genuinely ambiguous (probability
spread across categories, not an arbitrary tie-break) — using nothing more
than cosine similarity reshaped through Shepard's exponential gradient?

## Run it

```bash
python -m eidolon.labs.semantic_memory.demo
```

## Files

- `store.py`: `SemanticMemory`, `ConceptMatch`; imports `similarity` from
  `eidolon.labs.representation`.
- `demo.py`: the EM → SemanticMemory consolidation glue, learning a
  handful of cup/plate sightings, classifying a clear case and an
  ambiguous one (with and without a rejection floor).
- `tests/labs/semantic_memory/test_store.py`: construction/validation,
  new-label-vs-existing-label in `learn()`, the Luce-choice probabilities
  pinned to hand-computed values (verified independently before any test
  was written), the rejection-threshold semantics of `min_probability`,
  and `prototype`'s independence from classification.
