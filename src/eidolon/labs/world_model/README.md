# World Model Lab

## What this implements

A belief store: persists beliefs about objects' existence and last known
state, whether or not they're currently attended (`WorldModel`, `Belief`,
`UpdateReport`), plus re-identification of a new `track_id` as a
previously tracked, currently unobserved belief (`Reidentification`).

**Scene provides existence; working memory (and, later, semantic memory)
provide focus and interpretation, not existence.** `WorldModel.update()`
takes a `Scene` directly, never `working_memory` — so what's believed to
exist never shrinks to the ~4 items working memory happens to be attending,
and doesn't disappear the moment working memory's own rehearsal-driven
decay forgets something. When focus/interpretation signals from working
memory or semantic memory do get integrated (later), they will modulate an
*existing* belief's confidence or interpretation — never create or delete
one. Existence and focus are different questions, and this lab only answers
the first.

v0 is a **belief store**, not a **predictor**: it persists "the cup is
still there even if I don't see it," not "given a state and an action, what
happens next." A predictor needs actions, which don't exist yet, and is
deferred.

## The scenario this is built for

1. NAO observes an object.
2. The object leaves view.
3. The belief persists (`WorldModel` still has it).
4. Its confidence decays for as long as it goes unobserved.
5. Either it's re-observed under the *same* `track_id` (belief refreshes,
   confidence and position replaced outright), or it's re-observed under a
   *different* `track_id` — the tracker lost and re-acquired it, a common
   failure mode — and re-identified by appearance and position together
   (same outcome: the belief refreshes, now aliased to both ids), or it
   decays past `forget_threshold` and is forgotten, or the adapter reports
   having *looked at that exact spot and found nothing* (`Scene.coverage`),
   contradicting the belief immediately.

`demo.py` runs all of these: both non-re-id endings side by side, a
re-identification after a simulated occlusion, and the documented ambiguity
limitation when two objects look identical.

## Design notes

- **Contradicted vs. forgotten are different claims, not the same removal
  with two names.** `forgotten` is passive: no evidence either way, just
  decay past a threshold — absence of evidence, not evidence of absence.
  `contradicted` is active: the adapter claims to have observed the exact
  region the belief occupies (`Scene.coverage`) and found nothing there —
  real negative evidence. `UpdateReport` keeps them as separate fields, and
  contradiction is checked first: a belief that is both covered-and-absent
  and would also have decayed past the threshold is reported as
  contradicted, never forgotten, because explicit negative evidence is
  stronger than passive decay.
- **A belief's stored `confidence`/`last_observed_at` are never mutated by
  decay.** They're only ever replaced by a real observation (refresh), or
  the belief is removed (contradicted/forgotten). This is deliberate, not
  an oversight: `confidence` always means "confidence at
  `last_observed_at`" — the true last observation — so decay for any
  `Scene.observed_at` can be computed fresh, directly from that fixed
  origin, via `Belief.confidence_at(observed_at, decay_rate) = confidence *
  (1 - decay_rate) ** (observed_at - last_observed_at)`. If `update()`
  instead overwrote the stored confidence with a decayed snapshot each call
  while leaving `last_observed_at` fixed, the *next* call would decay that
  already-decayed number again from the same origin — silently
  double-counting elapsed time. Not mutating avoids the bug by
  construction rather than by getting the bookkeeping right every time.
  One consequence: `wm.get("cup").confidence` reflects the confidence *as
  last observed*, not a live-decaying number — call `.confidence_at(now,
  wm.decay_rate)` for that.
- **`update()` validates before it mutates anything.** Every object's and
  coverage region's `frame` is checked against the model's configured
  `frame` in a first pass; only if all pass does the second pass touch
  `_beliefs`. A rejected call (bad frame, or `observed_at` decreasing)
  changes nothing — same guarantee `visual_attention`'s clock check makes.
- **`update()` replaces, it doesn't combine**, on a match: the same
  precedent as `working_memory.add()`. A re-observed object's position and
  confidence become the new observation's, discarding the old ones
  outright rather than averaging or otherwise combining them.
- **No coverage means no contradiction is ever possible for that call** —
  a conscious limitation, not a default to fill in casually. Without an
  adapter that reports what it actually looked at, this lab has no negative
  evidence and can only lose confidence in a belief passively, never
  confirm it's actually gone.
- **A belief with no `position` can only decay, never be contradicted.**
  There's nothing to check its absence against.
- **No motion model in v0.** An unobserved belief doesn't move; it persists
  at its last known position until re-observed, contradicted, or forgotten.
- **Identity is still mostly the adapter's job — re-identification only
  covers one specific failure mode.** A new `track_id` is still a new
  belief *by default*. `reidentify_threshold`/`reidentify_max_distance`
  (both required together, off unless both are given — appearance alone is
  not sufficient evidence; see below) let an otherwise-unrecognized object
  be folded into an existing, currently unobserved belief instead, when a
  single best match clears both gates. This recovers specifically from
  "the tracker lost the object and re-acquired it under a new id" — it does
  not attempt general re-identification across arbitrary gaps, categories,
  or viewpoints.
- **New dependency: `world_model → representation`.** Re-identification
  imports `most_similar` from `eidolon.labs.representation` rather than
  reimplementing ranking or cross-space validation — the second consumer of
  that lab, after `episodic_memory.recall_by_similarity`.
- **Appearance and position are both required.** A candidate belief must
  have an `embedding` *and* a `position` within `reidentify_max_distance`
  of the new percept's — appearance alone would conflate "looks like a cup"
  with "is this cup", which is exactly the "similarity ≠ identity"
  limitation `representation` already documents.
- **`update()` runs in three ordered passes — refresh, then
  contradiction/forgetting, then re-identification — and the order is
  load-bearing, not an implementation detail.** A belief that's gone this
  call, by *either* kind of evidence, must not be offered as a
  re-identification candidate:
  - **Time** reuses existing decay rather than a third parameter or a
    separate staleness check: because the contradiction/forgetting pass
    runs to completion *before* re-identification, a candidate belief whose
    `confidence_at(scene.observed_at, decay_rate)` has dropped below
    `forget_threshold` — even within this same call — is already deleted
    from the store by the time re-identification looks for candidates.
  - **Contradiction outranks re-identification**, the same way it already
    outranks passive forgetting: a belief with explicit negative evidence
    against it (the adapter looked at its exact position and found
    nothing) cannot be re-identified onto by a new, appearance-matching
    object in the same scene — it's removed in the same earlier pass,
    before re-identification runs. This was a real bug in an earlier draft
    of this round (a contradicted belief could still be silently
    re-identified onto), caught by asking specifically whether the
    staleness test would fail under pass reordering — it wouldn't have,
    which is what led to finding this.
  - **Why `_find_reidentification_match` has no confidence check of its
    own**, not just "it happens to be redundant": the contradiction/
    forgetting pass runs unconditionally to completion before the
    re-identification pass starts, using the exact same `scene.observed_at`
    (one value for the whole call) and the exact same decay formula — so
    the old per-candidate check could never evaluate to true against
    anything re-identification would ever see. It isn't a parallel
    mechanism that happens to agree; it's the *only* mechanism.
    **⚠️ If a future refactor removes, skips, or reorders the
    contradiction/forgetting pass relative to re-identification, this
    guarantee silently breaks** — stale or contradicted beliefs would
    become re-identifiable again, and the explicit per-candidate check
    would need to come back. `test_a_belief_too_stale_to_still_be_live_is_excluded_even_within_the_same_call`
    and `test_a_contradicted_belief_cannot_be_reidentified_onto_in_the_same_call`
    both fail under such a reordering (verified), so a refactor that
    breaks this should be caught — but the invariant lives in *pass
    order*, not in a line of code that says so.
- **Two indistinguishable candidates: documented limitation, not
  detected.** Among qualifying candidates, the single most similar one
  wins (`most_similar(..., k=1)`); ties go to whichever belief was created
  first — the same insertion-order convention used everywhere else in this
  project. Two simultaneously unobserved, equally similar objects (e.g. two
  identical mugs) can therefore be misattributed. This isn't checked for or
  flagged; it's `representation`'s "similarity ≠ identity" limitation made
  concrete and consequential here, not a new one. See `demo.py`, scenario D.
- **A belief matched this call can't be claimed twice.** If two new objects
  in the same `Scene` could both match the same existing belief, the first
  (in `scene.objects` order) claims it; the second either matches a
  different candidate or becomes new. A belief already refreshed by an
  exact `track_id` match this call is also removed from the candidate pool
  — it can't simultaneously be refreshed directly and re-identified onto.
- **The alias, and what `Belief.track_id` means after one exists.** The key
  a belief is first created under never changes and is never itself
  exposed; `get()` resolves *any* `track_id` ever folded into that belief
  (original or re-identified) to the same `Belief`. But `Belief.track_id`
  reads the *most recently observed* percept's `track_id` — so after a
  re-identification, `world_model.get("original-id").track_id` can return
  a different string than the key used to look it up. This is deliberate,
  not patched with a separate "canonical id" field: `.track_id` answers
  "what was this last observed as", and `get()`'s alias resolution already
  answers "what is this believed to be, under any id it's ever had".
- **Confidence on a fresh observation is taken as-is**, even if it's below
  `forget_threshold` — a real (if weak) sensor reading isn't pruned the way
  a decayed-away belief is; `forget_threshold` only governs unobserved
  decay.

## Sources (framing inspired by, not implemented)

- N. Wojke, A. Bewley, D. Paulus. *Simple Online and Realtime Tracking with
  a Deep Association Metric.* ICIP 2017 (DeepSORT). Combines an appearance
  embedding with motion (Mahalanobis distance over a Kalman filter state)
  to re-associate tracks across occlusion — the engineering analogue of
  what this lab does with appearance + a fixed-radius position gate. We use
  a plain distance threshold, not a motion model or Mahalanobis distance;
  there is no filter, no velocity, no uncertainty ellipse.
- C. Fields. *The Principle of Persistence, Leibniz's Law, and the
  Computational Task of Object Re-Identification.* Human Development 56(3),
  147–166 (2013). Verified against the full text before citing (an earlier
  draft of this note attributed specific claims — a "causal continuator"
  phrase, a "BIC model", a hippocampal-reactivation mechanism — that are
  not actually in this paper; they were dropped once checked). What the
  paper actually argues: that object re-identification across a gap in
  observation is computationally intractable in general, so it is solved by
  *heuristic best guesses* combining featural (appearance) similarity with
  the plausibility of an object's possible causal history during the gap —
  not a clean, principled three-way split into appearance/position/time.
  It also argues against treating "persistence" itself as an explanation
  (Baillargeon's "principle of persistence") rather than an effect to be
  explained. Our implementation is a synthesis, not an implementation, of
  either source: a specific, narrow heuristic (appearance + a position
  gate + decay-based staleness), not general-purpose tracking and not a
  model of infant cognition.
- **Related work, not cited for specific claims:** C. Fields (2012a), *The
  very same thing: Extending the object token concept to incorporate causal
  continuity* — cited by Fields (2013) as the likely source of a "causal
  continuity" framing; we have not read it, so nothing from it is
  attributed here. Worth reading if this lab's heuristic needs to get more
  principled about causal plausibility later.

## Question this experiment asks

Does a belief store with three simple rules — replace on observation, decay
when unseen, contradict when explicitly disconfirmed — reproduce object
permanence well enough to support "the cup is still there even if I can't
see it, until I have a reason to think otherwise"?

## Run it

```bash
python -m eidolon.labs.world_model.demo
```

## Files

- `store.py`: `WorldModel`, `Belief`, `UpdateReport`, `Reidentification`;
  imports `most_similar` from `eidolon.labs.representation`.
- `demo.py`: the cup scenario (decay vs. contradiction endings), a
  re-identification after simulated occlusion, and the two-identical-cups
  ambiguity limitation.
- `tests/labs/world_model/`: `test_store.py` (types: frozen `Belief`,
  `UpdateReport` defaults, `WorldModel` construction/validation),
  `test_dynamics.py` (phenomena: new/refreshed, decay without compounding,
  forgetting, contradiction and its priority over decay, frame/clock
  validation), and `test_reidentification.py` (the position/appearance/
  staleness gates, the same-call race with forgetting, claim exclusivity
  within one call, alias resolution via `get()`, and the ambiguity
  limitation).
