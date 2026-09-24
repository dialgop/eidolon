# World Model Lab

## What this implements

A belief store: persists beliefs about objects' existence and last known
state, whether or not they're currently attended (`WorldModel`, `Belief`,
`UpdateReport`).

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
5. Either it's re-observed (belief refreshes, confidence and position
   replaced outright), or it decays past `forget_threshold` and is
   forgotten, or the adapter reports having *looked at that exact spot and
   found nothing* (`Scene.coverage`), contradicting the belief immediately.

`demo.py` runs both endings side by side.

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
- **Identity is the adapter's job.** A new `track_id` is a new belief, so
  losing tracking (occlusion, re-identification failure) can create a
  duplicate belief for the same real object. Re-identification is out of
  scope here.
- **Confidence on a fresh observation is taken as-is**, even if it's below
  `forget_threshold` — a real (if weak) sensor reading isn't pruned the way
  a decayed-away belief is; `forget_threshold` only governs unobserved
  decay.

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

- `store.py`: `WorldModel`, `Belief`, `UpdateReport`.
- `demo.py`: the cup scenario, both endings (decay vs. contradiction).
- `tests/labs/world_model/`: `test_store.py` (types: frozen `Belief`,
  `UpdateReport` defaults, `WorldModel` construction/validation) and
  `test_dynamics.py` (phenomena: new/refreshed, decay without compounding,
  forgetting, contradiction and its priority over decay, frame/clock
  validation).
