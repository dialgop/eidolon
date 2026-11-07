# Cognitive Micro-Laboratories

A **lab** is a small, isolated experiment that implements one cognitive
mechanism (episodic memory, working memory, attention, a world model, a
self-model, goal/motivation, planning, metacognition, uncertainty, a global
workspace, predictive processing, ...) well enough to run and probe on its
own, before it is ever wired into the larger cognitive core.

## Convention

Each lab lives in its own subpackage:

```
labs/
  <lab_name>/
    __init__.py
    README.md        # what it implements, which theory it's based on, what to observe
    <implementation>.py
    demo.py           # a runnable script/experiment showing it in action
```

Tests for a lab live under `tests/labs/<lab_name>/`, mirroring the package.

A lab should:

- Depend only on `eidolon.core` primitives it actually needs (or nothing, early on).
- Be runnable and observable on its own (`python -m eidolon.labs.<lab_name>.demo`).
- State in its README which theory/paper it draws from and what question the
  experiment is trying to answer.
- Stay small. Integration happens later, deliberately, in `eidolon.core`.

## Status

- `working_memory/` — implemented. A capacity-limited, decaying buffer with
  rehearsal. See its README for details.
- `episodic_memory/` — implemented. Consolidates everything that leaves
  `working_memory` (capacity eviction and decay forgetting alike) into an
  unbounded store with exact/subset-match cued recall, plus
  `recall_by_similarity` (ranked, via `representation`). See its README for
  details, and `/docs/architecture.md` for how the labs' interfaces are
  defined.
- `visual_attention/` — implemented. Object-level bottom-up/top-down
  selection blended by a factor `t`, with inhibition of return. Its input
  contract (`PerceivedObject`, `Scene`) moved out to `eidolon.percepts`
  once `world_model` became a second consumer. See its README for sources
  and for what is our own adaptation.
- `world_model/` — implemented. A belief store: persists object existence
  and last known state independent of attention, fed by `Scene` directly
  (not by `working_memory`). Decays and can be forgotten when unobserved,
  contradicted immediately by explicit negative evidence
  (`Scene.coverage`), or re-identified across a tracker id change by
  appearance + position + staleness together (via `representation`). See
  its README — in particular "Scene provides existence; working memory
  provides focus, not existence."

- `representation/` — implemented. Cosine similarity over adapter-supplied
  embeddings (`Embedding` lives in `eidolon.percepts`; Eidolon consumes
  embeddings and never produces them), nearest-neighbour retrieval, and the
  recognition hit / false-alarm trade-off. See its README.
- `semantic_memory/` — implemented. Category learning and classification,
  fed from `episodic_memory` the same way `episodic_memory` is fed from
  `working_memory`. Unlike the other labs' "framing inspired by" use of
  Shepard/Nosofsky/Rosch, this one implements their actual equations (a
  Luce choice rule over summed exemplar similarity, not nearest-anything),
  with the adaptations needed to run them on raw embeddings instead of
  multidimensional-scaling-fitted coordinates. See its README for the
  verbatim source equations and what we added.

- `procedural_memory/` — implemented. Skill learning and action selection,
  a peer of `semantic_memory` (both fed from `episodic_memory` directly,
  neither importing the other). Reuses the Luce-choice shape over its own
  strength mechanism rather than Nosofsky's similarity sum. Newell &
  Rosenbloom's power law and Luce's choice rule are genuinely cited and
  verified; the strength-accumulation mechanism itself is explicitly
  labeled as our own design, not sourced — an earlier proposal's citation
  for it (Logan, 1988) turned out to be a real misattribution, caught by
  reading the actual paper. See its README.

- `self_model/` — implemented. The agent's own body state (pose, battery,
  per-limb engine temperature, task status), kept independently of
  `world_model`'s beliefs about everything else, fed incrementally so each
  field ages on its own clock. Reuses `Belief.confidence_at`'s *pattern*
  (pure function of stored value + timestamp and a query time) but not its
  decay *mechanism* — self-state staleness is a reporting gap, not an
  eroding belief, so `freshness()` returns a raw elapsed time, not a
  decayed confidence. See its README for the `SelfState`/`SelfSnapshot`
  split (and the round-trip footgun it structurally closes off) and the
  documented, deliberately unresolved pose/`task_status` staleness
  ambiguity.

All three follow-ups `representation` unlocked are now built: similarity
recall in episodic memory, re-identification in the world model, and
semantic memory itself. `procedural_memory` and `self_model` are new
additions alongside them, not among the three. The general, non-visual
attention lab stays deferred: there is no second use case yet.
