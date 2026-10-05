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
  or contradicted immediately by explicit negative evidence
  (`Scene.coverage`). See its README — in particular "Scene provides
  existence; working memory provides focus, not existence."

- `representation/` — implemented. Cosine similarity over adapter-supplied
  embeddings (`Embedding` lives in `eidolon.percepts`; Eidolon consumes
  embeddings and never produces them), nearest-neighbour retrieval, and the
  recognition hit / false-alarm trade-off. See its README.

Follow-ups it unlocks: similarity recall in episodic memory is now also
built (above); re-identification in the world model and semantic memory are
each still their own round. The general, non-visual attention lab stays
deferred: there is no second use case yet.
