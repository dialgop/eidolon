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

No labs implemented yet. Per the project roadmap (see `/docs/architecture.md`),
the first lab is expected to be **episodic/working memory**, since later labs
(attention, world model, self-model, ...) are meant to build on top of it.
