# Procedural Memory Lab

## What this implements

Skill learning and context-sensitive action selection:
`ProceduralMemory.practice(skill, context, timestamp)` to record a practice
event, `select(context, now)` to pick the single most probable skill for a
situation (a Luce choice rule over accumulated, context- and time-weighted
strength), `strength(skill, context, now)` to inspect that evidence
directly, and `curve(skill)` for the shape of its practice-speed
improvement.

Like `semantic_memory`, this lab doesn't take arbitrary input directly and
doesn't import any other lab's type — it's fed by caller-orchestrated glue,
the same `working_memory → episodic_memory → (semantic_memory |
procedural_memory)` pattern, one branch further along.

## Sources — one holds up, one doesn't, one is honestly ours

- **A. Newell & P. S. Rosenbloom, *Mechanisms of Skill Acquisition and the
  Law of Practice*, in *Cognitive Skills and Their Acquisition* (Erlbaum,
  1981), pp. 1–51.** The power law of practice. Verified against the
  equation as quoted in Heathcote, Brown & Mewhort's *The Power Law of
  Practice: Repealed* — the **actual** "general power function" is
  four-parameter:

  ```
  E(RT_N) = A_P + B_P · (N + E)^(−β)
  ```

  A_P is an asymptote (a floor — physical/motor limits mean execution time
  never reaches zero no matter how practiced); B_P is the range of
  improvement from the start of learning to that floor; E is a **free,
  fitted parameter** representing "the subject's prior learning due to
  practice before experimental measurement" (credited to Seibel, 1963) —
  it is not a fixed shift, and "most subsequent researchers... fit the
  simpler three-parameter version... with E fixed at zero," which does
  **not** resolve N=0 (still a singularity there).

  **`curve()` implements neither version.** It drops the asymptote
  entirely (so values go all the way to 0 as N grows, not to a realistic
  floor) and uses `(N+1)` as the N=0 fix, not Newell & Rosenbloom's E (a
  fitted quantity with its own meaning, not a generic smoothing constant).
  This was checked explicitly, not assumed: the common E=0 simplification
  doesn't handle N=0 either, so `(N+1)` is **our own documented choice**,
  made only after confirming theirs wasn't simply "+1". `curve()` therefore
  returns a **dimensionless relative-speed multiplier**
  (`(N+1)**(-alpha)`, 1.0 at N=0, decreasing toward 0), not a calibrated
  predicted duration — there is no B_P or A_P parameter here for a caller
  to get an actual time out of this without supplying their own baseline.

- **Luce (1959)** for the choice rule itself in `select()` — the same
  normalization-across-competing-options move `semantic_memory` makes via
  Nosofsky's Equation 5. This part is genuinely grounded.

- **The accumulated-strength mechanism `practice()`/`strength()` build is
  our own design, explicitly not sourced.** An earlier draft of this
  round's proposal cited Logan (1988), *Toward an Instance Theory of
  Automatization* (Psychological Review 95(4), 492–527), for a
  "context→action habit strength" idea. Verified against the full text
  (grepped for "Luce," "habit strength," "choice rule," "context... action"
  — none of it appears) before accepting the citation, and it turned out to
  be a real misattribution, not an imprecise paraphrase: Logan's paper is
  explicit that "instance theories have been pitted **against** strength
  theories" — it's the alternative account, a race among stored per-
  exposure traces (the fastest-retrieved instance wins; the power-law
  speedup falls out of order statistics as traces accumulate), not an
  accumulated scalar reinforced by each practice. Citing it for a
  strength-accumulation mechanism would have been the same category of
  error as the earlier Fields (2013) misattribution in `world_model`'s
  README. **No replacement citation is given here** — a real one (classical
  Hull-style habit strength, or a specific ACT-R production-utility paper)
  would need its own verification pass, the same rigor every other source
  in this project gets, and that pass hasn't been done. Same reasoning
  that left `semantic_memory.sensitivity` uncited: say plainly what's ours
  when nothing fits cleanly, rather than reach for a citation that sounds
  right.

## Design notes

- **No `Skill` type, no separate registration step.** `practice(skill:
  str, ...)` on an unseen name creates it; on an existing name, it just
  adds to its history — mirroring `semantic_memory.learn(label,
  embedding)`'s new-label-vs-existing-label behavior exactly. None of this
  lab's three demo scenarios need parameters, preconditions, or an effect
  attached to a skill, so none are stubbed in: if a later lab needs
  executable preconditions, that's a real, separate extension, added when
  something concrete needs it.
- **Context is `Embedding`, compared via `representation.similarity` —
  not a `Mapping[str, Any]`.** `episodic_memory`'s dict-based
  `context`/`provenance` support *subset*-match filtering; this lab needs a
  *graded* similarity to generalize a habit to a merely-similar situation,
  which needs a vector space. Reusing `representation.similarity` rather
  than inventing a metric keeps the "don't duplicate a tested metric"
  rule this whole project follows.
- **No internal clock, and critically, `strength`/`select` enforce no time
  ordering at all — only `practice()` does.** `practice()`'s `timestamp`
  must not decrease across calls (checked and tested); `strength`/`select`
  take an arbitrary caller-supplied `now` and compute fresh, exactly like
  `Belief.confidence_at` — a pure function of stored data and the time
  asked about, nothing mutated, nothing to double-decay. This is a direct
  application of the lesson `world_model`'s re-identification round paid
  for: that bug existed because decay was checked in one place and applied
  in another; here there's no mutating decay step to get out of sync with
  anything in the first place. One consequence worth knowing, not
  guarded against: querying `now` *before* a skill's practice timestamp
  gives a negative elapsed time and an inflated, meaningless value — the
  demo's own first draft hit exactly this and got fixed by choosing
  sensible `now` values, not by adding a check to the store (see `demo.py`).
- **A context-free practice (`context=None`) contributes fully to every
  contextual query, not a reduced or excluded amount.** It represents
  reinforcement that wasn't tied to any particular situation (drilled
  abstractly, or the situation simply wasn't tracked) — it's real evidence
  that this skill is practiced, just uninformative about *which* contexts
  it applies to, so it shouldn't be penalized by context mismatch. The
  same rule applies symmetrically if the *query* context is `None`.
- **Dimension is validated globally across every skill, not per-skill** —
  same reasoning as `semantic_memory`: `select()` compares contexts across
  every known skill in one call, so a mismatch anywhere would break it.
  Only `practice()` can *establish* the dimension lock from real stored
  data; `strength()`/`select()` only ever validate against one already
  set, never set it themselves — otherwise querying before any practice
  ever happened would silently fix the dimension from a throwaway query
  context instead of from actual data.
- **Interference is a free consequence of `select()`'s Luce normalization,
  not a separate mechanism** — exactly as asked for ("derived from this;
  not added as ad hoc"). Two equally well-practiced, context-identical
  skills split probability 50/50 automatically; nothing models "practicing
  A actively suppresses B," because nothing needs to.
- **`min_probability` on `select()` is a rejection threshold on the
  already-computed winner, not a candidate filter** — the same contract
  `semantic_memory.classify` uses: it can never change which skill would
  have won, only whether it's reported. `strength()` has no equivalent
  parameter; it's a diagnostic read, not a decision.
- **`alpha` is global, not per-skill**, required with no default — a
  per-skill value would need its own calibration data nothing in v0
  supplies, the same reasoning that kept `sensitivity` global in
  `semantic_memory`. One practical consequence: every skill's `curve()` has
  the identical shape, differing only in how many points it has (its own
  practice count) — tested explicitly.
- **No `compile_from_episodes` on the store.** Every cross-lab ingestion
  in this project takes plain fields on the receiving lab's own method,
  with the assumption about the sending lab's shape living in glue code
  outside both labs (`working_memory → episodic_memory`,
  `episodic_memory → semantic_memory`). Adding a method that takes an
  `episodic_memory.Episode` directly would be the first exception to that
  discipline. If a demo wants to show "learn a skill from repeated
  episodes," that's glue code reading `episodic_memory.Episode`, the same
  pattern `semantic_memory/demo.py` already uses — not a store method.

## Question this experiment asks

Does accumulating every practice event (never averaging into a running
score, the same "keep every instance" shape `semantic_memory` uses for
concepts), weighted by how recent it is and how similar its context is to
the current one, and selected by a Luce choice rule across competing
skills, reproduce the two basic phenomena procedural memory is known for —
faster execution with practice, and the usual (not exact) choice of skill
degrading into genuine ambiguity when a comparably-practiced competitor
looks just as applicable?

## Run it

```bash
python -m eidolon.labs.procedural_memory.demo
```

## Files

- `store.py`: `ProceduralMemory`, `Practice`, `SkillMatch`; imports
  `similarity` from `eidolon.labs.representation`.
- `demo.py`: a practice curve, interference between two context-identical
  skills, and decay distinguishing a recent habit from an old one (with a
  note on why every `now` chosen is after both skills' practice events).
- `tests/labs/procedural_memory/test_store.py`: construction/validation,
  new-skill-vs-existing-skill in `practice()`, strength/select pinned to
  hand-computed values (verified independently before any test was
  written), the null-context contribution rule, the non-mutating
  dimension-lock discipline, interference as a free consequence of
  normalization, and `curve()`'s shape/length/monotonicity.
