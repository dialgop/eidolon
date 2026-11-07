# Self-Model Lab

## What this implements

A representation of the agent's own body state — pose, battery, per-limb
engine temperature, task status — for `goals`/`planning` to consult, kept
independently of `world_model`'s beliefs about everything else.
`SelfModel.update(state)` ingests an incremental message from the adapter;
`query(field)` and `freshness(field, now)` are how a consumer reads it back,
and `state()` is a read-only full snapshot for logging, demos and
visualization only.

Like `semantic_memory` and `procedural_memory`, this lab takes no arbitrary
input directly and imports no other lab's type — fed by adapter-owned glue
code, not a direct read of `world_model` or `eidolon.percepts`.

## Sources

No academic citation grounds this lab. Unlike `semantic_memory` and
`procedural_memory`, it doesn't implement an equation from the cognitive-
science or psychophysics literature — it's a representation-layer
engineering decision about what body state to expose to `goals`/`planning`,
reusing one *pattern* already established in this codebase
(`Belief.confidence_at`'s shape: a pure function of stored `(value,
timestamp)` and a given query time, nothing mutated by the read) rather than
grounding a mechanism in a cited theory. If a later round wants to connect
pieces of this to a specific theory of self-modeling (minimal self, body
schema, efference-copy accounts), that needs its own verification pass — not
retrofitted here, the same discipline `procedural_memory`'s strength
mechanism documents for the same reason.

## Design notes

- **Reuses the decay *pattern*, not the *mechanism* — and the two labs'
  staleness means something different.** `WorldModel.confidence_at` decays
  a belief about the external world: unobserved doesn't mean unchanged, so
  confidence erodes over time because *the world may have moved on*.
  `SelfModel.freshness` returns a raw elapsed time, not a decayed
  confidence, because the right reading of "my battery reading is 10
  seconds old" is *"the sensor hasn't reported in 10 seconds,"* not *"I am
  10-seconds-less-sure what my battery level is."* There is deliberately no
  `decay_rate` parameter here — nothing is being discounted, so there is
  nothing to calibrate a rate for. `goals`/`planning` must know this
  distinction: a stale self-state field is a reporting gap, not an eroding
  belief, and "how stale is too stale" is a decision left entirely to the
  consumer.
- **`SelfState` lives in `eidolon.labs.self_model`, not `eidolon.percepts`.**
  Proprioception isn't perception of the external world, so it doesn't
  belong on the shared `Scene`/`PerceivedObject` contract — a deliberate
  non-extraction, not an oversight. If a second lab besides `self_model`
  ever needs this type, that's when the Rule of Three (the same reasoning
  that moved `PerceivedObject`/`Scene` into `eidolon.percepts` once
  `world_model` became a second consumer) would apply here too.
- **`SelfState` (input) and `SelfSnapshot` (output) are deliberately two
  different types, not one type serving both roles.** `SelfState` is what
  `update()` takes: `observed_at` plus whichever fields the adapter has
  something new to report, where `None` means "nothing new in *this*
  message," not "unknown." `state()` returns a `SelfSnapshot` instead of a
  `SelfState` so that `sm.update(sm.state())` is not merely discouraged, it
  is structurally broken: `SelfSnapshot` has no `observed_at` attribute, so
  passing one to `update()` raises `AttributeError` immediately. This
  replaced an earlier version of the design that reused one type for both
  roles, which had a real footgun — feeding `state()`'s own `SelfState`
  back into `update()` would have collapsed every field's independent
  timestamp down to one shared value, silently destroying the
  per-field-freshness property that's the entire point of this lab.
  **`state()` is for logging, demos, and visualization. A consumer making a
  decision should use `query()` + `freshness()`, never `state()`.**
- **`SelfSnapshot.as_of` is not a per-field timestamp.** It's the maximum of
  the timestamps among whichever fields have ever been reported (`None` if
  none have) — a rough "how fresh is anything here at all" signal for a log
  line, nothing more. Per-field precision is `freshness(field, now)`'s job,
  never `as_of`'s.
- **`query`/`freshness` take a `Field = Literal["pose", "battery",
  "engine_temperature", "task_status"]`, with `@overload`s on `query` so a
  type checker narrows its return type per field and flags an unknown field
  name statically.** `Literal` is a static-only guarantee; both methods also
  validate `field` at runtime against the same four values and raise
  `ValueError` for anything else, so a typo is caught at both layers, not
  just one. `freshness` has no `@overload`s of its own — its return type
  (`int | None`) doesn't vary by field, only its accepted values do.
- **`query(field)` returns unconditionally, aside from `freshness(field)` —
  the same contract `WorldModel.get()`/`Belief.confidence_at` use: never
  hidden for being stale.** There is currently only `freshness`, not
  `confidence`: this lab has exactly one source per field (the adapter), so
  a confidence dimension on top of freshness would have nothing to express.
  If a field ever gets multiple independent sources, it will need one —
  noted here as the natural future extension, not built now.
- **The pose/`task_status` staleness ambiguity is documented, not resolved,
  and deliberately has no `is_moving` field.** A stale `pose` cannot by
  itself distinguish "idle, the sensor just hasn't re-reported" from "moved,
  the sensor hasn't caught up" — a consumer needing that distinction cross-
  checks `task_status` itself; see `demo.py` scenario B for exactly this.
  Adding a boolean to resolve it on `self_model`'s side was considered and
  rejected: it would be redundant with information `task_status` already
  carries in every case this lab's own scope covers.
- **`engine_temperature` is a closed set of five anatomical groups
  (`JOINT_GROUPS = ("head", "l_arm", "r_arm", "l_leg", "r_leg")`), not an
  open dict and not a scalar.** `SelfState.__post_init__` rejects any
  mapping whose key set isn't exactly these five. These are **our own
  anatomical grouping for this lab, not verified against any specific
  robot's SDK** (in particular, not claimed to match NAOqi's own chain
  naming) — mapping a real platform's per-joint or per-motor sensors onto
  these five keys is the adapter's job, done once, outside this lab. If
  platform fidelity (e.g. literal NAOqi chain names) ever matters, that gets
  the same verify-before-citing treatment as an academic source, at that
  time — not asserted here.
- **`battery` carries `level` and `charging` together, not as two
  independently-tracked fields.** Both numbers come from one sensor read in
  practice (a single battery-state message), so they share one timestamp;
  splitting them would invite asking "is the charging flag older than the
  level?", a question with no real answer given how the data actually
  arrives. `15%, not charging` and `15%, charging` are a very different
  situation for a planner, which is exactly why `charging` exists as its own
  field rather than being left for a consumer to infer.
- **`task_status` is an open string, by design, not a closed `Literal`.**
  `goals`/`planning` don't exist yet, so there's no real consumer dictating
  what the vocabulary should be — closing it now risks guessing wrong, the
  same reasoning that kept `semantic_memory.sensitivity` and
  `procedural_memory.alpha` uncited/undefaulted rather than invented. **To
  be closed into a fixed `Literal` once `goals`/`planning` exist and can say
  what values they actually need**, not before.
- **Validation in `__post_init__` catches structurally malformed data, not
  implausible-but-well-formed data.** `Battery.level` must be in `[0, 1]`;
  `Pose.orientation` must be a unit quaternion (`|x²+y²+z²+w² − 1| ≤ 1e-3`,
  loose enough to tolerate float round-trip over an adapter/ROS 2 bridge,
  tight enough to catch a zeroed or garbage quaternion); every float in
  `Pose`, `Battery`, and `engine_temperature` must be finite
  (`math.isfinite`); `engine_temperature`'s key set must be exactly
  `JOINT_GROUPS`. What is **not** validated, explicitly: physical
  plausibility — a temperature within the robot's real operating range, a
  position within reachable workspace. Catching that is the adapter's job;
  this lab only guards against malformed data (NaN, a non-unit quaternion,
  a wrong key set, an out-of-domain battery level), not implausible-but-
  well-formed readings.
- **Units: `observed_at` (on `SelfState`), `as_of` (on `SelfSnapshot`), and
  `now` (passed to `freshness`) are all milliseconds, on one monotonic
  clock the adapter and every consumer must share.** This is a precondition
  of the contract, not something `self_model` can enforce or abstract over:
  **mixing a Python monotonic clock with ROS 2 time in the same deployment
  breaks freshness semantics silently** — `freshness` will still return a
  number, it just won't mean what the consumer thinks it means. No clock
  abstraction is added to guard against this; it's a deployment discipline,
  documented here because it's easy to get wrong across process boundaries
  and would fail silently if it were.
- **Single-threaded by contract.** `update()` is not safe to call
  concurrently with itself or with any `query`/`freshness`/`state()` call
  that needs a consistent read — there is no internal locking. A ROS 2
  deployment with concurrent callbacks must serialize calls into one
  `SelfModel` itself (a single-threaded executor callback group, or
  external locking); this lab does not pretend otherwise and does not do it
  for you.
- **Only `update()` enforces time ordering** (`observed_at` must not
  decrease across calls — the one mutating method, rejecting the whole call
  and changing nothing on a decrease); `query`/`freshness`/`state()` take no
  ordering constraint of their own, matching `Belief.confidence_at` and
  `procedural_memory.strength`'s contract exactly. One consequence, not
  guarded against: calling `freshness(field, now)` with `now` before that
  field's last-reported timestamp gives a negative elapsed time — the same
  deliberate non-guard every other decay-adjacent read in this project
  documents rather than checks for.
- **An "empty" `update()` (every field `None`) is allowed and advances the
  shared clock without touching any stored value** — the same precedent
  `WorldModel.update()` sets by accepting a `Scene` with no objects.
- **No `confidence_at`-style decay formula, and no `CognitiveModule`
  shared-interface attempt.** This lab's time handling (independent
  per-field timestamps, a shared monotonic clock, no internal tick) is a
  fifth, genuinely separate convention from every other lab's — see the
  Clocks section in `docs/architecture.md` — reinforcing the existing
  decision there that a shared base interface isn't warranted yet.

## Question this experiment asks

Does giving `goals`/`planning` a small, closed set of independently-
timestamped self-state fields — rather than one atomic, uniformly-aged
snapshot, and rather than a decayed confidence score borrowed from
`world_model`'s belief semantics — let a consumer reason correctly about the
agent's own condition (where am I, how much power do I have and is it
running out, is anything overheating, what am I doing) while keeping "I
don't know yet" distinct from "nothing has changed," and keeping a reporting
gap distinct from an eroding belief?

## Run it

```bash
python -m eidolon.labs.self_model.demo
```

## Files

- `store.py`: `SelfModel`, `SelfState`, `SelfSnapshot`, `Pose`, `Battery`,
  `Field`, `JOINT_GROUPS`.
- `demo.py`: independent per-field freshness after incremental updates, the
  pose/`task_status` staleness ambiguity, and `state()` vs.
  `query()`/`freshness()`.
- `tests/labs/self_model/test_store.py`: construction/validation for `Pose`
  (length, finiteness, unit-quaternion tolerance), `Battery` (range,
  finiteness), and `SelfState` (`engine_temperature`'s closed key set);
  `update()`'s monotonicity, frame validation, and per-field partial
  merging; `query`/`freshness`'s unknown-field rejection and unconditional-
  read contract; `state()`'s `as_of` computation and the structural
  round-trip rejection.
