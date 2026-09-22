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
- Semantic memory *(added after the original roadmap; not yet in scope —
  see note below)*

Semantic memory (conceptual/categorical knowledge — "what a cup is", not
"the cup I saw") wasn't in the original roadmap above but belongs on it: a
belief store gives an object identity and persistence, not meaning. It's
deliberately deferred, not forgotten — it needs a representation layer
(embeddings) that doesn't exist yet, and needs temporal continuity (the
world model) to have raw material to generalize from. See the world model's
contract below for how these two will eventually connect.

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
  context, provenance)` (frozen; `context`/`provenance` are
  `MappingProxyType`, read-only).
- **Operations:** `encode(content, context, provenance) -> Episode`,
  `recall_by_content(cue, *, newest_first=False) -> list[Episode]`,
  `recall_by_context(cue, *, newest_first=False) -> list[Episode]`,
  `recall_by_provenance(cue, *, newest_first=False) -> list[Episode]`,
  `.episodes -> list[Episode]` (read-only snapshot), `len(em)`.
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
  exact-equality-or-nothing. This is the two-phase roadmap decided for this
  lab: **v0 exact/subset matching now, v1 similarity later**, once a
  representation lab (embeddings — not built) exists to feed it. Adding v1
  means a new `recall_by_similarity` method alongside these, not a
  replacement — but note an arbitrary `content`/cue doesn't carry a
  similarity-computable representation on its own; v1 will need a separate
  representation (e.g. an `embedding` field), not something derived from
  `content` by changing the comparison.
- **Recall order:** oldest-first (encoding order) by default; every
  `recall_by_*` takes `newest_first=True` to reverse it.
- **Deliberately out of scope for v0:** `action`, `outcome`, `keys`
  (multi-field cue indexing), `embedding`. Not stubbed as unused fields —
  simply not part of `Episode` yet.

### `eidolon.labs.visual_attention`

The input front end. Eidolon stays a pure cognition library: an embodiment
or adapter (scripted scenes today, Webots/NAO later) reports what it
perceives as `PerceivedObject`s, and this lab decides which ones get
selected. It does not know where objects come from, and detection, tracking
and feature extraction stay on the adapter side.

```text
adapter (outside Eidolon) → Scene → VisualAttention.select() → caller adds selected to WM
```

`select` returns its choice; it never writes to working memory. The caller
does, the same pattern as episodic consolidation, so labs stay decoupled.

- **Exposes:** `PerceivedObject`, `Scene`, `VisualAttention`, `Selection`,
  `learn_weights`.
- **`PerceivedObject`** (frozen): `track_id` (required, non-empty),
  `observed_at: int`, and optional `label`, `confidence` in `[0, 1]`, flat
  `features: Mapping[str, float]` (read-only), `position` + `frame`
  (`frame` required when `position` is given), `source`. **Equality and
  hashing use `track_id` only.** `position` is unused by v0 attention.
- **`Scene`** (frozen): `observed_at: int` and `objects: tuple[...]`. A
  list raises `TypeError`; `track_id`s must be unique within a scene.
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

### Clocks

Three separate clocks exist today, by conscious choice rather than
oversight: `working_memory`'s logical tick (advanced by `tick()`),
`episodic_memory`'s encode counter (advanced by `encode()`), and
`visual_attention`'s caller-supplied `Scene.observed_at`. None is wall-clock
time and none is shared, so each lab stays testable on its own and
deterministic. The consequence is that "3 ticks" means something different
in each lab. Unifying them behind one agent clock is deferred until
embodiment provides one; when it does, `PerceivedObject.observed_at` is the
natural event time for the `occurred_at`/`encoded_at` split in episodic
memory.

### Shared base interface (e.g. a `CognitiveModule` protocol)

Not defined yet, deliberately. With three labs built the common shape is
starting to show (an explicit time step, plus a read-only view of state),
but the labs advance time in three different ways (`tick()`, `encode()`,
caller-supplied `observed_at`), which is exactly the kind of difference a
premature protocol would paper over. Revisit once the clocks question is
settled.

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

## Long-term vision

Not to claim NAO becomes conscious, but to construct an increasingly
sophisticated artificial agent whose architecture contains functional
mechanisms inspired by theories of consciousness, place those mechanisms in
an embodied system, and experimentally study the resulting behavior.

> From cognitive concepts, to computational mechanisms, to an integrated
> cognitive architecture, to an embodied artificial agent.
