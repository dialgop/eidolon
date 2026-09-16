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
- **Time semantics:** `tick()` is a logical step, not wall-clock time.
  `decay_rate`/`forget_threshold` are only meaningful relative to how often
  the caller ticks — a consumer driving this from a real control loop (e.g.
  ROS 2 at ~10 Hz) must recalibrate these constants against that tick rate,
  not reuse the lab's demo defaults.
- **Not yet decided:** whether future consumers need similarity-based (not
  just exact-match) recall. This is deferred to when the episodic memory
  lab is scoped, since it determines whether `working_memory` needs a
  `get_by_similarity`-style operation or stays exact-match-only while
  episodic memory adds similarity on its own store.

### `eidolon.labs.episodic_memory` — not started yet

Retrieval strategy decided as a two-phase roadmap, not a one-time
exact-vs-similarity choice:

- **v0 (to build now): exact-match cued recall**, consistent with
  `working_memory`. An episode is retrieved by matching a cue against
  stored content/context with `==`-style comparison — no embeddings, no
  similarity metric. This is the only retrieval operation v0 implements.
- **v1 (later, optional): similarity-based cued recall**, added once a
  representation lab (something that produces embeddings/feature vectors —
  not built yet) exists to feed it. Not implemented now; the interface
  should not need to be redesigned to add it later.
- **What "prepared for both" means concretely:** cues are passed as
  arbitrary values (no `str`/hashable-only typing that would preclude a
  vector cue later), and recall is exposed as its own method (e.g.
  `recall(cue) -> list[Episode]`) rather than inlined dict/key lookups, so
  a v1 `recall_by_similarity(cue) -> list[Episode]` can be added alongside
  it without changing how v0's `recall` is called. Same reversibility
  discipline as `working_memory`'s `_find`: exact-match is v0's
  implementation choice, not a permanent contract.

### Shared base interface (e.g. a `CognitiveModule` protocol)

Not defined yet, deliberately. With only one lab built, any shared
interface would be guessed rather than extracted from real overlap.
Revisit once episodic memory exists and the two labs' actual common shape
(likely something like `tick()` plus a read-only view of state) is visible.

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
