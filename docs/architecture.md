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
