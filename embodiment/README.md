# Embodiment Layer (C++, not yet started)

This directory will hold the ROS 2 / Webots / simulated-NAO interface that
lets the Python cognitive core (`src/eidolon/core/`, once it exists) drive a
physical/simulated body.

It is expected to be adapted from
[`spin_the_bottle`](https://github.com/dialgop/spin_the_bottle), reusing:

- ROS 2 integration
- The simulated NAO in Webots
- `ros2_control` for robot joints
- NAO head and arm control
- The Webots-based robot environment
- The computer-vision pipeline (C++17)
- Existing perception ↔ robot-action interfaces

The game-specific logic from that project is not carried over — only the
infrastructure needed to communicate with the simulated NAO and Webots. See
`/docs/architecture.md` for how this layer fits into the overall system and
how it talks to the Python side over ROS 2.

Nothing is implemented here yet: this layer starts once enough cognitive
micro-labs exist to be worth embodying.
