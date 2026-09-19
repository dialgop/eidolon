# Eidolon — Consciousness Lab

An experimental laboratory for studying artificial cognition and the
functional mechanisms associated with consciousness.

Rather than attempting to directly create or claim artificial consciousness,
this project progressively implements cognitive mechanisms inspired by
theories of consciousness (memory, attention, world models, self-models,
goals, planning, metacognition, a global workspace, predictive processing,
...) and studies how they interact once integrated into an **embodied**
agent — a simulated NAO humanoid robot running in Webots, controlled through
ROS 2.

See [`docs/architecture.md`](docs/architecture.md) for the full layered
architecture, the language split (Python for cognition, C++ for embodiment,
ROS 2 as the bridge), and the long-term roadmap.

## Layout

```
src/eidolon/
  labs/          cognitive micro-laboratories (Python) — isolated experiments
  core/          integrated cognitive core (Python) — not started yet
embodiment/      ROS 2 / Webots / NAO interface (C++) — not started yet,
                 will reuse infrastructure from spin_the_bottle
docs/            architecture and design notes
tests/           tests, mirroring src/eidolon/
```

## Status

- `labs/working_memory` — implemented (capacity-limited, decaying buffer).
- `labs/episodic_memory` — implemented (consolidates from working memory,
  exact-match cued recall).
- Everything else — not started yet.

See [`src/eidolon/labs/README.md`](src/eidolon/labs/README.md) for the
convention each lab follows.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

If this machine also has ROS 2 sourced in your shell (e.g. `/opt/ros/*/setup.bash`
in `.bashrc`), it'll leak into `PYTHONPATH` and can break the venv's `pytest`
plugin autoloading. Run with `PYTHONPATH= pytest` in that case, or use a
sub-shell that doesn't source ROS.

## Related

The embodiment layer will reuse infrastructure (ROS 2 setup, Webots NAO
simulation, `ros2_control`, vision pipeline) from
[`spin_the_bottle`](https://github.com/dialgop/spin_the_bottle), an existing
independent robotics/computer-vision project — without carrying over its
game-specific logic.
