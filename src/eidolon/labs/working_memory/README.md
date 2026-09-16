# Working Memory Lab

## What this implements

A minimal, capacity-limited, decaying buffer inspired by the general idea of
working memory as a small "workbench" that holds a handful of active items
for a short time, distinct from long-term (episodic) storage.

- **Limited capacity** — a fixed number of item slots (default 4).
- **Decay** — every item's activation shrinks each tick; once it drops below
  `forget_threshold`, the item is forgotten and removed.
- **Rehearsal** — actively attending to an item resets its activation, so it
  can persist indefinitely if it keeps getting rehearsed while others decay.
- **Displacement** — inserting a new item when the buffer is already full
  evicts the *least-active* item, not necessarily the oldest one.

This is deliberately simple: no semantic content, no embeddings, no
consolidation into a longer-term store yet. It exists to be an inspectable
building block that later labs (attention, episodic memory) can read from or
write into.

## Question this experiment asks

Does a small set of decay + capacity + rehearsal rules reproduce the two
basic working-memory phenomena we'd expect: (1) new information can bump out
old information once the buffer is full, and (2) information that is
actively rehearsed resists both decay and eviction, while unattended
information fades out on its own?

## Run it

```bash
python -m eidolon.labs.working_memory.demo
```

## Files

- `buffer.py` — `WorkingMemory` and `Item`.
- `demo.py` — runnable scenario exercising capacity eviction, decay, and
  rehearsal.
- `tests/labs/working_memory/` — unit tests for the same behaviors.