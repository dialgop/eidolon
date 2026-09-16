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

## Design notes

- **Content is discrete/symbolic, not vector-based.** `Item.content` is
  compared with `==`; there's no embedding space or similarity metric. This
  is a real fork in the road (symbolic vs. connectionist), made implicitly
  by "keep it minimal" rather than decided on purpose — worth revisiting
  once a lab needs cue-based (rather than exact-match) recall.
- **Eviction is activation-only, not "activation + recency."** It might look
  like a hybrid of a Cowan-style activation model and LRU, but it isn't one
  in practice: since `decay_rate` is a single constant shared by every item
  in a buffer, activation is a strictly monotonic function of
  `clock_now - last_accessed`. Two items can only tie in activation if
  they also tie in `last_accessed` — so a `last_accessed` tiebreaker can
  never resolve anything `activation` alone didn't already leave tied. Real
  ties only happen when items were last touched (added or rehearsed) in the
  exact same tick, and those are broken by plain insertion order (oldest
  inserted evicted first), not by a separate recency signal.
- **`add()` refreshes, it doesn't duplicate.** Adding content that's already
  present resets that item's activation instead of creating a second
  `Item`. The buffer models a set of currently-active concepts, not a
  token stream, so a duplicate would just waste a capacity slot without
  encoding new information.
- **Decay is per logical tick, not wall-clock time or per-access.** `tick()`
  must be called explicitly; nothing decays on its own. This keeps the
  model deterministic and easy to test, but means "one tick" has no fixed
  real-world duration yet — that mapping will matter once this is driven by
  an actual control loop.

## Run it

```bash
python -m eidolon.labs.working_memory.demo
```

## Files

- `buffer.py` — `WorkingMemory` and `Item`.
- `demo.py` — runnable scenario exercising capacity eviction, decay, and
  rehearsal.
- `tests/labs/working_memory/` — unit tests for the same behaviors.