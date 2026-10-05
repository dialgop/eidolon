"""Run with: python -m eidolon.labs.episodic_memory.demo

Shows the chosen v0 design: episodic memory doesn't take arbitrary input
directly. It consolidates whatever working_memory loses — both items
evicted by capacity (add()) and items forgotten by decay (tick()) — with no
selectivity: v0 consolidates everything that leaves working memory.
Selectivity in *encoding* is deferred until attention/goals exist; v0 only
has selectivity in *retrieval*, via recall cues.

Note `EpisodicMemory.encode()` never sees a working_memory `Item`, and never
receives wall-clock time — this script extracts plain fields (`content`)
and supplies `provenance` (why the episode was stored); `context` stays
`{}` in v0, since there's no attention/goals yet to produce real episodic
context, and `occurred_at` is stamped by episodic memory's own clock.

A second scenario shows `recall_by_similarity`: it ranks, it doesn't filter,
so unlike the other `recall_by_*` methods its default order is similarity,
not encoding time — `newest_first` only reverses the tie-break between
equally similar episodes, shown here both ways.
"""

from eidolon.labs.episodic_memory import EpisodicMemory
from eidolon.labs.working_memory import WorkingMemory
from eidolon.percepts import Embedding


def consolidate(em: EpisodicMemory, content: object, reason: str) -> None:
    episode = em.encode(content=content, context={}, provenance={"reason": reason})
    print(f"consolidated: {episode.content!r} (occurred_at={episode.occurred_at}, reason={reason})")


def main() -> None:
    wm = WorkingMemory(capacity=2, decay_rate=0.4, forget_threshold=0.05)
    em = EpisodicMemory()

    wm.add("keys")
    wm.add("wallet")  # same tick as "keys": activation tie
    _, evicted = wm.add("phone")  # capacity exceeded: activation tie broken by insertion order
    consolidate(em, evicted.content, reason="capacity_eviction")

    for step in range(1, 9):
        forgotten = wm.tick()
        for item in forgotten:
            consolidate(em, item.content, reason="decay_forgotten")

    print(f"\nepisodic memory now holds {len(em)} episodes")
    print("recall_by_content('keys'):", em.recall_by_content("keys"))
    print(
        "recall_by_provenance({'reason': 'decay_forgotten'}), newest first:",
        em.recall_by_provenance({"reason": "decay_forgotten"}, newest_first=True),
    )
    print(f"recall_by_context({{}}) — empty cue is a subset of everything: {len(em.recall_by_context({}))} results")

    print("\n--- recall_by_similarity: ranks, doesn't filter ---")
    em2 = EpisodicMemory()
    cup_shaped = Embedding(vector=(1.0, 0.1), space="demo-space")
    em2.encode(content="mug", context={}, provenance={}, embedding=cup_shaped)
    em2.encode(content="identical-looking mug", context={}, provenance={}, embedding=cup_shaped)
    em2.encode(content="plate", context={}, provenance={}, embedding=Embedding(vector=(0.1, 1.0), space="demo-space"))
    em2.encode(content="no embedding at all", context={}, provenance={})

    query = Embedding(vector=(1.0, 0.0), space="demo-space")
    print("oldest-first tie-break (default):")
    for match in em2.recall_by_similarity(query, k=3):
        print(f"  {match.key.content!r}: similarity={match.similarity:.3f}")
    print("newest-first tie-break (reverses only the tie, not the ranking):")
    for match in em2.recall_by_similarity(query, k=3, newest_first=True):
        print(f"  {match.key.content!r}: similarity={match.similarity:.3f}")


if __name__ == "__main__":
    main()
