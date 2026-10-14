"""Run with: python -m eidolon.labs.semantic_memory.demo

Shows the chosen v0 design: semantic memory doesn't take arbitrary input
directly, any more than episodic memory does. It's fed from episodic
memory, the same way episodic memory is fed from working memory — this
script is the glue, assuming `episode.content` is a `PerceivedObject` (v0's
documented assumption) and pulling out `.label`/`.embedding` itself, rather
than `SemanticMemory` importing `PerceivedObject` or `Episode` at all.

Two scenarios: learning a handful of cup/plate sightings and classifying a
new one (the Luce-choice rule, not nearest-neighbor), and the rejection
threshold on an ambiguous query that doesn't clearly belong to either.
"""

from eidolon.labs.episodic_memory import EpisodicMemory
from eidolon.labs.semantic_memory import SemanticMemory
from eidolon.percepts import Embedding, PerceivedObject

SPACE = "demo-space"


def seen(label: str, track_id: str, *vector: float) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id, observed_at=0, label=label,
        embedding=Embedding(vector=vector, space=SPACE),
    )


def consolidate_into_semantic_memory(em: EpisodicMemory, sm: SemanticMemory) -> None:
    """The EM -> SemanticMemory glue: skips episodes with no embedding (not
    every episode has one) and assumes `content` is a `PerceivedObject` to
    reach its `.label` -- SemanticMemory itself knows neither type."""
    for episode in em.episodes:
        if episode.embedding is None:
            continue
        sm.learn(episode.content.label, episode.embedding)


def main() -> None:
    em = EpisodicMemory()
    for content in [
        seen("cup", "c1", 1.0, 0.1),
        seen("cup", "c2", 1.0, 0.05),
        seen("cup", "c3", 0.9, -0.1),
        seen("plate", "p1", 0.1, 1.0),
        seen("plate", "p2", 0.05, 1.0),
    ]:
        em.encode(content=content, context={}, provenance={"reason": "demo"}, embedding=content.embedding)

    sm = SemanticMemory(space=SPACE, sensitivity=2.0)
    consolidate_into_semantic_memory(em, sm)
    print(f"learned {len(sm)} concepts: {sm.labels}")
    for label in sm.labels:
        print(f"  {label}: {sm.exemplar_count(label)} exemplars, prototype={sm.prototype(label)}")

    print("\n--- classify a clear cup ---")
    clear_cup = Embedding(vector=(1.0, 0.0), space=SPACE)
    for match in sm.similar_concepts(clear_cup, k=2):
        print(f"  {match.label}: probability={match.probability:.3f}")
    print(f"classify(): {sm.classify(clear_cup)!r}")

    print("\n--- classify something ambiguous, with a rejection floor ---")
    ambiguous = Embedding(vector=(0.6, 0.6), space=SPACE)
    for match in sm.similar_concepts(ambiguous, k=2):
        print(f"  {match.label}: probability={match.probability:.3f}")
    print(f"classify() with no floor: {sm.classify(ambiguous)!r}")
    print(f"classify(min_probability=0.9): {sm.classify(ambiguous, min_probability=0.9)!r}")


if __name__ == "__main__":
    main()
