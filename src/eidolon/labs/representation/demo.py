"""Run with: python -m eidolon.labs.representation.demo

Synthetic 4-dimensional "appearance" vectors stand in for what an adapter
would get from a real model (e.g. CLIP); nothing here needs a GPU. Four
scenarios:

  A. Nearest neighbours: which stored objects look most like a new one?
  B. The phenomenon: recognition as a hit / false-alarm trade-off. Sweeping
     the threshold gives a curve. With well-separated categories some
     threshold is perfect; with overlapping ones none is.
  C. Similarity is not identity: two different objects can look identical.
  D. Embeddings from different spaces refuse to be compared.
"""

import itertools

from eidolon.labs.representation import most_similar, recognition_rates, similarity
from eidolon.percepts import Embedding

SPACE = "synthetic-4d"


def emb(*vector: float, space: str = SPACE) -> Embedding:
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


CUPS = [emb(1, 0.05, 0, 0), emb(1, -0.05, 0, 0), emb(1, 0, 0.05, 0), emb(1, 0, -0.05, 0)]
PLATES = [emb(0.05, 1, 0, 0), emb(-0.05, 1, 0, 0), emb(0, 1, 0.05, 0), emb(0, 1, -0.05, 0)]
MUGS = [emb(1, 0.6, 0, 0), emb(1, 0.4, 0.3, 0), emb(1, 0.8, -0.3, 0), emb(1, 0.6, 0, 0.3)]
BOWLS = [emb(0.6, 1, 0, 0), emb(0.4, 1, 0.3, 0), emb(0.8, 1, -0.3, 0), emb(0.6, 1, 0, 0.3)]

THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]


def pairs_within(*groups):
    return [pair for group in groups for pair in itertools.combinations(group, 2)]


def pairs_between(first, second):
    return [(a, b) for a in first for b in second]


def scenario_neighbours() -> None:
    print("--- A. nearest neighbours of a new cup ---")
    stored = [(f"cup-{i}", e) for i, e in enumerate(CUPS)] + [(f"plate-{i}", e) for i, e in enumerate(PLATES)]
    new_cup = emb(1, 0.02, 0.01, 0)
    for match in most_similar(new_cup, stored, k=3):
        print(f"  {match.key}: similarity {match.similarity:.3f}")
    kept = most_similar(new_cup, stored, k=8, min_similarity=0.5)
    print(f"  with min_similarity=0.5, {len(kept)} of {len(stored)} stored objects qualify")


def curve(title: str, same, different) -> None:
    print(f"  {title}")
    print("    threshold   hits                  false alarms")
    for threshold in THRESHOLDS:
        rates = recognition_rates(same, different, threshold)
        hits = f"{rates.hit_rate:4.2f} {'#' * round(rates.hit_rate * 10):<10}"
        false_alarms = f"{rates.false_alarm_rate:4.2f} {'#' * round(rates.false_alarm_rate * 10):<10}"
        print(f"    {threshold:>8.2f}    {hits}    {false_alarms}")
    grid = [t / 100 for t in range(-100, 101)]
    perfect = [
        t for t in grid
        if (r := recognition_rates(same, different, t)).hit_rate == 1.0 and r.false_alarm_rate == 0.0
    ]
    if perfect:
        print(f"    a perfect threshold exists: {perfect[0]:.2f} to {perfect[-1]:.2f}")
    else:
        print("    no perfect threshold exists: every choice trades hits against false alarms")


def scenario_tradeoff() -> None:
    print("--- B. recognition threshold sweep ---")
    curve("well-separated categories (cups vs plates)", pairs_within(CUPS, PLATES), pairs_between(CUPS, PLATES))
    print()
    curve("overlapping categories (mugs vs bowls)", pairs_within(MUGS, BOWLS), pairs_between(MUGS, BOWLS))


def scenario_identity() -> None:
    print("--- C. similarity is not identity ---")
    mug = emb(0.3, 0.9, 0.1, 0.2)
    left, right = ("mug-on-the-left", mug), ("mug-on-the-right", mug)
    for match in most_similar(mug, [left, right], k=2):
        print(f"  {match.key}: similarity {match.similarity:.3f}")
    print("  two different objects, perfect match: telling them apart needs position and time too")


def scenario_spaces() -> None:
    print("--- D. different spaces are not comparable ---")
    try:
        similarity(emb(1, 0, 0, 0, space="clip-vit-b32"), emb(1, 0, 0, 0, space="dinov2-small"))
    except ValueError as error:
        print(f"  refused: {error}")


def main() -> None:
    scenario_neighbours()
    print()
    scenario_tradeoff()
    print()
    scenario_identity()
    print()
    scenario_spaces()


if __name__ == "__main__":
    main()
