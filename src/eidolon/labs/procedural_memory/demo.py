"""Run with: python -m eidolon.labs.procedural_memory.demo

Three scenarios: a practice curve (execution speed improving with N, the
shape of Newell & Rosenbloom's power law, not the full model — see the
README), interference between two context-similar skills (a free
consequence of select()'s Luce normalization, not a separate mechanism),
and decay turning a recent habit into one that's effectively gone.
"""

from eidolon.labs.procedural_memory import ProceduralMemory
from eidolon.percepts import Embedding

SPACE = "demo-space"


def ctx(*vector: float) -> Embedding:
    return Embedding(vector=vector, space=SPACE)


def scenario_practice_curve() -> None:
    print("--- A. practice curve: (N+1)^-alpha, our own N=0 handling ---")
    pm = ProceduralMemory(space=SPACE, alpha=0.5, sensitivity=2.0, decay_rate=0.1)
    for n in range(4):
        pm.practice("grasp_cup", ctx(1.0, 0.0), timestamp=n)
    for n, relative_speed in pm.curve("grasp_cup"):
        print(f"  N={n}: relative execution time = {relative_speed:.3f}")


def scenario_interference() -> None:
    print("--- B. interference: a free consequence of Luce normalization ---")
    solo = ProceduralMemory(space=SPACE, alpha=0.5, sensitivity=2.0, decay_rate=0.1)
    solo.practice("grasp_cup", ctx(1.0, 0.0), timestamp=0)
    print(f"  one skill known: {solo.select(ctx(1.0, 0.0), now=1)}")

    contested = ProceduralMemory(space=SPACE, alpha=0.5, sensitivity=2.0, decay_rate=0.1)
    contested.practice("grasp_cup", ctx(1.0, 0.0), timestamp=0)
    contested.practice("grasp_mug", ctx(1.0, 0.0), timestamp=0)  # identical context, equal practice
    print(f"  a second, equally-practiced, context-identical skill exists:")
    print(f"    {contested.select(ctx(1.0, 0.0), now=1)}")
    print("  same evidence, now split between two competing skills — nothing else changed")


def scenario_decay_and_habit() -> None:
    print("--- C. decay: a recent habit vs. an old one ---")
    print("  (strength() is a pure function of elapsed time with no clock of its")
    print("   own — querying 'now' before a skill was ever practiced gives a")
    print("   negative elapsed time and an inflated, meaningless value, by design,")
    print("   the same way Belief.confidence_at never guards against this either;")
    print("   so every 'now' below is chosen to be after both skills exist.)")
    pm = ProceduralMemory(space=SPACE, alpha=0.5, sensitivity=2.0, decay_rate=0.3)
    pm.practice("old_habit", ctx(1.0, 0.0), timestamp=0)
    pm.practice("recent_habit", ctx(1.0, 0.0), timestamp=20)

    for now in (20, 21, 30, 50):
        print(f"  now={now}:")
        print(f"    old_habit strength:    {pm.strength('old_habit', ctx(1.0, 0.0), now=now):.4f}")
        print(f"    recent_habit strength: {pm.strength('recent_habit', ctx(1.0, 0.0), now=now):.4f}")


def main() -> None:
    scenario_practice_curve()
    print()
    scenario_interference()
    print()
    scenario_decay_and_habit()


if __name__ == "__main__":
    main()
