"""Run with: python -m eidolon.labs.world_model.demo

The motivating scenario: NAO sees a cup, the cup leaves view, the belief
persists (and fades) while unseen, and two different endings are shown —
decaying away on its own, versus being contradicted by looking at the
exact spot and finding nothing there.
"""

from eidolon.labs.world_model import WorldModel
from eidolon.percepts import CoverageRegion, PerceivedObject, Scene


def cup(observed_at: int, position=(1.0, 0.0, 0.0)) -> PerceivedObject:
    return PerceivedObject(
        track_id="cup", observed_at=observed_at, label="cup", position=position, frame="map"
    )


def report(wm: WorldModel, label: str) -> None:
    belief = wm.get("cup")
    if belief is None:
        print(f"{label}: no belief about the cup")
    else:
        print(f"{label}: cup believed at {belief.percept.position}, confidence={belief.confidence:.2f}")


def scenario_decays_away() -> None:
    print("--- A. unseen, decays away on its own ---")
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.1)
    wm.update(Scene(observed_at=0, objects=(cup(0),)))
    report(wm, "t=0, seen")

    for t in (3, 6, 9, 11):
        result = wm.update(Scene(observed_at=t, objects=()))
        belief = wm.get("cup")
        confidence = f"{belief.confidence_at(t, wm.decay_rate):.3f}" if belief else "n/a"
        print(f"t={t}, not seen: computed confidence={confidence}, forgotten={[b.track_id for b in result.forgotten]}")
    report(wm, "final")


def scenario_contradicted_by_looking() -> None:
    print("--- B. unseen, then contradicted by looking at its spot ---")
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.1)
    wm.update(Scene(observed_at=0, objects=(cup(0),)))
    report(wm, "t=0, seen")

    wm.update(Scene(observed_at=1, objects=()))
    report(wm, "t=1, not seen, no coverage yet")

    looked_at_the_shelf = CoverageRegion(center=(1.0, 0.0, 0.0), radius=0.3, frame="map")
    result = wm.update(Scene(observed_at=2, objects=(), coverage=(looked_at_the_shelf,)))
    print(f"t=2, looked at the shelf and it's empty: contradicted={[b.track_id for b in result.contradicted]}")
    report(wm, "final")


def main() -> None:
    scenario_decays_away()
    print()
    scenario_contradicted_by_looking()


if __name__ == "__main__":
    main()
