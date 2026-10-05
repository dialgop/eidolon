"""Run with: python -m eidolon.labs.world_model.demo

The motivating scenario: NAO sees a cup, the cup leaves view, the belief
persists (and fades) while unseen, and two different endings are shown —
decaying away on its own, versus being contradicted by looking at the
exact spot and finding nothing there.

A third scenario shows re-identification: the cup's tracker loses it and
re-detects it under a new track id (occlusion, a common tracking failure)
near where it was last seen — appearance plus position together fold the
new id into the existing belief instead of spawning a duplicate. A fourth
shows the documented limitation: two indistinguishable objects can't be
told apart by appearance, so re-identification picks one.
"""

from eidolon.labs.world_model import WorldModel
from eidolon.percepts import CoverageRegion, Embedding, PerceivedObject, Scene


def cup(observed_at: int, position=(1.0, 0.0, 0.0)) -> PerceivedObject:
    return PerceivedObject(
        track_id="cup", observed_at=observed_at, label="cup", position=position, frame="map"
    )


def report(wm: WorldModel, label: str, track_id: str = "cup") -> None:
    belief = wm.get(track_id)
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


def cup_shaped(track_id: str, observed_at: int, position=(1.0, 0.0, 0.0)) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id, observed_at=observed_at, label="cup", position=position, frame="map",
        embedding=Embedding(vector=(1.0, 0.1), space="demo-space"),
    )


def scenario_reidentification() -> None:
    print("--- C. tracker loses the cup, re-detects it under a new id ---")
    wm = WorldModel(
        frame="map", decay_rate=0.2, forget_threshold=0.1,
        reidentify_threshold=0.9, reidentify_max_distance=0.5,
    )
    wm.update(Scene(observed_at=0, objects=(cup_shaped("cup-7", 0, position=(1.0, 0.0, 0.0)),)))
    report(wm, "t=0, tracked as cup-7", track_id="cup-7")

    wm.update(Scene(observed_at=1, objects=()))
    print("t=1, tracker lost it (occlusion)")

    result = wm.update(Scene(observed_at=2, objects=(cup_shaped("cup-12", 2, position=(1.05, 0.0, 0.0)),)))
    print(f"t=2, re-detected as cup-12: reidentified={result.reidentified}")
    print(f"get('cup-7') is get('cup-12'): {wm.get('cup-7') is wm.get('cup-12')}")
    print(f"the belief's own track_id is now: {wm.get('cup-7').track_id!r} (last observed as, not the original key)")


def scenario_reidentification_ambiguity() -> None:
    print("--- D. two indistinguishable cups: a known limitation ---")
    wm = WorldModel(
        frame="map", decay_rate=0.2, forget_threshold=0.1,
        reidentify_threshold=0.9, reidentify_max_distance=10.0,
    )
    wm.update(Scene(observed_at=0, objects=(
        cup_shaped("cup-left", 0, position=(0.0, 0.0, 0.0)),
        cup_shaped("cup-right", 0, position=(2.0, 0.0, 0.0)),
    )))
    wm.update(Scene(observed_at=1, objects=()))
    result = wm.update(Scene(observed_at=2, objects=(cup_shaped("cup-new", 2, position=(1.0, 0.0, 0.0)),)))
    print(f"equidistant from both, identical appearance: reidentified against {result.reidentified[0].canonical_track_id!r}")
    print("(earliest-created wins — a real misattribution is possible here, not detected)")


def main() -> None:
    scenario_decays_away()
    print()
    scenario_contradicted_by_looking()
    print()
    scenario_reidentification()
    print()
    scenario_reidentification_ambiguity()


if __name__ == "__main__":
    main()
