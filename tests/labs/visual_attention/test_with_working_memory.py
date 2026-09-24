from eidolon.percepts import PerceivedObject
from eidolon.labs.working_memory import WorkingMemory


def seen(observed_at, position):
    return PerceivedObject(
        track_id="cup-1", observed_at=observed_at, position=position, frame="robot_base"
    )


def test_same_tracked_object_refreshes_instead_of_duplicating_in_working_memory():
    wm = WorkingMemory()
    wm.add(seen(0, (1.0, 0.0, 0.0)))
    _, evicted = wm.add(seen(1, (1.0, 0.5, 0.0)))
    assert evicted is None
    assert len(wm) == 1


def test_working_memory_keeps_the_latest_observation():
    """Working memory's `add` replaces the stored content when a duplicate
    is re-added, so a re-perceived object's newer observation is kept.

    Formerly called `test_working_memory_as_is_keeps_the_stale_observation`,
    when it asserted the opposite to document the pre-round behavior;
    flipped and renamed in the working memory round."""
    wm = WorkingMemory()
    wm.add(seen(0, (1.0, 0.0, 0.0)))
    item, _ = wm.add(seen(1, (1.0, 0.5, 0.0)))
    assert item.content.observed_at == 1
    assert item.content.position == (1.0, 0.5, 0.0)
