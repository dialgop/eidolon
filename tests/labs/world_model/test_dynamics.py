import pytest

from eidolon.labs.world_model import WorldModel
from eidolon.percepts import CoverageRegion, PerceivedObject, Scene


def obj(track_id, observed_at, position=None, frame=None, confidence=1.0):
    return PerceivedObject(
        track_id=track_id, observed_at=observed_at, position=position, frame=frame, confidence=confidence
    )


def seen_at(observed_at, *objects, coverage=()):
    return Scene(observed_at=observed_at, objects=tuple(objects), coverage=coverage)


def cup(observed_at, position=(1.0, 0.0, 0.0), confidence=1.0):
    return obj("cup", observed_at, position=position, frame="map", confidence=confidence)


# --- observing: new, refreshed, replace-not-combine -------------------------


def test_first_observation_is_new():
    wm = WorldModel(frame="map")
    report = wm.update(seen_at(0, cup(0)))
    assert [b.track_id for b in report.new] == ["cup"]
    assert report.refreshed == report.contradicted == report.forgotten == ()
    assert len(wm) == 1


def test_second_observation_of_the_same_object_is_refreshed_not_new():
    wm = WorldModel(frame="map")
    wm.update(seen_at(0, cup(0)))
    report = wm.update(seen_at(1, cup(1)))
    assert report.new == ()
    assert [b.track_id for b in report.refreshed] == ["cup"]
    assert len(wm) == 1


def test_re_observation_replaces_position_and_confidence_outright():
    wm = WorldModel(frame="map")
    wm.update(seen_at(0, cup(0, position=(1.0, 0.0, 0.0), confidence=0.4)))
    wm.update(seen_at(1, cup(1, position=(2.0, 0.0, 0.0), confidence=0.9)))
    belief = wm.get("cup")
    assert belief.percept.position == (2.0, 0.0, 0.0)
    assert belief.confidence == 0.9
    assert belief.last_observed_at == 1


# --- decay: stored value never mutated for a surviving-but-unobserved belief


def test_unobserved_belief_survives_and_keeps_its_stored_confidence_unchanged():
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.1)
    wm.update(seen_at(0, cup(0)))
    wm.update(seen_at(5, ))
    belief = wm.get("cup")
    assert belief is not None
    assert belief.confidence == 1.0  # not decayed in storage
    assert belief.last_observed_at == 0  # not bumped by decay-only calls


def test_confidence_at_computes_correctly_from_the_true_origin():
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.1)
    wm.update(seen_at(0, cup(0)))
    wm.update(seen_at(5, ))
    belief = wm.get("cup")
    assert belief.confidence_at(5, 0.2) == pytest.approx(0.8**5)
    assert belief.confidence_at(8, 0.2) == pytest.approx(0.8**8)


def test_decay_does_not_compound_across_several_update_calls():
    """The regression this guards: if update() mutated the stored
    confidence in place each call while leaving last_observed_at fixed,
    repeated calls would double-decay. It must match one direct
    computation over the full elapsed span."""
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.01)
    wm.update(seen_at(0, cup(0)))
    for t in (1, 2, 3, 4, 5, 6, 7, 8):
        wm.update(seen_at(t))
    assert wm.get("cup").confidence_at(8, 0.2) == pytest.approx(0.8**8)


def test_belief_is_forgotten_once_decayed_confidence_crosses_the_threshold():
    wm = WorldModel(frame="map", decay_rate=0.2, forget_threshold=0.1)
    wm.update(seen_at(0, cup(0)))
    for t in range(1, 11):
        report = wm.update(seen_at(t))
        assert report.forgotten == ()
    assert len(wm) == 1
    report = wm.update(seen_at(11))
    assert [b.track_id for b in report.forgotten] == ["cup"]
    assert len(wm) == 0


def test_object_with_no_position_can_only_decay_never_be_contradicted():
    wm = WorldModel(frame="map", decay_rate=0.5, forget_threshold=0.1)
    wm.update(seen_at(0, obj("ghost", 0, confidence=1.0)))
    region = CoverageRegion(center=(0.0, 0.0, 0.0), radius=100.0, frame="map")
    report = wm.update(seen_at(1, coverage=(region,)))
    assert report.contradicted == ()
    assert wm.get("ghost") is not None


# --- contradiction: negative evidence from coverage --------------------------


def test_covered_and_absent_object_is_contradicted_not_decayed():
    wm = WorldModel(frame="map", decay_rate=0.01, forget_threshold=0.001)
    wm.update(seen_at(0, cup(0)))
    region = CoverageRegion(center=(1.0, 0.0, 0.0), radius=0.5, frame="map")
    report = wm.update(seen_at(1, coverage=(region,)))
    assert [b.track_id for b in report.contradicted] == ["cup"]
    assert report.forgotten == ()
    assert wm.get("cup") is None


def test_object_outside_coverage_radius_is_not_contradicted():
    wm = WorldModel(frame="map", decay_rate=0.01, forget_threshold=0.001)
    wm.update(seen_at(0, cup(0, position=(10.0, 0.0, 0.0))))
    region = CoverageRegion(center=(0.0, 0.0, 0.0), radius=0.5, frame="map")
    report = wm.update(seen_at(1, coverage=(region,)))
    assert report.contradicted == ()
    assert wm.get("cup") is not None


def test_no_coverage_means_no_contradiction_ever_possible():
    """Documents the conscious limitation: without a coverage claim there
    is no negative evidence, so a belief can only decay, never be
    contradicted, no matter how obviously it should be gone."""
    wm = WorldModel(frame="map", decay_rate=0.01, forget_threshold=0.001)
    wm.update(seen_at(0, cup(0)))
    report = wm.update(seen_at(1))
    assert report.contradicted == ()
    assert wm.get("cup") is not None


def test_contradiction_takes_priority_over_a_decay_that_would_also_fire():
    wm = WorldModel(frame="map", decay_rate=0.9, forget_threshold=0.5)
    wm.update(seen_at(0, cup(0)))
    region = CoverageRegion(center=(1.0, 0.0, 0.0), radius=0.5, frame="map")
    report = wm.update(seen_at(5, coverage=(region,)))  # would also have decayed past threshold
    assert [b.track_id for b in report.contradicted] == ["cup"]
    assert report.forgotten == ()


def test_re_observing_inside_a_covered_region_is_refreshed_not_contradicted():
    wm = WorldModel(frame="map", decay_rate=0.01, forget_threshold=0.001)
    wm.update(seen_at(0, cup(0)))
    region = CoverageRegion(center=(1.0, 0.0, 0.0), radius=0.5, frame="map")
    report = wm.update(seen_at(1, cup(1), coverage=(region,)))
    assert [b.track_id for b in report.refreshed] == ["cup"]
    assert report.contradicted == ()


# --- validation ---------------------------------------------------------------


def test_observed_at_must_not_decrease():
    wm = WorldModel(frame="map")
    wm.update(seen_at(5))
    with pytest.raises(ValueError):
        wm.update(seen_at(4))


def test_rejected_update_changes_nothing():
    wm = WorldModel(frame="map")
    wm.update(seen_at(5, cup(5)))
    with pytest.raises(ValueError):
        wm.update(seen_at(4))
    assert wm.get("cup").last_observed_at == 5


def test_object_frame_mismatch_raises_before_any_mutation():
    wm = WorldModel(frame="map")
    with pytest.raises(ValueError):
        wm.update(seen_at(0, obj("x", 0, position=(0.0, 0.0, 0.0), frame="odom")))
    assert len(wm) == 0


def test_coverage_frame_mismatch_raises_before_any_mutation():
    wm = WorldModel(frame="map")
    wrong_frame_region = CoverageRegion(center=(0.0, 0.0, 0.0), radius=1.0, frame="odom")
    with pytest.raises(ValueError):
        wm.update(seen_at(0, cup(0), coverage=(wrong_frame_region,)))
    assert len(wm) == 0


def test_object_without_position_needs_no_frame():
    wm = WorldModel(frame="map")
    report = wm.update(seen_at(0, obj("ghost", 0)))
    assert [b.track_id for b in report.new] == ["ghost"]
