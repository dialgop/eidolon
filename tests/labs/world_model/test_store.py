import dataclasses

import pytest

from eidolon.labs.world_model import Belief, UpdateReport, WorldModel
from eidolon.percepts import PerceivedObject


def percept(track_id="cup", **kwargs):
    return PerceivedObject(track_id=track_id, observed_at=0, **kwargs)


def belief(track_id="cup", confidence=1.0, last_observed_at=0, **percept_kwargs):
    return Belief(
        percept=percept(track_id, **percept_kwargs),
        confidence=confidence,
        last_observed_at=last_observed_at,
    )


# --- Belief ----------------------------------------------------------------


def test_belief_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        belief().confidence = 0.5


def test_belief_track_id_and_label_are_read_from_the_percept():
    b = belief(track_id="cup", label="a cup")
    assert b.track_id == "cup"
    assert b.label == "a cup"


def test_belief_label_is_none_when_percept_has_no_label():
    assert belief().label is None


def test_belief_confidence_must_be_within_unit_interval():
    with pytest.raises(ValueError):
        belief(confidence=1.5)
    with pytest.raises(ValueError):
        belief(confidence=-0.1)


# --- UpdateReport ------------------------------------------------------------


def test_update_report_defaults_to_all_empty():
    report = UpdateReport()
    assert report.new == report.refreshed == report.contradicted == report.forgotten == ()


def test_update_report_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        UpdateReport().new = (belief(),)


# --- WorldModel construction and read-only accessors -------------------------


def test_world_model_starts_empty():
    wm = WorldModel(frame="robot_base")
    assert wm.beliefs == []
    assert len(wm) == 0
    assert wm.get("cup") is None


def test_world_model_rejects_empty_frame():
    with pytest.raises(ValueError):
        WorldModel(frame="")


def test_world_model_rejects_negative_decay_rate():
    with pytest.raises(ValueError):
        WorldModel(frame="robot_base", decay_rate=-0.1)


def test_world_model_rejects_forget_threshold_outside_unit_interval():
    with pytest.raises(ValueError):
        WorldModel(frame="robot_base", forget_threshold=0.0)
    with pytest.raises(ValueError):
        WorldModel(frame="robot_base", forget_threshold=1.1)


def test_world_model_accepts_forget_threshold_of_one():
    WorldModel(frame="robot_base", forget_threshold=1.0)
