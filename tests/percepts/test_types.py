import dataclasses

import pytest

from eidolon.percepts import CoverageRegion, PerceivedObject, Scene


def obj(track_id="a", observed_at=0, **kwargs):
    return PerceivedObject(track_id=track_id, observed_at=observed_at, **kwargs)


def region(**kwargs):
    defaults = dict(center=(0.0, 0.0, 0.0), radius=1.0, frame="robot_base")
    return CoverageRegion(**{**defaults, **kwargs})


def test_equality_and_hash_use_track_id_only():
    first = obj("a", observed_at=1, label="cup", confidence=0.9, features={"hue": 0.1})
    later = obj("a", observed_at=5, label="mug", confidence=0.4, features={"hue": 0.9})
    assert first == later
    assert hash(first) == hash(later)
    assert len({first, later}) == 1


def test_different_track_id_is_a_different_object():
    assert obj("a") != obj("b")


def test_perceived_object_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj("a").label = "cup"


def test_features_are_copied_and_read_only():
    features = {"hue": 0.1}
    percept = obj("a", features=features)
    features["hue"] = 0.9
    assert percept.features["hue"] == 0.1
    with pytest.raises(TypeError):
        percept.features["hue"] = 0.5


def test_empty_track_id_rejected():
    with pytest.raises(ValueError):
        obj("")


def test_confidence_must_be_within_unit_interval():
    with pytest.raises(ValueError):
        obj("a", confidence=1.5)


def test_position_requires_frame():
    with pytest.raises(ValueError):
        obj("a", position=(0.0, 0.0, 0.0))
    assert obj("a", position=(0.0, 0.0, 0.0), frame="robot_base").frame == "robot_base"


def test_non_finite_feature_rejected():
    with pytest.raises(ValueError):
        obj("a", features={"hue": float("nan")})


def test_scene_requires_tuple_of_objects():
    with pytest.raises(TypeError):
        Scene(observed_at=0, objects=[obj("a")])


def test_scene_rejects_duplicate_track_ids():
    with pytest.raises(ValueError):
        Scene(observed_at=0, objects=(obj("a"), obj("a")))


def test_scene_is_frozen():
    scene = Scene(observed_at=0, objects=(obj("a"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        scene.observed_at = 1


def test_scene_coverage_defaults_to_empty():
    scene = Scene(observed_at=0, objects=(obj("a"),))
    assert scene.coverage == ()


def test_scene_requires_tuple_of_coverage_regions():
    with pytest.raises(TypeError):
        Scene(observed_at=0, objects=(), coverage=[region()])


def test_coverage_region_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        region().radius = 2.0


def test_coverage_region_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        region(radius=0.0)
    with pytest.raises(ValueError):
        region(radius=-1.0)


def test_coverage_region_rejects_empty_frame():
    with pytest.raises(ValueError):
        region(frame="")
