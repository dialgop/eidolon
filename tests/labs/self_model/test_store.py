import pytest

from eidolon.labs.self_model import JOINT_GROUPS, Battery, Pose, SelfModel, SelfState
from eidolon.labs.self_model.store import SelfSnapshot

FRAME = "nao_base_link"
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def pose(*, position=(0.0, 0.0, 0.0), orientation=IDENTITY, frame=FRAME):
    return Pose(position=position, orientation=orientation, frame=frame)


def temps(value=30.0, **overrides):
    out = {group: value for group in JOINT_GROUPS}
    out.update(overrides)
    return out


# --- Pose validation ---------------------------------------------------------


def test_pose_rejects_wrong_position_length():
    with pytest.raises(ValueError):
        Pose(position=(0.0, 0.0), orientation=IDENTITY, frame=FRAME)


def test_pose_rejects_wrong_orientation_length():
    with pytest.raises(ValueError):
        Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), frame=FRAME)


def test_pose_rejects_empty_frame():
    with pytest.raises(ValueError):
        Pose(position=(0.0, 0.0, 0.0), orientation=IDENTITY, frame="")


def test_pose_rejects_non_finite_position():
    with pytest.raises(ValueError):
        Pose(position=(float("nan"), 0.0, 0.0), orientation=IDENTITY, frame=FRAME)


def test_pose_rejects_non_finite_orientation():
    with pytest.raises(ValueError):
        Pose(position=(0.0, 0.0, 0.0), orientation=(float("inf"), 0.0, 0.0, 1.0), frame=FRAME)


def test_pose_accepts_unit_quaternion():
    pose(orientation=(0.0, 0.0, 0.70710678, 0.70710678))  # does not raise


def test_pose_rejects_non_unit_quaternion():
    with pytest.raises(ValueError):
        pose(orientation=(1.0, 1.0, 1.0, 1.0))


def test_pose_tolerates_float_roundoff_in_quaternion():
    pose(orientation=(0.0, 0.0, 0.0, 1.0000000000000002))  # does not raise


# --- Battery validation -------------------------------------------------------


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_battery_accepts_level_within_range(level):
    Battery(level=level, charging=False)  # does not raise


@pytest.mark.parametrize("level", [-0.01, 1.01])
def test_battery_rejects_level_outside_range(level):
    with pytest.raises(ValueError):
        Battery(level=level, charging=False)


def test_battery_rejects_non_finite_level():
    with pytest.raises(ValueError):
        Battery(level=float("nan"), charging=False)


# --- SelfState validation -----------------------------------------------------


def test_self_state_accepts_all_fields_none():
    SelfState(observed_at=0)  # does not raise


def test_self_state_accepts_exact_joint_group_keys():
    SelfState(observed_at=0, engine_temperature=temps())  # does not raise


def test_self_state_rejects_missing_joint_group_key():
    bad = temps()
    del bad["head"]
    with pytest.raises(ValueError):
        SelfState(observed_at=0, engine_temperature=bad)


def test_self_state_rejects_unknown_joint_group_key():
    bad = temps()
    bad["tail"] = 30.0
    with pytest.raises(ValueError):
        SelfState(observed_at=0, engine_temperature=bad)


def test_self_state_rejects_non_finite_temperature():
    with pytest.raises(ValueError):
        SelfState(observed_at=0, engine_temperature=temps(head=float("nan")))


# --- SelfModel construction ---------------------------------------------------


def test_self_model_starts_with_nothing_reported():
    sm = SelfModel(frame=FRAME)
    snapshot = sm.state()
    assert snapshot == SelfSnapshot(
        pose=None, battery=None, engine_temperature=None, task_status=None, as_of=None
    )
    assert sm.query("pose") is None
    assert sm.freshness("pose", now=100) is None


# --- update(): monotonicity ---------------------------------------------------


def test_update_rejects_decreasing_observed_at():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=10, task_status="idle"))
    with pytest.raises(ValueError):
        sm.update(SelfState(observed_at=9, task_status="executing"))


def test_update_accepts_equal_observed_at():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=10, task_status="idle"))
    sm.update(SelfState(observed_at=10, battery=Battery(level=0.5, charging=False)))  # no raise


def test_rejected_update_changes_nothing():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=10, task_status="idle"))
    with pytest.raises(ValueError):
        sm.update(SelfState(observed_at=5, task_status="executing"))
    assert sm.query("task_status") == "idle"
    assert sm.freshness("task_status", now=10) == 0


# --- update(): frame validation -----------------------------------------------


def test_update_rejects_pose_in_a_different_frame():
    sm = SelfModel(frame=FRAME)
    with pytest.raises(ValueError):
        sm.update(SelfState(observed_at=0, pose=pose(frame="odom")))


# --- update(): partial, independent per-field merging -------------------------


def test_update_only_touches_fields_present_in_the_message():
    sm = SelfModel(frame=FRAME)
    sm.update(
        SelfState(
            observed_at=0,
            pose=pose(),
            battery=Battery(level=0.9, charging=False),
        )
    )
    sm.update(SelfState(observed_at=100, pose=pose(position=(1.0, 0.0, 0.0))))

    assert sm.freshness("pose", now=150) == 50
    assert sm.freshness("battery", now=150) == 150
    assert sm.query("battery") == Battery(level=0.9, charging=False)


def test_an_empty_update_still_advances_the_clock_and_touches_nothing():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=0, task_status="idle"))
    sm.update(SelfState(observed_at=50))  # no fields given
    assert sm.query("task_status") == "idle"
    assert sm.freshness("task_status", now=50) == 50


def test_an_empty_update_still_rejects_a_decrease():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=50))
    with pytest.raises(ValueError):
        sm.update(SelfState(observed_at=40))


# --- query() -------------------------------------------------------------------


def test_query_rejects_unknown_field():
    sm = SelfModel(frame=FRAME)
    with pytest.raises(ValueError):
        sm.query("velocity")


def test_query_returns_unconditionally_even_when_stale():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=0, task_status="idle"))
    assert sm.query("task_status") == "idle"
    assert sm.freshness("task_status", now=10_000_000) == 10_000_000
    assert sm.query("task_status") == "idle"  # never hidden for being stale


# --- freshness() ---------------------------------------------------------------


def test_freshness_rejects_unknown_field():
    sm = SelfModel(frame=FRAME)
    with pytest.raises(ValueError):
        sm.freshness("velocity", now=0)


def test_freshness_never_reported_is_none():
    sm = SelfModel(frame=FRAME)
    assert sm.freshness("battery", now=0) is None


def test_freshness_is_raw_elapsed_time_no_decay():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=100, battery=Battery(level=0.5, charging=True)))
    assert sm.freshness("battery", now=100) == 0
    assert sm.freshness("battery", now=150) == 50


def test_freshness_with_now_before_timestamp_is_negative_by_design():
    # Not guarded against, the same way Belief.confidence_at and
    # procedural_memory.strength don't guard against it either — see README.
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=100, battery=Battery(level=0.5, charging=True)))
    assert sm.freshness("battery", now=0) == -100


# --- state() --------------------------------------------------------------------


def test_state_merges_every_field_independently_of_timing():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=0, battery=Battery(level=0.9, charging=False)))
    sm.update(SelfState(observed_at=100, task_status="executing"))

    snapshot = sm.state()
    assert snapshot.battery == Battery(level=0.9, charging=False)
    assert snapshot.task_status == "executing"
    assert snapshot.pose is None
    assert snapshot.engine_temperature is None
    assert snapshot.as_of == 100  # max of the two reported fields' timestamps


def test_state_as_of_is_none_when_nothing_reported():
    sm = SelfModel(frame=FRAME)
    assert sm.state().as_of is None


def test_state_does_not_round_trip_into_update():
    sm = SelfModel(frame=FRAME)
    sm.update(SelfState(observed_at=0, task_status="idle"))
    snapshot = sm.state()
    with pytest.raises(AttributeError):
        sm.update(snapshot)  # SelfSnapshot has no .observed_at — structurally rejected
