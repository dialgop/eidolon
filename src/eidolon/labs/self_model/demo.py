"""Run with: python -m eidolon.labs.self_model.demo

Three scenarios: independent per-field freshness after an incremental
update stream, the pose/task_status staleness ambiguity this lab
documents rather than resolves, and the state()-vs-query()/freshness()
split (one for logging/demos, the other for decisions).

All timestamps below are milliseconds, on one shared clock — see the
lab's README for why that unit is load-bearing and not just cosmetic.
"""

from eidolon.labs.self_model import JOINT_GROUPS, Battery, Pose, SelfModel, SelfState

FRAME = "nao_base_link"


def flat_temperatures(value: float) -> dict[str, float]:
    return {group: value for group in JOINT_GROUPS}


def scenario_independent_freshness() -> None:
    print("--- A. incremental updates: each field ages on its own clock ---")
    sm = SelfModel(frame=FRAME)

    sm.update(
        SelfState(
            observed_at=0,
            pose=Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), frame=FRAME),
            battery=Battery(level=0.9, charging=False),
        )
    )
    # Only pose gets refreshed at t=1000; battery is left alone.
    sm.update(
        SelfState(
            observed_at=1000,
            pose=Pose(position=(0.5, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), frame=FRAME),
        )
    )

    now = 1500
    print(f"  now={now}:")
    print(f"    freshness(pose)    = {sm.freshness('pose', now)} ms (refreshed at t=1000)")
    print(f"    freshness(battery) = {sm.freshness('battery', now)} ms (last set at t=0)")
    print(f"    freshness(task_status) = {sm.freshness('task_status', now)} (never reported)")


def scenario_pose_staleness_ambiguity() -> None:
    print("--- B. a stale pose is ambiguous on its own ---")
    sm = SelfModel(frame=FRAME)
    sm.update(
        SelfState(
            observed_at=0,
            pose=Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), frame=FRAME),
            task_status="idle",
        )
    )

    now = 60_000
    print(f"  now={now}, freshness(pose) = {sm.freshness('pose', now)} ms")
    print("  By itself this cannot distinguish 'idle, sensor just hasn't re-reported'")
    print("  from 'moved, sensor hasn't caught up' — that's why there's no is_moving")
    print(f"  field. Cross-checking: task_status = {sm.query('task_status')!r}")
    print("  ...tells us it's the former, not the latter, in this particular case.")


def scenario_state_vs_query() -> None:
    print("--- C. state() for logging, query()/freshness() for decisions ---")
    sm = SelfModel(frame=FRAME)
    sm.update(
        SelfState(
            observed_at=0,
            battery=Battery(level=0.2, charging=True),
            engine_temperature=flat_temperatures(35.0),
            task_status="executing",
        )
    )

    snapshot = sm.state()
    print(f"  state() -> battery={snapshot.battery}, task_status={snapshot.task_status!r}, "
          f"as_of={snapshot.as_of}")
    print("  (as_of is the max of the fields included above, not a per-field value;")
    print("   state()'s return type can't be fed back into update() — see README)")

    now = 5000
    battery = sm.query("battery")
    assert battery is not None
    print(f"  a decision would instead ask: query('battery') = {battery}, "
          f"freshness('battery', {now}) = {sm.freshness('battery', now)} ms")


def main() -> None:
    scenario_independent_freshness()
    print()
    scenario_pose_staleness_ambiguity()
    print()
    scenario_state_vs_query()


if __name__ == "__main__":
    main()
