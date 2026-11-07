import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, overload

Field = Literal["pose", "battery", "engine_temperature", "task_status"]

JOINT_GROUPS: tuple[str, ...] = ("head", "l_arm", "r_arm", "l_leg", "r_leg")

_FIELDS: frozenset[str] = frozenset(("pose", "battery", "engine_temperature", "task_status"))
_QUATERNION_TOLERANCE = 1e-3


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_field(field_name: str) -> None:
    if field_name not in _FIELDS:
        raise ValueError(f"unknown field {field_name!r}; must be one of {sorted(_FIELDS)}")


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # quaternion (x, y, z, w)
    frame: str

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("position must have exactly 3 components")
        if len(self.orientation) != 4:
            raise ValueError("orientation must have exactly 4 components (x, y, z, w)")
        if not self.frame:
            raise ValueError("frame must be non-empty")
        for v in self.position:
            _check_finite(v, "position component")
        for v in self.orientation:
            _check_finite(v, "orientation component")
        norm_sq = sum(c * c for c in self.orientation)
        if abs(norm_sq - 1.0) > _QUATERNION_TOLERANCE:
            raise ValueError(
                f"orientation must be a unit quaternion (x²+y²+z²+w² ≈ 1), got {norm_sq!r}"
            )


@dataclass(frozen=True)
class Battery:
    level: float
    charging: bool

    def __post_init__(self) -> None:
        _check_finite(self.level, "battery level")
        if not (0.0 <= self.level <= 1.0):
            raise ValueError(f"battery level must be within [0, 1], got {self.level!r}")


def _check_engine_temperature(engine_temperature: Mapping[str, float]) -> None:
    if set(engine_temperature) != set(JOINT_GROUPS):
        raise ValueError(
            f"engine_temperature must have exactly the keys {JOINT_GROUPS}, "
            f"got {tuple(engine_temperature)}"
        )
    for group, value in engine_temperature.items():
        _check_finite(value, f"engine_temperature[{group!r}]")


@dataclass(frozen=True)
class SelfState:
    """An incremental update from the adapter. A field left None means
    "nothing new to report in this message", not "unknown" — SelfModel
    keeps whatever it last knew for that field until a new value arrives.
    observed_at is in milliseconds, on a monotonic clock the adapter and
    every consumer calling SelfModel.freshness() must share (see the lab's
    README: a documented precondition, not something this lab enforces)."""

    observed_at: int
    pose: Pose | None = None
    battery: Battery | None = None
    engine_temperature: Mapping[str, float] | None = None
    task_status: str | None = None

    def __post_init__(self) -> None:
        if self.engine_temperature is not None:
            _check_engine_temperature(self.engine_temperature)


@dataclass(frozen=True)
class SelfSnapshot:
    """A read-only, full snapshot of currently known self-state, for
    logging, demos and visualization. Deliberately a different type from
    SelfState: it does not round-trip into SelfModel.update(), and as_of is
    not a per-field timestamp (see freshness() for that). A consumer making
    decisions should use query()/freshness(), not this."""

    pose: Pose | None
    battery: Battery | None
    engine_temperature: Mapping[str, float] | None
    task_status: str | None
    as_of: int | None


@dataclass
class SelfModel:
    """The agent's own body state, parallel in spirit to a world model but
    about the self rather than the world, and fed incrementally rather than
    as one atomic Scene-like snapshot: each field remembers its own
    last-updated time independently, so one field can be fresh while
    another is stale.

    Single-threaded by contract: update() is not safe to call concurrently
    with itself or with any query/freshness/state() call that needs a
    consistent read. A ROS 2 deployment with concurrent callbacks must
    serialize calls into one SelfModel itself (e.g. a single-threaded
    executor callback group, or external locking) — this lab does not do
    it for you.
    """

    frame: str
    _values: dict[str, object] = field(default_factory=dict, init=False)
    _timestamps: dict[str, int] = field(default_factory=dict, init=False)
    _last_timestamp: int | None = field(default=None, init=False)

    def update(self, state: SelfState) -> None:
        if self._last_timestamp is not None and state.observed_at < self._last_timestamp:
            raise ValueError(
                f"observed_at must not decrease: got {state.observed_at}, "
                f"last was {self._last_timestamp}"
            )
        self._last_timestamp = state.observed_at

        if state.pose is not None:
            if state.pose.frame != self.frame:
                raise ValueError(
                    f"pose frame {state.pose.frame!r} does not match "
                    f"SelfModel frame {self.frame!r}"
                )
            self._values["pose"] = state.pose
            self._timestamps["pose"] = state.observed_at

        if state.battery is not None:
            self._values["battery"] = state.battery
            self._timestamps["battery"] = state.observed_at

        if state.engine_temperature is not None:
            self._values["engine_temperature"] = dict(state.engine_temperature)
            self._timestamps["engine_temperature"] = state.observed_at

        if state.task_status is not None:
            self._values["task_status"] = state.task_status
            self._timestamps["task_status"] = state.observed_at

    def state(self) -> SelfSnapshot:
        as_of = max(self._timestamps.values()) if self._timestamps else None
        return SelfSnapshot(
            pose=self._values.get("pose"),
            battery=self._values.get("battery"),
            engine_temperature=self._values.get("engine_temperature"),
            task_status=self._values.get("task_status"),
            as_of=as_of,
        )

    @overload
    def query(self, field: Literal["pose"]) -> Pose | None: ...
    @overload
    def query(self, field: Literal["battery"]) -> Battery | None: ...
    @overload
    def query(self, field: Literal["engine_temperature"]) -> Mapping[str, float] | None: ...
    @overload
    def query(self, field: Literal["task_status"]) -> str | None: ...
    def query(self, field: Field) -> Pose | Battery | Mapping[str, float] | str | None:
        _check_field(field)
        return self._values.get(field)

    def freshness(self, field: Field, now: int) -> int | None:
        _check_field(field)
        timestamp = self._timestamps.get(field)
        if timestamp is None:
            return None
        return now - timestamp
