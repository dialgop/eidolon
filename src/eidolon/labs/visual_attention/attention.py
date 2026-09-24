"""Object-level visual attention with bottom-up and top-down cues, blended
by a top-down factor `t` (see this lab's README for sources and for which
parts are our own adaptation)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from eidolon.percepts import PerceivedObject, Scene

_EPSILON = 1e-12
# Wide enough to leave the weight range reported in the original paper
# (~0.001 to ~77) untouched; only guards against 0 and division blow-ups.
_WEIGHT_LIMIT = 1000.0


@dataclass(frozen=True)
class Selection:
    percept: PerceivedObject
    salience: float


def _contrast_maps(scene: Scene) -> dict[str, list[float]]:
    """Per feature, two maps aligned with `scene.objects`: `<f>+` (object is
    higher than the mean of the *other* objects) and `<f>-` (lower). Objects
    lacking a feature get 0 in its maps; fewer than two holders -> all 0."""
    objects = scene.objects
    maps: dict[str, list[float]] = {}
    for name in sorted({n for o in objects for n in o.features}):
        holders = [i for i, o in enumerate(objects) if name in o.features]
        on = [0.0] * len(objects)
        off = [0.0] * len(objects)
        if len(holders) >= 2:
            total = sum(objects[i].features[name] for i in holders)
            for i in holders:
                value = objects[i].features[name]
                diff = value - (total - value) / (len(holders) - 1)
                if diff > 0:
                    on[i] = diff
                else:
                    off[i] = -diff
        maps[f"{name}+"] = on
        maps[f"{name}-"] = off
    return maps


def _normalize(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else [0.0] * len(values)


def learn_weights(training_scene: Scene, target_track_id: str) -> Mapping[str, float]:
    """Per contrast map, the ratio of the target's response to the mean
    response of the other objects. >1 means the map helps find the target,
    <1 means it's more present in the background. Keys match select()'s maps."""
    ids = [o.track_id for o in training_scene.objects]
    if target_track_id not in ids:
        raise ValueError(f"target {target_track_id!r} is not in the training scene")
    if len(ids) < 2:
        raise ValueError("training scene needs at least one object besides the target")
    target = ids.index(target_track_id)
    if not training_scene.objects[target].features:
        raise ValueError(f"target {target_track_id!r} has no features to learn from")
    weights: dict[str, float] = {}
    for name, values in _contrast_maps(training_scene).items():
        background = [v for i, v in enumerate(values) if i != target]
        ratio = (values[target] + _EPSILON) / (sum(background) / len(background) + _EPSILON)
        weights[name] = min(max(ratio, 1.0 / _WEIGHT_LIMIT), _WEIGHT_LIMIT)
    return MappingProxyType(weights)


@dataclass
class VisualAttention:
    """`suppression_duration` is in the caller's `Scene.observed_at` units.
    `peak_ratio`: a response counts as a peak of its map if it's at least
    this fraction of the map's maximum (used for uniqueness weighting)."""

    suppression_duration: int = 4
    peak_ratio: float = 0.5

    _suppressed_until: dict[str, int] = field(default_factory=dict, init=False)
    _last_observed_at: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.suppression_duration < 0:
            raise ValueError("suppression_duration must be >= 0")
        if not 0.0 < self.peak_ratio <= 1.0:
            raise ValueError("peak_ratio must be within (0, 1]")

    def select(
        self,
        scene: Scene,
        k: int,
        t: float = 0.0,
        weights: Mapping[str, float] | None = None,
    ) -> list[Selection]:
        """Up to `k` objects, most salient first (ties keep scene order).

        salience = (1 - t) * bottom_up + t * top_down, each map normalized
        by its maximum. `t=0` is pure bottom-up, `t=1` pure top-down;
        `weights` (from `learn_weights`) are required when `t > 0`.
        Objects already selected within the last `suppression_duration`
        units of `scene.observed_at` are excluded; everything returned here
        becomes suppressed. `observed_at` must not decrease between calls.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be an int >= 1")
        if not 0.0 <= t <= 1.0:
            raise ValueError("t must be within [0, 1]")
        if t > 0.0 and weights is None:
            raise ValueError("weights are required when t > 0")
        if weights is not None and any(not (w > 0 and math.isfinite(w)) for w in weights.values()):
            raise ValueError("weights must be positive and finite")
        if self._last_observed_at is not None and scene.observed_at < self._last_observed_at:
            raise ValueError("scene.observed_at must not decrease between calls")

        self._last_observed_at = scene.observed_at
        self._suppressed_until = {
            track_id: until
            for track_id, until in self._suppressed_until.items()
            if until > scene.observed_at
        }

        maps = _contrast_maps(scene)
        salience = self._bottom_up(maps, len(scene.objects))
        if t > 0.0:
            top_down = self._top_down(maps, weights, len(scene.objects))
            salience = [(1.0 - t) * b + t * d for b, d in zip(salience, top_down)]

        eligible = [
            i for i, o in enumerate(scene.objects) if o.track_id not in self._suppressed_until
        ]
        chosen = sorted(eligible, key=lambda i: -salience[i])[:k]
        for i in chosen:
            self._suppressed_until[scene.objects[i].track_id] = (
                scene.observed_at + self.suppression_duration
            )
        return [Selection(scene.objects[i], salience[i]) for i in chosen]

    def _bottom_up(self, maps: dict[str, list[float]], n: int) -> list[float]:
        total = [0.0] * n
        for values in maps.values():
            top = max(values, default=0.0)
            if top <= 0:
                continue
            peaks = sum(1 for v in values if v >= self.peak_ratio * top)
            scale = 1.0 / math.sqrt(peaks)
            for i, v in enumerate(values):
                total[i] += v * scale
        return _normalize(total)

    @staticmethod
    def _top_down(maps: dict[str, list[float]], weights: Mapping[str, float], n: int) -> list[float]:
        excitation = [0.0] * n
        inhibition = [0.0] * n
        for name, values in maps.items():
            w = weights.get(name, 1.0)
            if w > 1.0:
                for i, v in enumerate(values):
                    excitation[i] += w * v
            elif w < 1.0:
                for i, v in enumerate(values):
                    inhibition[i] += v / w
        return _normalize([max(e - i, 0.0) for e, i in zip(excitation, inhibition)])
