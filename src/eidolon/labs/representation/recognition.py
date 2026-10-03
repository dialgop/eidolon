"""Measuring recognition as a hit / false-alarm trade-off at a threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eidolon.labs.representation.similarity import similarity
from eidolon.percepts import Embedding


@dataclass(frozen=True)
class RecognitionRates:
    """`hit_rate`: fraction of same-category pairs judged "same".
    `false_alarm_rate`: fraction of different-category pairs also judged
    "same". Both rise and fall together as the threshold moves."""

    hit_rate: float
    false_alarm_rate: float


def recognition_rates(
    same_pairs: Iterable[tuple[Embedding, Embedding]],
    different_pairs: Iterable[tuple[Embedding, Embedding]],
    threshold: float,
) -> RecognitionRates:
    """Judge every pair "same" when its cosine similarity is at or above
    `threshold` (inclusive), and report how many of each kind were so
    judged. `threshold` must be within [-1, 1], and both kinds of pair must
    be non-empty (a rate over no pairs is undefined). Pairs from different
    spaces raise, as in `similarity`."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [-1, 1]")
    return RecognitionRates(
        hit_rate=_fraction_at_or_above(same_pairs, threshold, "same"),
        false_alarm_rate=_fraction_at_or_above(different_pairs, threshold, "different"),
    )


def _fraction_at_or_above(
    pairs: Iterable[tuple[Embedding, Embedding]], threshold: float, kind: str
) -> float:
    total = judged_same = 0
    for a, b in pairs:
        total += 1
        judged_same += similarity(a, b) >= threshold
    if total == 0:
        raise ValueError(f"no {kind} pairs: the rate is undefined")
    return judged_same / total
