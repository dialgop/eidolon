import itertools

import pytest

from eidolon.labs.representation import RecognitionRates, recognition_rates, similarity
from eidolon.percepts import Embedding


def emb(*vector, space="test-space"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


def within(group):
    return list(itertools.combinations(group, 2))


def between(first, second):
    return [(a, b) for a in first for b in second]


# Two categories with a clear gap: every same-category pair is far more
# similar (min ~0.995) than any different-category pair (max ~0.10).
CUPS = [emb(1, 0.05, 0, 0), emb(1, -0.05, 0, 0), emb(1, 0, 0.05, 0), emb(1, 0, -0.05, 0)]
PLATES = [emb(0.05, 1, 0, 0), emb(-0.05, 1, 0, 0), emb(0, 1, 0.05, 0), emb(0, 1, -0.05, 0)]

# Two categories whose spread overlaps: the least similar same-category pair
# (~0.836) is less similar than the most similar different-category pair
# (~0.977), so no threshold can be perfect.
MUGS = [emb(1, 0.6, 0, 0), emb(1, 0.4, 0.3, 0), emb(1, 0.8, -0.3, 0), emb(1, 0.6, 0, 0.3)]
BOWLS = [emb(0.6, 1, 0, 0), emb(0.4, 1, 0.3, 0), emb(0.8, 1, -0.3, 0), emb(0.6, 1, 0, 0.3)]

SEPARATED = (within(CUPS) + within(PLATES), between(CUPS, PLATES))
OVERLAPPING = (within(MUGS) + within(BOWLS), between(MUGS, BOWLS))


def sims(pairs):
    return [similarity(a, b) for a, b in pairs]


# --- the measurement ---------------------------------------------------------


def test_rates_are_hit_rate_and_false_alarm_rate():
    same, different = SEPARATED
    result = recognition_rates(same, different, threshold=0.5)
    assert isinstance(result, RecognitionRates)
    assert result.hit_rate == pytest.approx(1.0)
    assert result.false_alarm_rate == pytest.approx(0.0)


def test_rates_are_fractions_of_the_pairs_at_or_above_the_threshold():
    same, different = OVERLAPPING
    threshold = 0.9
    expected_hits = sum(s >= threshold for s in sims(same)) / len(same)
    expected_false_alarms = sum(s >= threshold for s in sims(different)) / len(different)
    result = recognition_rates(same, different, threshold=threshold)
    assert result.hit_rate == pytest.approx(expected_hits)
    assert result.false_alarm_rate == pytest.approx(expected_false_alarms)
    assert 0.0 < result.hit_rate < 1.0
    assert 0.0 < result.false_alarm_rate < 1.0


def test_threshold_is_inclusive():
    orthogonal_pair = [(emb(1, 0), emb(0, 1))]
    identical_pair = [(emb(1, 0), emb(1, 0))]
    result = recognition_rates(orthogonal_pair, identical_pair, threshold=0.0)
    assert result.hit_rate == 1.0
    assert result.false_alarm_rate == 1.0


def test_lowest_possible_threshold_accepts_everything():
    same, different = OVERLAPPING
    result = recognition_rates(same, different, threshold=-1.0)
    assert (result.hit_rate, result.false_alarm_rate) == (1.0, 1.0)


def test_a_threshold_above_every_similarity_accepts_nothing():
    same, different = OVERLAPPING
    result = recognition_rates(same, different, threshold=1.0)
    assert (result.hit_rate, result.false_alarm_rate) == (0.0, 0.0)


# --- the phenomenon: the hit / false-alarm trade-off ---------------------------


@pytest.mark.parametrize("clusters", [SEPARATED, OVERLAPPING], ids=["separated", "overlapping"])
def test_raising_the_threshold_never_raises_either_rate(clusters):
    same, different = clusters
    thresholds = [t / 20 for t in range(-20, 21)]
    rates = [recognition_rates(same, different, t) for t in thresholds]
    for lower, higher in zip(rates, rates[1:]):
        assert higher.hit_rate <= lower.hit_rate
        assert higher.false_alarm_rate <= lower.false_alarm_rate


def test_separated_categories_admit_a_perfect_threshold():
    same, different = SEPARATED
    threshold = (min(sims(same)) + max(sims(different))) / 2
    result = recognition_rates(same, different, threshold)
    assert (result.hit_rate, result.false_alarm_rate) == (1.0, 0.0)


def test_overlapping_categories_force_a_trade_off_no_threshold_is_perfect():
    same, different = OVERLAPPING
    assert min(sims(same)) < max(sims(different))  # the overlap that causes it

    catch_every_same_pair = recognition_rates(same, different, threshold=min(sims(same)))
    assert catch_every_same_pair.hit_rate == 1.0
    assert catch_every_same_pair.false_alarm_rate > 0.0

    reject_every_different_pair = recognition_rates(same, different, threshold=max(sims(different)) + 1e-9)
    assert reject_every_different_pair.false_alarm_rate == 0.0
    assert reject_every_different_pair.hit_rate < 1.0

    grid = [t / 100 for t in range(-100, 101)]
    assert not any(
        r.hit_rate == 1.0 and r.false_alarm_rate == 0.0
        for r in (recognition_rates(same, different, t) for t in grid)
    )


# --- validation ----------------------------------------------------------------


def test_no_same_pairs_raises_because_the_hit_rate_is_undefined():
    _, different = SEPARATED
    with pytest.raises(ValueError):
        recognition_rates([], different, threshold=0.5)


def test_no_different_pairs_raises_because_the_false_alarm_rate_is_undefined():
    same, _ = SEPARATED
    with pytest.raises(ValueError):
        recognition_rates(same, [], threshold=0.5)


@pytest.mark.parametrize("bad", [-1.1, 1.1])
def test_threshold_outside_the_cosine_range_raises(bad):
    same, different = SEPARATED
    with pytest.raises(ValueError):
        recognition_rates(same, different, threshold=bad)


def test_a_pair_from_two_spaces_raises():
    cross_space = [(emb(1, 0, space="a"), emb(1, 0, space="b"))]
    _, different = SEPARATED
    with pytest.raises(ValueError):
        recognition_rates(cross_space, different, threshold=0.5)


def test_pairs_may_be_one_shot_generators():
    same, different = SEPARATED
    result = recognition_rates((p for p in same), (p for p in different), threshold=0.5)
    assert (result.hit_rate, result.false_alarm_rate) == (1.0, 0.0)
