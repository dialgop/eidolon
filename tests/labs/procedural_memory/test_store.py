import math

import pytest

from eidolon.labs.procedural_memory import Practice, ProceduralMemory, SkillMatch
from eidolon.percepts import Embedding


def emb(*vector, space="s"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


def pm(alpha=0.5, sensitivity=2.0, decay_rate=0.1, space="s"):
    return ProceduralMemory(space=space, alpha=alpha, sensitivity=sensitivity, decay_rate=decay_rate)


# --- construction ----------------------------------------------------------


def test_space_must_be_a_non_empty_string():
    with pytest.raises(ValueError):
        ProceduralMemory(space="", alpha=0.5, sensitivity=1.0, decay_rate=0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_alpha_must_be_positive(bad):
    with pytest.raises(ValueError):
        ProceduralMemory(space="s", alpha=bad, sensitivity=1.0, decay_rate=0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_sensitivity_must_be_positive(bad):
    with pytest.raises(ValueError):
        ProceduralMemory(space="s", alpha=0.5, sensitivity=bad, decay_rate=0.1)


def test_decay_rate_must_be_non_negative():
    with pytest.raises(ValueError):
        ProceduralMemory(space="s", alpha=0.5, sensitivity=1.0, decay_rate=-0.1)


def test_starts_empty():
    store = pm()
    assert store.skills == []
    assert len(store) == 0
    assert store.practice_count("grasp") == 0


# --- practice: new vs. existing skill, mirroring semantic_memory.learn -----


def test_practice_a_new_skill_creates_it():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 0)
    assert store.skills == ["grasp"]
    assert len(store) == 1
    assert store.practice_count("grasp") == 1


def test_practice_an_existing_skill_adds_an_event_not_a_new_skill():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 0)
    store.practice("grasp", emb(0.9, 0.1), 1)
    assert store.skills == ["grasp"]
    assert len(store) == 1
    assert store.practice_count("grasp") == 2


def test_practice_rejects_empty_skill_name():
    with pytest.raises(ValueError):
        pm().practice("", emb(1.0, 0.0), 0)


def test_practice_timestamp_must_not_decrease():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 5)
    with pytest.raises(ValueError):
        store.practice("grasp", emb(1.0, 0.0), 4)


def test_practice_timestamp_may_stay_equal():
    store = pm()
    store.practice("a", emb(1.0, 0.0), 5)
    store.practice("b", emb(1.0, 0.0), 5)
    assert store.practice_count("a") == store.practice_count("b") == 1


def test_rejected_practice_stores_nothing():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 5)
    with pytest.raises(ValueError):
        store.practice("grasp", emb(1.0, 0.0), 4)
    assert store.practice_count("grasp") == 1


def test_practice_rejects_mismatched_context_space():
    store = pm(space="a")
    with pytest.raises(ValueError):
        store.practice("grasp", emb(1.0, 0.0, space="b"), 0)


def test_practice_rejects_mismatched_context_dimension_across_any_skill():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 0)
    with pytest.raises(ValueError):
        store.practice("reach", emb(1.0, 0.0, 0.0), 1)


def test_practice_with_null_context_is_allowed():
    store = pm()
    store.practice("grasp", None, 0)
    assert store.practice_count("grasp") == 1


# --- strength: pinned to hand-computed values -------------------------------


def learned(decay_rate=0.1, sensitivity=2.0):
    store = pm(decay_rate=decay_rate, sensitivity=sensitivity)
    store.practice("grasp_cup", emb(1.0, 0.05), 0)
    store.practice("grasp_cup", emb(1.0, -0.05), 5)
    store.practice("grasp_mug", emb(0.9, 0.2), 8)
    store.practice("grasp_cup", emb(0.95, 0.1), 10)
    return store


def test_strength_matches_the_hand_computed_value():
    store = learned()
    assert store.strength("grasp_cup", emb(1.0, 0.0), now=12) == pytest.approx(1.5599781082454858)
    assert store.strength("grasp_mug", emb(1.0, 0.0), now=12) == pytest.approx(0.6255850780462288)


def test_strength_of_an_unpracticed_skill_is_zero_not_an_error():
    store = learned()
    assert store.strength("never_practiced", emb(1.0, 0.0), now=12) == 0.0


def test_strength_queries_do_not_mutate_anything():
    """A pure read: calling strength() repeatedly, or before any practice()
    at all, must not change what a later practice() or strength() sees —
    specifically, it must not establish the dimension lock."""
    store = pm()
    store.strength("anything", emb(1.0, 0.0, 0.0), now=0)  # 3-dim query, no data yet
    store.practice("grasp", emb(1.0, 0.0), 0)  # 2-dim: must succeed, not clash with the query above
    assert store.practice_count("grasp") == 1


def test_strength_rejects_mismatched_context_space():
    store = learned()
    with pytest.raises(ValueError):
        store.strength("grasp_cup", emb(1.0, 0.0, space="other"), now=12)


def test_null_context_in_strength_ignores_context_weighting_entirely():
    store = pm()
    store.practice("grasp", emb(1.0, 0.0), 0)
    store.practice("grasp", emb(-1.0, 0.0), 1)  # opposite context
    # with context=None, both practices count fully (time-weight only)
    expected = (1.0 - 0.1) ** (5 - 0) + (1.0 - 0.1) ** (5 - 1)
    assert store.strength("grasp", None, now=5) == pytest.approx(expected)


def test_null_practice_context_contributes_fully_regardless_of_query_context():
    store = pm()
    store.practice("grasp", None, 0)
    # a context-free practice must contribute at full context-weight (1.0)
    # even when queried with a context that would otherwise score low
    expected_time_weight = (1.0 - 0.1) ** 5
    assert store.strength("grasp", emb(1.0, 0.0), now=5) == pytest.approx(expected_time_weight)


def test_strength_decays_to_near_zero_with_enough_elapsed_time():
    store = pm(decay_rate=0.5)
    store.practice("grasp", None, 0)
    assert store.strength("grasp", None, now=50) == pytest.approx((0.5) ** 50)
    assert store.strength("grasp", None, now=50) < 1e-10


def test_more_recent_practice_contributes_more_than_older_practice():
    store = pm(decay_rate=0.3)
    store.practice("grasp", None, 0)
    recent_only = pm(decay_rate=0.3)
    recent_only.practice("grasp", None, 9)
    assert store.strength("grasp", None, now=10) < recent_only.strength("grasp", None, now=10)


# --- select: Luce choice rule, pinned, with interference as a free effect ---


def test_select_matches_the_hand_computed_probability():
    store = learned()
    match = store.select(emb(1.0, 0.0), now=12)
    assert match == SkillMatch(
        skill="grasp_cup",
        strength=pytest.approx(1.5599781082454858),
        probability=pytest.approx(0.7137648172470956),
    )


def test_select_with_nothing_practiced_returns_none():
    assert pm().select(emb(1.0, 0.0), now=0) is None


def test_select_rejects_mismatched_context_space():
    store = learned()
    with pytest.raises(ValueError):
        store.select(emb(1.0, 0.0, space="other"), now=12)


def test_select_min_probability_is_a_rejection_threshold_not_a_filter():
    store = learned()
    assert store.select(emb(1.0, 0.0), now=12).skill == "grasp_cup"
    assert store.select(emb(1.0, 0.0), now=12, min_probability=0.99) is None


def test_select_accepts_exactly_at_the_floor_inclusive():
    store = learned()
    winning_probability = store.select(emb(1.0, 0.0), now=12).probability
    match = store.select(emb(1.0, 0.0), now=12, min_probability=winning_probability)
    assert match is not None and match.skill == "grasp_cup"


def test_select_ties_keep_first_practiced_skill():
    store = pm()
    store.practice("first", None, 0)
    store.practice("second", None, 0)
    match = store.select(emb(1.0, 0.0), now=0)
    assert match.skill == "first"


def test_interference_is_a_free_consequence_of_the_luce_normalization():
    """No separate cross-decay mechanism exists; a well-practiced skill
    with a near-identical competitor must show reduced (not 100%)
    probability purely from select()'s normalization across both."""
    solo = pm()
    solo.practice("grasp_cup", emb(1.0, 0.0), 0)
    solo_match = solo.select(emb(1.0, 0.0), now=1)
    assert solo_match.probability == pytest.approx(1.0)

    contested = pm()
    contested.practice("grasp_cup", emb(1.0, 0.0), 0)
    contested.practice("grasp_mug", emb(1.0, 0.0), 0)  # identical context, equal practice
    contested_match = contested.select(emb(1.0, 0.0), now=1)
    assert contested_match.probability == pytest.approx(0.5)
    assert contested_match.probability < solo_match.probability


# --- curve: dimensionless (N+1)^-alpha, our own N=0 handling ----------------


def test_curve_matches_the_hand_computed_values():
    store = learned()
    assert store.curve("grasp_cup") == [
        (0, pytest.approx(1.0)),
        (1, pytest.approx(0.7071067811865476)),
        (2, pytest.approx(0.5773502691896257)),
        (3, pytest.approx(0.5)),
    ]


def test_curve_of_an_unpracticed_or_nonexistent_skill_is_just_the_baseline_point():
    store = pm()
    assert store.curve("never_practiced") == [(0, 1.0)]


def test_curve_is_monotonically_non_increasing():
    store = learned()
    values = [v for _, v in store.curve("grasp_cup")]
    assert values == sorted(values, reverse=True)


def test_curve_length_matches_practice_count_plus_one():
    store = learned()
    assert len(store.curve("grasp_cup")) == store.practice_count("grasp_cup") + 1


def test_curve_is_the_same_shape_for_every_skill_since_alpha_is_global():
    """alpha is a store-wide hyperparameter, not per-skill (Q8): the only
    thing that can differ between two skills' curves is their length."""
    store = learned()
    cup_curve = dict(store.curve("grasp_cup"))
    mug_curve = dict(store.curve("grasp_mug"))
    for n in mug_curve:
        assert cup_curve[n] == pytest.approx(mug_curve[n])


# --- Practice / SkillMatch are plain frozen records -------------------------


def test_practice_is_frozen():
    p = Practice(context=emb(1.0, 0.0), timestamp=0)
    with pytest.raises(Exception):
        p.timestamp = 1


def test_skill_match_is_frozen():
    m = SkillMatch(skill="grasp", strength=1.0, probability=1.0)
    with pytest.raises(Exception):
        m.skill = "other"
