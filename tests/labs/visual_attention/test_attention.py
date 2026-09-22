import pytest

from eidolon.labs.visual_attention import PerceivedObject, Scene, VisualAttention, learn_weights


def obj(track_id, observed_at=0, **features):
    return PerceivedObject(track_id=track_id, observed_at=observed_at, features=features)


def scene(observed_at, *objects):
    return Scene(observed_at=observed_at, objects=tuple(objects))


def red_among_greens(observed_at=0):
    return scene(
        observed_at,
        obj("red", red=1.0, green=0.0),
        *[obj(f"green{i}", red=0.0, green=1.0) for i in range(4)],
    )


def shelf(observed_at=0):
    """A big pop-out distractor `popout` and a faintly green `target`."""
    return scene(
        observed_at,
        *[obj(f"d{i}", greenness=0.0, size=0.5) for i in range(1, 5)],
        obj("popout", greenness=0.0, size=1.0),
        obj("target", greenness=0.3, size=0.5),
    )


def shelf_weights():
    training = scene(
        0,
        obj("target", greenness=0.3, size=0.5),
        *[obj(f"x{i}", greenness=0.0, size=0.5) for i in range(4)],
    )
    return learn_weights(training, "target")


def top_id(selections):
    return selections[0].percept.track_id


# --- bottom-up ---------------------------------------------------------


def test_bottom_up_picks_red_among_greens():
    result = VisualAttention().select(red_among_greens(), k=1)
    assert top_id(result) == "red"
    assert result[0].salience == 1.0


def test_bottom_up_distractors_are_less_salient_than_the_outlier():
    result = VisualAttention().select(red_among_greens(), k=5)
    assert [s.percept.track_id for s in result][0] == "red"
    assert all(s.salience < 0.2 for s in result[1:])


def test_uniform_scene_has_zero_salience_and_keeps_scene_order():
    uniform = scene(0, *[obj(f"o{i}", hue=0.5) for i in range(3)])
    result = VisualAttention().select(uniform, k=3)
    assert [s.percept.track_id for s in result] == ["o0", "o1", "o2"]
    assert all(s.salience == 0.0 for s in result)


def test_single_object_has_nothing_to_contrast_with():
    result = VisualAttention().select(scene(0, obj("only", hue=1.0)), k=1)
    assert result[0].salience == 0.0


def test_empty_scene_selects_nothing():
    assert VisualAttention().select(scene(0), k=3) == []


def test_objects_lacking_a_feature_get_no_response_for_it():
    mixed = scene(0, obj("a", hue=1.0), obj("b", hue=0.0), obj("c", size=5.0))
    result = VisualAttention().select(mixed, k=3)
    assert result[-1].percept.track_id == "c"
    assert result[-1].salience == 0.0


# --- k, t, weights validation -------------------------------------------


def test_k_is_a_maximum_not_an_exact_count():
    assert len(VisualAttention().select(red_among_greens(), k=99)) == 5


def test_k_below_one_rejected():
    with pytest.raises(ValueError):
        VisualAttention().select(red_among_greens(), k=0)


@pytest.mark.parametrize("t", [-0.1, 1.1])
def test_t_outside_unit_interval_rejected(t):
    with pytest.raises(ValueError):
        VisualAttention().select(shelf(), k=1, t=t, weights=shelf_weights())


def test_positive_t_requires_weights():
    with pytest.raises(ValueError):
        VisualAttention().select(shelf(), k=1, t=0.5)


def test_non_positive_weight_rejected():
    with pytest.raises(ValueError):
        VisualAttention().select(shelf(), k=1, t=0.5, weights={"greenness+": 0.0})


def test_constructor_validates_parameters():
    with pytest.raises(ValueError):
        VisualAttention(suppression_duration=-1)
    with pytest.raises(ValueError):
        VisualAttention(peak_ratio=0.0)


# --- learn_weights -------------------------------------------------------


def test_learn_weights_excites_target_features_and_inhibits_background_ones():
    weights = shelf_weights()
    assert weights["greenness+"] > 1
    assert weights["greenness-"] < 1
    assert weights["size+"] == 1.0
    assert weights["size-"] == 1.0


def test_learn_weights_result_is_read_only():
    with pytest.raises(TypeError):
        shelf_weights()["greenness+"] = 1.0


def test_learn_weights_unknown_target_rejected():
    with pytest.raises(ValueError):
        learn_weights(shelf(), "missing")


def test_learn_weights_rejects_a_target_without_features():
    """Missing data is not a zero response: silently learning from it would
    either inhibit every background feature or return no weights at all."""
    featureless = scene(0, obj("target"), obj("a", hue=0.0), obj("b", hue=1.0))
    with pytest.raises(ValueError):
        learn_weights(featureless, "target")


def test_learn_weights_needs_a_background():
    with pytest.raises(ValueError):
        learn_weights(scene(0, obj("only", hue=1.0)), "only")


# --- top-down blend ------------------------------------------------------


def test_t0_is_bottom_up_and_finds_the_popout_not_the_target():
    result = VisualAttention().select(shelf(), k=2, t=0.0, weights=shelf_weights())
    assert [s.percept.track_id for s in result] == ["popout", "target"]


def test_t1_is_top_down_and_finds_the_target_despite_the_popout():
    result = VisualAttention().select(shelf(), k=1, t=1.0, weights=shelf_weights())
    assert top_id(result) == "target"
    assert result[0].salience == 1.0


def test_t_half_mixes_both_cues_and_still_finds_the_target():
    result = VisualAttention().select(shelf(), k=2, t=0.5, weights=shelf_weights())
    assert [s.percept.track_id for s in result] == ["target", "popout"]


def test_top_down_needs_enough_t_to_override_the_popout():
    weights = shelf_weights()
    assert top_id(VisualAttention().select(shelf(), k=1, t=0.2, weights=weights)) == "popout"
    assert top_id(VisualAttention().select(shelf(), k=1, t=0.3, weights=weights)) == "target"


def test_weights_for_unknown_maps_are_ignored():
    result = VisualAttention().select(shelf(), k=1, t=1.0, weights={"nonexistent+": 50.0})
    assert result[0].salience == 0.0


# --- cross-frame inhibition of return ------------------------------------


def test_inhibition_of_return_moves_attention_on_and_then_expires():
    attention = VisualAttention(suppression_duration=3)
    order = [top_id(attention.select(shelf(t), k=1)) for t in range(5)]
    assert order == ["popout", "target", "d1", "popout", "target"]


def test_second_select_in_the_same_frame_excludes_what_the_first_returned():
    attention = VisualAttention(suppression_duration=3)
    assert top_id(attention.select(shelf(0), k=1)) == "popout"
    assert top_id(attention.select(shelf(0), k=1)) == "target"


def test_everything_returned_by_a_multi_object_select_is_suppressed():
    attention = VisualAttention(suppression_duration=3)
    first = attention.select(shelf(0), k=3)
    assert [s.percept.track_id for s in first] == ["popout", "target", "d1"]
    assert top_id(attention.select(shelf(0), k=1)) == "d2"


def test_zero_duration_means_no_suppression():
    attention = VisualAttention(suppression_duration=0)
    assert [top_id(attention.select(shelf(t), k=1)) for t in range(3)] == ["popout"] * 3


def test_suppressed_objects_still_shape_contrast_for_the_rest():
    attention = VisualAttention(suppression_duration=10)
    attention.select(shelf(0), k=1)  # suppresses "popout"
    result = attention.select(shelf(1), k=1)
    assert result[0].percept.track_id == "target"
    assert result[0].salience == pytest.approx(0.654, abs=1e-3)


# --- clock ---------------------------------------------------------------


def test_observed_at_must_not_decrease():
    attention = VisualAttention()
    attention.select(shelf(5), k=1)
    with pytest.raises(ValueError):
        attention.select(shelf(4), k=1)


def test_equal_observed_at_is_allowed_and_a_rejected_call_changes_nothing():
    attention = VisualAttention(suppression_duration=3)
    attention.select(shelf(5), k=1)
    with pytest.raises(ValueError):
        attention.select(shelf(4), k=1)
    assert top_id(attention.select(shelf(5), k=1)) == "target"
