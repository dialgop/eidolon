import math

import pytest

from eidolon.labs.semantic_memory import ConceptMatch, SemanticMemory
from eidolon.percepts import Embedding


def emb(*vector, space="s"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


def sm(sensitivity=2.0, space="s"):
    return SemanticMemory(space=space, sensitivity=sensitivity)


CUPS = [emb(1.0, 0.1), emb(1.0, 0.05)]
PLATES = [emb(0.1, 1.0), emb(0.05, 1.0)]


def learned(sensitivity=2.0):
    store = sm(sensitivity)
    for e in CUPS:
        store.learn("cup", e)
    for e in PLATES:
        store.learn("plate", e)
    return store


# --- construction -------------------------------------------------------


def test_space_must_be_a_non_empty_string():
    with pytest.raises(ValueError):
        SemanticMemory(space="", sensitivity=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_sensitivity_must_be_positive(bad):
    with pytest.raises(ValueError):
        SemanticMemory(space="s", sensitivity=bad)


def test_starts_empty():
    store = sm()
    assert store.labels == []
    assert len(store) == 0
    assert store.exemplar_count("cup") == 0


# --- learn: new label vs. existing label ---------------------------------


def test_learn_a_new_label_creates_a_category():
    store = sm()
    store.learn("cup", emb(1.0, 0.0))
    assert store.labels == ["cup"]
    assert len(store) == 1
    assert store.exemplar_count("cup") == 1


def test_learn_an_existing_label_adds_an_exemplar_not_a_new_category():
    store = sm()
    store.learn("cup", emb(1.0, 0.0))
    store.learn("cup", emb(0.9, 0.1))
    assert store.labels == ["cup"]
    assert len(store) == 1
    assert store.exemplar_count("cup") == 2


def test_learn_is_unbounded_every_exemplar_is_kept():
    store = sm()
    for i in range(50):
        store.learn("cup", emb(1.0, i / 1000))
    assert store.exemplar_count("cup") == 50


def test_learn_rejects_empty_label():
    with pytest.raises(ValueError):
        sm().learn("", emb(1.0, 0.0))


def test_learn_rejects_mismatched_space():
    store = sm(space="a")
    with pytest.raises(ValueError):
        store.learn("cup", emb(1.0, 0.0, space="b"))


def test_learn_rejects_mismatched_dimension_across_any_label():
    store = sm()
    store.learn("cup", emb(1.0, 0.0))
    with pytest.raises(ValueError):
        store.learn("plate", emb(1.0, 0.0, 0.0))


def test_rejected_learn_does_not_store_anything():
    store = sm()
    store.learn("cup", emb(1.0, 0.0))
    with pytest.raises(ValueError):
        store.learn("cup", emb(1.0, 0.0, 0.0))
    assert store.exemplar_count("cup") == 1


# --- similar_concepts / classify: pinned to hand-computed values ----------


def test_similar_concepts_matches_the_hand_computed_luce_probabilities():
    store = learned(sensitivity=2.0)
    matches = store.similar_concepts(emb(1.0, 0.0), k=2)
    assert matches == [
        ConceptMatch("cup", pytest.approx(0.8633181359716224)),
        ConceptMatch("plate", pytest.approx(0.1366818640283776)),
    ]


def test_probabilities_sum_to_one_across_all_known_labels():
    store = learned()
    matches = store.similar_concepts(emb(0.5, 0.5), k=10)
    assert sum(m.probability for m in matches) == pytest.approx(1.0)


def test_classify_picks_the_highest_probability_label():
    store = learned()
    assert store.classify(emb(1.0, 0.0)) == "cup"
    assert store.classify(emb(0.0, 1.0)) == "plate"


def test_classify_min_probability_is_a_rejection_threshold_not_a_filter():
    """It must not change *which* label would have won — only whether the
    winner is accepted."""
    store = learned()
    assert store.classify(emb(1.0, 0.0)) == "cup"
    assert store.classify(emb(1.0, 0.0), min_probability=0.999) is None


def test_classify_accepts_exactly_at_the_floor_inclusive():
    store = learned()
    winning_probability = store.similar_concepts(emb(1.0, 0.0), k=1)[0].probability
    assert store.classify(emb(1.0, 0.0), min_probability=winning_probability) == "cup"


def test_classify_with_nothing_learned_returns_none():
    assert sm().classify(emb(1.0, 0.0)) is None


def test_similar_concepts_with_nothing_learned_returns_empty():
    assert sm().similar_concepts(emb(1.0, 0.0), k=5) == []


def test_k_is_a_maximum():
    store = learned()
    assert len(store.similar_concepts(emb(1.0, 0.0), k=1)) == 1
    assert len(store.similar_concepts(emb(1.0, 0.0), k=99)) == 2


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_k_must_be_a_positive_int(bad_k):
    with pytest.raises(ValueError):
        learned().similar_concepts(emb(1.0, 0.0), k=bad_k)


def test_similar_concepts_rejects_mismatched_space():
    store = learned()
    with pytest.raises(ValueError):
        store.similar_concepts(emb(1.0, 0.0, space="other"), k=1)


def test_sensitivity_affects_how_sharply_probability_concentrates():
    """Higher sensitivity should make the clearer match more dominant, not
    less — a basic sanity check on the direction of the exponential."""
    low = learned(sensitivity=0.1).similar_concepts(emb(1.0, 0.0), k=1)[0].probability
    high = learned(sensitivity=5.0).similar_concepts(emb(1.0, 0.0), k=1)[0].probability
    assert high > low


def test_an_unknown_category_never_appears_in_results():
    store = learned()
    assert {m.label for m in store.similar_concepts(emb(1.0, 0.0), k=10)} == {"cup", "plate"}


# --- prototype: flat, unweighted, re-normalized centroid -------------------


def test_prototype_of_unknown_label_is_none():
    assert sm().prototype("cup") is None


def test_prototype_matches_the_hand_computed_centroid():
    store = learned()
    proto = store.prototype("cup")
    assert proto.vector == pytest.approx((0.9971993098884564, 0.07478994824163425))
    assert proto.space == "s"


def test_prototype_is_unit_length():
    store = learned()
    proto = store.prototype("cup")
    norm = math.sqrt(sum(x * x for x in proto.vector))
    assert norm == pytest.approx(1.0)


def test_prototype_of_a_single_exemplar_equals_its_normalized_self():
    store = sm()
    store.learn("cup", emb(3.0, 4.0))  # norm 5
    assert store.prototype("cup").vector == pytest.approx((0.6, 0.8))


def test_prototype_plays_no_role_in_classification():
    """Exemplar-based (Nosofsky) and prototype-based (Rosch) are different
    mechanisms here, not the same thing under two names: classify() must
    give the same answer whether or not prototype() has ever been called."""
    store = learned()
    before = store.classify(emb(1.0, 0.0))
    store.prototype("cup")
    store.prototype("plate")
    after = store.classify(emb(1.0, 0.0))
    assert before == after == "cup"
