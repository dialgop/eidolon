
import math

import pytest

from eidolon.labs.representation import Match, most_similar, similarity
from eidolon.percepts import Embedding


def emb(*vector, space="test-space"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


# --- similarity (cosine) -----------------------------------------------------


def test_identical_vectors_have_similarity_one():
    assert similarity(emb(1, 2, 3), emb(1, 2, 3)) == pytest.approx(1.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert similarity(emb(1, 2, 3), emb(-1, -2, -3)) == pytest.approx(-1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert similarity(emb(1, 0), emb(0, 1)) == pytest.approx(0.0)


def test_similarity_matches_the_known_cosine():
    assert similarity(emb(1, 0), emb(1, 1)) == pytest.approx(1 / math.sqrt(2))


def test_similarity_is_symmetric():
    a, b = emb(1, 2, 3), emb(3, -1, 2)
    assert similarity(a, b) == pytest.approx(similarity(b, a))


def test_similarity_ignores_vector_magnitude():
    assert similarity(emb(1, 2, 3), emb(10, 20, 30)) == pytest.approx(1.0)
    assert similarity(emb(1, 0), emb(1, 1)) == pytest.approx(similarity(emb(5, 0), emb(0.5, 0.5)))


def test_similarity_stays_within_range_despite_float_error():
    vectors = [emb(0.1, 0.2, 0.3), emb(1e-9, 2e-9, 3e-9), emb(1e9, 2e9, 3e9), emb(-3, 5, 7)]
    for a in vectors:
        for b in vectors:
            assert -1.0 <= similarity(a, b) <= 1.0


def test_clamp_absorbs_floating_point_error_at_the_boundary():
    """For this vector the unclamped dot product of its unit vector with
    itself is 1.0000000000000002. The clamp exists for exactly this and is
    not a modelling choice, so the result must never exceed the range."""
    e = emb(0.9034701816518086, 0.09401229776087457)
    assert similarity(e, e) <= 1.0
    assert similarity(e, emb(-0.9034701816518086, -0.09401229776087457)) >= -1.0


def test_similarity_across_spaces_raises():
    with pytest.raises(ValueError, match="clip.*dino|dino.*clip"):
        similarity(emb(1, 2, space="clip"), emb(1, 2, space="dino"))


def test_similarity_with_mismatched_dimensions_in_one_space_raises():
    with pytest.raises(ValueError):
        similarity(emb(1, 2), emb(1, 2, 3))


# --- most_similar ------------------------------------------------------------


def candidates():
    return [
        ("far", emb(0, 1)),
        ("near", emb(1, 0.1)),
        ("nearest", emb(1, 0.01)),
    ]


def test_most_similar_ranks_by_descending_similarity():
    result = most_similar(emb(1, 0), candidates(), k=3)
    assert [m.key for m in result] == ["nearest", "near", "far"]
    assert [m.similarity for m in result] == sorted((m.similarity for m in result), reverse=True)


def test_most_similar_returns_matches_carrying_their_similarity():
    result = most_similar(emb(1, 0), candidates(), k=1)
    assert isinstance(result[0], Match)
    assert result[0].similarity == pytest.approx(similarity(emb(1, 0), emb(1, 0.01)))


def test_match_key_is_opaque_the_caller_decides_what_it_is():
    """Keys need not be hashable or comparable in any particular way: a
    track id, an index, a whole episode, or an unhashable dict all work."""
    unhashable = {"episode": 7}
    result = most_similar(emb(1, 0), [(unhashable, emb(1, 0)), (42, emb(0, 1)), (("a", 1), emb(1, 1))], k=3)
    assert result[0].key is unhashable
    assert {type(m.key) for m in result} == {dict, int, tuple}


def test_k_is_a_maximum_not_an_exact_count():
    assert len(most_similar(emb(1, 0), candidates(), k=99)) == 3
    assert len(most_similar(emb(1, 0), candidates(), k=2)) == 2


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_k_must_be_an_int_of_at_least_one(k):
    with pytest.raises(ValueError):
        most_similar(emb(1, 0), candidates(), k=k)


def test_ties_keep_candidate_order():
    same = emb(1, 2, 3)
    result = most_similar(same, [("first", same), ("second", same), ("third", same)], k=3)
    assert [m.key for m in result] == ["first", "second", "third"]


def test_identical_appearance_cannot_distinguish_two_different_objects():
    """Similarity is not identity: two distinct mugs with the same embedding
    are both perfect matches. Telling them apart needs position and time,
    not appearance alone."""
    mug = emb(0.3, 0.9, 0.1)
    result = most_similar(mug, [("mug-on-left", mug), ("mug-on-right", mug)], k=2)
    assert [m.similarity for m in result] == pytest.approx([1.0, 1.0])


def test_min_similarity_keeps_only_matches_at_or_above_it():
    result = most_similar(emb(1, 0), candidates(), k=3, min_similarity=0.9)
    assert [m.key for m in result] == ["nearest", "near"]


def test_min_similarity_is_inclusive():
    result = most_similar(emb(1, 0), [("orthogonal", emb(0, 1))], k=1, min_similarity=0.0)
    assert [m.key for m in result] == ["orthogonal"]


def test_min_similarity_none_applies_no_filter():
    assert len(most_similar(emb(1, 0), [("opposite", emb(-1, 0))], k=1, min_similarity=None)) == 1


@pytest.mark.parametrize("bad", [-1.1, 1.1])
def test_min_similarity_outside_the_cosine_range_raises(bad):
    with pytest.raises(ValueError):
        most_similar(emb(1, 0), candidates(), k=1, min_similarity=bad)


def test_no_candidates_gives_no_matches():
    assert most_similar(emb(1, 0), [], k=3) == []


def test_candidates_may_be_a_one_shot_generator():
    result = most_similar(emb(1, 0), ((key, e) for key, e in candidates()), k=2)
    assert [m.key for m in result] == ["nearest", "near"]


def test_a_candidate_from_another_space_raises_instead_of_being_skipped():
    mixed = [("ok", emb(1, 0)), ("stranger", emb(1, 0, space="other-model"))]
    with pytest.raises(ValueError):
        most_similar(emb(1, 0), mixed, k=1)


def test_a_bad_candidate_raises_even_when_it_would_not_make_the_top_k():
    mixed = [("best", emb(1, 0)), ("worst-and-foreign", emb(-1, 0, space="other-model"))]
    with pytest.raises(ValueError):
        most_similar(emb(1, 0), mixed, k=1)


def test_a_candidate_with_a_mismatched_dimension_raises():
    with pytest.raises(ValueError):
        most_similar(emb(1, 0), [("wrong-dim", emb(1, 0, 0))], k=1)
