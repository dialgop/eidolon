import dataclasses

import pytest

from eidolon.percepts import Embedding, PerceivedObject


def emb(*vector, space="test-space"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


# --- Embedding ---------------------------------------------------------------


def test_embedding_holds_vector_and_space():
    e = Embedding(vector=(1.0, 2.0, 3.0), space="clip-vit-b32")
    assert e.vector == (1.0, 2.0, 3.0)
    assert e.space == "clip-vit-b32"


def test_embedding_dim_is_the_vector_length():
    assert emb(1, 2, 3).dim == 3


def test_embedding_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        emb(1, 2).space = "other"


def test_embedding_vector_must_be_a_tuple_not_a_list():
    with pytest.raises(TypeError):
        Embedding(vector=[1.0, 2.0], space="test-space")


def test_embedding_rejects_an_empty_vector():
    with pytest.raises(ValueError):
        Embedding(vector=(), space="test-space")


def test_embedding_rejects_non_finite_components():
    with pytest.raises(ValueError):
        emb(1.0, float("nan"))
    with pytest.raises(ValueError):
        emb(1.0, float("inf"))


def test_embedding_rejects_a_zero_vector_because_cosine_is_undefined():
    with pytest.raises(ValueError):
        emb(0.0, 0.0, 0.0)


@pytest.mark.parametrize("space", ["", None, 3])
def test_embedding_space_must_be_a_non_empty_string(space):
    with pytest.raises(ValueError):
        Embedding(vector=(1.0,), space=space)


def test_embeddings_are_equal_by_vector_and_space():
    assert emb(1, 2) == emb(1, 2)
    assert emb(1, 2) != emb(1, 3)
    assert emb(1, 2, space="a") != emb(1, 2, space="b")


def test_embedding_is_hashable():
    assert len({emb(1, 2), emb(1, 2)}) == 1


# --- PerceivedObject.embedding -----------------------------------------------


def percept(track_id="cup", **kwargs):
    return PerceivedObject(track_id=track_id, observed_at=0, **kwargs)


def test_perceived_object_embedding_defaults_to_none():
    assert percept().embedding is None


def test_perceived_object_carries_an_embedding():
    e = emb(1, 2, 3)
    assert percept(embedding=e).embedding is e


def test_embedding_does_not_take_part_in_perceived_object_equality():
    """Identity is still `track_id` only: the same tracked object seen again
    with a different appearance vector is still the same object."""
    a = percept(embedding=emb(1, 0))
    b = percept(embedding=emb(0, 1))
    assert a == b
    assert hash(a) == hash(b)


def test_perceived_object_rejects_a_bare_tuple_as_embedding():
    with pytest.raises(TypeError):
        percept(embedding=(1.0, 2.0))
