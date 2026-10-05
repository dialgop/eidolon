import pytest

from eidolon.labs.episodic_memory import EpisodicMemory
from eidolon.labs.representation import Match
from eidolon.percepts import Embedding


def emb(*vector, space="test-space"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


def encode(em, content, embedding=None):
    return em.encode(content=content, context={}, provenance={}, embedding=embedding)


def test_ranks_by_descending_similarity():
    em = EpisodicMemory()
    encode(em, "far", emb(0.0, 1.0))
    encode(em, "near", emb(1.0, 0.1))
    encode(em, "nearest", emb(1.0, 0.01))
    result = em.recall_by_similarity(emb(1.0, 0.0), k=3)
    assert [m.key.content for m in result] == ["nearest", "near", "far"]


def test_returns_matches_with_episode_as_key_and_a_similarity_score():
    em = EpisodicMemory()
    episode = encode(em, "keys", emb(1.0, 0.0))
    result = em.recall_by_similarity(emb(1.0, 0.0), k=1)
    assert len(result) == 1
    assert isinstance(result[0], Match)
    assert result[0].key is episode
    assert result[0].similarity == pytest.approx(1.0)


def test_episodes_without_an_embedding_are_skipped_not_an_error():
    em = EpisodicMemory()
    encode(em, "no-embedding")
    encode(em, "has-embedding", emb(1.0, 0.0))
    result = em.recall_by_similarity(emb(1.0, 0.0), k=10)
    assert [m.key.content for m in result] == ["has-embedding"]


def test_all_episodes_without_embeddings_gives_no_matches():
    em = EpisodicMemory()
    encode(em, "a")
    encode(em, "b")
    assert em.recall_by_similarity(emb(1.0, 0.0), k=10) == []


def test_k_is_a_maximum():
    em = EpisodicMemory()
    for i in range(5):
        encode(em, f"e{i}", emb(1.0, float(i)))
    assert len(em.recall_by_similarity(emb(1.0, 0.0), k=2)) == 2
    assert len(em.recall_by_similarity(emb(1.0, 0.0), k=99)) == 5


def test_min_similarity_filters_out_distant_episodes():
    em = EpisodicMemory()
    encode(em, "close", emb(1.0, 0.0))
    encode(em, "far", emb(0.0, 1.0))
    result = em.recall_by_similarity(emb(1.0, 0.0), k=10, min_similarity=0.5)
    assert [m.key.content for m in result] == ["close"]


def test_candidate_from_a_different_space_raises():
    em = EpisodicMemory()
    encode(em, "same-space", emb(1.0, 0.0, space="clip"))
    encode(em, "other-space", emb(1.0, 0.0, space="dino"))
    with pytest.raises(ValueError):
        em.recall_by_similarity(emb(1.0, 0.0, space="clip"), k=10)


# --- ordering: ranking is primary, newest_first only reverses the tie-break


def test_default_tie_break_is_oldest_first():
    em = EpisodicMemory()
    same = emb(1.0, 2.0, 3.0)
    first = encode(em, "first", same)
    second = encode(em, "second", same)
    result = em.recall_by_similarity(same, k=2)
    assert [m.key for m in result] == [first, second]
    assert result[0].similarity == result[1].similarity == pytest.approx(1.0)


def test_newest_first_reverses_only_the_tie_break():
    em = EpisodicMemory()
    same = emb(1.0, 2.0, 3.0)
    first = encode(em, "first", same)
    second = encode(em, "second", same)
    result = em.recall_by_similarity(same, k=2, newest_first=True)
    assert [m.key for m in result] == [second, first]


def test_newest_first_does_not_change_ranking_across_distinct_scores():
    """newest_first must not turn this into "most recently encoded first":
    a more similar, older episode still outranks a less similar, newer one."""
    em = EpisodicMemory()
    older_closer = encode(em, "older-closer", emb(1.0, 0.0))
    encode(em, "newer-farther", emb(0.2, 1.0))
    result = em.recall_by_similarity(emb(1.0, 0.0), k=2, newest_first=True)
    assert result[0].key is older_closer
