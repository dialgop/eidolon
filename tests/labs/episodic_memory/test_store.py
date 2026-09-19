import pytest

from eidolon.labs.episodic_memory import EpisodicMemory


def test_encode_stores_episode_with_given_fields():
    em = EpisodicMemory()
    episode = em.encode(content="keys", context={"place": "kitchen"}, provenance={"reason": "test"})
    assert episode.content == "keys"
    assert episode.occurred_at == 1
    assert dict(episode.context) == {"place": "kitchen"}
    assert dict(episode.provenance) == {"reason": "test"}
    assert len(em) == 1


def test_occurred_at_is_own_monotonic_clock_not_wall_time():
    em = EpisodicMemory()
    e1 = em.encode(content="a", context={}, provenance={})
    e2 = em.encode(content="b", context={}, provenance={})
    assert (e1.occurred_at, e2.occurred_at) == (1, 2)


def test_encode_does_not_deduplicate():
    em = EpisodicMemory()
    em.encode(content="keys", context={}, provenance={})
    em.encode(content="keys", context={}, provenance={})
    assert len(em) == 2


def test_encode_does_not_mutate_caller_dicts():
    em = EpisodicMemory()
    context = {"place": "kitchen"}
    episode = em.encode(content="keys", context=context, provenance={})
    context["place"] = "hallway"
    assert dict(episode.context) == {"place": "kitchen"}


def test_encode_copy_is_shallow_nested_mutation_still_reaches_episode():
    """Documents a known v0 limitation (not a desired behavior): the copy
    in encode() is shallow, so mutating a *nested* structure the caller
    passed in still changes the stored episode. See Episode's docstring."""
    em = EpisodicMemory()
    context = {"tags": {"place": "kitchen"}}
    episode = em.encode(content="keys", context=context, provenance={})
    context["tags"]["place"] = "hallway"
    assert dict(episode.context["tags"]) == {"place": "hallway"}


def test_episode_context_and_provenance_are_immutable():
    em = EpisodicMemory()
    episode = em.encode(content="keys", context={"place": "kitchen"}, provenance={})
    with pytest.raises(TypeError):
        episode.context["place"] = "hallway"
    with pytest.raises(TypeError):
        episode.provenance["x"] = 1


def test_episode_itself_is_frozen():
    em = EpisodicMemory()
    episode = em.encode(content="keys", context={}, provenance={})
    with pytest.raises(Exception):
        episode.content = "wallet"


def test_recall_by_content_exact_match():
    em = EpisodicMemory()
    em.encode(content="keys", context={}, provenance={"reason": "a"})
    em.encode(content="wallet", context={}, provenance={"reason": "b"})
    results = em.recall_by_content("keys")
    assert [e.content for e in results] == ["keys"]


def test_recall_by_content_no_match_returns_empty():
    em = EpisodicMemory()
    em.encode(content="keys", context={}, provenance={})
    assert em.recall_by_content("phone") == []


def test_recall_by_content_order_default_oldest_first_and_newest_first():
    em = EpisodicMemory()
    em.encode(content="a", context={}, provenance={})
    em.encode(content="a", context={}, provenance={})
    ordered = em.recall_by_content("a")
    assert [e.occurred_at for e in ordered] == [1, 2]
    reversed_order = em.recall_by_content("a", newest_first=True)
    assert [e.occurred_at for e in reversed_order] == [2, 1]


def test_recall_by_context_is_subset_match_not_equality():
    em = EpisodicMemory()
    em.encode(content="keys", context={"place": "kitchen", "time_of_day": "morning"}, provenance={})
    results = em.recall_by_context({"place": "kitchen"})
    assert [e.content for e in results] == ["keys"]


def test_recall_by_context_empty_cue_matches_everything():
    em = EpisodicMemory()
    em.encode(content="keys", context={"place": "kitchen"}, provenance={})
    em.encode(content="wallet", context={}, provenance={})
    assert len(em.recall_by_context({})) == 2


def test_recall_by_context_no_match_returns_empty():
    em = EpisodicMemory()
    em.encode(content="keys", context={"place": "kitchen"}, provenance={})
    assert em.recall_by_context({"place": "office"}) == []


def test_recall_by_context_and_provenance_do_not_cross_match():
    em = EpisodicMemory()
    em.encode(content="keys", context={"place": "kitchen"}, provenance={"reason": "test"})
    assert em.recall_by_context({"reason": "test"}) == []
    assert em.recall_by_provenance({"place": "kitchen"}) == []


def test_recall_by_provenance_is_subset_match():
    em = EpisodicMemory()
    em.encode(content="keys", context={}, provenance={"reason": "capacity_eviction"})
    em.encode(content="wallet", context={}, provenance={"reason": "decay_forgotten"})
    em.encode(content="phone", context={}, provenance={"reason": "decay_forgotten"})
    results = em.recall_by_provenance({"reason": "decay_forgotten"})
    assert {e.content for e in results} == {"wallet", "phone"}
