import pytest

from eidolon.labs.world_model import Reidentification, WorldModel
from eidolon.percepts import CoverageRegion, Embedding, PerceivedObject, Scene


def obj(track_id, observed_at, position=(1.0, 0.0, 0.0), embedding=None, confidence=1.0):
    return PerceivedObject(
        track_id=track_id, observed_at=observed_at, position=position, frame="map",
        embedding=embedding, confidence=confidence,
    )


def seen_at(observed_at, *objects):
    return Scene(observed_at=observed_at, objects=tuple(objects))


def emb(*vector, space="s"):
    return Embedding(vector=tuple(float(v) for v in vector), space=space)


CUP = emb(1.0, 0.1)
PLATE = emb(0.1, 1.0)


def reidentifying_wm(**overrides):
    kwargs = dict(frame="map", decay_rate=0.2, forget_threshold=0.1,
                  reidentify_threshold=0.9, reidentify_max_distance=0.5)
    kwargs.update(overrides)
    return WorldModel(**kwargs)


# --- validation ---------------------------------------------------------


def test_reidentification_is_off_by_default():
    wm = WorldModel(frame="map")
    assert wm.reidentify_threshold is None
    assert wm.reidentify_max_distance is None


def test_threshold_without_max_distance_raises():
    with pytest.raises(ValueError):
        WorldModel(frame="map", reidentify_threshold=0.9)


def test_max_distance_without_threshold_raises():
    with pytest.raises(ValueError):
        WorldModel(frame="map", reidentify_max_distance=0.5)


@pytest.mark.parametrize("bad", [-1.1, 1.1])
def test_threshold_outside_cosine_range_raises(bad):
    with pytest.raises(ValueError):
        WorldModel(frame="map", reidentify_threshold=bad, reidentify_max_distance=0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_max_distance_not_positive_raises(bad):
    with pytest.raises(ValueError):
        WorldModel(frame="map", reidentify_threshold=0.9, reidentify_max_distance=bad)


# --- the basic case -------------------------------------------------------


def test_reidentifies_a_new_track_id_as_a_currently_unobserved_belief():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, position=(1.0, 0.0, 0.0), embedding=CUP)))
    wm.update(seen_at(1))  # "a" goes unobserved, decays once
    report = wm.update(seen_at(2, obj("b", 2, position=(1.05, 0.0, 0.0), embedding=CUP)))
    assert report.reidentified == (Reidentification("b", "a", pytest.approx(1.0)),)
    assert report.new == ()
    assert len(wm) == 1


def test_get_resolves_both_the_original_and_the_new_track_id_to_the_same_belief():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    wm.update(seen_at(1))
    wm.update(seen_at(2, obj("b", 2, embedding=CUP)))
    assert wm.get("a") is wm.get("b")
    assert wm.get("a") is not None


def test_belief_track_id_reflects_the_most_recently_observed_id():
    """Documented, intentional: the dict key a belief was first created
    under stays stable (both ids keep resolving via get()), but
    Belief.track_id reads the latest percept, so it can differ from
    whichever key was used to look it up."""
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    wm.update(seen_at(1))
    wm.update(seen_at(2, obj("b", 2, embedding=CUP)))
    assert wm.get("a").track_id == "b"


def test_reidentification_replaces_position_and_confidence_outright():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, position=(1.0, 0.0, 0.0), embedding=CUP, confidence=0.4)))
    wm.update(seen_at(1))
    wm.update(seen_at(2, obj("b", 2, position=(1.2, 0.0, 0.0), embedding=CUP, confidence=0.9)))
    belief = wm.get("a")
    assert belief.percept.position == (1.2, 0.0, 0.0)
    assert belief.confidence == 0.9


def test_once_aliased_a_later_sighting_under_the_new_id_is_a_plain_refresh():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    wm.update(seen_at(1))
    wm.update(seen_at(2, obj("b", 2, embedding=CUP)))  # reidentified here
    report = wm.update(seen_at(3, obj("b", 3, position=(1.0, 0.0, 0.0), embedding=CUP)))
    assert report.reidentified == ()
    assert [belief.track_id for belief in report.refreshed] == ["b"]
    assert len(wm) == 1


# --- gates: position, appearance, staleness --------------------------------


def test_too_far_away_is_not_reidentified_becomes_new():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, position=(1.0, 0.0, 0.0), embedding=CUP)))
    report = wm.update(seen_at(1, obj("b", 1, position=(5.0, 0.0, 0.0), embedding=CUP)))
    assert report.reidentified == ()
    assert [belief.track_id for belief in report.new] == ["b"]
    assert len(wm) == 2


def test_distance_at_exactly_max_distance_is_allowed_inclusive():
    wm = reidentifying_wm(reidentify_max_distance=1.0)
    wm.update(seen_at(0, obj("a", 0, position=(0.0, 0.0, 0.0), embedding=CUP)))
    report = wm.update(seen_at(1, obj("b", 1, position=(1.0, 0.0, 0.0), embedding=CUP)))
    assert [r.new_track_id for r in report.reidentified] == ["b"]


def test_dissimilar_appearance_is_not_reidentified_becomes_new():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    report = wm.update(seen_at(1, obj("b", 1, embedding=PLATE)))
    assert report.reidentified == ()
    assert [belief.track_id for belief in report.new] == ["b"]


def test_candidate_without_embedding_cannot_be_reidentified_against():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=None)))
    report = wm.update(seen_at(1, obj("b", 1, embedding=CUP)))
    assert report.reidentified == ()
    assert len(wm) == 2


def test_new_percept_without_embedding_cannot_trigger_reidentification():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    report = wm.update(seen_at(1, obj("b", 1, embedding=None)))
    assert report.reidentified == ()
    assert len(wm) == 2


def test_candidate_without_position_cannot_be_reidentified_against():
    wm = reidentifying_wm()
    wm.update(seen_at(0, obj("a", 0, position=None, embedding=CUP)))
    report = wm.update(seen_at(1, obj("b", 1, embedding=CUP)))
    assert report.reidentified == ()


def test_a_belief_too_stale_to_still_be_live_is_excluded_even_within_the_same_call():
    """Regression for the pass ordering within one update() call:
    contradiction/forgetting runs before re-identification, specifically so
    that a belief whose confidence *just* crossed below forget_threshold
    this same call is already gone from `_beliefs` by the time
    re-identification looks for candidates — not caught by a separate
    staleness check (there isn't one; `_find_reidentification_match` has no
    confidence check at all), but because it was already deleted a few
    lines earlier in the same call. If the two passes were reordered, this
    belief would still be sitting in `_beliefs` when re-identification ran
    and nothing would stop it from being matched — this test would then
    fail (`reidentified` would be non-empty, `new` would be `[]`)."""
    wm = reidentifying_wm(decay_rate=0.5, forget_threshold=0.2, reidentify_max_distance=10.0)
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    wm.update(seen_at(1))
    assert wm.get("a").confidence_at(2, 0.5) >= 0.2  # still live going into the race call
    report = wm.update(seen_at(3, obj("b", 3, embedding=CUP)))  # confidence_at(3) = 0.125 < 0.2
    assert report.reidentified == ()
    assert [belief.track_id for belief in report.new] == ["b"]
    assert [belief.track_id for belief in report.forgotten] == ["a"]


def test_a_contradicted_belief_cannot_be_reidentified_onto_in_the_same_call():
    """A belief with explicit negative evidence against it (the adapter
    looked at its exact position and found nothing) must not be offered as
    a re-identification candidate, even if a new, appearance-matching
    object shows up at that same spot in the same scene — contradiction
    outranks re-identification the same way it already outranks passive
    decay. Without the pass ordering in update(), this would silently
    reidentify "b" as "a" instead of reporting the contradiction."""
    wm = reidentifying_wm(decay_rate=0.01, forget_threshold=0.001, reidentify_max_distance=10.0)
    wm.update(seen_at(0, obj("a", 0, position=(1.0, 0.0, 0.0), embedding=CUP)))
    region = CoverageRegion(center=(1.0, 0.0, 0.0), radius=0.5, frame="map")
    report = wm.update(Scene(
        observed_at=1,
        objects=(obj("b", 1, position=(1.0, 0.0, 0.0), embedding=CUP),),
        coverage=(region,),
    ))
    assert report.reidentified == ()
    assert [belief.track_id for belief in report.contradicted] == ["a"]
    assert [belief.track_id for belief in report.new] == ["b"]


# --- alias resolution: always one hop, never chained -----------------------


def test_a_third_track_id_aliases_directly_to_the_canonical_not_to_the_middle_one():
    """Chains never form: _beliefs keys never change, so whichever track_id
    a re-identification matches against (read straight from _beliefs.items())
    is already the true canonical id — the alias recorded for a third id is
    always a direct, one-hop pointer to it, never to an intermediate alias."""
    wm = reidentifying_wm(reidentify_max_distance=10.0)
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    wm.update(seen_at(1))
    wm.update(seen_at(2, obj("b", 2, embedding=CUP)))  # "b" aliases to "a"
    wm.update(seen_at(3))
    report = wm.update(seen_at(4, obj("c", 4, embedding=CUP)))  # "c" must alias to "a", not "b"
    assert [r.canonical_track_id for r in report.reidentified] == ["a"]
    assert wm._aliases["c"] == "a"
    assert wm.get("a") is wm.get("b") is wm.get("c")


# --- claiming within one call ----------------------------------------------


def test_two_new_objects_in_one_scene_cannot_claim_the_same_belief():
    wm = reidentifying_wm(reidentify_max_distance=10.0)
    wm.update(seen_at(0, obj("a", 0, position=(1.0, 0.0, 0.0), embedding=CUP)))
    wm.update(seen_at(1))
    report = wm.update(seen_at(
        2,
        obj("b", 2, position=(1.0, 0.0, 0.0), embedding=CUP),
        obj("c", 2, position=(1.0, 0.0, 0.0), embedding=CUP),
    ))
    claimed = [r.new_track_id for r in report.reidentified]
    assert claimed == ["b"]
    assert [belief.track_id for belief in report.new] == ["c"]


def test_a_belief_refreshed_by_exact_track_id_cannot_also_be_reidentified_onto():
    wm = reidentifying_wm(reidentify_max_distance=10.0)
    wm.update(seen_at(0, obj("a", 0, embedding=CUP)))
    # "a" is refreshed directly (exact id match) in the same scene as a new "b":
    report = wm.update(seen_at(1, obj("a", 1, embedding=CUP), obj("b", 1, embedding=CUP)))
    assert [belief.track_id for belief in report.refreshed] == ["a"]
    assert [belief.track_id for belief in report.new] == ["b"]
    assert report.reidentified == ()


# --- ambiguity: documented limitation, not detected -------------------------


def test_ambiguous_identical_candidates_resolve_to_the_earliest_created():
    wm = reidentifying_wm(reidentify_max_distance=10.0)
    wm.update(seen_at(
        0,
        obj("a1", 0, position=(1.0, 0.0, 0.0), embedding=CUP),
        obj("a2", 0, position=(2.0, 0.0, 0.0), embedding=CUP),
    ))
    wm.update(seen_at(1))
    report = wm.update(seen_at(2, obj("b", 2, position=(1.5, 0.0, 0.0), embedding=CUP)))
    assert [r.canonical_track_id for r in report.reidentified] == ["a1"]
