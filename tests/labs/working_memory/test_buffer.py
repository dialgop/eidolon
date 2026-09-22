from dataclasses import dataclass, field

from eidolon.labs.working_memory import WorkingMemory


@dataclass(frozen=True)
class Sighting:
    """Equal by `id` only, like a tracked object: `note` is volatile."""

    id: str
    note: str = field(default="", compare=False)


def test_add_below_capacity_does_not_evict():
    wm = WorkingMemory(capacity=3)
    for content in ["a", "b"]:
        _, evicted = wm.add(content)
        assert evicted is None
    assert len(wm) == 2


def test_add_at_capacity_evicts_least_active():
    wm = WorkingMemory(capacity=2, decay_rate=0.5)
    wm.add("a")
    wm.add("b")
    wm.tick()  # decay "a" and "b" equally
    wm.rehearse("b")  # "b" is now more active than "a"
    _, evicted = wm.add("c")
    assert evicted is not None
    assert evicted.content == "a"
    assert {item.content for item in wm.items} == {"b", "c"}


def test_decay_forgets_item_below_threshold():
    wm = WorkingMemory(capacity=4, decay_rate=0.5, forget_threshold=0.2)
    wm.add("a")
    forgotten = wm.tick()
    assert forgotten == []
    forgotten = wm.tick()  # activation now 1 * 0.5 * 0.5 = 0.25, still above 0.2
    assert forgotten == []
    forgotten = wm.tick()  # activation now 0.125, below 0.2
    assert [f.content for f in forgotten] == ["a"]
    assert len(wm) == 0


def test_rehearsal_prevents_forgetting():
    wm = WorkingMemory(capacity=4, decay_rate=0.5, forget_threshold=0.2)
    wm.add("a")
    for _ in range(5):
        wm.rehearse("a")
        forgotten = wm.tick()
        assert forgotten == []
    assert len(wm) == 1


def test_rehearse_unknown_content_returns_none():
    wm = WorkingMemory()
    assert wm.rehearse("missing") is None


def test_add_existing_content_refreshes_instead_of_duplicating():
    wm = WorkingMemory(capacity=3, decay_rate=0.5)
    wm.add("a")
    wm.tick()  # "a" decays to 0.5
    item, evicted = wm.add("a")
    assert evicted is None
    assert len(wm) == 1
    assert item.activation == 1.0


def test_add_at_capacity_ties_evict_oldest_inserted():
    wm = WorkingMemory(capacity=2)
    wm.add("a")
    wm.add("b")  # same tick as "a": activation and last_accessed both tie
    _, evicted = wm.add("c")
    assert evicted is not None
    assert evicted.content == "a"


def test_recency_overrides_insertion_order():
    """Distinguishes created_at from last_accessed: "a" is the oldest
    inserted item, but refreshing it (without touching "b" or "c") makes
    its activation the highest, so it should survive eviction despite
    not being the most recently *inserted*."""
    wm = WorkingMemory(capacity=3, decay_rate=0.3)
    wm.add("a")
    wm.add("b")
    wm.add("c")
    wm.tick()  # "a", "b", "c" decay equally
    wm.add("a")  # refreshes "a": full activation, last_accessed bumped; created_at unchanged, still oldest
    _, evicted = wm.add("d")
    assert evicted is not None
    assert evicted.content != "a"


def test_add_existing_content_replaces_content_with_the_new_observation():
    wm = WorkingMemory()
    first = Sighting("cup", note="left of the plate")
    second = Sighting("cup", note="right of the plate")
    wm.add(first)
    item, evicted = wm.add(second)
    assert evicted is None
    assert len(wm) == 1
    assert item.content is second
    assert wm.items[0].content is second


def test_add_replaces_content_even_when_the_new_content_is_equal_to_the_old():
    wm = WorkingMemory()
    first = Sighting("cup", note="same")
    second = Sighting("cup", note="same")
    assert first == second and first is not second
    wm.add(first)
    item, _ = wm.add(second)
    assert item.content is second


def test_add_replacement_keeps_identity_position_and_created_at():
    wm = WorkingMemory(capacity=2)
    wm.add(Sighting("cup", note="old"))
    wm.add(Sighting("plate"))
    wm.tick()
    original = wm.items[0]
    item, _ = wm.add(Sighting("cup", note="new"))
    assert item is original
    assert wm.items[0] is original
    assert item.created_at == 0
    assert item.last_accessed == 1
    assert item.activation == 1.0


def test_rehearse_keeps_the_stored_content():
    wm = WorkingMemory()
    first = Sighting("cup", note="old")
    wm.add(first)
    item = wm.rehearse(Sighting("cup", note="new"))
    assert item is not None
    assert item.content is first


def test_evicted_and_forgotten_items_carry_the_latest_content():
    wm = WorkingMemory(capacity=1, decay_rate=0.9, forget_threshold=0.5)
    wm.add(Sighting("cup", note="old"))
    latest = Sighting("cup", note="latest")
    wm.add(latest)
    forgotten = wm.tick()
    assert [f.content for f in forgotten] == [latest]
    assert forgotten[0].content.note == "latest"
