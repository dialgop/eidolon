from eidolon.labs.working_memory import WorkingMemory


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