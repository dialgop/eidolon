"""Run with: python -m eidolon.labs.working_memory.demo

Two scenarios, each isolated so the mechanism it's showing off isn't muddied
by another one running at the same time:

  A. Eviction — a genuine same-tick tie (broken by insertion order), then a
     later eviction that's decided by activation instead of a tie.
  B. Decay/rehearsal/forgetting — a rehearsed item survives long enough to
     watch an un-rehearsed one actually cross the forget threshold and get
     dropped (not just decay partway, as a too-short loop would show).
"""

from eidolon.labs.working_memory import WorkingMemory


def report(wm: WorkingMemory, label: str) -> None:
    contents = [f"{i.content}({i.activation:.2f})" for i in wm.items]
    print(f"{label}: [{', '.join(contents)}]")


def scenario_eviction() -> None:
    print("--- A. eviction ---")
    wm = WorkingMemory(capacity=2)

    wm.add("apple")
    wm.add("ball")  # same tick as "apple": activation AND last_accessed tie
    report(wm, "buffer full, apple/ball tied")

    _, evicted = wm.add("cup")
    print(f"tied eviction: {evicted.content} evicted (oldest of the tie)")
    report(wm, "after adding 'cup'")

    item, evicted = wm.add("ball")
    print(f"re-adding 'ball': refreshed in place, evicted={evicted}")
    report(wm, "after re-adding 'ball'")

    wm.tick()
    wm.rehearse("cup")  # "cup" now clearly more active than "ball"
    _, evicted = wm.add("desk")
    print(f"non-tied eviction: {evicted.content} evicted (lower activation, not a tie)")
    report(wm, "after adding 'desk'")


def scenario_decay_and_forgetting() -> None:
    print("--- B. decay, rehearsal, forgetting ---")
    wm = WorkingMemory(capacity=4, decay_rate=0.4, forget_threshold=0.05)
    wm.add("keys")
    wm.add("wallet")
    report(wm, "start")

    for step in range(1, 9):
        wm.rehearse("keys")
        forgotten = wm.tick()
        if forgotten:
            print(f"tick {step}: forgotten -> {[f.content for f in forgotten]}")
        report(wm, f"tick {step}")


def main() -> None:
    scenario_eviction()
    print()
    scenario_decay_and_forgetting()


if __name__ == "__main__":
    main()