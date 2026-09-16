"""Run with: python -m eidolon.labs.working_memory.demo

Shows two things a capacity-limited, decaying buffer should do:
  1. Adding more items than `capacity` evicts the least-active one.
  2. An item that gets rehearsed survives decay that would otherwise
     forget it, while un-rehearsed items fade out over time.
"""

from eidolon.labs.working_memory import WorkingMemory


def report(wm: WorkingMemory, label: str) -> None:
    contents = [f"{i.content}({i.activation:.2f})" for i in wm.items]
    print(f"{label}: [{', '.join(contents)}]")


def main() -> None:
    wm = WorkingMemory(capacity=3, decay_rate=0.3, forget_threshold=0.05)

    for content in ["apple", "ball", "cup"]:
        wm.add(content)
    report(wm, "after filling buffer to capacity")

    _, evicted = wm.add("desk")
    print(f"adding 'desk' evicted: {evicted.content if evicted else None}")
    report(wm, "after adding 'desk'")

    for step in range(1, 6):
        wm.rehearse("desk")
        forgotten = wm.tick()
        if forgotten:
            print(f"tick {step}: forgotten -> {[f.content for f in forgotten]}")
        report(wm, f"tick {step}")


if __name__ == "__main__":
    main()