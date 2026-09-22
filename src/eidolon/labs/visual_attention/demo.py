"""Run with: python -m eidolon.labs.visual_attention.demo

Scripted scenes stand in for a perception adapter; Eidolon only ever sees
`PerceivedObject`/`Scene`. Three scenarios:

  A. Bottom-up exploration of a static scene: attention visits the pop-out
     first, then moves on because of cross-frame inhibition of return, and
     comes back to it once the suppression expires.
  B. Top-down search: a big distractor pops out bottom-up (t=0); with
     learned weights and enough top-down factor the faint target wins.
  C. How much top-down does it take to override the pop-out?
"""

from eidolon.labs.visual_attention import PerceivedObject, Scene, VisualAttention, learn_weights


def thing(track_id: str, label: str, observed_at: int, **features: float) -> PerceivedObject:
    return PerceivedObject(track_id=track_id, observed_at=observed_at, label=label, features=features)


def shelf(observed_at: int) -> Scene:
    return Scene(
        observed_at=observed_at,
        objects=(
            *(thing(f"box{i}", "box", observed_at, greenness=0.0, size=0.5) for i in range(1, 5)),
            thing("crate", "big crate", observed_at, greenness=0.0, size=1.0),
            thing("cup", "faintly green cup", observed_at, greenness=0.3, size=0.5),
        ),
    )


def names(selections) -> list[str]:
    return [f"{s.percept.track_id}({s.salience:.2f})" for s in selections]


def scenario_exploration() -> None:
    print("--- A. bottom-up exploration with inhibition of return (N=3) ---")
    attention = VisualAttention(suppression_duration=3)
    for frame in range(6):
        print(f"frame {frame}: attends {names(attention.select(shelf(frame), k=1))}")


def scenario_search(weights) -> None:
    print("--- B. top-down search for the cup ---")
    for t in (0.0, 0.5, 1.0):
        result = VisualAttention().select(shelf(0), k=3, t=t, weights=weights)
        print(f"t={t}: {names(result)}")


def scenario_override(weights) -> None:
    print("--- C. top-down factor needed to override the pop-out ---")
    for t in (0.1, 0.2, 0.3, 0.4):
        winner = VisualAttention().select(shelf(0), k=1, t=t, weights=weights)[0]
        print(f"t={t}: {winner.percept.track_id}")


def main() -> None:
    training = Scene(
        observed_at=0,
        objects=(
            thing("cup", "faintly green cup", 0, greenness=0.3, size=0.5),
            *(thing(f"box{i}", "box", 0, greenness=0.0, size=0.5) for i in range(1, 5)),
        ),
    )
    weights = learn_weights(training, "cup")
    print(f"learned weights for the cup: { {k: round(v, 3) for k, v in weights.items()} }\n")

    scenario_exploration()
    print()
    scenario_search(weights)
    print()
    scenario_override(weights)


if __name__ == "__main__":
    main()
