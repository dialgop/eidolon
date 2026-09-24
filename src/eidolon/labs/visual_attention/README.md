# Visual Attention Lab

## What this implements

An object-level visual attention mechanism that decides which perceived
objects get selected (`VisualAttention`, `learn_weights`, `Selection`).

Eidolon stays a pure cognition library. It never sees pixels, sensors or a
robot: an embodiment/adapter (Webots/NAO later, scripted scenes now) reports
what it perceives as `PerceivedObject`s (from `eidolon.percepts` — the input
contract shared across labs, not owned by this one; see below), and this lab
decides which ones matter. Detection, tracking and feature extraction are
the adapter's job; *selection* is cognition and lives here.

`PerceivedObject`/`Scene` used to live in this lab (`types.py`). They moved
to `eidolon.percepts` once `world_model` needed the same types — the Rule
of Three, not a speculative abstraction — with no compatibility shim: every
import site (this lab, its tests, its demo) was updated directly. See
`docs/architecture.md`'s `eidolon.percepts` section for the current home of
that contract, including `CoverageRegion` (added for `world_model`, unused
by this lab).

Two modes, as one continuous parameter `t` in `[0, 1]`:

- **Bottom-up exploration** (`t=0`, goal-free): objects whose features stand
  out from the rest of the scene are selected first.
- **Top-down search** (`t=1`, goal-directed): learned target weights boost
  features that identify the target and suppress features that are more
  present in the background. Values in between blend both.

## Sources ("inspired by", not a reimplementation)

- S. Frintrop, G. Backer, E. Rome. *Goal-directed search with a top-down
  modulated computational attention system.* DAGM 2005 (LNCS 3663). The
  source for the two-mode framework: uniqueness weighting, top-down weights
  from target vs. background, the excitation/inhibition maps, the
  `(1-t)·bottom-up + t·top-down` blend, and successive foci by selecting a
  region and inhibiting it.
- S. Frintrop, T. Werner, G. M. García. *Traditional Saliency Reloaded: A
  Good Old Model in New Shape.* CVPR 2015 (VOCUS2). Used only as inspiration
  for center-surround contrast per feature, with on-off/off-on kept
  separate. VOCUS2 itself has no top-down part and does not discuss
  inhibition of return.

Both papers work on pixel-level feature maps. Nothing here processes images;
what follows is our adaptation to discrete objects with numeric features.

## What maps to what

| Paper concept | Here |
|---|---|
| Feature/contrast map | Per feature `f`, two maps aligned with the scene's objects: `f+` (object is higher than the mean of the *other* objects) and `f-` (lower). Leave-one-out mean is our stand-in for the surround. |
| Uniqueness weight `X/√m` (m = number of peaks) | Same formula. A response is a peak if it is at least `peak_ratio` (default 0.5) of its map's maximum. |
| Bottom-up saliency `S_bu` | Sum of the uniqueness-weighted maps, normalized by its maximum. |
| Weight `w_i = m(target)/m(background)` | `learn_weights(training_scene, target_track_id)`: the target's response over the mean response of the other objects, per map, from raw contrast maps. |
| `E = Σ w_i X_i (w_i>1)`, `I = Σ X_i / w_i (w_i<1)`, `S_td = E − I` | Same, clipped at 0 and normalized by its maximum. |
| `S = (1−t)·S_bu + t·S_td` | Same, `t` in `[0, 1]`, `ValueError` outside. |
| Focus of attention, then inhibit and repeat | `select(scene, k)` returns the top-`k` in one call. |

## Our own adaptations (not from the papers)

- **Object-level contrast** with a leave-one-out surround, and the `f+`/`f-`
  map naming.
- **Normalization of `S_bu` and `S_td` by their maxima** (all-zero stays
  zero) so the `t` blend compares like with like. The paper normalizes
  `S_td` to `S_bu`'s range and does not say how.
- **Peak definition** (`peak_ratio` relative to the map maximum). It also
  means `m` can only be zero for an all-zero map, which contributes nothing,
  so there is no division by zero.
- **Weight clamping** to `[1/1000, 1000]` with a tiny epsilon: a feature
  absent from the target but present in the background would otherwise give
  a weight of 0 and an infinite inhibition. The limit is wide enough to
  leave the range the paper reports (roughly 0.001 to 77) untouched. A map
  with no information at all gets weight 1 and is neutral. The toy scenes in
  the demo hit the clamp exactly because their target's response is exactly
  zero somewhere; real feature data will mostly not.
- **Cross-frame inhibition of return** (below).
- **Raw contrast maps for learning and search.** The paper is not explicit
  about whether the top-down weights apply to the uniqueness-weighted maps;
  we use the raw ones.

## Design notes

- **`PerceivedObject` is frozen and equal by `track_id` only.** The same
  tracked object seen again (new position, new confidence, even a corrected
  `label`) is still the same object, so equality and hashing ignore
  everything else. `track_id` is required: without a stable identity,
  inhibition of return and working-memory refresh cannot tell objects
  apart. A tracker that has no ids must have its adapter generate them.
  `features` is a flat name-to-number mapping, copied into a read-only
  `MappingProxyType` (flat, so the shallow-copy limitation noted for
  episodic memory does not apply). `position` is carried for later labs but
  unused by v0 attention; when given, `frame` is required.
- **`Scene` is frozen and holds a tuple.** A list raises `TypeError` rather
  than being silently coerced, so nobody keeps a mutable handle on it.
  `track_id`s must be unique within a scene.
- **`k` is a maximum.** `select` returns `min(k, eligible)` objects, most
  salient first, ties in scene order (the same insertion-order fallback
  working memory uses). There is no salience threshold in v0, so objects
  with zero salience can be returned; a threshold can come later.
- **Inhibition of return has two parts.** The paper's part: within one
  scene, select a focus, inhibit it, repeat. That is the ranked top-`k`,
  since inhibiting an object does not change the other objects' salience.
  Our extension: objects returned by `select` are excluded from later
  `select` calls for `suppression_duration` units (default 4, an arbitrary
  choice with no source). Suppressed objects are excluded from the
  competition but still count as context for everyone else's contrast.
  Everything returned by one call becomes suppressed, and a second call in
  the same frame excludes what the first returned.
- **Time is `Scene.observed_at`, the caller's clock.** `VisualAttention` has
  no clock of its own and `select` never advances one: the world does.
  `suppression_duration` is therefore in whatever units the adapter uses for
  `observed_at` (`4` means 4 ms if the adapter counts milliseconds).
  `observed_at` must not decrease between calls (`ValueError`; the rejected
  call changes nothing); equal is fine. This is the third clock in the
  project, next to working memory's tick and episodic memory's encode
  counter, and it is a conscious choice; see `docs/architecture.md`.
- **Objects lacking a feature** get no response for it, and a feature held
  by fewer than two objects has no contrast at all. A scene with a single
  object therefore has nothing to contrast with and every salience is 0.
- **`weights` and `learn_weights` are plain `Mapping[str, float]`**, keyed
  by map name (`"greenness+"`, `"greenness-"`). `learn_weights` returns a
  read-only mapping. Maps missing from `weights` are neutral (weight 1);
  weights for maps that do not exist in the scene are ignored.
  `learn_weights` raises `ValueError` for an unknown target, for a training
  scene with no other object, and for a target with no features: missing
  data is not a zero response, and learning from it would either inhibit
  every background feature or silently return no weights at all.

## Known limitations

- **Working memory and re-perceived objects.** Adding a selected
  `PerceivedObject` to working memory again (same `track_id`) refreshes its
  activation *and* replaces the stored content with the newer observation
  (this was a limitation until working memory's own round; see its README).
  `tests/labs/visual_attention/test_with_working_memory.py` covers it.
- `learn_weights` takes a single training scene. The paper's geometric mean
  over several training images is deferred.
- No conspicuity-map grouping level, no orientation features, no search by
  category name ("find a cup") and no spatial use of `position`: all
  deferred until something needs them.
- No selection of *what to look for*: the caller supplies the target.
  Goals do not exist yet.

## Question this experiment asks

Does a small object-level version of the two-mode framework reproduce the
qualitative behavior that matters: pop-outs win bottom-up, learned
target weights let a faint target beat a stronger distractor as `t` grows,
and inhibition of return turns "always the same winner" into exploration?

## Run it

```bash
python -m eidolon.labs.visual_attention.demo
```

## Files

- `attention.py`: `VisualAttention`, `Selection`, `learn_weights` — imports
  `PerceivedObject`/`Scene` from `eidolon.percepts`.
- `demo.py`: bottom-up exploration with inhibition of return, top-down
  search, and the `t` needed to override a pop-out.
- `tests/labs/visual_attention/`: unit tests, including red-among-greens
  (bottom-up), faint target among a pop-out (top-down at `t=0`, `0.5`, `1`),
  inhibition of return and its expiry. `PerceivedObject`/`Scene`/
  `CoverageRegion`'s own tests live in `tests/percepts/`.
