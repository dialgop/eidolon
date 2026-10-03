# Representation Lab

## What this implements

Similarity over learned representations: cosine similarity between
embeddings, nearest-neighbour retrieval (`most_similar`), and the
measurement of recognition as a hit / false-alarm trade-off
(`recognition_rates`).

Eidolon **consumes** embeddings and never **produces** them. The perception
adapter runs the model (CLIP, DINOv2, a truncated detector backbone, on a
GPU if it wants one) and hands the result over as an `Embedding` on a
`PerceivedObject`. This lab defines what Eidolon does with them, and is
model-agnostic: any model that fills the same interface works. Nothing here
needs torch, CUDA or a GPU, and tests run on synthetic vectors.

Cosine plus top-k is the mechanism. The hit / false-alarm trade-off is the
phenomenon, and it is the actual content of the lab: with well-separated
categories some threshold recognizes perfectly, and with overlapping ones no
threshold does, so every choice trades hits against false alarms.

## Sources (framing inspired by, not implemented)

- R. N. Shepard. *Toward a universal law of generalization for
  psychological science.* Science 237, 1317–1323 (1987). Generalization
  decays exponentially with distance in a psychological space.
- R. M. Nosofsky. *Attention, similarity, and the identification–
  categorization relationship.* Journal of Experimental Psychology: General
  115(1), 39–57 (1986). The Generalized Context Model: classification by
  summed similarity to stored exemplars.
- E. Rosch and C. B. Mervis. *Family resemblances: Studies in the internal
  structure of categories.* Cognitive Psychology 7(4), 573–605 (1975).
  Prototype theory.
- D. M. Green and J. A. Swets. *Signal detection theory and psychophysics.*
  Wiley (1966). The vocabulary of hits, false alarms and their trade-off.

These frame *why similarity is worth having*; none is implemented here.
v0 thresholds raw cosine similarity: there is no exponential generalization
gradient (Shepard), no summed-exemplar classification (Nosofsky), no
prototypes or centroids (Rosch; that belongs to semantic memory), and no
sensitivity index or full ROC analysis (Green & Swets): `recognition_rates`
gives one point of the curve per threshold and the demo prints several.

## Design notes

- **`Embedding` carries its `space`.** It is `(vector: tuple[float, ...],
  space: str)`, where `space` names the model and version. Embeddings are
  only comparable within one space: same-dimension vectors from different
  models are unrelated coordinates, so a bare vector would let a mix-up
  return a plausible-looking wrong number. Comparing across spaces (or
  across dimensions within one) raises `ValueError`. `Embedding` lives in
  `eidolon.percepts` because it is part of the adapter contract; the
  operations live here.
- **`PerceivedObject.embedding` is its own field, not a feature.** It is a
  point in a learned space with its own semantics, unlike hue or size.
  Identity is still `track_id` only: the embedding takes no part in
  equality. A bare tuple is rejected with `TypeError`.
- **Cosine only in v0.** On L2-normalized vectors Euclidean distance ranks
  identically (`‖a−b‖² = 2 − 2cos`), so offering both is redundant. Euclidean
  waits for a model whose vector norms carry meaning.
- **The clamp to [-1, 1] absorbs floating-point error and nothing else.**
  The dot product of two unit vectors can land just outside the range: for
  the vector `(0.9034701816518086, 0.09401229776087457)` it is
  `1.0000000000000002`. `_cosine` clamps that, and a regression test uses
  exactly this vector. It is not a modelling choice. Vectors are also
  scaled by their largest component before normalizing, so very small or
  very large vectors cannot underflow or overflow the norm.
- **`Match.key` is opaque.** The caller decides what it is (a `track_id`, an
  index, an `Episode`); it is never hashed, compared or inspected. That
  keeps the lab independent of its callers, which adapt their own data to
  `(key, Embedding)` pairs. Callers also filter out objects with no
  embedding themselves.
- **Nothing is skipped silently.** Every candidate is checked against the
  query's space and dimension before ranking, so a bad one raises even if
  it would not have made the top `k`. `k` is a maximum (an int ≥ 1);
  `min_similarity` is inclusive and must be within [-1, 1]; ties keep
  candidate order (the same insertion-order fallback the other labs use).
- **`recognition_rates` reuses `similarity`.** Pairs are `(Embedding,
  Embedding)`, a pair is judged "same" at or above the threshold
  (inclusive), and both kinds of pair must be non-empty because a rate over
  no pairs is undefined.
- **Thresholds are per space, not universal.** What counts as "the same
  object" differs by model and by modality (image against image versus image
  against text), so no threshold is hardcoded here. Choosing one for a real
  model means calibrating it against that model's own hits and false
  alarms, which is what `recognition_rates` measures.
- **Similarity is not identity.** Two different mugs with the same
  embedding are both perfect matches (`demo.py`, scenario C, and a test).
  Re-identification needs position and time as well as appearance. This lab
  supplies the appearance evidence only. It is also worth remembering that
  a model's notion of similarity reflects its training data, not human
  perceptual similarity.

## Not here yet (each its own later round)

- **Episodic memory:** `Episode` gains an `embedding` field (deferred until
  now) and `recall_by_similarity(cue, k, min_similarity)` alongside the
  exact-match methods, built on `most_similar`.
- **World model:** re-identification of a new `track_id` against unobserved
  beliefs. It needs appearance, position and time together, an alias
  mapping, and a revision of the documented "identity is the adapter's job"
  limitation.
- **Semantic memory:** prototypes (centroids) and category structure.
- **A real-model fixture:** a few real CLIP vectors recorded once on the GPU
  machine and committed as JSON, so tests gain real geometry while staying
  hardware-free.
- Euclidean distance, and per-space threshold calibration tooling.

## How this was tested without a model

Hand-built low-dimensional vectors with known geometry (identical, opposite,
orthogonal, scaled), and two synthetic cluster pairs whose similarities were
computed independently with plain numpy before any expectation was written
down. The tests were also validated by deliberately breaking the code in a
scratch copy (an inclusive threshold made exclusive, the ranking reversed,
the space check removed, the clamp removed, and others) and confirming a
test failed each time.

## Question this experiment asks

Is cosine similarity over adapter-supplied embeddings enough to reproduce
the basic recognition phenomenon: judgments of "same" that trade hits
against false alarms as the threshold moves, with a perfect threshold only
when the categories are separated?

## Run it

```bash
python -m eidolon.labs.representation.demo
```

## Files

- `similarity.py`: `similarity`, `most_similar`, `Match`.
- `recognition.py`: `recognition_rates`, `RecognitionRates`.
- `demo.py`: nearest neighbours, the threshold curve for separated and
  overlapping categories, similarity is not identity, and the refusal to
  compare across spaces.
- `tests/labs/representation/` and `tests/percepts/test_embedding.py`: unit
  tests, including the trade-off phenomenon and the cross-space errors.
