"""Category learning and classification over adapter-supplied embeddings.

Follows Nosofsky's (1986) Generalized Context Model: an unbounded exemplar
store (every learned embedding is kept, nothing is averaged away during
learning) classified by a Luce choice rule — the summed similarity to every
exemplar of a category, normalized against the same sum across every known
category — not nearest-exemplar and not nearest-centroid classification.
See this lab's README for the exact source equations and what we adapted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from eidolon.labs.representation import similarity
from eidolon.percepts import Embedding


@dataclass(frozen=True)
class ConceptMatch:
    """A label and a Luce-choice response *probability* — not
    `eidolon.labs.representation.Match`, deliberately: `Match.similarity`
    means raw cosine everywhere else it's used, and a GCM response is a
    different quantity (normalized across every known label), not a
    similarity score. Reusing `Match` would conflate the two."""

    label: str
    probability: float


@dataclass
class SemanticMemory:
    """`space` is the single embedding space this store accepts — like
    `WorldModel.frame`, fixed and validated against every call, not
    inferred from the first one. `sensitivity` has no default: there is no
    value derivable from the source papers (it doesn't exist in Nosofsky's
    own equations — see the README), so a default would invite using it
    unexamined. The same reasoning kept `context`/`provenance` required
    with no default in `episodic_memory`. Callers calibrate it by looking
    at the actual hit/miss trade-off it produces, the same way a cosine
    recognition threshold is calibrated in `representation`.
    """

    space: str
    sensitivity: float

    _exemplars: dict[str, list[Embedding]] = field(default_factory=dict, init=False)
    _dim: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("space must be a non-empty string")
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")

    @property
    def labels(self) -> list[str]:
        return list(self._exemplars.keys())

    def exemplar_count(self, label: str) -> int:
        return len(self._exemplars.get(label, ()))

    def __len__(self) -> int:
        """Number of distinct labels known, not total exemplars — the
        thing this store holds is concepts; exemplars are each concept's
        internal structure. See `exemplar_count` for a label's own count."""
        return len(self._exemplars)

    def learn(self, label: str, embedding: Embedding) -> None:
        """Add one exemplar under `label`: a new label creates a category
        (its first exemplar); an existing label just grows its exemplar
        pool. Unbounded — faithful to Nosofsky's model, where every
        exemplar is retained, not averaged away (see the README's
        complexity note).

        `embedding.space` must match `self.space`, and its dimension must
        match every exemplar already stored (across *all* labels, not just
        this one — classification sums across every label together, so a
        mismatched dimension anywhere would break it), or this raises
        `ValueError` before storing anything.
        """
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        if embedding.space != self.space:
            raise ValueError(f"embedding space {embedding.space!r} does not match {self.space!r}")
        if self._dim is None:
            self._dim = embedding.dim
        elif embedding.dim != self._dim:
            raise ValueError(f"embedding dim {embedding.dim} does not match existing dim {self._dim}")
        self._exemplars.setdefault(label, []).append(embedding)

    def prototype(self, label: str) -> Embedding | None:
        """The flat, unweighted centroid of `label`'s exemplars, re-
        normalized to unit length — re-normalizing matters because `eta`
        (and hence everything this store does with the result) is defined
        in terms of cosine similarity, which treats unit vectors as the
        canonical representation. `None` if `label` is unknown. This is
        Rosch's prototype (central tendency); it plays no role in
        `classify`/`similar_concepts`, which are exemplar-based (Nosofsky),
        not prototype-based — the two are deliberately different
        mechanisms, not the same thing under two names.
        """
        exemplars = self._exemplars.get(label)
        if not exemplars:
            return None
        dim = exemplars[0].dim
        mean = tuple(sum(e.vector[i] for e in exemplars) / len(exemplars) for i in range(dim))
        norm = math.sqrt(sum(x * x for x in mean))
        return Embedding(vector=tuple(x / norm for x in mean), space=self.space)

    def similar_concepts(self, embedding: Embedding, k: int) -> list[ConceptMatch]:
        """Every known label ranked by Luce-choice response probability —
        the summed `eta` similarity to all of that label's exemplars,
        normalized by the same sum across every label (Equation 5, with
        category bias fixed uniform; see the README). `k` is a maximum, no
        `min_similarity`-style filter here: filtering *individual* matches
        on a quantity that's already normalized across every label doesn't
        have a clean meaning the way filtering raw cosine does (see
        `classify` for the actual floor this lab exposes instead). Ties
        keep label-insertion order (first learned). Empty if nothing has
        been learned yet, or if every summed similarity underflows to
        exactly zero (degenerate: nothing is close to anything known).
        """
        if embedding.space != self.space:
            raise ValueError(f"embedding space {embedding.space!r} does not match {self.space!r}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be an int >= 1")

        raw = {
            label: sum(self._eta(embedding, exemplar) for exemplar in exemplars)
            for label, exemplars in self._exemplars.items()
        }
        total = sum(raw.values())
        if total <= 0.0:
            return []
        ranked = sorted(raw.items(), key=lambda item: -item[1])
        return [ConceptMatch(label, score / total) for label, score in ranked[:k]]

    def classify(self, embedding: Embedding, min_probability: float | None = None) -> str | None:
        """The single most probable label, or `None` if nothing is known
        yet. `min_probability` is a *decision threshold*, not a candidate
        filter: the winner is computed exactly as `similar_concepts` would
        rank it, and only then rejected (returning `None`) if its
        probability falls short — it never changes which label would have
        won."""
        matches = self.similar_concepts(embedding, k=1)
        if not matches:
            return None
        best = matches[0]
        if min_probability is not None and best.probability < min_probability:
            return None
        return best.label

    def _eta(self, a: Embedding, b: Embedding) -> float:
        """Shepard's exponential generalization gradient (Equation 4b),
        expressed directly from cosine similarity via d^2 = 2(1 - cosine)
        for unit vectors — see the README for the derivation and why this
        counts as an adaptation, not a direct implementation of either
        source's exact equations."""
        return math.exp(-self.sensitivity * (1.0 - similarity(a, b)))
