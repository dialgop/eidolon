"""Cosine similarity over embeddings, and nearest-neighbour retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from eidolon.percepts import Embedding


@dataclass(frozen=True)
class Match:
    """`key` is opaque: whatever the caller paired with the embedding (a
    track id, an index, an episode). It is never hashed, compared or
    inspected here, which keeps this lab independent of who calls it."""

    key: Any
    similarity: float


def similarity(a: Embedding, b: Embedding) -> float:
    """Cosine similarity in [-1, 1]. Raises `ValueError` if the embeddings
    come from different spaces or have different dimensions."""
    _check_comparable(a, b)
    return _cosine(_unit(a), _unit(b))


def most_similar(
    query: Embedding,
    candidates: Iterable[tuple[Any, Embedding]],
    k: int,
    min_similarity: float | None = None,
) -> list[Match]:
    """Up to `k` candidates most similar to `query`, most similar first.

    `candidates` are `(key, embedding)` pairs (any iterable, read once).
    `k` is a maximum. `min_similarity`, if given, is inclusive. Ties keep
    candidate order. Every candidate is checked against `query`'s space and
    dimension before ranking, so a foreign or malformed one raises even if
    it would not have made the top `k`; nothing is skipped silently.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an int >= 1")
    if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be within [-1, 1]")

    unit_query = _unit(query)
    scored: list[Match] = []
    for key, embedding in candidates:
        _check_comparable(query, embedding)
        scored.append(Match(key, _cosine(unit_query, _unit(embedding))))

    ranked = sorted(scored, key=lambda match: -match.similarity)
    if min_similarity is not None:
        ranked = [match for match in ranked if match.similarity >= min_similarity]
    return ranked[:k]


def _check_comparable(a: Embedding, b: Embedding) -> None:
    if a.space != b.space:
        raise ValueError(f"cannot compare embeddings from different spaces: {a.space!r} vs {b.space!r}")
    if a.dim != b.dim:
        raise ValueError(f"embeddings in space {a.space!r} differ in dimension: {a.dim} vs {b.dim}")


def _unit(embedding: Embedding) -> np.ndarray:
    # Scale by the largest component first so squaring cannot underflow or
    # overflow for very small or very large vectors, then normalize.
    vector = np.asarray(embedding.vector, dtype=float)
    vector = vector / np.max(np.abs(vector))
    return vector / np.linalg.norm(vector)


def _cosine(unit_a: np.ndarray, unit_b: np.ndarray) -> float:
    # Clamping only absorbs floating-point error (a dot product of two unit
    # vectors can land at 1.0000000000000002); it is not a modelling choice.
    return max(-1.0, min(1.0, float(np.dot(unit_a, unit_b))))
