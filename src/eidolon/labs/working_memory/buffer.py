"""A minimal, capacity-limited, decaying working-memory buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Item:
    content: Any
    activation: float
    created_at: int
    last_accessed: int


@dataclass
class WorkingMemory:
    capacity: int = 4
    decay_rate: float = 0.15
    forget_threshold: float = 0.05
    rehearsal_activation: float = 1.0

    _items: list[Item] = field(default_factory=list, init=False)
    _clock: int = field(default=0, init=False)

    @property
    def items(self) -> list[Item]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, content: Any) -> tuple[Item, Item | None]:
        """Insert `content` at full activation.

        If an item matching `content` already exists, it is refreshed in
        place instead of duplicated (no eviction happens in that case), and
        its `content` is replaced with the new object even if it compares
        equal: `add` carries a new observation, so the buffer holds the
        latest one. The item keeps its position, `created_at` and identity.
        Otherwise, if the buffer is already at capacity, the least-active
        item is evicted first; ties (items last touched in the same tick)
        go to the oldest-inserted one. Returns (item, evicted_item_or_None).
        """
        existing = self._find(content)
        if existing is not None:
            existing.content = content
            existing.activation = self.rehearsal_activation
            existing.last_accessed = self._clock
            return existing, None

        evicted = None
        if len(self._items) >= self.capacity:
            evicted = min(self._items, key=lambda i: i.activation)
            self._items.remove(evicted)

        item = Item(
            content=content,
            activation=self.rehearsal_activation,
            created_at=self._clock,
            last_accessed=self._clock,
        )
        self._items.append(item)
        return item, evicted

    def rehearse(self, content: Any) -> Item | None:
        """Refresh the activation of an item matching `content`, if present.
        Unlike `add`, this keeps the stored content: rehearsal is internal
        and brings no new observation."""
        item = self._find(content)
        if item is not None:
            item.activation = self.rehearsal_activation
            item.last_accessed = self._clock
        return item

    def _find(self, content: Any) -> Item | None:
        for item in self._items:
            if item.content == content:
                return item
        return None

    def tick(self) -> list[Item]:
        """Advance one time step: decay every item, drop those that fall
        below `forget_threshold`. Returns the list of forgotten items."""
        self._clock += 1
        forgotten = []
        for item in self._items:
            item.activation *= 1.0 - self.decay_rate
        for item in list(self._items):
            if item.activation < self.forget_threshold:
                forgotten.append(item)
                self._items.remove(item)
        return forgotten