"""Generic name-to-class registry, reused by every model sub-package.

Mirrors the pattern already used by
:mod:`src.data.preprocessing.registry` and :mod:`src.data.datasets`
(``DATASET_REGISTRY``), generalized into a single reusable class so the
backbone, SPM, PLKA-fusion, neck-pooling, and ordinal-head registries all
share one implementation instead of five near-identical copies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from src.utils.logger import get_logger

__all__ = ["Registry"]

logger = get_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> class registry for one kind of pluggable component.

    Args:
        kind: Human-readable component kind, used only in error messages
            (for example ``"backbone"`` or ``"ordinal_head"``).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator that registers a class under ``name``.

        Args:
            name: Configuration key, in snake_case.

        Returns:
            The decorator.

        Raises:
            ValueError: If ``name`` is already registered to a different
                class, which would make configuration ambiguous.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries and self._entries[name] is not cls:
                raise ValueError(
                    f"{self._kind} name {name!r} is already registered to "
                    f"{self._entries[name].__name__}"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Look up a registered class.

        Args:
            name: Configuration key.

        Returns:
            The registered class.

        Raises:
            KeyError: If ``name`` is unknown -- expected for every
                gap-driven registry in this package until the corresponding
                Paper Gap is resolved and a concrete implementation is
                registered (see ``docs/milestone_04_paper_gaps.md``).
        """
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {sorted(self._entries)}"
            ) from exc

    def build(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate a registered class.

        Args:
            name: Configuration key.
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            The instantiated object.
        """
        instance = self.get(name)(*args, **kwargs)
        logger.debug("built %s '%s' (%s)", self._kind, name, type(instance).__name__)
        return instance

    def available(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self._entries)
