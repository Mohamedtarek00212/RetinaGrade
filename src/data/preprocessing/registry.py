"""Name-to-class registry for configuration-driven transform construction.

The registry exists so that adding a preprocessing step never requires editing
the pipeline builder: a new transform registers itself, gains a configuration
key, and becomes available. Without it, ``pipeline.py`` accumulates an
if/elif chain that every new dataset makes longer.

Example
-------
>>> from src.data.preprocessing.registry import build_transform, available_transforms
>>> "circular_crop" in available_transforms()
True
>>> build_transform("circular_crop", margin_ratio=0.02).margin_ratio
0.02
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from src.data.preprocessing.base import RetinaTransform
from src.data.preprocessing.geometry import BlackBorderRemoval, CircularCrop, FundusResize
from src.data.preprocessing.intensity import CLAHEEnhancement, IlluminationCorrection
from src.utils.logger import get_logger

__all__ = ["register_transform", "build_transform", "available_transforms", "get_transform_class"]

logger = get_logger(__name__)

T = TypeVar("T", bound=RetinaTransform)

#: Configuration key -> transform class.
_REGISTRY: dict[str, type[RetinaTransform]] = {}


def register_transform(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator that registers a transform under ``name``.

    Args:
        name: Configuration key, in snake_case.

    Returns:
        The decorator.

    Raises:
        ValueError: If the name is already registered, which would make the
            configuration ambiguous.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"transform name {name!r} is already registered to {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_transform_class(name: str) -> type[RetinaTransform]:
    """Look up a registered transform class.

    Args:
        name: Configuration key.

    Returns:
        The registered class.

    Raises:
        KeyError: If the name is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown transform {name!r}; registered: {sorted(_REGISTRY)}") from exc


def build_transform(name: str, **kwargs: Any) -> RetinaTransform:
    """Instantiate a registered transform.

    Args:
        name: Configuration key.
        **kwargs: Constructor arguments.

    Returns:
        The instantiated transform.
    """
    transform = get_transform_class(name)(**kwargs)
    logger.debug("built transform '%s' (%s)", name, transform.describe())
    return transform


def available_transforms() -> list[str]:
    """Return the sorted list of registered transform names."""
    return sorted(_REGISTRY)


# Built-in registrations. Declared here rather than as decorators on the classes
# themselves so the transform modules stay importable without the registry.
_REGISTRY.update(
    {
        "black_border_removal": BlackBorderRemoval,
        "circular_crop": CircularCrop,
        "resize": FundusResize,
        "clahe": CLAHEEnhancement,
        "illumination_correction": IlluminationCorrection,
    }
)
