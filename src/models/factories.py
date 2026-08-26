"""String-to-module factories for gap-driven configuration fields.

This module performs **no architectural decision-making**: it only
translates a user-supplied configuration string (for example ``"gelu"``)
into the corresponding ``torch.nn`` constructor. Which string to put in
``configs/model.yaml`` for ``plka.activation``, ``plka.normalization``, or
``neck.activation`` remains an unresolved Paper Gap (PG-07, PG-08, PG-12b
in ``docs/milestone_04_paper_gaps.md``) -- this module does not choose one
on the paper's behalf, it only prevents that lookup from being duplicated
at every call site that needs it.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

__all__ = ["activation_factory", "normalization_factory"]

#: Registered activation names. ``"identity"`` is included so a gapped
#: field can be explicitly set to "no activation" without special-casing
#: the wiring code.
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}

#: Registered normalization names, each a ``(num_channels) -> nn.Module``
#: factory so callers never need to special-case construction.
_NORMALIZATIONS: dict[str, Callable[[int], nn.Module]] = {
    "identity": lambda channels: nn.Identity(),
    "batch_norm_2d": nn.BatchNorm2d,
    "instance_norm_2d": nn.InstanceNorm2d,
    "group_norm_8": lambda channels: nn.GroupNorm(8, channels),
}


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Return a zero-argument activation-module factory for ``name``.

    Raises:
        KeyError: If ``name`` is not a registered activation.
    """
    try:
        return _ACTIVATIONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown activation {name!r}; available: {sorted(_ACTIVATIONS)}") from exc


def normalization_factory(name: str) -> Callable[[int], nn.Module]:
    """Return a ``(num_channels) -> nn.Module`` normalization factory for ``name``.

    Raises:
        KeyError: If ``name`` is not a registered normalization.
    """
    try:
        return _NORMALIZATIONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown normalization {name!r}; available: {sorted(_NORMALIZATIONS)}") from exc
