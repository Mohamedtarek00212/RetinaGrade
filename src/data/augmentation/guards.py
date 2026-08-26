"""Machine-enforced ban on augmentations that are unsafe for DR grading.

A prose warning in a README decays; a guard does not. Pipeline construction is
routed through :func:`assert_no_forbidden_transforms`, so the forbidden list in
``configs/data.yaml`` is a hard constraint rather than a suggestion, and a unit
test asserts that every built pipeline satisfies it.

Why each family is banned
-------------------------
**CutOut, CoarseDropout, GridDropout, RandomErasing.** These occlude
rectangular regions. Grade 1 (Mild) is *defined* by the presence of
microaneurysms alone -- often a handful of few-pixel lesions in one part of the
retina. Erasing a patch can therefore remove the only diagnostic evidence in an
image while leaving its label untouched, silently converting a correct sample
into a mislabelled one. The risk concentrates exactly where the dataset is
already weakest: Grades 1 and 3 hold 370 and 193 images respectively against
1,805 for Grade 0.

**MixUp and CutMix.** Both blend labels. DR grading is an *ordinal* task with a
clinically defined severity scale; a 0.5/0.5 blend of "No DR" and
"Proliferative DR" does not denote "Severe" and has no clinical referent.
CutMix additionally pastes a lesion patch from one eye into another, producing
spatially incoherent pathology that no retina exhibits -- and that directly
undermines the lesion-locality interpretation the model is expected to support
through Grad-CAM analysis.

Neither family appears in the reference paper's training recipe, so banning
them costs no fidelity to the reproduction target.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import albumentations as A

from src.utils.logger import get_logger

__all__ = ["ForbiddenAugmentationError", "DEFAULT_FORBIDDEN", "assert_no_forbidden_transforms"]

logger = get_logger(__name__)

#: Fallback list used when a caller supplies none. The configuration is the
#: source of truth; this exists so the guard is never accidentally a no-op.
DEFAULT_FORBIDDEN: tuple[str, ...] = (
    "MixUp",
    "CutMix",
    "CutOut",
    "Cutout",
    "CoarseDropout",
    "GridDropout",
    "RandomErasing",
    "Erasing",
    "XYMasking",
    "MaskDropout",
    "PixelDropout",
)


class ForbiddenAugmentationError(RuntimeError):
    """Raised when a pipeline contains an augmentation banned for this task."""


def _iter_transforms(transforms: Iterable[A.BasicTransform]) -> Iterable[A.BasicTransform]:
    """Yield every transform, descending into nested composition containers."""
    for transform in transforms:
        yield transform
        children = getattr(transform, "transforms", None)
        if children:
            yield from _iter_transforms(children)


def assert_no_forbidden_transforms(
    transforms: Sequence[A.BasicTransform] | A.BaseCompose,
    forbidden: Sequence[str] | None = None,
) -> None:
    """Raise if any forbidden augmentation is present.

    Nested containers (``OneOf``, ``Sequential``, ``Compose``) are inspected
    recursively, so wrapping a banned transform cannot smuggle it through.

    Args:
        transforms: A transform sequence or a composed pipeline.
        forbidden: Class names to reject; defaults to :data:`DEFAULT_FORBIDDEN`.

    Raises:
        ForbiddenAugmentationError: If a forbidden transform is found.

    Example:
        >>> import albumentations as A
        >>> assert_no_forbidden_transforms([A.HorizontalFlip(p=0.5)])
        >>> assert_no_forbidden_transforms([A.CoarseDropout(p=1.0)])
        Traceback (most recent call last):
        ...
        src.data.augmentation.guards.ForbiddenAugmentationError: ...
    """
    banned = {name.lower() for name in (forbidden or DEFAULT_FORBIDDEN)}
    candidates = getattr(transforms, "transforms", transforms)

    offenders = [
        type(transform).__name__
        for transform in _iter_transforms(candidates)
        if type(transform).__name__.lower() in banned
    ]
    if offenders:
        raise ForbiddenAugmentationError(
            f"forbidden augmentation(s) present: {sorted(set(offenders))}. "
            "Occlusion-based transforms can erase the only lesion evidence in Grade 1/3 "
            "images, and label-mixing transforms break ordinal label semantics."
        )
    logger.debug("augmentation guard passed: no forbidden transforms present")
