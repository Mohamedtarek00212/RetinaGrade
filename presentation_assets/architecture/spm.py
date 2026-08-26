"""Semantic Prior Modulation (SPM) interface.

Paper-explicit (Figure 1 caption): a sigmoid-activated gating signal,
computed from a "fusion matrix" combining text and visual priors, is
injected into the Swin backbone to guide visual feature extraction.

Paper Gap PG-06 (see ``docs/milestone_04_paper_gaps.md``): the fusion-
matrix algebra and the gate/feature combination rule (multiply / add /
FiLM-style scale-shift) are unspecified. This module therefore fixes
**only** the sigmoid nonlinearity -- the one paper-explicit detail -- in
:meth:`forward`, and leaves :meth:`fuse` and :meth:`apply_gate` abstract.
No concrete subclass ships in this milestone.

Paper Gap PG-05 / PG-05b: which single backbone stage receives this
module's output, and whether more than one stage should, are unresolved;
that decision lives in :class:`~src.models.config.SPMConfig` and
:class:`~src.models.dual_swinord.DualSwinOrd`, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

__all__ = ["SemanticPriorModulation", "DefaultSemanticPriorModulation"]


class SemanticPriorModulation(nn.Module, ABC):
    """Sigmoid-gated fusion of visual features with text-derived priors.

    Concrete subclasses must implement :meth:`fuse` and :meth:`apply_gate`
    to define the (currently paper-unspecified) fusion algebra and
    combination rule; :meth:`forward` wires them together with the one
    paper-explicit detail, the sigmoid nonlinearity.
    """

    @abstractmethod
    def fuse(self, visual_feat: Tensor, text_embeddings: Tensor) -> Tensor:
        """Combine visual and text-prior features into gate logits (PG-06).

        Args:
            visual_feat: ``[B, C, H, W]`` backbone stage feature map.
            text_embeddings: ``[P, D_text]`` frozen text-prompt embeddings.

        Returns:
            Gate logits, broadcastable against ``visual_feat``.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_gate(self, visual_feat: Tensor, gate: Tensor) -> Tensor:
        """Combine the sigmoid gate with the visual feature map (PG-06).

        Args:
            visual_feat: ``[B, C, H, W]`` backbone stage feature map.
            gate: Sigmoid-activated gate, same shape as returned by
                :meth:`fuse`.

        Returns:
            The modulated feature map, same shape as ``visual_feat``.
        """
        raise NotImplementedError

    def forward(self, visual_feat: Tensor, text_embeddings: Tensor) -> Tensor:
        """Apply SPM: paper-explicit sigmoid, gap-driven fuse/combine steps."""
        gate_logits = self.fuse(visual_feat, text_embeddings)
        gate = torch.sigmoid(gate_logits)  # paper-explicit (Figure 1 caption)
        return self.apply_gate(visual_feat, gate)


class DefaultSemanticPriorModulation(SemanticPriorModulation):
    """Engineering default resolving PG-06 -- **not a paper claim**.

    Chosen only so the model is trainable end-to-end while PG-06 (the
    "fusion matrix" algebra) remains textually unspecified: the pooled text
    embedding is linearly projected to the visual channel width, broadcast
    over space, added to a 1x1-conv projection of the visual feature map to
    form the gate logits (a standard FiLM-style conditioning), and the
    resulting sigmoid gate multiplies the visual feature map (a standard
    SE-block-style combination). Replace this class the moment PG-06 is
    resolved from the paper.

    Args:
        visual_channels: Channel width of the backbone stage this module
            gates.
        text_embedding_dim: Dimensionality of the text-adapter output.
    """

    def __init__(self, visual_channels: int, text_embedding_dim: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_embedding_dim, visual_channels)
        self.visual_projection = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)

    def fuse(self, visual_feat: Tensor, text_embeddings: Tensor) -> Tensor:
        pooled_text = text_embeddings.mean(dim=0)
        projected_text = self.text_projection(pooled_text).view(1, -1, 1, 1)
        return self.visual_projection(visual_feat) + projected_text

    def apply_gate(self, visual_feat: Tensor, gate: Tensor) -> Tensor:
        return visual_feat * gate
