"""Dual-SwinOrd: full architecture assembly.

Wires: SPM-modulated Swin backbone stage -> PLKA -> SharedFeatureNeck ->
DualHead, matching the four-component sequential order stated explicitly
in Figure 1 of the paper. See ``docs/milestone_04_paper_gaps.md`` for
every value/mechanism below that is configurable specifically because the
paper does not specify it (PG-01 through PG-14).

Implementation-boundary note (PG-05c): this assembly modulates the
backbone's already-extracted stage *output*, not the backbone's internal
transformer blocks -- ``timm`` (the chosen, already-installed backbone
vehicle) exposes no hook into the latter, and the paper gives no detail
that would justify a deeper, custom integration.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention.plka import PLKA
from src.models.backbones.base import RetinalBackbone
from src.models.dual_head import DualHead
from src.models.neck.shared_feature_neck import SharedFeatureNeck
from src.models.semantic_prior.spm import SemanticPriorModulation
from src.models.semantic_prior.text_adapter import TextAdapter

__all__ = ["DualSwinOrd"]


class DualSwinOrd(nn.Module):
    """The full Dual-SwinOrd architecture.

    Args:
        backbone: Hierarchical Swin backbone (PG-01, PG-02).
        spm: Semantic Prior Modulation module (PG-06).
        spm_inject_at_stage: 0-based index into the backbone's stage
            outputs that ``spm`` modulates (PG-05, PG-05b).
        text_adapter: Frozen text encoder; no concrete implementation
            ships in this milestone (PG-03), so callers must supply one
            (a test double, for now).
        text_prompts: Clinical text prompts encoded once at construction
            time (PG-04) -- never hardcoded here.
        plka: Parallel/Progressive Lesion-aware Kernel Attention module.
        plka_input_stage: 0-based index into the backbone's stage outputs
            (after any SPM modulation) that feeds ``plka`` (PG-10).
        neck: Shared Feature Neck.
        dual_head: Composed Classification + Ordinal heads.
    """

    def __init__(
        self,
        backbone: RetinalBackbone,
        spm: SemanticPriorModulation,
        spm_inject_at_stage: int,
        text_adapter: TextAdapter,
        text_prompts: list[str],
        plka: PLKA,
        plka_input_stage: int,
        neck: SharedFeatureNeck,
        dual_head: DualHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spm = spm
        self.spm_inject_at_stage = spm_inject_at_stage
        self.text_adapter = text_adapter
        self.text_prompts = list(text_prompts)
        self.plka = plka
        self.plka_input_stage = plka_input_stage
        self.neck = neck
        self.dual_head = dual_head

        # The text adapter is frozen (paper-explicit) and the prompt list is
        # fixed at construction time, so the embeddings are computed once
        # rather than recomputed on every forward pass.
        with torch.no_grad():
            text_embeddings = self.text_adapter.encode(self.text_prompts)
        self.register_buffer("_text_embeddings", text_embeddings, persistent=False)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run the full architecture on one image batch.

        Args:
            x: ``[B, 3, H, W]`` input batch.

        Returns:
            ``{"classification_logits": [B, K], "ordinal_logits": [B, K-1],
            "shared_embedding": [B, hidden_dim]}``. Intermediate tensors
            are not discarded from this dict so that a future Grad-CAM/SHAP
            consumer (out of scope for this milestone) is not forced to
            retrofit hooks into a finished model.
        """
        stage_features = list(self.backbone(x))
        stage_features[self.spm_inject_at_stage] = self.spm(
            stage_features[self.spm_inject_at_stage], self._text_embeddings
        )

        plka_output = self.plka(stage_features[self.plka_input_stage])
        shared_embedding = self.neck(plka_output)
        head_outputs = self.dual_head(shared_embedding)

        return {**head_outputs, "shared_embedding": shared_embedding}
