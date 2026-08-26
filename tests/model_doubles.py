"""Test-only concrete implementations of the model package's abstract slots.

**None of these are paper-faithful.** They exist solely so
``tests/test_dual_swinord.py`` can execute one real, end-to-end forward
pass through ``DualSwinOrd``, without pretending any of these choices
resolves the corresponding Paper Gap. See
``tests/fixtures/non_paper_test_config.yaml`` and
``docs/milestone_04_paper_gaps.md``.

None of these classes are registered in the module-level registries in
``src.models`` -- tests register them into local, disposable
``Registry`` instances instead, so production registries stay empty.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention.plka import PLKAFusion
from src.models.heads.ordinal_head import OrdinalHead
from src.models.neck.shared_feature_neck import NeckPooling
from src.models.semantic_prior.spm import SemanticPriorModulation
from src.models.semantic_prior.text_adapter import TextAdapter

__all__ = [
    "MockTextAdapter",
    "FakeSemanticPriorModulation",
    "FakePLKAFusion",
    "FakeNeckPooling",
    "FakeOrdinalHead",
]


class MockTextAdapter(TextAdapter):
    """Returns fixed, deterministically-seeded random embeddings. NOT PubMedCLIP."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, prompts: list[str]) -> Tensor:
        generator = torch.Generator().manual_seed(0)
        return torch.randn(len(prompts), self.embedding_dim, generator=generator)


class FakeSemanticPriorModulation(SemanticPriorModulation):
    """Arbitrary linear-projection fuse + multiplicative gate. NOT paper-faithful (PG-06)."""

    def __init__(self, visual_channels: int, text_embedding_dim: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_embedding_dim, visual_channels)

    def fuse(self, visual_feat: Tensor, text_embeddings: Tensor) -> Tensor:
        pooled_text = text_embeddings.mean(dim=0)
        projected = self.text_projection(pooled_text)
        return projected.view(1, -1, 1, 1).expand_as(visual_feat)

    def apply_gate(self, visual_feat: Tensor, gate: Tensor) -> Tensor:
        return visual_feat * gate


class FakePLKAFusion(PLKAFusion):
    """Arbitrary elementwise-sum fusion. NOT paper-faithful (PG-09)."""

    def forward(self, branch_outputs: list[Tensor]) -> Tensor:
        return torch.stack(branch_outputs, dim=0).sum(dim=0)


class FakeNeckPooling(NeckPooling):
    """Arbitrary global-average-pool. NOT paper-faithful (PG-11)."""

    def forward(self, feature_map: Tensor) -> Tensor:
        return feature_map.mean(dim=(2, 3))


class FakeOrdinalHead(OrdinalHead):
    """Arbitrary independent-per-threshold linear layer. NOT paper-faithful (PG-13/PG-14)."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__(num_classes)
        self.linear = nn.Linear(hidden_dim, self.num_thresholds)

    def forward(self, shared_embedding: Tensor) -> Tensor:
        return self.linear(shared_embedding)
