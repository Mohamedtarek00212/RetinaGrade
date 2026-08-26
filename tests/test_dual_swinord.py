"""One real, end-to-end forward pass through `DualSwinOrd`.

Uses `tests/fixtures/non_paper_test_config.yaml` and the test-only
implementations in `tests/model_doubles.py`, registered into local,
disposable `Registry` instances (never the module-level globals in
`src.models`) so this test never pollutes production registry state and
never pretends to resolve a Paper Gap. See
`docs/milestone_04_paper_gaps.md`.
"""

from __future__ import annotations

import torch

from src.models import build_model
from src.models.attention.plka import PLKAFusion
from src.models.config import ModelConfig
from src.models.dual_swinord import DualSwinOrd
from src.models.heads.ordinal_head import OrdinalHead
from src.models.neck.shared_feature_neck import NeckPooling
from src.models.registry import Registry
from src.models.semantic_prior.spm import SemanticPriorModulation
from tests.model_doubles import (
    FakeNeckPooling,
    FakeOrdinalHead,
    FakePLKAFusion,
    FakeSemanticPriorModulation,
    MockTextAdapter,
)


def _local_registries() -> dict[str, Registry]:
    """Fresh, disposable registries populated only with test doubles."""
    spm_registry: Registry[SemanticPriorModulation] = Registry("spm")
    spm_registry.register("test_spm")(FakeSemanticPriorModulation)

    plka_fusion_registry: Registry[PLKAFusion] = Registry("plka_fusion")
    plka_fusion_registry.register("test_plka_fusion")(FakePLKAFusion)

    neck_pooling_registry: Registry[NeckPooling] = Registry("neck_pooling")
    neck_pooling_registry.register("test_neck_pooling")(FakeNeckPooling)

    ordinal_head_registry: Registry[OrdinalHead] = Registry("ordinal_head")
    ordinal_head_registry.register("test_ordinal_head")(FakeOrdinalHead)

    return {
        "spm_registry": spm_registry,
        "plka_fusion_registry": plka_fusion_registry,
        "neck_pooling_registry": neck_pooling_registry,
        "ordinal_head_registry": ordinal_head_registry,
    }


def test_build_model_end_to_end_forward_pass(non_paper_model_config: ModelConfig) -> None:
    num_classes = 5
    text_adapter = MockTextAdapter(embedding_dim=non_paper_model_config.spm.text_embedding_dim)
    text_prompts = ["Microaneurysms", "Hard exudates", "Cotton wool spots"]

    model = build_model(
        non_paper_model_config,
        num_classes=num_classes,
        text_adapter=text_adapter,
        text_prompts=text_prompts,
        **_local_registries(),
    )
    assert isinstance(model, DualSwinOrd)

    batch_size = 2
    x = torch.randn(batch_size, 3, non_paper_model_config.backbone.image_size, non_paper_model_config.backbone.image_size)
    model.eval()
    with torch.no_grad():
        outputs = model(x)

    assert outputs["classification_logits"].shape == (batch_size, num_classes)
    assert outputs["ordinal_logits"].shape == (batch_size, num_classes - 1)
    assert outputs["shared_embedding"].shape == (batch_size, non_paper_model_config.neck.hidden_dim)
    assert not torch.isnan(outputs["classification_logits"]).any()
    assert not torch.isnan(outputs["ordinal_logits"]).any()


def test_build_model_leaves_global_registries_unchanged(non_paper_model_config: ModelConfig) -> None:
    """The production registries must not gain or lose keys from test doubles."""
    from src.models import ORDINAL_HEAD_REGISTRY, PLKA_FUSION_REGISTRY, SPM_REGISTRY, NECK_POOLING_REGISTRY

    before = {
        "spm": set(SPM_REGISTRY.available()),
        "plka": set(PLKA_FUSION_REGISTRY.available()),
        "neck": set(NECK_POOLING_REGISTRY.available()),
        "ordinal": set(ORDINAL_HEAD_REGISTRY.available()),
    }

    text_adapter = MockTextAdapter(embedding_dim=non_paper_model_config.spm.text_embedding_dim)
    build_model(
        non_paper_model_config,
        num_classes=5,
        text_adapter=text_adapter,
        text_prompts=["Microaneurysms"],
        **_local_registries(),
    )

    assert set(SPM_REGISTRY.available()) == before["spm"]
    assert set(PLKA_FUSION_REGISTRY.available()) == before["plka"]
    assert set(NECK_POOLING_REGISTRY.available()) == before["neck"]
    assert set(ORDINAL_HEAD_REGISTRY.available()) == before["ordinal"]
