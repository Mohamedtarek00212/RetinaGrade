"""Tests for `src.models.semantic_prior`: TextAdapter and SPM interfaces."""

from __future__ import annotations

import pytest
import torch

from src.models.semantic_prior.spm import SemanticPriorModulation
from src.models.semantic_prior.text_adapter import TextAdapter
from tests.model_doubles import FakeSemanticPriorModulation, MockTextAdapter


def test_text_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        TextAdapter()  # type: ignore[abstract]


def test_spm_is_abstract() -> None:
    with pytest.raises(TypeError):
        SemanticPriorModulation()  # type: ignore[abstract]


def test_mock_text_adapter_encodes_prompts_deterministically() -> None:
    adapter = MockTextAdapter(embedding_dim=16)
    embeddings = adapter.encode(["Microaneurysms", "Hard exudates"])
    assert embeddings.shape == (2, 16)

    repeat = adapter.encode(["Microaneurysms", "Hard exudates"])
    assert torch.equal(embeddings, repeat)


def test_spm_forward_applies_sigmoid_gate() -> None:
    """The one paper-explicit detail (sigmoid) must always be present."""
    spm = FakeSemanticPriorModulation(visual_channels=8, text_embedding_dim=16)
    visual_feat = torch.randn(2, 8, 4, 4)
    text_embeddings = torch.randn(3, 16)

    output = spm(visual_feat, text_embeddings)
    assert output.shape == visual_feat.shape

    gate_logits = spm.fuse(visual_feat, text_embeddings)
    expected = spm.apply_gate(visual_feat, torch.sigmoid(gate_logits))
    assert torch.allclose(output, expected)
