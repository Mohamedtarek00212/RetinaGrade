"""Tests for deployment checkpoint export."""

import pytest
import torch

from scripts.export_inference_checkpoint import build_inference_payload


def test_build_inference_payload_drops_training_state() -> None:
    model_state = {"layer.weight": torch.ones(2, 2)}
    checkpoint = {
        "epoch": 24,
        "model_state_dict": model_state,
        "optimizer_state_dict": {"state": {1: {"momentum": torch.ones(2, 2)}}},
        "scheduler_state_dict": {"last_epoch": 24},
        "metrics": {"val_qwk": 0.9177},
        "extra": {"run": "final"},
    }

    payload = build_inference_payload(checkpoint)

    assert payload["format"] == "retinagrade-inference-v1"
    assert payload["model_state_dict"] is model_state
    assert payload["metadata"]["epoch"] == 24
    assert "optimizer_state_dict" not in payload
    assert "scheduler_state_dict" not in payload


def test_build_inference_payload_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="dictionary"):
        build_inference_payload(["not", "a", "checkpoint"])
