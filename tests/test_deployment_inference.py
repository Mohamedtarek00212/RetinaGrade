"""Tests for deployment-specific model configuration."""

from dataclasses import replace

from deployment.inference import prepare_inference_config


def test_prepare_inference_config_disables_pretrained_download(non_paper_model_config) -> None:
    training_config = replace(
        non_paper_model_config,
        backbone=replace(non_paper_model_config.backbone, pretrained=True),
    )
    inference_config = prepare_inference_config(training_config)

    assert inference_config.backbone.pretrained is False
    assert training_config.backbone.pretrained is True
