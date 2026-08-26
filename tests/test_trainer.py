"""One real, end-to-end `Trainer.fit()` loop.

Uses `tests/fixtures/non_paper_test_config.yaml` (model) and
`tests/fixtures/non_paper_training_config.yaml` (training), plus the
test-only implementations in `tests/model_doubles.py`, exactly like
`tests/test_dual_swinord.py` -- see that file and
`docs/milestone_04_paper_gaps.md` before relying on any of these values
outside a test.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models import build_model
from src.models.config import ModelConfig
from src.models.registry import Registry
from src.training.config import load_training_config
from src.training.trainer import Trainer
from tests.model_doubles import (
    FakeNeckPooling,
    FakeOrdinalHead,
    FakePLKAFusion,
    FakeSemanticPriorModulation,
    MockTextAdapter,
)


def _local_registries() -> dict[str, Registry]:
    spm_registry = Registry("spm")
    spm_registry.register("test_spm")(FakeSemanticPriorModulation)

    plka_fusion_registry = Registry("plka_fusion")
    plka_fusion_registry.register("test_plka_fusion")(FakePLKAFusion)

    neck_pooling_registry = Registry("neck_pooling")
    neck_pooling_registry.register("test_neck_pooling")(FakeNeckPooling)

    ordinal_head_registry = Registry("ordinal_head")
    ordinal_head_registry.register("test_ordinal_head")(FakeOrdinalHead)

    return {
        "spm_registry": spm_registry,
        "plka_fusion_registry": plka_fusion_registry,
        "neck_pooling_registry": neck_pooling_registry,
        "ordinal_head_registry": ordinal_head_registry,
    }


class _RandomImageDataset(Dataset):
    """Random images with balanced integer labels -- not a real corpus."""

    def __init__(self, num_samples: int, image_size: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=generator)
        self.labels = torch.arange(num_samples) % num_classes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}


def test_trainer_fit_runs_full_loop(non_paper_model_config: ModelConfig, tmp_path: Path) -> None:
    num_classes = 5
    text_adapter = MockTextAdapter(embedding_dim=non_paper_model_config.spm.text_embedding_dim)

    model = build_model(
        non_paper_model_config,
        num_classes=num_classes,
        text_adapter=text_adapter,
        text_prompts=["Microaneurysms"],
        **_local_registries(),
    )

    fixtures_dir = Path(__file__).parent / "fixtures"
    training_config = load_training_config(
        path=fixtures_dir / "non_paper_training_config.yaml",
        overrides={
            "checkpoint": {"dir": str(tmp_path / "checkpoints")},
            "logging": {"log_dir": str(tmp_path / "logs")},
        },
    )

    image_size = non_paper_model_config.backbone.image_size
    train_dataset = _RandomImageDataset(8, image_size, num_classes, seed=0)
    val_dataset = _RandomImageDataset(4, image_size, num_classes, seed=1)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    trainer = Trainer(
        model=model,
        config=training_config,
        num_classes=num_classes,
        device=torch.device("cpu"),
        project_root=tmp_path,
    )
    result = trainer.fit(train_loader, val_loader)

    assert len(result.history) == training_config.epochs
    for epoch_metrics in result.history:
        assert "train_loss" in epoch_metrics
        assert "val_qwk" in epoch_metrics
        assert epoch_metrics["train_loss"] == epoch_metrics["train_loss"]  # not NaN

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert trainer.csv_logger.path.exists()
