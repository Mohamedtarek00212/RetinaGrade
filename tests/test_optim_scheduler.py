"""Tests for `src.training.optim` and `src.training.scheduler`."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.config import OptimizerConfig, SchedulerConfig
from src.training.optim import apply_frozen_patterns, build_optimizer, split_param_groups
from src.training.scheduler import build_scheduler


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=True)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


def test_split_param_groups_routes_bias_and_norm_to_no_decay() -> None:
    model = _TinyModel()
    groups = split_param_groups(model, weight_decay=0.01, no_decay_patterns=["bias", "norm"])

    decay_group, no_decay_group = groups
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0
    # linear.weight should be the only decayed parameter.
    assert len(decay_group["params"]) == 1
    assert len(no_decay_group["params"]) == 3  # linear.bias, norm.weight, norm.bias


def test_apply_frozen_patterns_disables_grad() -> None:
    model = _TinyModel()
    frozen_count = apply_frozen_patterns(model, ["norm"])
    assert frozen_count == 2
    assert not model.norm.weight.requires_grad
    assert not model.norm.bias.requires_grad
    assert model.linear.weight.requires_grad


def test_apply_frozen_patterns_noop_when_empty() -> None:
    model = _TinyModel()
    assert apply_frozen_patterns(model, []) == 0
    assert model.linear.weight.requires_grad


def test_build_optimizer_returns_adamw() -> None:
    model = _TinyModel()
    config = OptimizerConfig(lr=1e-3, weight_decay=1e-4)
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_build_optimizer_unknown_name_raises() -> None:
    model = _TinyModel()
    config = OptimizerConfig(name="sgd")
    with pytest.raises(KeyError):
        build_optimizer(model, config)


def test_build_scheduler_cosine_annealing() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SchedulerConfig(eta_min=1e-5)
    scheduler = build_scheduler(optimizer, config, epochs=10)

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 10
    assert scheduler.eta_min == pytest.approx(1e-5)


def test_build_scheduler_respects_explicit_t_max() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SchedulerConfig(eta_min=1e-5, t_max=25)
    scheduler = build_scheduler(optimizer, config, epochs=10)
    assert scheduler.T_max == 25


def test_build_scheduler_unknown_name_raises() -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SchedulerConfig(eta_min=1e-5, name="step")
    with pytest.raises(KeyError):
        build_scheduler(optimizer, config, epochs=10)
