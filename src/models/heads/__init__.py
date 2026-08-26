"""Dual-Head package: Classification Head and Ordinal Head."""

from __future__ import annotations

from src.models.heads.base import PredictionHead
from src.models.heads.classification_head import ClassificationHead
from src.models.heads.ordinal_head import OrdinalHead

__all__ = ["PredictionHead", "ClassificationHead", "OrdinalHead"]
