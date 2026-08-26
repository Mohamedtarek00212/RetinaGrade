"""Semantic Prior Modulation (SPM) package.

See ``docs/milestone_04_paper_gaps.md`` (PG-03 through PG-06). No concrete
:class:`~src.models.semantic_prior.text_adapter.TextAdapter` or
:class:`~src.models.semantic_prior.spm.SemanticPriorModulation` subclass
ships in this milestone.
"""

from __future__ import annotations

from src.models.semantic_prior.spm import SemanticPriorModulation
from src.models.semantic_prior.text_adapter import TextAdapter

__all__ = ["TextAdapter", "SemanticPriorModulation"]
