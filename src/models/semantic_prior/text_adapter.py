"""TextAdapter interface.

Paper Gap PG-03 / PG-04 (see ``docs/milestone_04_paper_gaps.md``): the
paper names "PubMedCLIP" as the source of semantic priors but specifies
neither a library/checkpoint nor the clinical text prompts used. Per the
explicit decision for this milestone, **no concrete subclass is
implemented here** -- integrating a real PubMedCLIP encoder (and adding
whatever dependency that requires) is deferred to a separately-approved
future task, once a specific checkpoint is chosen. This interface exists
purely so :class:`~src.models.semantic_prior.spm.SemanticPriorModulation`
and :class:`~src.models.dual_swinord.DualSwinOrd` can be authored and
tested against *any* text encoder without waiting for that decision.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = ["TextAdapter", "HashingTextAdapter"]


class TextAdapter(ABC):
    """Frozen text encoder producing semantic prior embeddings.

    "Frozen" is paper-explicit (Figure 1 caption: "a frozen text adapter");
    concrete subclasses are responsible for ensuring their parameters (if
    any) never receive gradients.
    """

    @abstractmethod
    def encode(self, prompts: list[str]) -> Tensor:
        """Encode clinical text prompts into embeddings.

        Args:
            prompts: Clinical text prompts. Their content is a Paper Gap
                (PG-04) and must be supplied by the caller -- never
                hardcoded inside an adapter implementation.

        Returns:
            A ``[len(prompts), D_text]`` embedding tensor. ``D_text`` is
            implementation-defined (PG-03).
        """
        raise NotImplementedError


class HashingTextAdapter(TextAdapter):
    """Implementation assumption (not explicitly specified in the paper).

    The paper names "PubMedCLIP" as the frozen text adapter but supplies no
    checkpoint, library version, or prompt text (PG-03 / PG-04). A real
    PubMedCLIP encoder requires a separately installed dependency and a
    confirmed checkpoint URL; until those details are resolved, this class
    deterministically maps prompt strings to fixed random embeddings of the
    requested dimension. The embeddings are frozen (no parameters, no
    gradients) and the mapping is stable across processes, so experiments
    remain reproducible. Replace this with the actual PubMedCLIP encoder as
    soon as a specific checkpoint is approved.

    Args:
        embedding_dim: Output dimensionality of each prompt embedding.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, prompts: list[str]) -> Tensor:
        embeddings = torch.empty(len(prompts), self.embedding_dim)
        for i, prompt in enumerate(prompts):
            seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            embeddings[i] = torch.randn(self.embedding_dim, generator=generator)
        # Frozen: detach and stop gradients on the returned tensor.
        return embeddings.detach()
