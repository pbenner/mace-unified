from __future__ import annotations

from typing import Any

import torch
from mace_model.core.modules.backends import use_backend
from mace_model.core.modules.embeddings import (
    GenericJointEmbedding as CoreGenericJointEmbedding,
)

from .backends import TORCH_BACKEND


@use_backend(TORCH_BACKEND)
class GenericJointEmbedding(CoreGenericJointEmbedding, torch.nn.Module):
    """
    Torch unified generic joint embedding block.
    """

    base_dim: int
    embedding_specs: dict[str, Any] | None
    out_dim: int | None = None

    def __init__(
        self,
        *,
        base_dim: int,
        embedding_specs: dict[str, Any] | None,
        out_dim: int | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.init(
            base_dim=base_dim,
            embedding_specs=embedding_specs,
            out_dim=out_dim,
            rngs=None,
        )

    forward = CoreGenericJointEmbedding.forward


__all__ = ["GenericJointEmbedding"]
