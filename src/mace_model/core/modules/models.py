"""Public shared MACE model composition for unified backends."""

from __future__ import annotations

from .model_assembly import MACEModelAssembly
from .model_forward import MACEModelForward
from .model_init import MACEModelInit


class MACEModel(
    MACEModelInit,
    MACEModelAssembly,
    MACEModelForward,
):
    """Shared helpers for Torch/JAX MACE model wrappers."""


__all__ = ["MACEModel"]
