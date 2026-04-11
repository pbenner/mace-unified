"""e3nn.o3 adapters that remain after enforcing cue-only paths."""

from __future__ import annotations

import jax.numpy as jnp
from mace_model.core.modules.e3nn_adapter_utils import wigner_3j_coefficients

from ..irreps import Irrep, Irreps
from ._spherical_harmonics import SphericalHarmonics


def wigner_3j(l1: int, l2: int, l3: int, dtype=None):
    return jnp.asarray(wigner_3j_coefficients(l1, l2, l3), dtype=dtype)


__all__ = ["Irrep", "Irreps", "SphericalHarmonics", "wigner_3j"]
