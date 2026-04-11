"""Spherical harmonics layer matching the readable e3nn.o3 API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cuequivariance as cue
import cuequivariance_jax as cuex
import cuequivariance_jax.spherical_harmonics as cue_spherical_harmonics
import jax.numpy as jnp
from flax import nnx
from mace_model.core.modules.e3nn_adapter_utils import (
    apply_spherical_harmonics_normalization,
    build_spherical_harmonics_plan,
    validate_layout_str,
)

from mace_model.jax.adapters.nnx.torch import nxx_auto_import_from_torch

from ..irreps import Irreps
from ...cuequivariance.utility import ir_mul_to_mul_ir


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SphericalHarmonics(nnx.Module):
    irreps_out: int | Sequence[int] | str | Any
    normalize: bool
    normalization: str = "integral"
    irreps_in: Any = None

    def __init__(
        self,
        irreps_out: int | Sequence[int] | str | Any,
        normalize: bool,
        normalization: str = "integral",
        irreps_in: Any = None,
        *,
        layout_str: str = "mul_ir",
    ) -> None:
        self.irreps_out = irreps_out
        self.normalize = normalize
        self.normalization = normalization
        self.layout_str = validate_layout_str(layout_str)
        self.irreps_in = irreps_in

        self._plan = build_spherical_harmonics_plan(
            Irreps(self.irreps_out),
            Irreps("1o" if self.irreps_in is None else self.irreps_in),
            self.normalization,
        )
        if self._plan.lmax > 11:
            raise NotImplementedError(
                f"spherical_harmonics maximum l implemented is 11, got {self._plan.lmax}"
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.normalize:
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
        vector = cuex.RepArray(Irreps(self._plan.irreps_in), x, cue.mul_ir)
        sh = cue_spherical_harmonics(self._plan.degrees, vector, normalize=False)
        array = sh.array
        if not self._plan.is_range_lmax:
            pieces = []
            offset = 0
            for l_value in self._plan.degrees:
                width = 2 * l_value + 1
                pieces.append(array[..., offset : offset + width])
                offset += width
            array = jnp.concatenate(pieces, axis=-1)
        array = apply_spherical_harmonics_normalization(
            array,
            self._plan,
            asarray=jnp.asarray,
        )
        if self.layout_str == "ir_mul":
            return ir_mul_to_mul_ir(array, self._plan.canonical_irreps_out)
        return array
