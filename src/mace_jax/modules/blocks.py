from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

from mace_core.modules.blocks import NonLinearReadoutBlock as CoreNonLinearReadoutBlock

from .backends import JAX_BACKEND


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearReadoutBlock(CoreNonLinearReadoutBlock, nnx.Module):
    """
    Flax/JAX unified readout block.

    Signature mirrors mace_jax.modules.blocks.NonLinearReadoutBlock.
    """

    BACKEND = JAX_BACKEND

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable | None
    irrep_out: Irreps = Irreps("0e")
    num_heads: int = 1
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable | None,
        irrep_out: Irreps = Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    def __call__(self, x: IrrepsArray, heads: jnp.ndarray | None = None) -> IrrepsArray:
        return self.forward(x, heads=heads)
