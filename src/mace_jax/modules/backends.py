from __future__ import annotations

from e3nn_jax import Irreps

from mace_jax.adapters.e3nn import nn
from mace_jax.modules.irreps_tools import mask_head
from mace_jax.modules.wrapper_ops import Linear

from mace_core.modules.backends import ModelBackend


def _make_irreps(value):
    return Irreps(value)


def _make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
    return Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        cueq_config=cueq_config,
        rngs=rngs,
    )


def _make_activation(*, hidden_irreps, gate, cueq_config):
    return nn.Activation(
        irreps_in=hidden_irreps,
        acts=[gate],
        layout_str=getattr(cueq_config, "layout_str", "mul_ir"),
    )


JAX_BACKEND = ModelBackend(
    name="jax",
    make_irreps=_make_irreps,
    make_linear=_make_linear,
    make_activation=_make_activation,
    mask_head=mask_head,
)
