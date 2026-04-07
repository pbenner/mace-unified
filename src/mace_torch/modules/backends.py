from __future__ import annotations

from e3nn import nn

from mace.modules.irreps_tools import mask_head
from mace.modules.wrapper_ops import Linear
from mace.tools.compile import simplify_if_compile

from mace_core.modules.backends import ModelBackend


def _make_irreps(value):
    return value


def _make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
    del rngs
    return Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        cueq_config=cueq_config,
    )


def _make_activation(*, hidden_irreps, gate, cueq_config):
    del cueq_config
    return simplify_if_compile(nn.Activation)(
        irreps_in=hidden_irreps,
        acts=[gate],
    )


TORCH_BACKEND = ModelBackend(
    name="torch",
    make_irreps=_make_irreps,
    make_linear=_make_linear,
    make_activation=_make_activation,
    mask_head=mask_head,
)
