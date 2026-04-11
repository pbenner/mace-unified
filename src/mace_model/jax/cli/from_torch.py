"""Torch -> JAX import helpers.

The heavy lifting lives in ``mace_model.jax.tools.model_builder`` and the
NNX Torch import adapters.
"""

from __future__ import annotations

from typing import Any

from flax import nnx

from mace_model.jax.adapters.cuequivariance import CuEquivarianceConfig
from mace_model.jax.nnx_utils import state_to_pure_dict, state_to_serializable_dict
from mace_model.jax.tools.model_builder import (
    _as_irreps,
    _build_jax_model,
    _normalize_atomic_config,
    _prepare_template_data,
    build_model,
    coerce_irreps,
    normalize_atomic_config,
    prepare_template_data,
)


def convert_model(
    torch_model,
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
):
    jax_model = build_model(
        config,
        cueq_config=cueq_config,
        rngs=nnx.Rngs(0),
    )
    template_data = prepare_template_data(config)
    graphdef, state = nnx.split(jax_model)
    pure = state_to_pure_dict(state)
    updated = jax_model.__class__.import_from_torch(torch_model, pure)
    if updated is not None:
        updated.pop("_normalize2mom_consts_var", None)
        nnx.replace_by_pure_dict(state, updated)
        jax_model = nnx.merge(graphdef, state)
    variables = state_to_serializable_dict(state)
    return jax_model, variables, template_data


__all__ = [
    "build_model",
    "coerce_irreps",
    "normalize_atomic_config",
    "prepare_template_data",
    "_as_irreps",
    "_build_jax_model",
    "_normalize_atomic_config",
    "_prepare_template_data",
    "convert_model",
]
