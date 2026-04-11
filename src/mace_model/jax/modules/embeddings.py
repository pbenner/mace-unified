from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import nnx
from mace_model.core.modules.backends import use_backend
from mace_model.core.modules.embeddings import (
    GenericJointEmbedding as CoreGenericJointEmbedding,
)

from ..adapters.nnx.torch import nxx_register_module
from .backends import JAX_BACKEND


@use_backend(JAX_BACKEND)
@nxx_register_module("mace_model.torch.modules.embeddings.GenericJointEmbedding")
@nxx_register_module("mace.modules.embeddings.GenericJointEmbedding")
class GenericJointEmbedding(CoreGenericJointEmbedding, nnx.Module):
    """
    Flax/JAX unified generic joint embedding block.
    """

    base_dim: int
    embedding_specs: dict[str, Any] | None
    out_dim: int | None = None

    def __init__(
        self,
        base_dim: int,
        embedding_specs: dict[str, Any] | None,
        out_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            base_dim=base_dim,
            embedding_specs=embedding_specs,
            out_dim=out_dim,
            rngs=rngs,
        )

    __call__ = CoreGenericJointEmbedding.forward

    @classmethod
    def import_from_torch(cls, torch_module, variables):
        params = variables

        def assign(scope: str, key: str, value):
            node = params
            for part in scope.split("/"):
                if not part:
                    continue
                if part.isdigit():
                    part = int(part)
                if part not in node:
                    raise KeyError(f"Unknown NNX parameter scope {scope!r}")
                node = node[part]
            if key not in node:
                raise KeyError(f"Unknown NNX parameter key {key!r} at {scope!r}")
            node[key] = jnp.asarray(value, dtype=node[key].dtype)

        for name, spec in torch_module.specs.items():
            if spec["type"] == "categorical":
                embed = torch_module.embedders[name]
                assign(
                    f"embedders/{name}",
                    "embedding",
                    embed.weight.detach().cpu().numpy(),
                )
            elif spec["type"] == "continuous":
                seq = torch_module.embedders[name]
                lin1 = seq[0]
                lin2 = seq[2]
                assign(
                    f"embedders/{name}/lin1",
                    "kernel",
                    lin1.weight.detach().cpu().numpy().T,
                )
                if lin1.bias is not None:
                    assign(
                        f"embedders/{name}/lin1",
                        "bias",
                        lin1.bias.detach().cpu().numpy(),
                    )
                assign(
                    f"embedders/{name}/lin2",
                    "kernel",
                    lin2.weight.detach().cpu().numpy().T,
                )
                if lin2.bias is not None:
                    assign(
                        f"embedders/{name}/lin2",
                        "bias",
                        lin2.bias.detach().cpu().numpy(),
                    )
            else:
                raise ValueError(f"Unknown feature type {spec['type']!r}")

        project = getattr(torch_module, "project")
        if hasattr(project, "__getitem__"):
            proj = project[0]
        else:
            proj = project
        assign(
            "project/lin",
            "kernel",
            proj.weight.detach().cpu().numpy().T,
        )

        return params


__all__ = ["GenericJointEmbedding"]
