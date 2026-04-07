from __future__ import annotations

from typing import Any, Callable

from .backends import ModelBackend


class NonLinearReadoutBlock:
    """
    Shared base class inherited by both Torch and JAX readout wrappers.

    Subclasses provide a class-level BACKEND operation bundle.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        mlp_irreps: Any,
        gate: Callable | None,
        irrep_out: Any,
        num_heads: int,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = getattr(self, "BACKEND", None)
        if backend is None:
            raise RuntimeError(
                "NonLinearReadoutBlock requires a class-level BACKEND."
            )
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_activation = backend.require("make_activation")

        self.irreps_in = make_irreps(irreps_in)
        self.MLP_irreps = make_irreps(mlp_irreps)
        self.gate = gate
        self.irrep_out = make_irreps(irrep_out)
        self.num_heads = int(num_heads)
        self.cueq_config = cueq_config

        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity = make_activation(
            hidden_irreps=self.hidden_irreps,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.linear_2 = make_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any, heads: Any = None) -> Any:
        backend = self.BACKEND
        mask_head = backend.require("mask_head")
        x = self.linear_1(x)
        x = self.non_linearity(x)
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)
