"""Flax port of the e3nn gated non-linearity using cue-backed irreps."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from mace_model.core.modules.e3nn_adapter_utils import (
    apply_gate_blocks,
    build_gate_plan,
    validate_layout_str,
)

from ..irreps import Irreps, IrrepsArray
from ._activation import Activation
from ._extract import Extract


class Gate:
    irreps_scalars: any
    act_scalars: Sequence[Callable | None]
    irreps_gates: any
    act_gates: Sequence[Callable | None]
    irreps_gated: any

    def __init__(
        self,
        irreps_scalars,
        act_scalars: Sequence[Callable | None],
        irreps_gates,
        act_gates: Sequence[Callable | None],
        irreps_gated,
        normalize_act: bool = True,
        layout_str: str = "mul_ir",
    ) -> None:
        self.act_scalars = tuple(act_scalars)
        self.act_gates = tuple(act_gates)
        self.normalize_act = normalize_act
        self.layout_str = validate_layout_str(layout_str)

        plan = build_gate_plan(
            Irreps(irreps_scalars),
            Irreps(irreps_gates),
            Irreps(irreps_gated),
        )
        self.irreps_scalars = plan.irreps_scalars
        self.irreps_gates = plan.irreps_gates
        self.irreps_gated = plan.irreps_gated
        self._gate_blocks = plan.gate_blocks

        self._scalar_activation = Activation(
            self.irreps_scalars,
            self.act_scalars,
            normalize_act=self.normalize_act,
            layout_str=self.layout_str,
        )
        self._gate_activation = Activation(
            self.irreps_gates,
            self.act_gates,
            normalize_act=self.normalize_act,
            layout_str=self.layout_str,
        )
        self._sortcut = Extract(
            plan.irreps_in,
            [self.irreps_scalars, self.irreps_gates, self.irreps_gated],
            list(plan.split_instructions),
        )
        self.irreps_in = plan.irreps_in
        self.irreps_out = self._scalar_activation.irreps_out + self.irreps_gated

    def __call__(self, features: IrrepsArray | jnp.ndarray):
        if isinstance(features, IrrepsArray):
            if self.layout_str != "mul_ir":
                raise ValueError(
                    "Gate expects mul_ir layout when passing an IrrepsArray."
                )
            array = features.array
            return_irreps = True
        else:
            array = features
            if array.shape[-1] != self.irreps_in.dim:
                raise ValueError(
                    f"Invalid input shape: expected last dim {self.irreps_in.dim}, got {array.shape[-1]}"
                )
            return_irreps = False

        scalars, gates, gated = self._sortcut(array)
        scalars_act = self._scalar_activation(scalars)
        gates_act = self._gate_activation(gates) if gates.shape[-1] > 0 else gates

        outputs = []
        if scalars_act.shape[-1] > 0:
            outputs.append(scalars_act)
        if gates_act.shape[-1] > 0 and gated.shape[-1] > 0:
            gated_prod = apply_gate_blocks(
                gated,
                gates_act,
                self._gate_blocks,
                layout_str=self.layout_str,
                concatenate=lambda parts: jnp.concatenate(parts, axis=-1),
            )
            if gated_prod.shape[-1] > 0:
                outputs.append(gated_prod)

        concatenated = (
            jnp.concatenate(outputs, axis=-1)
            if len(outputs) > 1
            else outputs[0]
            if outputs
            else jnp.zeros(array.shape[:-1] + (0,), dtype=array.dtype)
        )
        if return_irreps:
            return IrrepsArray(
                self.irreps_out, concatenated, layout_str=self.layout_str
            )
        return concatenated
