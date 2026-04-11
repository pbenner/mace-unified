from __future__ import annotations

import torch
from mace_model.core.modules.e3nn_adapter_utils import (
    apply_gate_blocks,
    build_gate_plan,
)

from .. import o3
from ._activation import Activation
from ._extract import Extract


class Gate(torch.nn.Module):
    def __init__(
        self,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
    ):
        super().__init__()
        plan = build_gate_plan(
            o3.Irreps(irreps_scalars),
            o3.Irreps(irreps_gates),
            o3.Irreps(irreps_gated),
        )
        self._extract = Extract(
            plan.irreps_in,
            [plan.irreps_scalars, plan.irreps_gates, plan.irreps_gated],
            list(plan.split_instructions),
        )
        self.irreps_scalars = plan.irreps_scalars
        self.irreps_gates = plan.irreps_gates
        self.irreps_gated = plan.irreps_gated
        self._gate_blocks = plan.gate_blocks
        self.act_scalars = Activation(self.irreps_scalars, act_scalars)
        self.act_gates = Activation(self.irreps_gates, act_gates)
        self.irreps_in = plan.irreps_in
        self.irreps_out = self.act_scalars.irreps_out + self.irreps_gated

    def forward(self, features):
        scalars, gates, gated = self._extract(features)
        scalars = self.act_scalars(scalars)
        if gates.shape[-1] == 0 or gated.shape[-1] == 0:
            return scalars
        gated_out = apply_gate_blocks(
            gated,
            self.act_gates(gates),
            self._gate_blocks,
            layout_str="mul_ir",
            concatenate=lambda parts: torch.cat(parts, dim=-1),
        )
        return torch.cat([scalars, gated_out], dim=-1)
