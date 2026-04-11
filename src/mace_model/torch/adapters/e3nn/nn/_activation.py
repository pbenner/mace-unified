from __future__ import annotations

import torch
from mace_model.core.modules.e3nn_adapter_utils import (
    build_irreps_block_slices,
    infer_activation_irreps_out,
)

from .. import o3
from ..math import normalize2mom


class Activation(torch.nn.Module):
    def __init__(self, irreps_in, acts):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        def _activation_parity(ir, act):
            if ir.p != -1:
                return ir.p
            x = torch.linspace(0.0, 10.0, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < 1e-5:
                return 1
            if (a1 + a2).abs().max() < 1e-5:
                return -1
            raise ValueError("Activation: parity is violated for odd scalar input.")

        self.irreps_out = o3.Irreps(
            infer_activation_irreps_out(self.irreps_in, acts, _activation_parity)
        )
        self._block_slices = build_irreps_block_slices(self.irreps_in)
        self.acts = torch.nn.ModuleList(acts)

    def forward(self, features, dim=-1):
        moved = dim != -1
        array = features.movedim(dim, -1) if moved else features

        output = []
        for block_slice, act in zip(self._block_slices, self.acts):
            block = array[..., block_slice]
            if act is not None:
                output.append(act(block))
            else:
                output.append(block)
        if len(output) > 1:
            activated = torch.cat(output, dim=-1)
        elif len(output) == 1:
            activated = output[0]
        else:
            activated = array[..., :0]

        return activated.movedim(-1, dim) if moved else activated
