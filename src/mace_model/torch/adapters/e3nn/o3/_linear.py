from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

from ..irreps import Irreps


class Linear(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        shared_weights: bool = True,
        internal_weights: bool = True,
        biases: bool = False,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
        self.biases = biases
        self.linear = cuet.Linear(
            self.irreps_in,
            self.irreps_out,
            layout=cue.mul_ir,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            method="naive",
        )
        self.weight_numel = self.linear.weight_numel
        bias_slices = []
        offset = 0
        scalar_dim = 0
        for mul, ir in self.irreps_out:
            width = mul * ir.dim
            if ir.l == 0:
                bias_slices.append((slice(offset, offset + width), width))
                scalar_dim += width
            offset += width
        self._bias_slices = tuple(bias_slices)
        self.bias = (
            torch.nn.Parameter(torch.zeros(scalar_dim, dtype=torch.get_default_dtype()))
            if biases and scalar_dim > 0
            else None
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.linear(x, weight)
        if self.bias is None:
            return out
        out = out.clone()
        offset = 0
        for sl, width in self._bias_slices:
            out[..., sl] = out[..., sl] + self.bias[offset : offset + width]
            offset += width
        return out
