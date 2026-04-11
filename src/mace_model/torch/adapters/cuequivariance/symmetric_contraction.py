from __future__ import annotations

import types

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

from mace_model.torch.adapters.e3nn import o3

from .utility import _cue_irreps


def SymmetricContraction(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    num_elements: int | None = None,
    layout: object = cue.mul_ir,
    group: object = cue.O3,
    use_reduced_cg: bool = True,
):
    module = cuet.SymmetricContraction(
        _cue_irreps(group, irreps_in),
        _cue_irreps(group, irreps_out),
        layout_in=cue.ir_mul,
        layout_out=layout,
        contraction_degree=correlation,
        num_elements=num_elements,
        original_mace=(not use_reduced_cg),
        dtype=torch.get_default_dtype(),
        math_dtype=torch.get_default_dtype(),
        method="naive",
    )
    module.original_forward = module.forward

    def forward(self, x, attrs):
        indices = attrs
        features = x
        if isinstance(features, torch.Tensor):
            if features.ndim == 3:
                features = features.transpose(-1, -2).reshape(features.shape[0], -1)
            elif features.ndim > 3:
                features = features.reshape(features.shape[0], -1)
        if isinstance(attrs, torch.Tensor):
            if attrs.ndim != 1:
                if attrs.shape[-1] == 1:
                    indices = torch.zeros(
                        attrs.shape[0],
                        dtype=torch.int32,
                        device=attrs.device,
                    )
                else:
                    indices = torch.argmax(attrs, dim=-1)
            indices = indices.to(torch.int32)
        return self.original_forward(features, indices)

    module.forward = types.MethodType(forward, module)
    return module


__all__ = ["SymmetricContraction"]
