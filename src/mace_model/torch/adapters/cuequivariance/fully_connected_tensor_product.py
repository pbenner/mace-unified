from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet

from mace_model.torch.adapters.e3nn import o3

from .utility import _cue_irreps


def FullyConnectedTensorProduct(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    layout: object = cue.mul_ir,
    group: object = cue.O3,
):
    return cuet.FullyConnectedTensorProduct(
        _cue_irreps(group, irreps_in1),
        _cue_irreps(group, irreps_in2),
        _cue_irreps(group, irreps_out),
        layout=layout,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        method="naive",
    )


__all__ = ["FullyConnectedTensorProduct"]
