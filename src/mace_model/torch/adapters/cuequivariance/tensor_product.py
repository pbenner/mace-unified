from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch

from mace_model.torch.adapters.e3nn import o3

from .utility import _cue_irreps, _enable_cueq_conv_fusion


def TensorProduct(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions=None,
    shared_weights: bool = False,
    internal_weights: bool = False,
    layout: object = cue.mul_ir,
    group: object = cue.O3,
    conv_fusion: bool = False,
):
    del instructions
    irreps1 = _cue_irreps(group, irreps_in1)
    irreps2 = _cue_irreps(group, irreps_in2)
    irreps_out_cue = _cue_irreps(group, irreps_out)
    if conv_fusion:
        descriptor = (
            cue.descriptors.channelwise_tensor_product(
                irreps1,
                irreps2,
                irreps_out_cue,
            )
            .flatten_coefficient_modes()
            .squeeze_modes()
            .polynomial
        )
        return _enable_cueq_conv_fusion(
            cuet.SegmentedPolynomial(
                descriptor,
                math_dtype=torch.get_default_dtype(),
                method="uniform_1d",
            )
        )
    return cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        [ir for _, ir in irreps_out_cue],
        layout=layout,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        dtype=torch.get_default_dtype(),
        math_dtype=torch.get_default_dtype(),
        method="naive",
    )


__all__ = ["TensorProduct"]
