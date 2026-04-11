"""Cue-equivariant adapters for Torch."""

from __future__ import annotations

import dataclasses
from typing import Optional

from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.tools.scatter import scatter_sum
from .fully_connected_tensor_product import (
    FullyConnectedTensorProduct as _CueFullyConnectedTensorProduct,
)
from .linear import Linear as _CueLinear
from .symmetric_contraction import SymmetricContraction as _CueSymmetricContraction
from .tensor_product import TensorProduct as _CueTensorProduct
from .utility import (
    TransposeIrrepsLayoutWrapper,
    _cue_layout,
    _resolve_cue_group,
    _validate_cue_group,
)


@dataclasses.dataclass
class CuEquivarianceConfig:
    enabled: bool = False
    layout: str = "mul_ir"
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False

    def __post_init__(self):
        if isinstance(self.layout, str):
            self.layout_str = self.layout
        else:
            self.layout_str = getattr(self.layout, "name", None) or getattr(
                self.layout, "__name__", None
            )
            if self.layout_str is None:
                self.layout_str = str(self.layout)


@dataclasses.dataclass
class OEQConfig:
    enabled: bool = False
    optimize_all: bool = False
    optimize_channelwise: bool = False
    conv_fusion: Optional[str] = "atomic"


def Linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    cueq_config: CuEquivarianceConfig | None = None,
):
    group_value = getattr(cueq_config, "group", None) if cueq_config else None
    _validate_cue_group(group_value, context="Linear")
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    layout = getattr(cueq_config, "layout", "mul_ir") if cueq_config else "mul_ir"
    linear_kwargs = dict(
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        layout=layout,
    )
    if group is not None:
        linear_kwargs["group"] = group
    return _CueLinear(irreps_in, irreps_out, **linear_kwargs)


def TensorProduct(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions=None,
    shared_weights: bool = False,
    internal_weights: bool = False,
    cueq_config: CuEquivarianceConfig | None = None,
    oeq_config: OEQConfig | None = None,
):
    del oeq_config
    group_value = getattr(cueq_config, "group", None) if cueq_config else None
    _validate_cue_group(group_value, context="TensorProduct")
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    conv_fusion = (
        bool(getattr(cueq_config, "conv_fusion", False)) if cueq_config else False
    )
    tp_kwargs = dict(
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        conv_fusion=conv_fusion,
    )
    if cueq_config is not None and getattr(cueq_config, "layout", None) is not None:
        tp_kwargs["layout"] = getattr(cueq_config, "layout")
    if group is not None:
        tp_kwargs["group"] = group
    return _CueTensorProduct(irreps_in1, irreps_in2, irreps_out, **tp_kwargs)


def FullyConnectedTensorProduct(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    cueq_config: CuEquivarianceConfig | None = None,
):
    group_value = getattr(cueq_config, "group", None) if cueq_config else None
    _validate_cue_group(group_value, context="FullyConnectedTensorProduct")
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    fctp_kwargs = dict(
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    if cueq_config is not None and getattr(cueq_config, "layout", None) is not None:
        fctp_kwargs["layout"] = getattr(cueq_config, "layout")
    if group is not None:
        fctp_kwargs["group"] = group
    return _CueFullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        **fctp_kwargs,
    )


def SymmetricContractionWrapper(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    num_elements: int | None = None,
    cueq_config: CuEquivarianceConfig | None = None,
    oeq_config: OEQConfig | None = None,
    use_reduced_cg: bool = True,
):
    del oeq_config
    group_value = getattr(cueq_config, "group", None) if cueq_config else None
    if cueq_config is not None:
        _validate_cue_group(group_value, context="SymmetricContraction")
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    sc_kwargs = dict(
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
        layout=_cue_layout(cueq_config),
    )
    if group is not None:
        sc_kwargs["group"] = group
    return _CueSymmetricContraction(irreps_in, irreps_out, **sc_kwargs)


__all__ = [
    "CuEquivarianceConfig",
    "OEQConfig",
    "Linear",
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "SymmetricContractionWrapper",
    "TransposeIrrepsLayoutWrapper",
    "scatter_sum",
]
