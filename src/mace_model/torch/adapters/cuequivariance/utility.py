from __future__ import annotations

import types
from typing import Optional

import cuequivariance as cue
import torch

from mace_model.torch.adapters.e3nn import o3

_SUPPORTED_CUE_GROUPS = {"O3", "O3_e3nn"}


def _group_name(group_value: object | None) -> str | None:
    if group_value is None:
        return None
    if isinstance(group_value, str):
        return group_value
    return getattr(group_value, "__name__", None) or str(group_value)


def _resolve_cue_group(cueq_config) -> object:
    if cueq_config is None:
        return cue.O3
    group_value = getattr(cueq_config, "group", None)
    if group_value is None:
        return cue.O3
    if isinstance(group_value, str):
        if group_value == "O3_e3nn":
            return cue.O3
        try:
            return getattr(cue, group_value)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported cuequivariance group '{group_value}'."
            ) from exc
    return group_value


def _validate_cue_group(group_value: object | None, *, context: str) -> None:
    name = _group_name(group_value)
    if name is None:
        return
    if name not in _SUPPORTED_CUE_GROUPS:
        raise ValueError(
            f"{context} only supports the 'O3' or 'O3_e3nn' groups; "
            f"received {group_value!r}."
        )


def _normalize_cue_layout(layout: object):
    if isinstance(layout, str):
        return getattr(cue, layout)
    return layout


def _cue_layout(cueq_config) -> object:
    if cueq_config is None:
        return cue.mul_ir
    layout = getattr(cueq_config, "layout", cue.mul_ir)
    return _normalize_cue_layout(layout)


def _cue_group(cueq_config) -> object:
    return _resolve_cue_group(cueq_config)


def _cue_irreps(group: object, irreps) -> cue.Irreps:
    return cue.Irreps(group, irreps)


def _enable_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


class TransposeIrrepsLayoutWrapper(torch.nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        source: str,
        target: str,
        cueq_config: Optional[object] = None,
    ):
        super().__init__()
        del cueq_config
        self.irreps = o3.Irreps(irreps)
        self.source = source
        self.target = target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == self.target:
            return x

        dims = [ir.dim for _, ir in self.irreps]
        muls = [mul for mul, _ in self.irreps]
        ix = 0
        fields = []
        for mul, dim in zip(muls, dims):
            field = x[:, ix : ix + mul * dim]
            ix += mul * dim
            if self.source == "mul_ir" and self.target == "ir_mul":
                field = field.reshape(x.shape[0], mul, dim).transpose(1, 2)
            elif self.source == "ir_mul" and self.target == "mul_ir":
                field = field.reshape(x.shape[0], dim, mul).transpose(1, 2)
            else:
                raise ValueError(
                    f"Unsupported irreps layout transpose {self.source!r} -> {self.target!r}."
                )
            fields.append(field.reshape(x.shape[0], mul * dim))
        return torch.cat(fields, dim=-1)


__all__ = ["TransposeIrrepsLayoutWrapper"]
