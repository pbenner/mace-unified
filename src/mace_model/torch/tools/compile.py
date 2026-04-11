from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Sequence

import torch

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from torch import autograd, nn
from torch.fx import symbolic_trace

from mace_model.torch.adapters.e3nn import (
    get_optimization_defaults,
    set_optimization_defaults,
)

ModuleFactory = Callable[..., nn.Module]
GraphData = dict[str, torch.Tensor]
_FORCE_OUTPUT_KEYS = frozenset(
    {"forces", "stress", "virials", "hessian", "edge_forces"}
)
_DISPLACEMENT_OUTPUT_KEYS = frozenset({"stress", "virials", "atomic_stresses"})


@contextmanager
def disable_e3nn_codegen():
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)


def prepare(func: ModuleFactory, allow_autograd: bool = True) -> ModuleFactory:
    if dynamo is not None:
        if allow_autograd:
            dynamo.allow_in_graph(autograd.grad)
        else:
            dynamo.disallow_in_graph(autograd.grad)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_e3nn_codegen():
            model = func(*args, **kwargs)
        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: nn.Module) -> nn.Module:
    _SIMPLIFY_REGISTRY.add(module)
    return module


def simplify(module: nn.Module) -> nn.Module:
    simplify_types = tuple(_SIMPLIFY_REGISTRY)
    for name, child in module.named_children():
        if isinstance(child, simplify_types):
            traced = symbolic_trace(child)
            setattr(module, name, traced)
        else:
            simplify(child)
    return module


def _normalize_output_keys(output_keys: Sequence[str] | None) -> tuple[str, ...]:
    if output_keys is None:
        return ("energy",)
    keys = tuple(str(key) for key in output_keys)
    if not keys:
        raise ValueError("output_keys must contain at least one output name.")
    return keys


def _head_tensor_from_graph(data: GraphData) -> torch.Tensor:
    head = data.get("head")
    if head is None:
        batch = data["batch"]
        return torch.empty((0,), dtype=torch.int64, device=batch.device)
    return torch.as_tensor(head, dtype=torch.int64, device=data["batch"].device)


def graph_to_inference_args(data: GraphData) -> tuple[torch.Tensor, ...]:
    """Convert a standard Torch graph dict into positional inference arguments."""
    return (
        data["positions"],
        data["node_attrs"],
        data["edge_index"],
        data["shifts"],
        data["unit_shifts"],
        data["cell"],
        data["batch"],
        data["ptr"],
        _head_tensor_from_graph(data),
    )


class TorchInferenceWrapper(nn.Module):
    """Tensor-only inference wrapper for ``torch.compile`` / ``torch.export``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        output_keys: Sequence[str] | None = None,
        training: bool = False,
        lammps_mliap: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_keys = _normalize_output_keys(output_keys)
        self.training = bool(training)
        self.lammps_mliap = bool(lammps_mliap)

    def forward(
        self,
        positions: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        unit_shifts: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        head: torch.Tensor,
    ):
        data: GraphData = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "batch": batch,
            "ptr": ptr,
        }
        if head.numel() > 0:
            data["head"] = head

        outputs = self.model(
            data,
            training=self.training,
            compute_force=any(key in _FORCE_OUTPUT_KEYS for key in self.output_keys),
            compute_virials="virials" in self.output_keys,
            compute_stress="stress" in self.output_keys,
            compute_displacement=any(
                key in _DISPLACEMENT_OUTPUT_KEYS for key in self.output_keys
            ),
            compute_hessian="hessian" in self.output_keys,
            compute_edge_forces="edge_forces" in self.output_keys,
            compute_atomic_stresses="atomic_stresses" in self.output_keys,
            lammps_mliap=self.lammps_mliap,
            compute_node_feats="node_feats" in self.output_keys,
        )
        return tuple(outputs[key] for key in self.output_keys)


def make_inference_wrapper(
    model: nn.Module,
    *,
    output_keys: Sequence[str] | None = None,
    training: bool = False,
    lammps_mliap: bool = False,
) -> TorchInferenceWrapper:
    """Create a tensor-only inference wrapper around a local Torch MACE model."""
    return TorchInferenceWrapper(
        model,
        output_keys=output_keys,
        training=training,
        lammps_mliap=lammps_mliap,
    )


def compile_model(
    model: nn.Module,
    *,
    output_keys: Sequence[str] | None = None,
    training: bool = False,
    lammps_mliap: bool = False,
    backend: str | Callable[..., Any] | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
) -> nn.Module:
    """Return a ``torch.compile``-ready inference module."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    wrapper = make_inference_wrapper(
        model,
        output_keys=output_keys,
        training=training,
        lammps_mliap=lammps_mliap,
    )
    return torch.compile(
        wrapper,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
        mode=mode,
        options=options,
    )


def export_model(
    model: nn.Module,
    example_graph: GraphData,
    *,
    output_keys: Sequence[str] | None = None,
    training: bool = False,
    lammps_mliap: bool = False,
    dynamic_shapes: Any = None,
    strict: bool = False,
    preserve_module_call_signature: tuple[str, ...] = (),
):
    """Export a local Torch MACE model via ``torch.export``."""
    if not hasattr(torch, "export"):
        raise RuntimeError("torch.export is not available in this PyTorch build.")
    wrapper = make_inference_wrapper(
        model,
        output_keys=output_keys,
        training=training,
        lammps_mliap=lammps_mliap,
    )
    return torch.export.export(
        wrapper,
        graph_to_inference_args(example_graph),
        dynamic_shapes=dynamic_shapes,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
    )
