from __future__ import annotations

import types
from functools import cache

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import torch

from mace_model.core.modules.native_symmetric_weights import (
    gather_native_reduced_weights,
    native_design_matrix,
)
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


def torch_target_design_matrix(
    target_module,
    *,
    basis_dim: int,
    inputs_np: np.ndarray,
    probe_indices: list[int] | tuple[int, ...] = (0,),
) -> np.ndarray:
    """Evaluate a local cue-backed Torch module on all its basis vectors."""
    if not hasattr(target_module, "weight"):
        raise ValueError("Target Torch symmetric-contraction module lacks 'weight'.")

    torch_inputs = torch.tensor(inputs_np, dtype=target_module.weight.dtype)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for idx in range(basis_dim):
            per_element_outputs: list[np.ndarray] = []
            for element_index in probe_indices:
                target_module.weight.zero_()
                target_module.weight[element_index, idx, 0] = 1.0
                torch_indices = torch.full(
                    (inputs_np.shape[0],),
                    int(element_index),
                    dtype=torch.int32,
                )
                out = target_module(torch_inputs, torch_indices).detach().cpu().numpy()
                per_element_outputs.append(np.asarray(out).reshape(-1))
            outputs.append(np.concatenate(per_element_outputs, axis=0))
    return np.stack(outputs, axis=1)


def _native_full_cg_context(
    native_module,
    irreps_in_str: str,
    correlation: int,
):
    native_dtype = native_module.contractions[0].weights_max.dtype
    cue_irreps_in = cue.Irreps(cue.O3, irreps_in_str)
    num_elements = int(native_module.contractions[0].weights_max.shape[0])
    mul = int(cue_irreps_in[0].mul)
    feature_dim = int(sum(term.ir.dim for term in cue_irreps_in))
    native_dim = gather_native_reduced_weights(
        native_module,
        correlation=correlation,
        mul_dim=mul,
        num_elements=num_elements,
    ).shape[1]
    solve_dtype = np.float64 if native_dtype == torch.float64 else np.float32
    return native_dtype, solve_dtype, mul, feature_dim, native_dim


def _solve_native_to_canonical_transform(
    *,
    native_module,
    canonical_matrix: np.ndarray,
    solve_dtype,
    mul: int,
    feature_dim: int,
    native_dim: int,
) -> np.ndarray:
    batch = max(int(canonical_matrix.shape[1]), native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(solve_dtype)
    native_matrix = native_design_matrix(
        native_module,
        basis_dim=native_dim,
        inputs_np=inputs_np,
        probe_indices=[0],
    ).astype(solve_dtype, copy=False)
    transform = np.linalg.lstsq(canonical_matrix, native_matrix, rcond=1e-12)[0]
    return transform.astype(np.float64)


_NATIVE_INSTANCE_TRANSFORM_CACHE: dict[tuple[str, str, int, int, str], np.ndarray] = {}


def _full_cg_transform_from_native_instance(native_module) -> np.ndarray:
    """Compute exact native->canonical full-CG transform from a module instance."""
    irreps_in = cue.Irreps(cue.O3, str(native_module.irreps_in))
    irreps_out = cue.Irreps(cue.O3, str(native_module.irreps_out))
    correlation = int(native_module.contractions[0].correlation)
    native_dtype, solve_dtype, mul, feature_dim, native_dim = _native_full_cg_context(
        native_module,
        str(irreps_in),
        correlation,
    )
    cache_key = (
        str(irreps_in),
        str(irreps_out),
        correlation,
        native_dim,
        str(native_dtype),
    )
    cached = _NATIVE_INSTANCE_TRANSFORM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    canonical_module = (
        SymmetricContraction(
            irreps_in=o3.Irreps(str(irreps_in)),
            irreps_out=o3.Irreps(str(irreps_out)),
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
        )
        .to(dtype=native_dtype)
        .eval()
    )
    batch = max(int(canonical_module.weight.shape[1]), native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(solve_dtype)
    canonical_matrix = torch_target_design_matrix(
        canonical_module,
        basis_dim=int(canonical_module.weight.shape[1]),
        inputs_np=inputs_np,
    ).astype(solve_dtype, copy=False)
    transform = _solve_native_to_canonical_transform(
        native_module=native_module,
        canonical_matrix=canonical_matrix,
        solve_dtype=solve_dtype,
        mul=mul,
        feature_dim=feature_dim,
        native_dim=native_dim,
    )
    _NATIVE_INSTANCE_TRANSFORM_CACHE[cache_key] = transform
    return transform


@cache
def _cached_full_cg_transform_from_native_torch(
    native_cls,
    native_irreps_cls,
    irreps_in_str: str,
    irreps_out_str: str,
    correlation: int,
) -> np.ndarray:
    native_module = native_cls(
        irreps_in=native_irreps_cls(irreps_in_str),
        irreps_out=native_irreps_cls(irreps_out_str),
        correlation=correlation,
        use_reduced_cg=False,
        num_elements=1,
    ).eval()
    native_dtype, solve_dtype, mul, feature_dim, native_dim = _native_full_cg_context(
        native_module,
        irreps_in_str,
        correlation,
    )
    canonical_module = (
        SymmetricContraction(
            irreps_in=o3.Irreps(irreps_in_str),
            irreps_out=o3.Irreps(irreps_out_str),
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
        )
        .to(dtype=native_dtype)
        .eval()
    )
    batch = max(int(canonical_module.weight.shape[1]), native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(solve_dtype)
    canonical_matrix = torch_target_design_matrix(
        canonical_module,
        basis_dim=int(canonical_module.weight.shape[1]),
        inputs_np=inputs_np,
    ).astype(solve_dtype, copy=False)
    return _solve_native_to_canonical_transform(
        native_module=native_module,
        canonical_matrix=canonical_matrix,
        solve_dtype=solve_dtype,
        mul=mul,
        feature_dim=feature_dim,
        native_dim=native_dim,
    )


def native_full_to_canonical_weight(
    torch_module, native_weight: np.ndarray
) -> np.ndarray:
    """Reorder native full-CG weights into the local Torch canonical basis."""
    irreps_in = cue.Irreps(cue.O3, str(torch_module.irreps_in)).set_mul(1)
    irreps_out = cue.Irreps(cue.O3, str(torch_module.irreps_out)).set_mul(1)
    correlation = int(torch_module.contractions[0].correlation)
    try:
        transform = _cached_full_cg_transform_from_native_torch(
            type(torch_module),
            type(torch_module.irreps_in),
            str(irreps_in),
            str(irreps_out),
            correlation,
        ).astype(native_weight.dtype, copy=False)
    except Exception as cached_exc:
        try:
            transform = _full_cg_transform_from_native_instance(torch_module).astype(
                native_weight.dtype,
                copy=False,
            )
        except Exception as instance_exc:
            raise RuntimeError(
                "Failed to compute exact native full-CG -> canonical basis "
                "mapping for symmetric-contraction weights. "
                f"(cached_transform_error={type(cached_exc).__name__}: {cached_exc}; "
                f"instance_transform_error={type(instance_exc).__name__}: {instance_exc})"
            ) from instance_exc
    return np.einsum("ab,zbu->zau", transform, native_weight, optimize=True)


__all__ = [
    "SymmetricContraction",
    "native_full_to_canonical_weight",
    "torch_target_design_matrix",
]
