"""Utilities for importing PyTorch module parameters into NNX modules."""

from __future__ import annotations

import re
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

_IMPORT_MAPPERS: dict[str, Callable] = {}

_FIRST_CAP_RE = re.compile(r"(.)([A-Z][a-z]+)")
_ALL_CAP_RE = re.compile(r"([a-z0-9])([A-Z])")


def camel_to_snake(value: str) -> str:
    """Return a snake_case version of ``value``."""
    step1 = _FIRST_CAP_RE.sub(r"\1_\2", value)
    step2 = _ALL_CAP_RE.sub(r"\1_\2", step1)
    return step2.lower()


def _state_to_pure_dict(state: nnx.State) -> dict[str, Any]:
    def _extract(value):
        if isinstance(value, nnx.Variable):
            return value.get_value()
        return value

    return nnx.to_pure_dict(state, extract_fn=_extract)


def nxx_register_import_mapper(torch_type: str) -> Callable[[Callable], Callable]:
    """Register a function that copies parameters from a Torch module."""

    def decorator(fn: Callable) -> Callable:
        _IMPORT_MAPPERS[torch_type] = fn
        return fn

    return decorator


def _torch_module_name(module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__name__}"


def _copy_direct_parameters(torch_module, variables: MutableMapping) -> None:
    """Best-effort copy matching direct tensors to leaves in ``variables``."""
    if not variables:
        return

    def _maybe_assign(name: str, tensor) -> None:
        if name not in variables:
            return
        target = variables[name]
        array = jnp.asarray(tensor.detach().cpu().numpy())
        if hasattr(target, "dtype"):
            array = array.astype(target.dtype)
        target_shape = getattr(target, "shape", None)
        if target_shape is not None and target_shape != array.shape:
            if not target_shape and array.size == 1:
                array = array.reshape(())
            else:
                array = array.reshape(target_shape)
        variables[name] = array

    for name, param in torch_module.named_parameters(recurse=False):
        _maybe_assign(name, param)

    for name, buf in torch_module.named_buffers(recurse=False):
        if name in variables:
            _maybe_assign(name, buf)


def _iter_module_tree(module, path: tuple[str, ...] = ()):
    yield path, module
    for name, child in module.named_children():
        yield from _iter_module_tree(child, path + (name,))


def _resolve_scope(variables: MutableMapping, scope: Sequence[str]):
    node: Any = variables
    for key in scope:
        if isinstance(node, nnx.State):
            node = node
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Scope {'.'.join(scope)} not found in NNX state")
        node = node[key]
    if not isinstance(node, dict):
        raise KeyError(f"Scope {'.'.join(scope)} did not resolve to a dict")
    return node


def _apply_module_import(
    module,
    variables: MutableMapping,
    scope: tuple[str, ...],
    *,
    fallback: Callable | None,
) -> None:
    module_name = _torch_module_name(module)
    mapper = _IMPORT_MAPPERS.get(module_name)
    has_parameters = any(True for _ in module.parameters(recurse=False))
    has_buffers = any(True for _ in module.buffers(recurse=False))
    has_state = has_parameters or has_buffers

    if mapper is not None:
        mapper(module, variables, list(scope))
        return

    if not has_state:
        return

    if fallback is not None:
        try:
            target = _resolve_scope(variables, list(scope))
        except KeyError:
            fallback(module, variables)
        else:
            fallback(module, target)
        return

    raise NotImplementedError(
        f"No NNX import mapper registered for Torch module {module_name!r}"
    )


def nxx_auto_import_from_torch(
    *,
    allow_missing_mapper: bool = False,
    fallback: Callable | None = None,
):
    """Decorator adding a generic ``import_from_torch`` method to an NNX module."""

    def decorator(cls):
        fallback_fn = fallback or (
            _copy_direct_parameters if allow_missing_mapper else None
        )

        def _import_root_parameters(module, variables_mut) -> None:
            has_parameters = any(True for _ in module.parameters(recurse=False))
            has_buffers = any(True for _ in module.buffers(recurse=False))
            if not (has_parameters or has_buffers):
                return
            if fallback_fn is None:
                raise NotImplementedError(
                    f"No NNX import mapper registered for Torch module "
                    f"{_torch_module_name(module)!r}"
                )
            fallback_fn(module, variables_mut)

        @classmethod
        def _import_from_torch_impl(
            cls,
            torch_module,
            variables,
            *,
            skip_root: bool = False,
        ):
            variables_mut = variables

            if skip_root:
                _import_root_parameters(torch_module, variables_mut)

            for scope, module in _iter_module_tree(torch_module):
                if skip_root and not scope:
                    continue
                try:
                    _apply_module_import(
                        module,
                        variables_mut,
                        scope,
                        fallback=fallback_fn,
                    )
                except (KeyError, ValueError):
                    if fallback_fn is not None:
                        fallback_fn(module, variables_mut)
                    else:
                        raise

            return variables_mut

        @classmethod
        def import_from_torch(cls, torch_module, variables):
            return cls._import_from_torch_impl(
                torch_module,
                variables,
                skip_root=False,
            )

        cls.import_from_torch = import_from_torch
        cls._import_from_torch_impl = _import_from_torch_impl
        cls._torch_import_fallback = fallback_fn
        return cls

    return decorator


def nxx_register_module(torch_type: str):
    """Register an NNX module class to handle Torch weight imports."""

    def decorator(cls):
        def mapper(module, variables, scope: Sequence[str]):
            target = _resolve_scope(variables, scope)
            if hasattr(cls, "_import_from_torch_impl"):
                updated = cls._import_from_torch_impl(
                    module,
                    target,
                    skip_root=True,
                )
            else:
                updated = cls.import_from_torch(module, target)
            if updated is not None and updated is not target:
                target.clear()
                target.update(updated)

        _IMPORT_MAPPERS[torch_type] = mapper
        return cls

    return decorator


def init_from_torch(module: nnx.Module, torch_module):
    """Populate an NNX module's parameters from a Torch module in-place."""
    graphdef, state = nnx.split(module)
    pure = _state_to_pure_dict(state)
    updated = module.__class__.import_from_torch(torch_module, pure)
    if updated is not None:
        nnx.replace_by_pure_dict(state, updated)
        module = nnx.merge(graphdef, state)
    return module, state


@nxx_register_import_mapper("torch.nn.modules.linear.Linear")
def _import_linear(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    target["kernel"] = jnp.asarray(
        module.weight.detach().cpu().numpy().T, dtype=target["kernel"].dtype
    )
    if module.bias is not None and "bias" in target:
        target["bias"] = jnp.asarray(
            module.bias.detach().cpu().numpy(), dtype=target["bias"].dtype
        )


@nxx_register_import_mapper("torch.nn.modules.normalization.LayerNorm")
def _import_layernorm(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    target["scale"] = jnp.asarray(
        module.weight.detach().cpu().numpy(), dtype=target["scale"].dtype
    )
    target["bias"] = jnp.asarray(
        module.bias.detach().cpu().numpy(), dtype=target["bias"].dtype
    )


@nxx_register_import_mapper("torch.nn.modules.sparse.Embedding")
def _import_embedding(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    target["embedding"] = jnp.asarray(
        module.weight.detach().cpu().numpy(), dtype=target["embedding"].dtype
    )


@nxx_register_import_mapper("mace_model.torch.adapters.e3nn.nn._activation.Activation")
@nxx_register_import_mapper("e3nn.nn._activation.Activation")
def _import_e3nn_activation(module, variables, scope: Sequence[str]) -> None:
    del variables, scope
    try:
        from mace_model.jax.adapters.e3nn.math import register_normalize2mom_const
    except Exception:
        return None

    acts = getattr(module, "acts", None)
    if acts is None:
        return None
    for act in acts:
        const = getattr(act, "cst", None)
        if const is None:
            continue
        original = getattr(act, "f", None) or act
        register_normalize2mom_const(original, const)
    return None


@nxx_register_import_mapper(
    "mace_model.torch.adapters.e3nn.math._normalize_activation.normalize2mom"
)
@nxx_register_import_mapper("e3nn.math._normalize_activation.normalize2mom")
def _import_e3nn_normalize2mom(module, variables, scope: Sequence[str]) -> None:
    del variables, scope
    try:
        from mace_model.jax.adapters.e3nn.math import register_normalize2mom_const
    except Exception:
        return None

    const = getattr(module, "cst", None)
    if const is None:
        return None
    original = getattr(module, "f", None) or module
    register_normalize2mom_const(original, const)
    return None


@nxx_register_import_mapper("e3nn.o3._tensor_product._sub.ElementwiseTensorProduct")
def _import_e3nn_elementwise_tp(module, variables, scope: Sequence[str]) -> None:
    del module, variables, scope


@nxx_register_import_mapper("mace.modules.irreps_tools.reshape_irreps")
def _import_mace_reshape_irreps(module, variables, scope: Sequence[str]) -> None:
    del module, variables, scope


@nxx_register_import_mapper("mace.modules.radial.RadialMLP")
@nxx_register_import_mapper("mace_model.torch.modules.radial.RadialMLP")
def _import_mace_radial_mlp(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    net = target.get("net")
    if net is None:
        return None
    layers = net.get("layers")
    if layers is None:
        return None

    import torch  # local import to avoid hard dependency at module import time

    for name, child in module.net.named_children():
        if name not in layers:
            continue
        layer_vars = layers[name]
        if isinstance(child, torch.nn.Linear):
            layer_vars["kernel"] = jnp.asarray(
                child.weight.detach().cpu().numpy().T, dtype=layer_vars["kernel"].dtype
            )
            if child.bias is not None and "bias" in layer_vars:
                layer_vars["bias"] = jnp.asarray(
                    child.bias.detach().cpu().numpy(), dtype=layer_vars["bias"].dtype
                )
        elif isinstance(child, torch.nn.LayerNorm):
            layer_vars["scale"] = jnp.asarray(
                child.weight.detach().cpu().numpy(), dtype=layer_vars["scale"].dtype
            )
            layer_vars["bias"] = jnp.asarray(
                child.bias.detach().cpu().numpy(), dtype=layer_vars["bias"].dtype
            )
    return None


@nxx_register_import_mapper("mace_model.torch.adapters.e3nn.o3._linear.Linear")
@nxx_register_import_mapper("e3nn.o3._linear.Linear")
def _import_e3nn_linear(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    weight_tensor = getattr(module, "weight", None)
    if weight_tensor is None and hasattr(module, "linear"):
        weight_tensor = getattr(module.linear, "weight", None)
    if weight_tensor is None:
        state_dict = module.state_dict()
        weight_tensor = state_dict.get("linear.weight")
    if weight_tensor is None:
        raise AttributeError(f"Unable to locate Linear weights for module {module!r}.")
    weight = weight_tensor.detach().cpu().numpy()
    if "weight" in target:
        target["weight"] = jnp.asarray(
            weight.reshape(target["weight"].shape), dtype=target["weight"].dtype
        )
    if hasattr(module, "bias") and module.bias is not None and "bias" in target:
        target["bias"] = jnp.asarray(
            module.bias.detach().cpu().numpy(),
            dtype=target["bias"].dtype,
        )


@nxx_register_import_mapper("mace_model.torch.adapters.e3nn.nn._fc._Layer")
@nxx_register_import_mapper("e3nn.nn._fc._Layer")
def _import_e3nn_fc_layer(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    weight = jnp.asarray(module.weight.detach().cpu().numpy())
    if "weight" in target:
        target["weight"] = weight.astype(target["weight"].dtype, copy=False)
    elif "kernel" in target:
        reshaped = weight.reshape(target["kernel"].shape)
        target["kernel"] = reshaped.astype(target["kernel"].dtype, copy=False)
    if getattr(module, "act", None) is not None and "act_scale" in target:
        const = getattr(module.act, "cst", None)
        if const is not None:
            target["act_scale"] = jnp.asarray(const, dtype=target["act_scale"].dtype)
    if hasattr(module, "bias") and module.bias is not None:
        bias = jnp.asarray(module.bias.detach().cpu().numpy())
        if "bias" in target:
            target["bias"] = bias.astype(target["bias"].dtype, copy=False)


@nxx_register_import_mapper("mace_model.torch.adapters.e3nn.nn._fc.FullyConnectedNet")
@nxx_register_import_mapper("e3nn.nn._fc.FullyConnectedNet")
def _import_e3nn_fc(module, variables, scope: Sequence[str]) -> None:
    target = _resolve_scope(variables, scope)
    layers = target.get("layers")
    if isinstance(layers, dict):
        for name, child in module.named_children():
            if not name.startswith("layer"):
                continue
            suffix = name[len("layer") :]
            if not suffix.isdigit():
                continue
            idx = int(suffix)
            if idx not in layers:
                continue
            _import_e3nn_fc_layer(child, layers, [idx])


@nxx_register_import_mapper("e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct")
@nxx_register_import_mapper(
    "cuequivariance_torch.operations.fully_connected_tensor_product.FullyConnectedTensorProduct"
)
def _import_fctp(module, variables, scope: Sequence[str]) -> None:
    try:
        from mace_model.jax.adapters.cuequivariance.fully_connected_tensor_product import (
            _map_fctp as upstream_map_fctp,
        )
    except Exception:
        target = _resolve_scope(variables, scope)
        if hasattr(module, "weight") and "weight" in target:
            weight = module.weight.detach().cpu().numpy()
            target["weight"] = jnp.asarray(weight, dtype=target["weight"].dtype)
        return
    upstream_map_fctp(module, variables, scope)


@nxx_register_import_mapper(
    "cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction"
)
def _import_cue_symmetric_contraction(module, variables, scope: Sequence[str]) -> None:
    try:
        from mace_model.jax.adapters.cuequivariance.symmetric_contraction import (
            _import_cue_symmetric_contraction as upstream_import,
        )
    except Exception:
        target = _resolve_scope(variables, scope)
        if hasattr(module, "weight"):
            weight = module.weight.detach().cpu().numpy()
            existing = target.get("weight")
            dtype = existing.dtype if existing is not None else weight.dtype
            target["weight"] = jnp.asarray(weight, dtype=dtype)
        return
    upstream_import(module, variables, scope)


@nxx_register_import_mapper("mace.modules.symmetric_contraction.SymmetricContraction")
def _import_native_symmetric_contraction(
    module, variables, scope: Sequence[str]
) -> None:
    try:
        from mace_model.jax.adapters.cuequivariance.symmetric_contraction import (
            _import_native_symmetric_contraction as upstream_import,
        )
    except Exception:
        target = _resolve_scope(variables, scope)
        if hasattr(module, "weight") and "weight" in target:
            weight = module.weight.detach().cpu().numpy()
            target["weight"] = jnp.asarray(weight, dtype=target["weight"].dtype)
        return
    upstream_import(module, variables, scope)


_GATE_MAP: dict[str, Callable | None] = {
    "silu": jnn.silu,
    "silu6": jnn.silu,
    "swish": jnn.silu,
    "relu": jnn.relu,
    "tanh": jnn.tanh,
    "sigmoid": jnn.sigmoid,
    "softplus": jnn.softplus,
    "softsign": jnn.soft_sign,
    "gelu": jnn.gelu,
    "abs": jnp.abs,
    "none": None,
}


def resolve_gate_callable(gate: Callable | str | None) -> Callable | None:
    if gate is None:
        return None
    if isinstance(gate, str):
        return _GATE_MAP.get(gate.lower(), None)
    name = getattr(gate, "__name__", None)
    if name is None:
        cls = getattr(gate, "__class__", None)
        name = getattr(cls, "__name__", None)
    if not name:
        return None
    return _GATE_MAP.get(name.lower(), None)


__all__ = [
    "nxx_auto_import_from_torch",
    "nxx_register_import_mapper",
    "nxx_register_module",
    "init_from_torch",
    "resolve_gate_callable",
]
