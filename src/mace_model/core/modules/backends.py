"""Backend registration and dispatch for shared MACE core modules.

`mace_model.core` keeps the backend-independent control flow for blocks and models,
while Torch and JAX provide the concrete tensor and module operations. This file
defines the registry object passed around by the shared code and the decorators
used to bind backend implementations to classes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable


@dataclass(frozen=True)
class ModelBackend:
    """
    Unified backend operation bundle for model modules.

    The readout mixin uses a subset of these operations today. Additional module
    mixins can reuse the same backend object as they are introduced.
    """

    name: str

    # Core operations already used by the readout block.
    make_irreps: Callable[[Any], Any] | None = None
    make_linear: Callable[..., Any] | None = None
    make_bias_linear: Callable[..., Any] | None = None
    make_activation: Callable[..., Any] | None = None
    make_gate: Callable[..., Any] | None = None

    # Radial embedding helpers.
    make_bessel_basis: Callable[..., Any] | None = None
    make_gaussian_basis: Callable[..., Any] | None = None
    make_chebychev_basis: Callable[..., Any] | None = None
    make_polynomial_cutoff: Callable[..., Any] | None = None
    make_agnesi_transform: Callable[..., Any] | None = None
    make_soft_transform: Callable[..., Any] | None = None

    mask_head: Callable[[Any, Any, int], Any] | None = None
    mask_head_stage1: Callable[[Any, Any, int], Any] | None = None

    # Shared tensor helpers.
    atleast_1d: Callable[[Any], Any] | None = None
    make_atomic_energies: Callable[..., Any] | None = None
    get_atomic_energies: Callable[..., Any] | None = None
    make_scale_shift: Callable[..., Any] | None = None
    get_scale_shift: Callable[..., Any] | None = None
    atleast_2d: Callable[[Any], Any] | None = None
    matmul: Callable[[Any, Any], Any] | None = None
    transpose: Callable[[Any], Any] | None = None
    to_numpy: Callable[[Any], Any] | None = None

    # Joint embedding helpers.
    make_joint_embedders: Callable[..., Any] | None = None
    make_joint_projection: Callable[..., Any] | None = None
    make_joint_categorical_indices: Callable[..., Any] | None = None

    # Additional module operations to support broader MACE blocks.
    make_tensor_product: Callable[..., Any] | None = None
    make_fully_connected_tensor_product: Callable[..., Any] | None = None
    make_fully_connected_net: Callable[..., Any] | None = None
    make_radial_mlp: Callable[..., Any] | None = None
    make_symmetric_contraction: Callable[..., Any] | None = None
    make_transpose_irreps_layout: Callable[..., Any] | None = None
    make_custom_gate: Callable[..., Any] | None = None
    tp_out_irreps_with_instructions: Callable[..., Any] | None = None
    reshape_irreps: Callable[..., Any] | None = None
    scatter_sum: Callable[..., Any] | None = None
    stack: Callable[..., Any] | None = None
    sum: Callable[..., Any] | None = None
    make_irrep: Callable[..., Any] | None = None
    init_uniform_: Callable[..., Any] | None = None
    tanh: Callable[..., Any] | None = None
    silu: Callable[..., Any] | None = None
    sigmoid: Callable[..., Any] | None = None
    cat: Callable[..., Any] | None = None
    make_zeros: Callable[..., Any] | None = None
    make_parameter: Callable[..., Any] | None = None
    lammps_mp_apply: Callable[..., Any] | None = None
    make_ones: Callable[..., Any] | None = None
    make_index_attrs: Callable[..., Any] | None = None
    transpose_mul_ir: Callable[..., Any] | None = None

    def require(self, field_name: str) -> Callable[..., Any]:
        """Return a backend operation or raise a precise error if it is missing."""
        fn = getattr(self, field_name, None)
        if fn is None:
            raise NotImplementedError(
                f"ModelBackend '{self.name}' is missing required operation "
                f"'{field_name}'."
            )
        return fn


def _require_backend(instance: Any, class_name: str) -> ModelBackend:
    """Fetch the backend bound to a shared class instance."""
    backend = getattr(instance, "BACKEND", None)
    if backend is None:
        raise RuntimeError(f"{class_name} requires a class-level BACKEND.")
    return backend


def _model_backend_field_names() -> tuple[str, ...]:
    """Return the backend-operation fields declared on `ModelBackend`."""
    return tuple(field.name for field in fields(ModelBackend) if field.name != "name")


def define_backend(*, name: str):
    """
    Decorator that turns a namespace of backend operations into `ModelBackend`.

    The decorated class is treated as a simple container of callables. Only
    operations declared on `ModelBackend` are copied into the resulting backend
    object.
    """

    def decorator(ops_cls):
        kwargs = {}
        for field_name in _model_backend_field_names():
            fn = getattr(ops_cls, field_name, None)
            if fn is not None:
                kwargs[field_name] = fn
        return ModelBackend(name=name, **kwargs)

    return decorator


def use_backend(backend: ModelBackend):
    """
    Decorator to bind a backend bundle to a shared block or model class.
    """

    def decorator(cls):
        cls.BACKEND = backend
        return cls

    return decorator
