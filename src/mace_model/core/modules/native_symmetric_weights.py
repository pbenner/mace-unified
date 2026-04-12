"""Helpers for importing native MACE symmetric-contraction weights.

The original Torch MACE implementation stores symmetric-contraction weights in
its own hierarchical basis. Local Torch/JAX cue-backed implementations store
weights in a different basis. This module computes the required change of basis
numerically from module responses, without depending on the old `mace` package
at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import cuequivariance as cue
import numpy as np
from cuequivariance.etc.linalg import round_to_sqrt_rational, triu_array
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)


def _probe_element_indices(num_elements: int) -> list[int]:
    """Return representative element rows for design-matrix probing."""
    if num_elements <= 2:
        return list(range(num_elements))
    return [0, num_elements - 1]


def native_symmetric_metadata(torch_module) -> dict[str, int]:
    """Extract shape metadata from a native Torch symmetric-contraction module."""
    irreps_in = cue.Irreps(cue.O3, str(torch_module.irreps_in))
    correlation = int(torch_module.contractions[0].correlation)
    num_elements = int(torch_module.contractions[0].weights_max.shape[0])
    mul = int(irreps_in[0].mul)
    feature_dim = int(sum(term.ir.dim for term in irreps_in))
    return {
        "correlation": correlation,
        "num_elements": num_elements,
        "mul": mul,
        "feature_dim": feature_dim,
    }


def _flatten(
    x: np.ndarray,
    axis_start: int | None = None,
    axis_end: int | None = None,
) -> np.ndarray:
    x = np.asarray(x)
    if axis_start is None:
        axis_start = 0
    if axis_end is None:
        axis_end = x.ndim
    return x.reshape(
        x.shape[:axis_start]
        + (np.prod(x.shape[axis_start:axis_end]),)
        + x.shape[axis_end:]
    )


def _stp_to_matrix(d) -> np.ndarray:
    matrix = np.zeros([operand.num_segments for operand in d.operands])
    for path in d.paths:
        matrix[path.indices] = path.coefficients
    return matrix


def _compute_u_cueq(
    irreps_in,
    irreps_out,
    correlation: int,
    *,
    dtype,
) -> list[np.ndarray | str]:
    irreps_in = cue.Irreps(cue.O3, str(irreps_in))
    irreps_out = cue.Irreps(cue.O3, str(irreps_out))

    out: list[np.ndarray | str] = []
    for _, ir in irreps_out:
        u_matrix = cue.reduced_symmetric_tensor_product_basis(
            irreps_in,
            correlation,
            keep_ir=ir,
            layout=cue.ir_mul,
        ).array
        if u_matrix.shape[-1] == 0:
            if ir.dim == 1:
                out_shape = (*([irreps_in.dim] * correlation), 1)
            else:
                out_shape = (ir.dim, *([irreps_in.dim] * correlation), 1)
            return [np.zeros(out_shape, dtype=dtype)]
        out.append(str(ir))
        u_matrix = np.asarray(
            u_matrix.reshape(*([irreps_in.dim] * correlation), ir.dim, -1),
            dtype=dtype,
        )
        u_matrix = np.moveaxis(u_matrix, -2, 0)
        if ir.dim == 1:
            u_matrix = u_matrix[0]
        out.append(u_matrix)
    return out


def _native_symmetric_polynomial(irreps_in, irreps_out, degree: int):
    mul = irreps_in.muls[0]
    assert all(mul == m for m in irreps_in.muls)
    assert all(mul == m for m in irreps_out.muls)
    irreps_in = irreps_in.set_mul(1)
    irreps_out = irreps_out.set_mul(1)

    input_operands = range(1, degree + 1)
    output_operand = degree + 1

    abc = "abcdefgh"[:degree]
    descriptor = cue.SegmentedTensorProduct.from_subscripts(
        f"u_{'_'.join(f'{a}' for a in abc)}_i+{abc}ui"
    )

    for operand in input_operands:
        descriptor.add_segment(operand, (irreps_in.dim,))
    for _, ir in irreps_out:
        u = _compute_u_cueq(
            irreps_in,
            cue.Irreps(cue.O3, [(1, ir)]),
            degree,
            dtype=np.float64,
        )[-1]
        u = np.asarray(u)
        if ir.dim == 1:
            u = np.expand_dims(u, axis=0)
        u = np.moveaxis(u, 0, -1)

        if u.shape[-2] == 0:
            descriptor.add_segment(output_operand, {"i": ir.dim})
        else:
            u = triu_array(u, degree)
            descriptor.add_path(None, *(0,) * degree, None, c=u)

    descriptor = descriptor.flatten_coefficient_modes()
    descriptor = descriptor.append_modes_to_all_operands("u", {"u": mul})

    [weights, x], y = descriptor.operands[:2], descriptor.operands[-1]
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(weights.size), cue.ir_mul),
            cue.IrrepsAndLayout(mul * irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(mul * irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial(
            [weights, x],
            [y],
            [(cue.Operation([0] + [1] * degree + [2]), descriptor)],
        ),
    )


@cache
def _native_reduced_projection(
    irreps_in_str: str,
    irreps_out_str: str,
    degrees: tuple[int, ...],
) -> np.ndarray:
    irreps_in = cue.Irreps(cue.O3, irreps_in_str)
    irreps_out = cue.Irreps(cue.O3, irreps_out_str)

    poly1 = cue.EquivariantPolynomial.stack(
        [
            cue.EquivariantPolynomial.stack(
                [
                    _native_symmetric_polynomial(
                        irreps_in,
                        irreps_out[i : i + 1],
                        degree,
                    )
                    for degree in reversed(degrees)
                ],
                [True, False, False],
            )
            for i in range(len(irreps_out))
        ],
        [True, False, True],
    )
    poly2 = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, degrees)
    a1, a2 = [
        np.concatenate(
            [
                _flatten(
                    _stp_to_matrix(
                        descriptor.symmetrize_operands(
                            range(1, descriptor.num_operands - 1)
                        )
                    ),
                    1,
                    None,
                )
                for _, descriptor in poly.polynomial.operations
            ],
            axis=1,
        )
        for poly in (poly1, poly2)
    ]
    nonzeros = np.nonzero(np.any(a1 != 0, axis=0) | np.any(a2 != 0, axis=0))[0]
    a1 = a1[:, nonzeros]
    a2 = a2[:, nonzeros]
    projection = a1 @ np.linalg.pinv(a2)
    projection = round_to_sqrt_rational(projection)
    np.testing.assert_allclose(a1, projection @ a2, atol=1e-7)
    return np.asarray(projection, dtype=np.float64)


def gather_native_reduced_weights(
    torch_module,
    *,
    correlation: int,
    mul_dim: int,
    num_elements: int,
) -> np.ndarray:
    """Stack native-use weights in the order expected by the conversion solve."""
    base_array = torch_module.contractions[0].weights_max.detach().cpu().numpy()
    dtype = np.asarray(base_array).dtype

    if correlation <= 0:
        return np.zeros((num_elements, 0, mul_dim), dtype=dtype)

    native_blocks: list[np.ndarray] = []
    for contraction in torch_module.contractions:
        degree_blocks: list[np.ndarray] = []

        for degree in range(correlation, 0, -1):
            if degree == correlation:
                weight_param = contraction.weights_max
                zeroed = bool(getattr(contraction, "weights_max_zeroed", False))
            else:
                idx = correlation - degree - 1
                weight_param = contraction.weights[idx]
                zeroed = bool(getattr(contraction, f"weights_{idx}_zeroed", False))

            array = np.asarray(weight_param.detach().cpu().numpy(), dtype=dtype)
            if zeroed:
                array = np.zeros_like(array)

            if array.shape[1] == 0:
                continue
            degree_blocks.append(array)

        if degree_blocks:
            native_blocks.append(np.concatenate(degree_blocks, axis=1))

    if native_blocks:
        stacked = np.concatenate(native_blocks, axis=1)
    else:
        stacked = np.zeros((num_elements, 0, mul_dim), dtype=dtype)

    if stacked.shape[0] != num_elements or stacked.shape[2] != mul_dim:
        raise ValueError("Native SymmetricContraction weights shape mismatch.")

    return stacked


def assign_native_basis(
    torch_module,
    *,
    basis_vector: np.ndarray,
    correlation: int,
    element_index: int = 0,
    mul_index: int = 0,
) -> None:
    """Populate a native Torch symmetric-contraction module from a flat basis."""
    import torch  # noqa: PLC0415

    offset = 0
    with torch.no_grad():
        for contraction in torch_module.contractions:
            for degree in range(correlation, 0, -1):
                if degree == correlation:
                    target = contraction.weights_max
                else:
                    idx = correlation - degree - 1
                    target = contraction.weights[idx]

                width = int(target.shape[1])
                slice_vals = basis_vector[offset : offset + width]
                target.zero_()
                target[element_index, :width, mul_index] = torch.tensor(
                    slice_vals,
                    dtype=target.dtype,
                )
                offset += width

    if offset != basis_vector.size:
        raise ValueError("Basis vector length mismatch while scattering weights.")


def native_design_matrix(
    torch_module,
    *,
    basis_dim: int,
    inputs_np: np.ndarray,
    probe_indices: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Evaluate the native module on all its basis vectors."""
    import torch  # noqa: PLC0415

    correlation = int(torch_module.contractions[0].correlation)
    num_elements = int(torch_module.contractions[0].weights_max.shape[0])
    dtype = torch_module.contractions[0].weights_max.dtype
    torch_inputs = torch.tensor(inputs_np, dtype=dtype)

    if probe_indices is None:
        probe_indices = _probe_element_indices(num_elements)
    outputs: list[np.ndarray] = []
    for idx in range(basis_dim):
        per_element_outputs: list[np.ndarray] = []
        for element_index in probe_indices:
            basis = np.zeros(basis_dim, dtype=np.asarray(inputs_np).dtype)
            basis[idx] = 1.0
            assign_native_basis(
                torch_module,
                basis_vector=basis,
                correlation=correlation,
                element_index=element_index,
            )
            torch_attrs = torch.zeros((inputs_np.shape[0], num_elements), dtype=dtype)
            torch_attrs[:, element_index] = 1.0
            with torch.no_grad():
                out = torch_module(torch_inputs, torch_attrs).detach().cpu().numpy()
            per_element_outputs.append(np.asarray(out).reshape(-1))
        outputs.append(np.concatenate(per_element_outputs, axis=0))

    return np.stack(outputs, axis=1)


def torch_target_design_matrix(
    target_module,
    *,
    basis_dim: int,
    inputs_np: np.ndarray,
    probe_indices: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Evaluate a local cue-backed Torch module on all its basis vectors."""
    import torch  # noqa: PLC0415

    if not hasattr(target_module, "weight"):
        raise ValueError("Target Torch symmetric-contraction module lacks 'weight'.")

    torch_inputs = torch.tensor(inputs_np, dtype=target_module.weight.dtype)
    num_elements = int(target_module.weight.shape[0])
    if probe_indices is None:
        probe_indices = _probe_element_indices(num_elements)

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


def _with_zero_weights(params: dict) -> dict:
    """Return a params pytree with all learnable weights zeroed."""
    import jax.numpy as jnp  # noqa: PLC0415

    params_mutable = dict(params)
    params_mutable["weight"] = jnp.zeros_like(params_mutable["weight"])
    return params_mutable


def _canonical_design_matrix(
    graphdef,
    params_zero: dict,
    inputs,
    indices,
) -> np.ndarray:
    """Evaluate the JAX module on the canonical cue basis vectors."""
    import jax.numpy as jnp  # noqa: PLC0415

    weight_shape = np.asarray(params_zero["weight"]).shape
    basis_dim = int(weight_shape[1])

    outputs: list[np.ndarray] = []
    for idx in range(basis_dim):
        weight = np.zeros(weight_shape, dtype=np.float32)
        weight[0, idx, 0] = 1.0

        params_idx = dict(params_zero)
        params_idx["weight"] = jnp.asarray(weight)
        out, _ = graphdef.apply(params_idx)(inputs, indices)
        outputs.append(np.asarray(out).reshape(-1))

    return np.stack(outputs, axis=1)


def _compute_full_cg_transform(
    irreps_in,
    irreps_out,
    correlation: int,
    *,
    native_module=None,
) -> np.ndarray:
    """Return the native -> canonical transform for a full-CG setup."""
    base_in = cue.Irreps(cue.O3, str(irreps_in)).set_mul(1)
    base_out = cue.Irreps(cue.O3, str(irreps_out)).set_mul(1)
    if native_module is not None:
        return _cached_full_cg_transform_from_native(
            type(native_module),
            type(native_module.irreps_in),
            str(base_in),
            str(base_out),
            int(correlation),
        )
    return _cached_full_cg_transform(str(base_in), str(base_out), int(correlation))


@cache
def _cached_full_cg_transform_from_native(
    native_cls,
    native_irreps_cls,
    irreps_in_str: str,
    irreps_out_str: str,
    correlation: int,
) -> np.ndarray:
    """Compute the native -> canonical transform from the real native class."""
    try:
        import torch  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
        from flax import nnx  # noqa: PLC0415

        from mace_model.jax.adapters.cuequivariance.symmetric_contraction import (  # noqa: PLC0415
            SymmetricContraction as JaxSymmetricContraction,
        )
        from mace_model.jax.adapters.e3nn import Irreps as JaxIrreps  # noqa: PLC0415
        from mace_model.jax.nnx_utils import state_to_pure_dict  # noqa: PLC0415

        native_module = native_cls(
            irreps_in=native_irreps_cls(irreps_in_str),
            irreps_out=native_irreps_cls(irreps_out_str),
            correlation=correlation,
            use_reduced_cg=False,
            num_elements=1,
        ).eval()

        native_dtype = native_module.contractions[0].weights_max.dtype
        native_module = native_module.to(dtype=native_dtype)
        jax_module = JaxSymmetricContraction(
            irreps_in=JaxIrreps(irreps_in_str),
            irreps_out=JaxIrreps(irreps_out_str),
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
            rngs=nnx.Rngs(0),
        )

        mul = int(cue.Irreps(cue.O3, irreps_in_str)[0].mul)
        feature_dim = int(
            sum(term.ir.dim for term in cue.Irreps(cue.O3, irreps_in_str))
        )

        native_dim = gather_native_reduced_weights(
            native_module,
            correlation=correlation,
            mul_dim=mul,
            num_elements=1,
        ).shape[1]

        graphdef, state = nnx.split(jax_module)
        params = state_to_pure_dict(state)
        params_zero = _with_zero_weights(params)
        canonical_dim = int(np.asarray(params_zero["weight"]).shape[1])

        solve_dtype = np.float64 if native_dtype == torch.float64 else np.float32
        batch = max(canonical_dim, native_dim)
        rng = np.random.default_rng(0)
        inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(solve_dtype)

        canonical_matrix = _canonical_design_matrix(
            graphdef,
            params_zero,
            jnp.asarray(inputs_np),
            jnp.zeros((batch,), dtype=jnp.int32),
        ).astype(solve_dtype, copy=False)
        native_matrix = native_design_matrix(
            native_module,
            basis_dim=native_dim,
            inputs_np=inputs_np,
            probe_indices=[0],
        ).astype(solve_dtype, copy=False)
        transform = np.linalg.lstsq(canonical_matrix, native_matrix, rcond=1e-12)[0]
        return transform.astype(np.float64)
    except Exception:
        return _cached_full_cg_transform_from_native_torch(
            native_cls,
            native_irreps_cls,
            irreps_in_str,
            irreps_out_str,
            correlation,
        )


@cache
def _cached_full_cg_transform_from_native_torch(
    native_cls,
    native_irreps_cls,
    irreps_in_str: str,
    irreps_out_str: str,
    correlation: int,
) -> np.ndarray:
    """Compute the native -> canonical transform using only local Torch code."""
    import torch  # noqa: PLC0415

    from mace_model.torch.adapters.cuequivariance import (  # noqa: PLC0415
        SymmetricContractionWrapper as TorchSymmetricContraction,
    )
    from mace_model.torch.adapters.e3nn import o3  # noqa: PLC0415

    native_module = native_cls(
        irreps_in=native_irreps_cls(irreps_in_str),
        irreps_out=native_irreps_cls(irreps_out_str),
        correlation=correlation,
        use_reduced_cg=False,
        num_elements=1,
    ).eval()

    native_dtype = native_module.contractions[0].weights_max.dtype
    native_module = native_module.to(dtype=native_dtype)
    canonical_module = (
        TorchSymmetricContraction(
            irreps_in=o3.Irreps(irreps_in_str),
            irreps_out=o3.Irreps(irreps_out_str),
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
        )
        .to(dtype=native_dtype)
        .eval()
    )

    mul = int(cue.Irreps(cue.O3, irreps_in_str)[0].mul)
    feature_dim = int(sum(term.ir.dim for term in cue.Irreps(cue.O3, irreps_in_str)))
    native_dim = gather_native_reduced_weights(
        native_module,
        correlation=correlation,
        mul_dim=mul,
        num_elements=1,
    ).shape[1]
    canonical_dim = int(canonical_module.weight.shape[1])

    solve_dtype = np.float64 if native_dtype == torch.float64 else np.float32
    batch = max(canonical_dim, native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(solve_dtype)

    canonical_matrix = torch_target_design_matrix(
        canonical_module,
        basis_dim=canonical_dim,
        inputs_np=inputs_np,
        probe_indices=[0],
    ).astype(solve_dtype, copy=False)
    native_matrix = native_design_matrix(
        native_module,
        basis_dim=native_dim,
        inputs_np=inputs_np,
        probe_indices=[0],
    ).astype(solve_dtype, copy=False)
    transform = np.linalg.lstsq(canonical_matrix, native_matrix, rcond=1e-12)[0]
    return transform.astype(np.float64)


@cache
def _cached_full_cg_transform(
    irreps_in_str: str,
    irreps_out_str: str,
    correlation: int,
) -> np.ndarray:
    """Compute the native -> canonical transform for full-CG weights."""
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    from mace_model.jax.adapters.cuequivariance.symmetric_contraction import (  # noqa: PLC0415
        SymmetricContraction as JaxSymmetricContraction,
    )
    from mace_model.jax.adapters.e3nn import Irreps as JaxIrreps  # noqa: PLC0415
    from mace_model.jax.nnx_utils import state_to_pure_dict  # noqa: PLC0415
    from mace_model.torch.adapters.cuequivariance import (  # noqa: PLC0415
        SymmetricContractionWrapper as TorchSymmetricContraction,
    )
    from mace_model.torch.adapters.e3nn import o3  # noqa: PLC0415

    torch_module = (
        TorchSymmetricContraction(
            irreps_in=o3.Irreps(irreps_in_str),
            irreps_out=o3.Irreps(irreps_out_str),
            correlation=correlation,
            num_elements=1,
            use_reduced_cg=False,
        )
        .float()
        .eval()
    )

    jax_module = JaxSymmetricContraction(
        irreps_in=JaxIrreps(irreps_in_str),
        irreps_out=JaxIrreps(irreps_out_str),
        correlation=correlation,
        num_elements=1,
        use_reduced_cg=False,
        rngs=nnx.Rngs(0),
    )

    mul = int(cue.Irreps(cue.O3, irreps_in_str)[0].mul)
    feature_dim = int(sum(term.ir.dim for term in cue.Irreps(cue.O3, irreps_in_str)))

    native_dim = gather_native_reduced_weights(
        torch_module,
        correlation=correlation,
        mul_dim=mul,
        num_elements=1,
    ).shape[1]

    graphdef, state = nnx.split(jax_module)
    params = state_to_pure_dict(state)
    params_zero = _with_zero_weights(params)
    canonical_dim = int(np.asarray(params_zero["weight"]).shape[1])

    batch = max(canonical_dim, native_dim)
    rng = np.random.default_rng(0)
    inputs_np = rng.standard_normal((batch, mul, feature_dim)).astype(np.float32)
    inputs_jax = jnp.asarray(inputs_np)
    indices_jax = jnp.zeros((batch,), dtype=jnp.int32)

    canonical_matrix = _canonical_design_matrix(
        graphdef,
        params_zero,
        inputs_jax,
        indices_jax,
    )
    native_matrix = native_design_matrix(
        torch_module,
        basis_dim=native_dim,
        inputs_np=inputs_np,
    )
    transform = np.linalg.lstsq(canonical_matrix, native_matrix, rcond=1e-12)[0]
    return transform.astype(np.float64)


def convert_native_symmetric_weights(
    torch_module,
    *,
    target_template,
    target_design_matrix_fn: Callable[[int, np.ndarray], np.ndarray],
    target_basis_kind: str | None = None,
) -> np.ndarray:
    """Convert native Torch weights to a target basis using design matrices."""
    template = np.asarray(target_template)
    if template.ndim != 3:
        raise ValueError(
            "Target symmetric-contraction template must have shape "
            "(num_elements, basis_dim, mul)."
        )

    basis_dim = int(template.shape[1])
    mul_dim = int(template.shape[2])
    if basis_dim == 0 or mul_dim == 0:
        return np.zeros_like(template)

    metadata = native_symmetric_metadata(torch_module)
    native_weight = gather_native_reduced_weights(
        torch_module,
        correlation=metadata["correlation"],
        mul_dim=mul_dim,
        num_elements=int(template.shape[0]),
    )
    native_dim = int(native_weight.shape[1])
    irreps_in = cue.Irreps(cue.O3, str(torch_module.irreps_in))
    irreps_out = cue.Irreps(cue.O3, str(torch_module.irreps_out))
    degrees = tuple(range(1, metadata["correlation"] + 1))

    reduced_projection = _native_reduced_projection(
        str(irreps_in),
        str(irreps_out),
        degrees,
    ).astype(native_weight.dtype, copy=False)
    _, descriptor_projection = cue_mace_symmetric_contraction(
        irreps_in,
        irreps_out,
        degrees,
    )
    descriptor_projection = np.asarray(
        descriptor_projection,
        dtype=native_weight.dtype,
    )

    native_projection_dim = int(reduced_projection.shape[0])
    reduced_dim = int(reduced_projection.shape[1])
    full_dim = int(descriptor_projection.shape[0])

    if target_basis_kind not in {None, "reduced", "native_full", "canonical_full"}:
        raise ValueError(
            "target_basis_kind must be one of None, 'reduced', 'native_full', "
            f"or 'canonical_full'; got {target_basis_kind!r}."
        )

    if target_basis_kind in {None, "reduced"} and basis_dim == reduced_dim:
        if native_dim == reduced_dim:
            converted = np.einsum(
                "zau,ab->zbu",
                native_weight,
                reduced_projection,
                optimize=True,
            )
            return converted.astype(template.dtype, copy=False)
        if native_dim == full_dim:
            transform = _compute_full_cg_transform(
                irreps_in,
                irreps_out,
                metadata["correlation"],
                native_module=torch_module,
            ).astype(native_weight.dtype, copy=False)
            canonical = np.einsum(
                "ab,zbu->zau",
                transform,
                native_weight,
                optimize=True,
            )
            converted = np.einsum(
                "zau,ab->zbu",
                canonical,
                descriptor_projection,
                optimize=True,
            )
            return converted.astype(template.dtype, copy=False)

    if basis_dim == full_dim:
        if target_basis_kind == "canonical_full" and native_dim == full_dim:
            transform = _compute_full_cg_transform(
                irreps_in,
                irreps_out,
                metadata["correlation"],
                native_module=torch_module,
            ).astype(native_weight.dtype, copy=False)
            converted = np.einsum(
                "ab,zbu->zau",
                transform,
                native_weight,
                optimize=True,
            )
            return converted.astype(template.dtype, copy=False)

        if target_basis_kind == "native_full" and native_dim == full_dim:
            return native_weight.astype(template.dtype, copy=False)

    solve_dtype = np.result_type(template.dtype, native_weight.dtype, np.float64)
    batch = max(16 * max(basis_dim, native_dim), 128)
    target_matrices: list[np.ndarray] = []
    native_matrices: list[np.ndarray] = []
    best_transform = None
    best_residual = None
    best_rank = 0
    for seed in range(16):
        rng = np.random.default_rng(seed)
        inputs_np = rng.standard_normal(
            (batch, metadata["mul"], metadata["feature_dim"])
        ).astype(solve_dtype, copy=False)

        target_matrix_seed = np.asarray(
            target_design_matrix_fn(basis_dim, inputs_np),
            dtype=solve_dtype,
        )
        native_matrix_seed = np.asarray(
            native_design_matrix(
                torch_module,
                basis_dim=native_dim,
                inputs_np=inputs_np,
            ),
            dtype=solve_dtype,
        )
        target_matrices.append(target_matrix_seed)
        native_matrices.append(native_matrix_seed)
        target_matrix = np.concatenate(target_matrices, axis=0)
        native_matrix = np.concatenate(native_matrices, axis=0)

        rank = int(np.linalg.matrix_rank(target_matrix))
        best_rank = max(best_rank, rank)
        transform = np.linalg.lstsq(target_matrix, native_matrix, rcond=1e-12)[0]
        residual = np.linalg.norm(target_matrix @ transform - native_matrix)
        residual /= max(np.linalg.norm(native_matrix), np.finfo(float).eps)
        if best_residual is None or residual < best_residual:
            best_transform = transform
            best_residual = residual
        if residual <= 1e-12:
            break

    if best_transform is None:
        raise RuntimeError(
            "Failed to compute a native symmetric-contraction basis solve. "
            f"Best observed rank was {best_rank} for basis dimension {basis_dim}."
        )

    transform = best_transform
    transform = transform.astype(native_weight.dtype, copy=False)
    converted = np.einsum("ab,zbu->zau", transform, native_weight, optimize=True)
    return converted.astype(template.dtype, copy=False)


__all__ = [
    "assign_native_basis",
    "convert_native_symmetric_weights",
    "gather_native_reduced_weights",
    "native_design_matrix",
    "native_symmetric_metadata",
    "torch_target_design_matrix",
]
