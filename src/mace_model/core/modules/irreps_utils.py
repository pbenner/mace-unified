"""Small backend-agnostic helpers for working with irreps layouts.

These functions intentionally stay free of framework types and implement the
shared bookkeeping needed by both the Torch and JAX adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable


def _unpack_mul_ir(mul_ir: Any) -> tuple[Any, Any]:
    """
    Return (multiplicity, irrep) for both e3nn (tuple-like) and e3nn_jax (MulIrrep).
    """

    if hasattr(mul_ir, "mul") and hasattr(mul_ir, "ir"):
        return mul_ir.mul, mul_ir.ir
    mul, ir = mul_ir
    return mul, ir


def _make_irreps_terms(
    make_irreps: Callable[[Any], Any], irreps: Any
) -> list[tuple[Any, Any]]:
    """Normalize an irreps object into a list of `(multiplicity, irrep)` pairs."""
    return [_unpack_mul_ir(mul_ir) for mul_ir in make_irreps(irreps)]


def _make_zero_irrep(make_irreps: Callable[[Any], Any]) -> Any:
    """Return the scalar even irrep used as a gate carrier."""
    _, zero_irrep = _unpack_mul_ir(next(iter(make_irreps("0e"))))
    return zero_irrep


def _build_gated_irreps(
    *,
    make_irreps: Callable[[Any], Any],
    hidden_irreps: Any,
    irreps_out: Any,
) -> tuple[Any, Any, Any]:
    """Split hidden irreps into scalar, gate, and gated parts for gated MLPs."""
    hidden_terms = _make_irreps_terms(make_irreps, hidden_irreps)

    irreps_scalars = make_irreps(
        [(mul, ir) for mul, ir in hidden_terms if ir.l == 0 and ir in irreps_out]
    )
    irreps_gated = make_irreps(
        [(mul, ir) for mul, ir in hidden_terms if ir.l > 0 and ir in irreps_out]
    )
    zero_irrep = _make_zero_irrep(make_irreps)
    irreps_gates = make_irreps(
        [(mul, zero_irrep) for mul, _ in _make_irreps_terms(make_irreps, irreps_gated)]
    )
    return irreps_scalars, irreps_gates, irreps_gated


def _tp_out_irreps_with_instructions(
    *,
    make_irreps: Callable[[Any], Any],
    irreps1: Any,
    irreps2: Any,
    target_irreps: Any,
) -> tuple[Any, list[tuple[int, int, int, str, bool]]]:
    """Enumerate tensor-product outputs and the matching instruction table."""
    irreps1_value = make_irreps(irreps1)
    irreps2_value = make_irreps(irreps2)
    target_irreps_value = make_irreps(target_irreps)
    trainable = True
    irreps_out_list: list[tuple[int, Any]] = []
    instructions: list[tuple[int, int, int, str, bool]] = []

    for i, mul_ir_in in enumerate(irreps1_value):
        mul, ir_in = _unpack_mul_ir(mul_ir_in)
        for j, mul_ir_edge in enumerate(irreps2_value):
            _, ir_edge = _unpack_mul_ir(mul_ir_edge)
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps_value:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    irreps_out = make_irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda item: item[2])
    return irreps_out, instructions


def _init_reshape_irreps_state(
    *,
    make_irreps: Callable[[Any], Any],
    irreps: Any,
) -> tuple[Any, list[int], list[int], int]:
    """Precompute shape metadata used by layout conversion helpers."""
    irreps_value = make_irreps(irreps)
    dims: list[int] = []
    muls: list[int] = []
    for mul_ir in irreps_value:
        mul, ir = _unpack_mul_ir(mul_ir)
        dims.append(ir.dim)
        muls.append(mul)
    total_dim = sum(mul * dim for mul, dim in zip(muls, dims))
    return irreps_value, dims, muls, total_dim


def _validate_flat_irreps_input(array: Any, *, total_dim: int) -> None:
    """Check that a flattened irrep tensor has the expected second dimension."""
    if array.ndim < 2:
        raise ValueError(
            f"Expected tensor with at least 2 dimensions, got shape {array.shape}"
        )
    if array.shape[1] != total_dim:
        raise ValueError(
            f"Last dimension mismatch: expected {total_dim}, got {array.shape[1]}"
        )


def _reshape_irreps_tensor(
    *,
    array: Any,
    muls: Sequence[int],
    dims: Sequence[int],
    layout_str: str,
    concat_fields: Callable[[Sequence[Any], int], Any],
) -> Any:
    """Reshape a flattened irrep payload into `mul_ir` or `ir_mul` layout."""
    ix = 0
    batch = array.shape[0]
    fields: list[Any] = []
    for mul, dim in zip(muls, dims):
        field = array[:, ix : ix + mul * dim]
        ix += mul * dim
        if layout_str == "ir_mul":
            field = field.reshape(batch, dim, mul)
        else:
            field = field.reshape(batch, mul, dim)
        fields.append(field)
    cat_axis = -2 if layout_str == "ir_mul" else -1
    return concat_fields(fields, cat_axis)


class _CachedIrrepsReshaper:
    """Cache irreps layout metadata and apply the corresponding reshape."""

    def __init__(
        self,
        *,
        make_irreps: Callable[[Any], Any],
        irreps: Any,
        cueq_config: object | None = None,
    ) -> None:
        self.irreps, self.dims, self.muls, self.total_dim = _init_reshape_irreps_state(
            make_irreps=make_irreps,
            irreps=irreps,
        )
        self.cueq_config = cueq_config

    @property
    def layout_str(self) -> str:
        return self.cueq_config.layout_str if self.cueq_config is not None else "mul_ir"

    def reshape(
        self,
        array: Any,
        *,
        concat_fields: Callable[[Sequence[Any], int], Any],
        validate_input: bool = False,
    ) -> Any:
        if validate_input:
            _validate_flat_irreps_input(array, total_dim=self.total_dim)
        return _reshape_irreps_tensor(
            array=array,
            muls=self.muls,
            dims=self.dims,
            layout_str=self.layout_str,
            concat_fields=concat_fields,
        )
