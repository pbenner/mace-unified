"""Shared helpers for the local e3nn-compatible adapter layers.

These utilities cover the pieces that are genuinely backend-independent across
the Torch and JAX adapter implementations. Backend-specific tensor wrappers,
modules, and import machinery stay in the respective adapter packages.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import cuequivariance as cue
import numpy as np


def make_irreps(value: Any):
    """Coerce ``value`` into cue ``O3`` irreps."""
    if isinstance(value, cue.Irreps):
        return value
    return cue.Irreps(cue.O3, value)


def spherical_harmonics_irreps(max_ell: int, p: int = -1):
    """Return the standard spherical-harmonics irreps up to ``max_ell``."""
    return cue.Irreps(cue.O3, [(1, (l, int(p) ** l)) for l in range(int(max_ell) + 1)])


def make_irrep(l: Any, p: int | None = None):
    """Coerce ``l``/``p`` into a single cue ``O3`` irrep."""
    if p is None:
        if isinstance(l, cue.Irrep):
            return l
        if hasattr(l, "l") and hasattr(l, "p"):
            return cue.O3(int(l.l), int(l.p))
        if isinstance(l, (tuple, list)) and len(l) == 2:
            return cue.O3(int(l[0]), int(l[1]))
        parsed = make_irreps(l)
        if len(parsed) != 1 or parsed[0].mul != 1:
            raise ValueError(f"Cannot coerce {l!r} to a single irrep.")
        return parsed[0].ir
    return cue.O3(int(l), int(p))


def wigner_3j_coefficients(l1: int, l2: int, l3: int) -> np.ndarray:
    """Return real Wigner-3j coefficients as a NumPy array."""
    coeffs = cue.O3.clebsch_gordan(
        cue.O3(int(l1), 1),
        cue.O3(int(l2), 1),
        cue.O3(int(l3), 1),
    )[0]
    coeffs = coeffs / np.sqrt(2 * int(l3) + 1)
    return np.asarray(coeffs)


def compile_mode(mode: str):
    """Attach the e3nn compile-mode marker used by the local shims."""

    def decorator(obj):
        setattr(obj, "_e3nn_compile_mode", mode)
        return obj

    return decorator


def activation_key(f: Callable | None) -> str | None:
    """Return a stable backend-agnostic identifier for an activation callable."""
    if f is None:
        return None
    key = getattr(f, "_normalize2mom_key", None)
    if isinstance(key, str):
        return key
    if isinstance(f, partial):
        return activation_key(f.func)
    name = getattr(f, "__name__", None)
    if name is None:
        name = getattr(getattr(f, "__class__", None), "__name__", None)
    if not name:
        return None
    return str(name).replace("<lambda>", "lambda").lower()


def normalize2mom_identifier(identifier: str | Callable | None) -> str | None:
    """Normalise a string/callable activation identifier."""
    if identifier is None:
        return None
    if callable(identifier):
        return activation_key(identifier)
    return str(identifier).lower()


def estimate_silu_normalize2mom_const(
    identifier: str | Callable,
    *,
    seed: int = 0,
    samples: int = 1_000_000,
    dtype: np.dtype = np.float64,
) -> float:
    """Estimate the normalize2mom constant for supported scalar activations."""
    key = normalize2mom_identifier(identifier)
    if key not in {"silu", "swish"}:
        raise ValueError(f"Unsupported normalize2mom identifier: {key}")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=samples).astype(dtype)
    silu = values / (1.0 + np.exp(-values))
    return float(np.mean(silu * silu) ** -0.5)


def validate_extract_instructions(
    irreps_outs: Sequence[Any],
    instructions: Sequence[tuple[int, ...]],
) -> None:
    """Validate that each instruction tuple matches its target irreps length."""
    if len(irreps_outs) != len(instructions):
        raise ValueError(
            "Number of output irreps must match number of instruction sets."
        )
    for ir_out, ins in zip(irreps_outs, instructions):
        if len(ir_out) != len(ins):
            raise ValueError("Instruction length must match irreps length.")


def build_extract_slices(
    irreps_in: Any,
    instructions: Sequence[tuple[int, ...]],
) -> list[tuple[slice, ...]]:
    """Plan contiguous slices for e3nn-style ``Extract`` blocks."""
    dims = [mul_ir.mul * mul_ir.ir.dim for mul_ir in irreps_in]
    offsets = [0]
    for dim in dims:
        offsets.append(offsets[-1] + dim)

    slices_out: list[tuple[slice, ...]] = []
    for ins in instructions:
        slices_out.append(tuple(slice(offsets[idx], offsets[idx + 1]) for idx in ins))
    return slices_out


def validate_layout_str(layout_str: str) -> str:
    """Validate the local layout marker used by the adapter shims."""
    if layout_str not in {"mul_ir", "ir_mul"}:
        raise ValueError(
            f"layout_str must be either 'mul_ir' or 'ir_mul'; got {layout_str!r}."
        )
    return layout_str


def build_irreps_block_slices(irreps: Any) -> tuple[slice, ...]:
    """Return contiguous last-axis slices for each irrep block."""
    return tuple(
        block_slices[0]
        for block_slices in build_extract_slices(
            irreps,
            [(idx,) for idx in range(len(irreps))],
        )
    )


def infer_activation_irreps_out(
    irreps_in: Any,
    acts: Sequence[Callable | None],
    parity_fn: Callable[[Any, Callable], int],
) -> Any:
    """Infer the output irreps after scalar activations are applied."""
    if len(irreps_in) != len(acts):
        raise ValueError(
            "Irreps in and number of activation functions does not match: "
            f"{len(acts)}, ({irreps_in}, {acts})"
        )

    irreps_out = []
    for (mul, ir), act in zip(irreps_in, acts):
        if act is None:
            irreps_out.append((mul, ir))
            continue
        if ir.l != 0:
            raise ValueError(
                "Activation: cannot apply an activation function to a non-scalar input. "
                f"{irreps_in} {acts}"
            )
        irreps_out.append((mul, (0, parity_fn(ir, act))))
    return make_irreps(irreps_out)


@dataclass(frozen=True)
class GateBlockSpec:
    """Size metadata for one gated irrep block."""

    gated_mul: int
    gated_ir_dim: int
    gated_size: int
    gate_size: int


@dataclass(frozen=True)
class GatePlan:
    """Backend-independent structure of an ``e3nn.nn.Gate`` block."""

    irreps_scalars: Any
    irreps_gates: Any
    irreps_gated: Any
    split_instructions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    irreps_in: Any
    gate_blocks: tuple[GateBlockSpec, ...]


def build_gate_plan(
    irreps_scalars: Any,
    irreps_gates: Any,
    irreps_gated: Any,
) -> GatePlan:
    """Validate gate irreps and precompute the shared split/multiply plan."""
    irreps_scalars = irreps_scalars.simplify()
    irreps_gates = irreps_gates.simplify()
    irreps_gated = irreps_gated.simplify()

    if any(ir.l > 0 for _, ir in irreps_scalars):
        raise ValueError(f"Scalars must be l=0 irreps, got {irreps_scalars}")
    if any(ir.l > 0 for _, ir in irreps_gates):
        raise ValueError(f"Gate irreps must be l=0 irreps, got {irreps_gates}")
    if len(irreps_gates) != len(irreps_gated):
        raise ValueError(
            f"Mismatch: {len(irreps_gated)} irreps in gated, "
            f"{len(irreps_gates)} in gates"
        )

    gate_blocks = []
    for gated_irrep, gate_irrep in zip(irreps_gated, irreps_gates):
        gated_mul, gated_ir = gated_irrep
        gate_mul, gate_ir = gate_irrep
        if gate_mul != gated_mul or gate_ir.dim != 1:
            raise ValueError(
                "Gate multiplicities must match gated multiplicities and stay scalar."
            )
        gate_blocks.append(
            GateBlockSpec(
                gated_mul=gated_mul,
                gated_ir_dim=gated_ir.dim,
                gated_size=gated_mul * gated_ir.dim,
                gate_size=gate_mul,
            )
        )

    n_scalars = len(irreps_scalars)
    n_gates = len(irreps_gates)
    n_gated = len(irreps_gated)
    split_instructions = (
        tuple(range(0, n_scalars)),
        tuple(range(n_scalars, n_scalars + n_gates)),
        tuple(range(n_scalars + n_gates, n_scalars + n_gates + n_gated)),
    )

    return GatePlan(
        irreps_scalars=irreps_scalars,
        irreps_gates=irreps_gates,
        irreps_gated=irreps_gated,
        split_instructions=split_instructions,
        irreps_in=irreps_scalars + irreps_gates + irreps_gated,
        gate_blocks=tuple(gate_blocks),
    )


def apply_gate_blocks(
    gated: Any,
    gates: Any,
    gate_blocks: Sequence[GateBlockSpec],
    *,
    layout_str: str,
    concatenate: Callable[[Sequence[Any]], Any],
) -> Any:
    """Apply scalar gates to higher-order irreps along the last axis."""
    validate_layout_str(layout_str)

    leading_shape = gated.shape[:-1]
    gated_offset = 0
    gate_offset = 0
    pieces = []

    for spec in gate_blocks:
        gated_block = gated[..., gated_offset : gated_offset + spec.gated_size]
        gates_block = gates[..., gate_offset : gate_offset + spec.gate_size]
        gated_offset += spec.gated_size
        gate_offset += spec.gate_size

        if layout_str == "ir_mul":
            gated_block = gated_block.reshape(
                *leading_shape, spec.gated_ir_dim, spec.gated_mul
            )
            gates_block = gates_block.reshape(*leading_shape, 1, spec.gate_size)
        else:
            gated_block = gated_block.reshape(
                *leading_shape, spec.gated_mul, spec.gated_ir_dim
            )
            gates_block = gates_block.reshape(*leading_shape, spec.gate_size, 1)
        pieces.append(
            (gated_block * gates_block).reshape(*leading_shape, spec.gated_size)
        )

    return concatenate(pieces) if pieces else gated[..., :0]


@dataclass(frozen=True)
class SphericalHarmonicsPlan:
    """Backend-independent validation and metadata for spherical harmonics."""

    irreps_in: Any
    degrees: tuple[int, ...]
    canonical_irreps_out: Any
    lmax: int
    is_range_lmax: bool
    normalization: str
    norm_factors: tuple[float, ...]


def build_spherical_harmonics_plan(
    irreps_out: Any,
    irreps_in: Any,
    normalization: str,
    *,
    require_unique_sorted: bool = False,
) -> SphericalHarmonicsPlan:
    """Validate spherical-harmonics metadata shared by both backends."""
    if irreps_in not in (make_irreps("1x1o"), make_irreps("1x1e")):
        raise ValueError(
            f"irreps_in must be either 1x1o or 1x1e; received {irreps_in!s}"
        )
    if normalization not in {"integral", "component", "norm"}:
        raise ValueError(
            "normalization must be 'integral', 'component', or 'norm'; "
            f"got {normalization!r}"
        )

    input_parity = irreps_in[0].ir.p
    degrees = []
    for mul, ir in irreps_out:
        if ir.p != input_parity**ir.l:
            raise ValueError(
                "Output parity mismatch in SphericalHarmonics: "
                f"l={ir.l}, p={ir.p}, expected {input_parity**ir.l}"
            )
        degrees.extend([ir.l] * mul)

    if require_unique_sorted and degrees != sorted(set(degrees)):
        raise ValueError(
            "Cue spherical harmonics only supports unique, sorted degrees."
        )

    norm_factors = ()
    if normalization == "norm":
        factors = []
        for l_value in degrees:
            factors.extend([float(np.sqrt(2 * l_value + 1))] * (2 * l_value + 1))
        norm_factors = tuple(factors)

    return SphericalHarmonicsPlan(
        irreps_in=irreps_in,
        degrees=tuple(int(l) for l in degrees),
        canonical_irreps_out=make_irreps(
            [(1, (l_value, input_parity**l_value)) for l_value in degrees]
        ).simplify(),
        lmax=max(degrees) if degrees else 0,
        is_range_lmax=degrees == list(range(max(degrees) + 1)) if degrees else True,
        normalization=normalization,
        norm_factors=norm_factors,
    )


def apply_spherical_harmonics_normalization(
    array: Any,
    plan: SphericalHarmonicsPlan,
    *,
    asarray: Callable[..., Any],
) -> Any:
    """Apply the shared e3nn spherical-harmonics output normalization."""
    if plan.normalization == "integral":
        return array / np.sqrt(4.0 * np.pi)
    if plan.normalization == "norm":
        return array / asarray(plan.norm_factors, dtype=array.dtype)
    return array
