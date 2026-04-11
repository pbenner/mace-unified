"""Utilities for slicing arrays according to e3nn-style irreps layouts."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from mace_model.core.modules.e3nn_adapter_utils import (
    build_extract_slices,
    validate_extract_instructions,
)

from ..irreps import Irreps, IrrepsArray


class Extract:
    """Extract contiguous irreps slices following e3nn indexing rules."""

    def __init__(
        self,
        irreps_in,
        irreps_outs: Sequence,
        instructions: Sequence[tuple[int, ...]],
        squeeze_out: bool = False,
    ):
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = [Irreps(ir) for ir in irreps_outs]
        self.instructions = [tuple(ins) for ins in instructions]
        self.squeeze_out = squeeze_out

        validate_extract_instructions(self.irreps_outs, self.instructions)
        self._slices_out = build_extract_slices(self.irreps_in, self.instructions)

    def __call__(self, x: jnp.ndarray | IrrepsArray):
        array = x.array if isinstance(x, IrrepsArray) else x

        if array.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Invalid input shape: expected last dim {self.irreps_in.dim}, "
                f"got {array.shape[-1]}"
            )

        outputs: list[jnp.ndarray] = []
        for slices in self._slices_out:
            if not slices:
                arr = jnp.zeros(array.shape[:-1] + (0,), dtype=array.dtype)
            else:
                parts = [array[..., s] for s in slices]
                arr = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
            outputs.append(arr)

        if self.squeeze_out and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class ExtractIr(Extract):
    """Extract a single irrep from an IrrepsArray."""

    def __init__(self, irreps_in, ir) -> None:
        ir = Irreps(ir)[0].ir if isinstance(ir, str) else ir
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [
            tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)
        ]
        super().__init__(irreps_in, [irreps_out], instructions, squeeze_out=True)
