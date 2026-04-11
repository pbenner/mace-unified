"""Flax-friendly scalar activation logic with cue-backed irreps."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cuequivariance_jax.activation as cue_activation
import jax
import jax.numpy as jnp
from mace_model.core.modules.e3nn_adapter_utils import (
    build_irreps_block_slices,
    infer_activation_irreps_out,
    validate_layout_str,
)

from ..irreps import Irreps, IrrepsArray
from ..math import normalize2mom, register_normalize2mom_const


class Activation:
    def __init__(
        self,
        irreps_in,
        acts: Sequence[Callable | None],
        *,
        normalize_act: bool = True,
        layout_str: str = "mul_ir",
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.layout_str = validate_layout_str(layout_str)

        def _to_jax_act(act: Callable | None) -> Callable | None:
            if act is None:
                return None
            if hasattr(act, "__name__"):
                name = act.__name__.lower()
                if name in ("silu", "swish"):
                    return jax.nn.silu
            cls_name = getattr(getattr(act, "__class__", None), "__name__", "").lower()
            if cls_name in ("silu", "swish"):
                return jax.nn.silu
            wrapped = getattr(act, "f", None)
            if wrapped is not None and wrapped is not act:
                mapped = _to_jax_act(wrapped)
                if mapped is not None:
                    return mapped
            return act

        def _maybe_get_const(act):
            if act is None:
                return None, None
            const = getattr(act, "_normalize2mom_const", None)
            if const is None:
                const = getattr(act, "cst", None)
            orig = getattr(act, "_normalize2mom_original", None)
            if orig is None:
                orig = getattr(act, "f", None)
            return const, orig

        processed_acts: list[Callable | None] = []
        for act in acts:
            jax_act = _to_jax_act(act)
            const, orig = _maybe_get_const(act)
            if const is not None:
                register_normalize2mom_const(orig or jax_act or act, const)
            use_norm = normalize_act or const is not None
            processed_acts.append(
                normalize2mom(jax_act) if use_norm and jax_act is not None else jax_act
            )

        self._acts = tuple(processed_acts)
        self._block_slices = build_irreps_block_slices(self.irreps_in)

        def _activation_parity(ir, act):
            if ir.p != -1:
                return ir.p
            p_out = cue_activation.function_parity(act)
            if p_out == 0:
                raise ValueError(
                    "Activation parity is violated: odd scalar input requires an even or odd activation."
                )
            return p_out

        self.irreps_out = Irreps(
            infer_activation_irreps_out(self.irreps_in, self._acts, _activation_parity)
        )

    def __call__(self, features: jnp.ndarray | IrrepsArray, axis: int = -1):
        return_irreps = isinstance(features, IrrepsArray)
        array = features.array if return_irreps else features
        moved = False
        if axis != -1:
            array = jnp.moveaxis(array, axis, -1)
            moved = True
        if array.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Invalid input shape: expected last dimension {self.irreps_in.dim}, got {array.shape[-1]}"
            )

        segments = []
        for (_, ir), act, block_slice in zip(
            self.irreps_in, self._acts, self._block_slices
        ):
            segment = array[..., block_slice]
            if act is not None:
                if ir.l != 0:
                    raise ValueError(
                        "Activation can only apply non-linearities to scalar irreps."
                    )
                segment = act(segment)
            segments.append(segment)
        activated = jnp.concatenate(segments, axis=-1) if segments else array[..., :0]
        if moved:
            activated = jnp.moveaxis(activated, -1, axis)
        if return_irreps:
            return IrrepsArray(self.irreps_out, activated, layout_str=self.layout_str)
        return activated
