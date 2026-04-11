"""JAX-side e3nn-compatible irreps aliases plus ``IrrepsArray`` glue."""

from __future__ import annotations

from typing import Any

import cuequivariance as cue
import jax.numpy as jnp
from mace_model.core.modules.e3nn_adapter_utils import (
    make_irrep as Irrep,
    make_irreps as Irreps,
    spherical_harmonics_irreps,
)


_LAYOUTS = {
    "mul_ir": cue.mul_ir,
    "ir_mul": cue.ir_mul,
}
_IRREPS_TYPE = type(Irreps("0e"))


def layout_value(layout_str: str):
    try:
        return _LAYOUTS[layout_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported layout_str {layout_str!r}.") from exc


class IrrepsArray:
    def __init__(self, irreps: Any, array: Any, layout_str: str = "mul_ir") -> None:
        self.irreps = Irreps(irreps)
        self.array = jnp.asarray(array)
        self.layout_str = layout_str

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __array__(self):  # pragma: no cover - numpy interop
        return jnp.asarray(self.array)


def as_rep_array(
    value: jnp.ndarray | IrrepsArray,
    *,
    irreps: Any | None = None,
    layout_str: str = "mul_ir",
):
    import cuequivariance_jax as cuex

    if isinstance(value, IrrepsArray):
        irreps = value.irreps
        layout_str = value.layout_str
        value = value.array
    if irreps is None:
        raise ValueError("irreps is required when converting a raw array to RepArray.")
    return cuex.RepArray(Irreps(irreps), jnp.asarray(value), layout_value(layout_str))


__all__ = ["Irrep", "Irreps", "IrrepsArray", "as_rep_array", "layout_value"]

Irreps.spherical_harmonics = staticmethod(spherical_harmonics_irreps)  # type: ignore[attr-defined]
if not hasattr(_IRREPS_TYPE, "lmax"):
    _IRREPS_TYPE.lmax = property(  # type: ignore[attr-defined]
        lambda self: max((ir.l for _, ir in self), default=0)
    )
