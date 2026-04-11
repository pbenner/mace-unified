"""Torch-side e3nn-compatible irreps aliases backed by cue irreps objects."""

from __future__ import annotations

from mace_model.core.modules.e3nn_adapter_utils import (
    make_irrep as Irrep,
    make_irreps as Irreps,
    spherical_harmonics_irreps,
)


_IRREPS_TYPE = type(Irreps("0e"))

Irreps.spherical_harmonics = staticmethod(spherical_harmonics_irreps)  # type: ignore[attr-defined]
if not hasattr(_IRREPS_TYPE, "lmax"):
    _IRREPS_TYPE.lmax = property(  # type: ignore[attr-defined]
        lambda self: max((ir.l for _, ir in self), default=0)
    )

__all__ = ["Irrep", "Irreps"]
