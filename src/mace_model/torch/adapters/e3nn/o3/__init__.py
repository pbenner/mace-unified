from __future__ import annotations

import torch
from mace_model.core.modules.e3nn_adapter_utils import wigner_3j_coefficients

from ..irreps import Irrep, Irreps
from ._linear import Linear
from ._spherical_harmonics import SphericalHarmonics


def wigner_3j(l1: int, l2: int, l3: int, dtype=None):
    torch_dtype = dtype if dtype is not None else torch.get_default_dtype()
    return torch.tensor(wigner_3j_coefficients(l1, l2, l3), dtype=torch_dtype)


__all__ = ["Irrep", "Irreps", "Linear", "SphericalHarmonics", "wigner_3j"]
