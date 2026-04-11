from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cuequivariance_torch as cuet
import torch
from mace_model.core.modules.e3nn_adapter_utils import (
    apply_spherical_harmonics_normalization,
    build_spherical_harmonics_plan,
)

from ..irreps import Irreps


class SphericalHarmonics(torch.nn.Module):
    def __init__(
        self,
        irreps_out: int | Sequence[int] | str | Any,
        normalize: bool,
        normalization: str = "integral",
        irreps_in: Any = None,
    ) -> None:
        super().__init__()
        self.irreps_out = irreps_out
        self.normalize = normalize
        self.normalization = normalization
        self.irreps_in = irreps_in

        irreps_out = Irreps(self.irreps_out)
        self._plan = build_spherical_harmonics_plan(
            irreps_out,
            Irreps("1o" if self.irreps_in is None else self.irreps_in),
            self.normalization,
            require_unique_sorted=True,
        )
        self._lmax = self._plan.lmax
        self._is_range_lmax = self._plan.is_range_lmax
        self._sh = cuet.SphericalHarmonics(
            list(self._plan.degrees),
            normalize=normalize,
            method="naive",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_spherical_harmonics_normalization(
            self._sh(x),
            self._plan,
            asarray=lambda values, *, dtype: torch.tensor(
                values,
                dtype=dtype,
                device=x.device,
            ),
        )
