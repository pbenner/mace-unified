from __future__ import annotations

from collections.abc import Callable

import torch
from mace_model.core.modules.e3nn_adapter_utils import (
    activation_key as _activation_key,
    estimate_silu_normalize2mom_const,
    normalize2mom_identifier,
)

_CONST_OVERRIDES: dict[str, float] = {}


def register_normalize2mom_const(identifier: str | Callable, value: float) -> None:
    key = normalize2mom_identifier(identifier)
    if key is not None:
        _CONST_OVERRIDES[key] = float(value)


def estimate_normalize2mom_const(
    identifier: str | Callable,
    *,
    seed: int = 0,
    samples: int = 1_000_000,
) -> float:
    return estimate_silu_normalize2mom_const(
        identifier,
        seed=seed,
        samples=samples,
    )


def moment(f: Callable, n: int, dtype=None, device=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = "cpu"
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64).to(
        dtype=dtype,
        device=device,
    )
    return f(z).pow(n).mean()


class normalize2mom(torch.nn.Module):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None):
        super().__init__()
        key = _activation_key(f)
        override = _CONST_OVERRIDES.get(key, None) if key is not None else None
        with torch.no_grad():
            cst = (
                float(override)
                if override is not None
                else moment(f, 2, dtype=torch.float64, device="cpu").pow(-0.5).item()
            )
        self._is_id = abs(cst - 1.0) < 1e-4
        self.f = f
        self.cst = cst
        if key is not None:
            self._normalize2mom_key = key

    def forward(self, x):
        if self._is_id:
            return self.f(x)
        return self.f(x).mul(self.cst)
