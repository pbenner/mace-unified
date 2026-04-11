from __future__ import annotations

from mace_model.torch.adapters.e3nn import o3


def Linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    layout: object | None = None,
    group: object | None = None,
):
    del layout, group
    return o3.Linear(
        irreps_in,
        irreps_out,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        biases=False,
    )


__all__ = ["Linear"]
