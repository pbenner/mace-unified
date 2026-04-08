from __future__ import annotations

from typing import Callable

import torch
from e3nn import o3

from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig

from mace_core.modules.backends import use_backend
from mace_core.modules.blocks import NonLinearReadoutBlock as CoreNonLinearReadoutBlock

from .backends import TORCH_BACKEND


@use_backend(TORCH_BACKEND)
class NonLinearReadoutBlock(CoreNonLinearReadoutBlock, torch.nn.Module):
    """
    PyTorch unified readout block.

    Signature mirrors mace.modules.blocks.NonLinearReadoutBlock.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable | None,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,  # parity with mace API
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config

        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreNonLinearReadoutBlock.forward
