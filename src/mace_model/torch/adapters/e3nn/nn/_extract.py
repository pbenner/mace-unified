from __future__ import annotations

from collections.abc import Sequence

from mace_model.core.modules.e3nn_adapter_utils import build_extract_slices
import torch

from .. import o3


class Extract(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_outs: Sequence,
        instructions: Sequence[tuple[int, ...]],
        squeeze_out: bool = False,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = [o3.Irreps(ir) for ir in irreps_outs]
        self.instructions = [tuple(ins) for ins in instructions]
        self.squeeze_out = squeeze_out

        self._slices_out = build_extract_slices(self.irreps_in, self.instructions)

    def forward(self, x: torch.Tensor):
        outputs = []
        for slices in self._slices_out:
            if not slices:
                arr = torch.zeros(*x.shape[:-1], 0, dtype=x.dtype, device=x.device)
            else:
                parts = [x[..., s] for s in slices]
                arr = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            outputs.append(arr)
        if self.squeeze_out and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
