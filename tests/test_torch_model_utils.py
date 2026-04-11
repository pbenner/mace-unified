from __future__ import annotations

import numpy as np
import torch

from mace_model.torch.model_utils import (
    extract_torch_model_config,
    select_local_torch_model_head,
)
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.blocks import RealAgnosticInteractionBlock
from mace_model.torch.modules.models import ScaleShiftMACE


def _make_model(*, heads=None):
    torch.manual_seed(0)
    atomic_energies = (
        np.array([[-1.25, -2.0], [-1.0, -1.5]], dtype=np.float32)
        if heads is not None
        else np.array([-1.25, -2.0], dtype=np.float32)
    )
    atomic_inter_scale = [1.0, 1.1] if heads is not None else 1.0
    atomic_inter_shift = [0.0, 0.1] if heads is not None else 0.0
    return ScaleShiftMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=6.0,
        atomic_numbers=[11, 17],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=False,
        apply_cutoff=True,
        distance_transform="None",
        radial_type="bessel",
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        heads=heads,
    ).eval()


def test_extract_torch_model_config_returns_expected_basic_fields():
    model = _make_model()
    config = extract_torch_model_config(model)

    assert config["torch_model_class"] == "ScaleShiftMACE"
    assert config["num_bessel"] == 4
    assert config["num_interactions"] == 2
    assert config["atomic_numbers"] == [11, 17]
    assert config["heads"] == ["Default"]
    assert config["radial_type"] == "bessel"


def test_select_local_torch_model_head_reduces_to_single_head():
    model = _make_model(heads=["pt_head", "target"])
    selected = select_local_torch_model_head(model, head_to_keep="target")
    config = extract_torch_model_config(selected)

    assert selected.heads == ["target"]
    assert config["heads"] == ["target"]
    assert selected.scale_shift.scale.shape == torch.Size([])
    assert selected.scale_shift.shift.shape == torch.Size([])
    assert tuple(selected.atomic_energies_fn.atomic_energies.shape) == (1, 2)
