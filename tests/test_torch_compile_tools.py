from __future__ import annotations

from pathlib import Path

import torch

from mace_model.build import build_initial_model
from mace_model.config import load_build_request
from mace_model.torch import (
    compile_model,
    export_model,
    graph_to_inference_args,
    make_inference_wrapper,
)


TORCH_CONFIG = """
backend = "torch"
model_class = "ScaleShiftMACE"
seed = 0
output = "{output}"

[model]
r_max = 4.5
num_bessel = 4
num_polynomial_cutoff = 3
max_ell = 1
interaction_cls = "RealAgnosticInteractionBlock"
interaction_cls_first = "RealAgnosticInteractionBlock"
num_interactions = 2
hidden_irreps = "16x0e + 16x1o"
MLP_irreps = "8x0e"
atomic_numbers = [11, 17]
atomic_energies = [-1.25, -2.0]
avg_num_neighbors = 6.0
correlation = 2
gate = "silu"
pair_repulsion = false
apply_cutoff = true
use_reduced_cg = true
use_so3 = false
use_agnostic_product = false
use_last_readout_only = false
use_embedding_readout = false
distance_transform = "None"
radial_type = "bessel"
atomic_inter_scale = 1.0
atomic_inter_shift = 0.0
"""


def _make_tiny_torch_model(tmp_path: Path):
    config_path = tmp_path / "compile-tool.toml"
    config_path.write_text(TORCH_CONFIG.format(output=tmp_path / "unused"))
    request = load_build_request(config_path)
    result = build_initial_model(request)
    return result.model.eval()


def _example_graph() -> dict[str, torch.Tensor]:
    return {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "node_attrs": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.int64,
        ),
        "shifts": torch.zeros((2, 3), dtype=torch.float32),
        "unit_shifts": torch.zeros((2, 3), dtype=torch.float32),
        "cell": torch.zeros((1, 3, 3), dtype=torch.float32),
        "batch": torch.tensor([0, 0], dtype=torch.int64),
        "ptr": torch.tensor([0, 2], dtype=torch.int64),
    }


def test_compile_model_matches_eager_wrapper(tmp_path: Path):
    model = _make_tiny_torch_model(tmp_path)
    graph = _example_graph()
    args = graph_to_inference_args(graph)

    eager_wrapper = make_inference_wrapper(
        model,
        output_keys=("energy", "node_energy"),
    )
    eager_out = eager_wrapper(*args)

    compiled_wrapper = compile_model(
        model,
        output_keys=("energy", "node_energy"),
        backend="eager",
    )
    compiled_out = compiled_wrapper(*args)

    assert len(eager_out) == len(compiled_out) == 2
    for eager_value, compiled_value in zip(eager_out, compiled_out):
        assert torch.allclose(eager_value, compiled_value)


def test_export_model_matches_eager_wrapper(tmp_path: Path):
    if not hasattr(torch, "export"):
        import pytest

        pytest.skip("torch.export is not available in this PyTorch build.")

    model = _make_tiny_torch_model(tmp_path)
    graph = _example_graph()
    args = graph_to_inference_args(graph)

    eager_wrapper = make_inference_wrapper(
        model,
        output_keys=("energy", "node_energy"),
    )
    eager_out = eager_wrapper(*args)

    exported = export_model(
        model,
        graph,
        output_keys=("energy", "node_energy"),
        strict=False,
    )
    exported_out = exported.module()(*args)

    assert len(eager_out) == len(exported_out) == 2
    for eager_value, exported_value in zip(eager_out, exported_out):
        assert torch.allclose(eager_value, exported_value)
