from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from mace_model.jax.tools.bundle import load_model_bundle
from mace_model.build import build_initial_model, save_initialized_model
from mace_model.config import load_build_request
from mace_model.convert_cli import main as convert_main


TORCH_CONFIG = """
backend = "torch"
model_class = "ScaleShiftMACE"
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


def test_convert_cli_converts_serialized_torch_checkpoint_to_jax_bundle(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "reference.pt"
    output_dir = tmp_path / "converted-jax"
    config_path = tmp_path / "torch.toml"
    config_path.write_text(TORCH_CONFIG.format(output=tmp_path / "unused"))
    request = load_build_request(config_path)
    result = build_initial_model(request)
    torch.save(
        {
            "model_config": result.normalized_model_config,
            "state_dict": result.model.state_dict(),
        },
        checkpoint_path,
    )

    exit_code = convert_main([str(checkpoint_path), "--output", str(output_dir)])

    assert exit_code == 0
    bundle = load_model_bundle(str(output_dir), dtype="float32")
    assert bundle.config["model_class"] == "ScaleShiftMACE"
    assert bundle.config["atomic_numbers"] == [11, 17]


def test_convert_cli_converts_local_torch_bundle_dir_to_jax_bundle(tmp_path: Path):
    torch_dir = tmp_path / "torch-init"
    config_path = tmp_path / "torch.toml"
    config_path.write_text(TORCH_CONFIG.format(output=torch_dir))

    request = load_build_request(config_path)
    result = build_initial_model(request)
    save_initialized_model(result, torch_dir)

    output_dir = tmp_path / "converted-local-jax"
    exit_code = convert_main([str(torch_dir), "--output", str(output_dir)])

    assert exit_code == 0
    bundle = load_model_bundle(str(output_dir), dtype="float32")
    assert bundle.config["model_class"] == "ScaleShiftMACE"
    assert bundle.config["atomic_numbers"] == [11, 17]
