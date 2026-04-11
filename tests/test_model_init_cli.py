from __future__ import annotations

import json
from pathlib import Path

import torch

from mace_model.build import build_initial_model, save_initialized_model
from mace_model.cli import main
from mace_model.config import load_build_request


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


def _require_cue_jax():
    try:
        import cuequivariance_jax  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip(f"cuequivariance_jax is unavailable in this environment: {exc}")


JAX_CONFIG = """
backend = "jax"
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
replace_symmetric_contraction = false
replacement_depth = 2
replacement_use_species_conditioning = true
attn_num_heads = 4
attn_head_dim = 16
attn_gate_mode = "scalar"
collapse_hidden_irreps = true
"""


def test_build_and_save_torch_model_from_config(tmp_path: Path):
    output_dir = tmp_path / "torch-init"
    config_path = tmp_path / "torch.toml"
    config_path.write_text(TORCH_CONFIG.format(output=output_dir))

    request = load_build_request(config_path)
    result = build_initial_model(request)
    written = save_initialized_model(result, output_dir)

    assert (output_dir / "config.json") in written
    assert (output_dir / "state_dict.pt") in written

    config_saved = json.loads((output_dir / "config.json").read_text())
    assert config_saved["model_class"] == "ScaleShiftMACE"

    state_dict = torch.load(output_dir / "state_dict.pt", map_location="cpu")
    assert any(
        key.startswith("node_embedding.linear") and key.endswith(".weight")
        for key in state_dict
    )


def test_build_and_save_jax_model_from_config(tmp_path: Path):
    _require_cue_jax()
    from mace_model.jax.tools.bundle import load_model_bundle

    output_dir = tmp_path / "jax-init"
    config_path = tmp_path / "jax.toml"
    config_path.write_text(JAX_CONFIG.format(output=output_dir))

    request = load_build_request(config_path)
    result = build_initial_model(request)
    written = save_initialized_model(result, output_dir)

    assert (output_dir / "config.json") in written
    assert (output_dir / "params.msgpack") in written

    bundle = load_model_bundle(str(output_dir), dtype="float32")
    assert bundle.config["torch_model_class"] == "ScaleShiftMACE"


def test_cli_main_builds_torch_bundle(tmp_path: Path):
    output_dir = tmp_path / "cli-torch"
    config_path = tmp_path / "cli-torch.toml"
    config_path.write_text(TORCH_CONFIG.format(output=output_dir))

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "config.json").exists()
    assert (output_dir / "state_dict.pt").exists()


def test_cli_prints_example_config(capsys):
    exit_code = main(["--print-example-config"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'backend = "torch"' in captured.out
