from __future__ import annotations

from pathlib import Path

import torch

from mace_model.build import build_initial_model
from mace_model.config import load_build_request
from mace_model.foundation import download_foundation_model, save_foundation_model
from mace_model.foundation_cli import main as foundation_main

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


def _make_local_torch_model(tmp_path: Path):
    config_path = tmp_path / "foundation-local.toml"
    config_path.write_text(TORCH_CONFIG.format(output=tmp_path / "unused"))
    request = load_build_request(config_path)
    result = build_initial_model(request)
    return result


def _write_local_foundation_checkpoint(tmp_path: Path) -> tuple[torch.nn.Module, Path]:
    result = _make_local_torch_model(tmp_path)
    model = result.model
    checkpoint_path = tmp_path / "foundation-local.pt"
    torch.save(
        {
            "model_config": result.normalized_model_config,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return model, checkpoint_path


def _require_cue_jax():
    try:
        import cuequivariance_jax  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip(f"cuequivariance_jax is unavailable in this environment: {exc}")


def test_download_and_save_jax_foundation_bundle(tmp_path: Path):
    _require_cue_jax()
    from mace_model.jax.tools.bundle import load_model_bundle

    _model, checkpoint_path = _write_local_foundation_checkpoint(tmp_path)

    result = download_foundation_model(
        backend="jax",
        source="mp",
        model=str(checkpoint_path),
    )
    written = save_foundation_model(result, tmp_path / "jax-foundation")

    assert (tmp_path / "jax-foundation" / "config.json") in written
    assert (tmp_path / "jax-foundation" / "params.msgpack") in written

    bundle = load_model_bundle(str(tmp_path / "jax-foundation"), dtype="float32")
    assert bundle.config["torch_model_class"] == "ScaleShiftMACE"
    assert bundle.config["atomic_numbers"] == [11, 17]


def test_foundation_cli_exports_local_torch_model(tmp_path: Path):
    original_model, checkpoint_path = _write_local_foundation_checkpoint(tmp_path)

    output_dir = tmp_path / "torch-foundation"
    exit_code = foundation_main(
        [
            "--backend",
            "torch",
            "--source",
            "mp",
            "--model",
            str(checkpoint_path),
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    state_dict = torch.load(output_dir / "state_dict.pt", map_location="cpu")
    reference_state = original_model.state_dict()
    assert state_dict.keys() == reference_state.keys()
    for key in state_dict:
        assert torch.equal(state_dict[key], reference_state[key]), key
