from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.serialization import add_safe_globals

from mace_model.conversion import convert_torch_model, load_serialized_torch_model

add_safe_globals([slice])


def _require_legacy_mace():
    legacy_repo = Path(__file__).resolve().parents[2] / "mace"
    if not legacy_repo.exists():
        pytest.skip("Legacy mace repository is not available in this workspace.")
    sys.path.insert(0, str(legacy_repo))
    try:
        from e3nn import o3  # noqa: PLC0415
        from mace import data, modules, tools  # noqa: PLC0415
        from mace.tools import torch_geometric  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Legacy mace dependencies are unavailable: {exc}")
    return o3, data, modules, tools, torch_geometric


def _make_legacy_model_and_batch(*, use_reduced_cg: bool = True):
    o3, data, modules, tools, torch_geometric = _require_legacy_mace()
    batch = _make_legacy_water_batch(torch.float64)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = modules.ScaleShiftMACE(
            r_max=5.0,
            num_bessel=4,
            num_polynomial_cutoff=3,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([1.0, 3.0], dtype=float),
            avg_num_neighbors=3.0,
            atomic_numbers=[1, 8],
            correlation=2,
            radial_type="bessel",
            use_reduced_cg=use_reduced_cg,
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        ).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, batch


def _make_legacy_water_batch(
    dtype: torch.dtype,
    *,
    atomic_numbers_table: list[int] | tuple[int, ...] | None = None,
):
    _o3, data, _modules, tools, torch_geometric = _require_legacy_mace()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [
                    [0.0, -2.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            properties={
                "forces": np.zeros((3, 3)),
                "energy": -1.5,
            },
            property_weights={
                "forces": 1.0,
                "energy": 1.0,
            },
            weight=1.0,
        )
        if atomic_numbers_table is None:
            atomic_numbers_table = [1, 8]
        table = tools.AtomicNumberTable(list(atomic_numbers_table))
        atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
        return next(
            iter(
                torch_geometric.dataloader.DataLoader(
                    [atomic_data],
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
            )
        )
    finally:
        torch.set_default_dtype(previous_dtype)


def test_convert_legacy_torch_model_preserves_energy_outputs():
    legacy_model, batch = _make_legacy_model_and_batch()
    local_model = convert_torch_model(legacy_model, backend="torch").model.eval()

    legacy_out = legacy_model(batch.to_dict(), training=False)
    local_out = local_model(batch.to_dict(), training=False)

    assert torch.allclose(
        legacy_out["energy"],
        local_out["energy"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        legacy_out["node_energy"],
        local_out["node_energy"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_load_serialized_legacy_checkpoint_and_convert(tmp_path: Path):
    legacy_model, batch = _make_legacy_model_and_batch()
    checkpoint_path = tmp_path / "legacy-scale-shift-mace.pt"
    torch.save(legacy_model, checkpoint_path)

    loaded_model, _normalized = load_serialized_torch_model(checkpoint_path)
    local_model = convert_torch_model(loaded_model, backend="torch").model.eval()

    legacy_out = legacy_model(batch.to_dict(), training=False)
    local_out = local_model(batch.to_dict(), training=False)

    assert torch.allclose(
        legacy_out["energy"],
        local_out["energy"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_convert_real_foundation_checkpoint_preserves_energy_outputs():
    if os.environ.get("MACE_MODEL_RUN_FOUNDATION_PARITY", "0").lower() not in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip(
            "Set MACE_MODEL_RUN_FOUNDATION_PARITY=1 to run the heavy real-foundation parity test."
        )

    checkpoint = Path(
        "/home/pbenner/Env/mace-jax/mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model"
    )
    if not checkpoint.exists():
        pytest.skip("Local foundation checkpoint is not available in this workspace.")

    legacy_model, _normalized = load_serialized_torch_model(checkpoint)
    legacy_model = legacy_model.eval()
    try:
        batch_dtype = next(legacy_model.parameters()).dtype
    except StopIteration:
        batch_dtype = torch.get_default_dtype()
    batch = _make_legacy_water_batch(
        batch_dtype,
        atomic_numbers_table=getattr(legacy_model, "atomic_numbers", None),
    )
    local_model = convert_torch_model(legacy_model, backend="torch").model.eval()

    legacy_out = legacy_model(batch.to_dict(), training=False)
    local_out = local_model(batch.to_dict(), training=False)

    assert torch.allclose(
        legacy_out["energy"],
        local_out["energy"],
        atol=2e-2,
        rtol=2e-2,
    )
