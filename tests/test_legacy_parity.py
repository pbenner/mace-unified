from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import molecule
from mace_model.conversion import convert_torch_model, load_serialized_torch_model
from mace_model.foundation import download_foundation_model
from torch.serialization import add_safe_globals

add_safe_globals([slice])


def _require_legacy_mace():
    legacy_repo = Path(__file__).resolve().parents[2] / "mace"
    if not legacy_repo.exists():
        pytest.skip("Legacy mace repository is not available in this workspace.")
    sys.path.insert(0, str(legacy_repo))
    try:
        from mace.tools import torch_geometric  # noqa: PLC0415

        from e3nn import o3  # noqa: PLC0415
        from mace import data, modules, tools  # noqa: PLC0415
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


def _legacy_water_configurations(data_module):
    return [
        data_module.Configuration(
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
        ),
        data_module.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array(
                [
                    [0.1, -2.2, 0.1],
                    [1.1, 0.2, -0.1],
                    [-0.2, 0.9, 0.3],
                ]
            ),
            properties={
                "forces": np.zeros((3, 3)),
                "energy": -1.25,
            },
            property_weights={
                "forces": 1.0,
                "energy": 1.0,
            },
            weight=1.0,
        ),
    ]


def _make_legacy_batch_from_config(
    config,
    dtype: torch.dtype,
    *,
    atomic_numbers_table: list[int] | tuple[int, ...] | None = None,
):
    _o3, data, _modules, tools, torch_geometric = _require_legacy_mace()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        if atomic_numbers_table is None:
            atomic_numbers_table = [1, 8]
        table = tools.AtomicNumberTable(list(atomic_numbers_table))
        atomic_data = data.AtomicData.from_config(
            config,
            z_table=table,
            cutoff=3.0,
        )
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


def _make_legacy_water_batch(
    dtype: torch.dtype,
    *,
    atomic_numbers_table: list[int] | tuple[int, ...] | None = None,
):
    _o3, data, _modules, _tools, _torch_geometric = _require_legacy_mace()
    config = _legacy_water_configurations(data)[0]
    return _make_legacy_batch_from_config(
        config,
        dtype,
        atomic_numbers_table=atomic_numbers_table,
    )


def _make_legacy_water_batches(
    dtype: torch.dtype,
    *,
    atomic_numbers_table: list[int] | tuple[int, ...] | None = None,
):
    _o3, data, _modules, _tools, _torch_geometric = _require_legacy_mace()
    return [
        _make_legacy_batch_from_config(
            config,
            dtype,
            atomic_numbers_table=atomic_numbers_table,
        )
        for config in _legacy_water_configurations(data)
    ]


def _legacy_off_test_configurations(data_module):
    configs = []
    for i, name in enumerate(("H2O", "CH4", "NH3", "CH3OH")):
        atoms = molecule(name)
        positions = np.asarray(atoms.positions, dtype=float)
        if i:
            positions = positions + 0.02 * i
        configs.append(
            {
                "config": data_module.Configuration(
                    atomic_numbers=np.asarray(atoms.numbers, dtype=int),
                    positions=positions,
                    properties={
                        "forces": np.zeros_like(positions),
                        "energy": float(-(i + 1)),
                    },
                    property_weights={
                        "forces": 1.0,
                        "energy": 1.0,
                    },
                    weight=1.0,
                ),
                "compare_stress": False,
            }
        )

    periodic_atoms = molecule("CH3OH")
    periodic_positions = np.asarray(periodic_atoms.positions, dtype=float)
    periodic_positions[:, 0] += 0.15
    periodic_positions[:, 1] -= 0.08
    periodic_cell = np.array(
        [
            [8.0, 0.2, 0.0],
            [0.0, 7.6, 0.1],
            [0.0, 0.0, 8.4],
        ],
        dtype=float,
    )
    configs.append(
        {
            "config": data_module.Configuration(
                atomic_numbers=np.asarray(periodic_atoms.numbers, dtype=int),
                positions=periodic_positions,
                properties={
                    "forces": np.zeros_like(periodic_positions),
                    "energy": -5.0,
                },
                property_weights={
                    "forces": 1.0,
                    "energy": 1.0,
                },
                cell=periodic_cell,
                pbc=(True, True, True),
                weight=1.0,
            ),
            "compare_stress": True,
        }
    )
    return configs


def _make_legacy_off_batches(
    dtype: torch.dtype,
    *,
    atomic_numbers_table: list[int] | tuple[int, ...] | None = None,
):
    _o3, data, _modules, _tools, _torch_geometric = _require_legacy_mace()
    return [
        {
            "batch": _make_legacy_batch_from_config(
                item["config"],
                dtype,
                atomic_numbers_table=atomic_numbers_table,
            ),
            "compare_stress": bool(item["compare_stress"]),
        }
        for item in _legacy_off_test_configurations(data)
    ]


def _assert_outputs_close(
    legacy_out,
    local_out,
    *,
    energy_tol: float,
    force_tol: float,
    stress_tol: float,
):
    for key in ("energy", "node_energy", "interaction_energy"):
        assert torch.allclose(
            legacy_out[key],
            local_out[key],
            atol=energy_tol,
            rtol=energy_tol,
        ), key
    for key in ("forces",):
        assert torch.allclose(
            legacy_out[key],
            local_out[key],
            atol=force_tol,
            rtol=force_tol,
        ), key
    for key in ("stress",):
        assert torch.allclose(
            legacy_out[key],
            local_out[key],
            atol=stress_tol,
            rtol=stress_tol,
        ), key


def test_convert_legacy_torch_model_preserves_energy_outputs():
    legacy_model, batch = _make_legacy_model_and_batch()
    local_model = convert_torch_model(legacy_model, backend="torch").model.eval()
    batches = [batch, *_make_legacy_water_batches(torch.float64)[1:]]

    for item in batches:
        legacy_out = legacy_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        local_out = local_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        _assert_outputs_close(
            legacy_out,
            local_out,
            energy_tol=1e-8,
            force_tol=1e-8,
            stress_tol=1e-10,
        )


def test_convert_legacy_torch_model_full_cg_preserves_energy_outputs():
    legacy_model, batch = _make_legacy_model_and_batch(use_reduced_cg=False)
    local_model = convert_torch_model(legacy_model, backend="torch").model.eval()
    batches = [batch, *_make_legacy_water_batches(torch.float64)[1:]]

    for item in batches:
        legacy_out = legacy_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        local_out = local_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        _assert_outputs_close(
            legacy_out,
            local_out,
            energy_tol=1e-6,
            force_tol=1e-6,
            stress_tol=1e-8,
        )


def test_load_serialized_legacy_checkpoint_and_convert(tmp_path: Path):
    legacy_model, batch = _make_legacy_model_and_batch()
    checkpoint_path = tmp_path / "legacy-scale-shift-mace.pt"
    torch.save(legacy_model, checkpoint_path)

    loaded_model, _normalized = load_serialized_torch_model(checkpoint_path)
    local_model = convert_torch_model(loaded_model, backend="torch").model.eval()
    batches = [batch, *_make_legacy_water_batches(torch.float64)[1:]]

    for item in batches:
        legacy_out = legacy_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        local_out = local_model(
            item.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        _assert_outputs_close(
            legacy_out,
            local_out,
            energy_tol=1e-8,
            force_tol=1e-8,
            stress_tol=1e-10,
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
    batches = _make_legacy_water_batches(
        batch_dtype,
        atomic_numbers_table=getattr(legacy_model, "atomic_numbers", None),
    )
    local_model = convert_torch_model(legacy_model, backend="torch").model.eval()

    for batch in batches:
        legacy_out = legacy_model(
            batch.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        local_out = local_model(
            batch.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=True,
        )
        _assert_outputs_close(
            legacy_out,
            local_out,
            energy_tol=1e-5,
            force_tol=1e-5,
            stress_tol=1e-8,
        )


def test_off_foundation_model_energy_force_parity():
    if os.environ.get("MACE_MODEL_RUN_OFF_PARITY", "0").lower() not in {
        "1",
        "true",
        "yes",
    }:
        pytest.skip(
            "Set MACE_MODEL_RUN_OFF_PARITY=1 to run the heavy OFF foundation parity test."
        )

    _o3, _data, _modules, _tools, _torch_geometric = _require_legacy_mace()
    from mace.calculators import mace_off as legacy_mace_off  # noqa: PLC0415

    try:
        legacy_model = legacy_mace_off(
            model="medium",
            device="cpu",
            default_dtype="float64",
            return_raw_model=True,
        ).eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Unable to load legacy MACE-OFF model: {exc}")

    try:
        local_model = download_foundation_model(
            backend="torch",
            source="off",
            model="medium",
            device="cpu",
            default_dtype="float64",
        ).model.eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Unable to load local MACE-OFF model: {exc}")

    try:
        batch_dtype = next(legacy_model.parameters()).dtype
    except StopIteration:
        batch_dtype = torch.get_default_dtype()

    batches = _make_legacy_off_batches(
        batch_dtype,
        atomic_numbers_table=getattr(legacy_model, "atomic_numbers", None),
    )

    for item in batches:
        batch = item["batch"]
        compare_stress = bool(item["compare_stress"])
        legacy_out = legacy_model(
            batch.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=compare_stress,
        )
        local_out = local_model(
            batch.to_dict(),
            training=False,
            compute_force=True,
            compute_stress=compare_stress,
        )

        assert torch.allclose(
            legacy_out["energy"],
            local_out["energy"],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            legacy_out["forces"],
            local_out["forces"],
            atol=1e-5,
            rtol=1e-5,
        )
        if compare_stress:
            assert "stress" in legacy_out
            assert "stress" in local_out
            assert torch.allclose(
                legacy_out["stress"],
                local_out["stress"],
                atol=1e-5,
                rtol=1e-5,
            )
