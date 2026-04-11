from __future__ import annotations

import importlib
import sys

from mace_model.legacy_checkpoint import (
    get_mace_mp_names,
    legacy_checkpoint_imports,
    resolve_foundation_checkpoint,
)


def test_legacy_checkpoint_imports_are_scoped(tmp_path):
    marker = object()
    original = sys.modules.get("e3nn.o3._linear", marker)
    original_local = sys.modules.get("mace_torch.modules.models", marker)

    with legacy_checkpoint_imports():
        linear_mod = importlib.import_module("e3nn.o3._linear")
        models_mod = importlib.import_module("mace.modules.models")
        local_models_mod = importlib.import_module("mace_torch.modules.models")
        assert hasattr(linear_mod, "Linear")
        assert hasattr(models_mod, "ScaleShiftMACE")
        assert hasattr(local_models_mod, "ScaleShiftMACE")

    restored = sys.modules.get("e3nn.o3._linear", marker)
    restored_local = sys.modules.get("mace_torch.modules.models", marker)
    assert restored is original
    assert restored_local is original_local


def test_resolve_foundation_checkpoint_accepts_existing_local_path(tmp_path):
    model_path = tmp_path / "local.model"
    model_path.write_bytes(b"placeholder")

    for source in ("mp", "off", "omol", "anicc"):
        resolved = resolve_foundation_checkpoint(source=source, model=str(model_path))
        assert resolved == model_path.resolve()


def test_get_mace_mp_names_exposes_default_and_named_variants():
    names = get_mace_mp_names()
    assert names[0] is None
    assert "medium-mpa-0" in names
