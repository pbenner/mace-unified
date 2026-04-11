"""Public API for mace-model."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BuildRequest",
    "BuildResult",
    "TorchConversionResult",
    "FoundationResult",
    "core",
    "jax",
    "torch",
    "build_initial_model",
    "convert_torch_model",
    "download_foundation_model",
    "get_mace_mp_names",
    "load_serialized_torch_model",
    "save_initialized_model",
    "save_converted_model",
    "save_foundation_model",
    "load_foundation_torch_model",
    "load_config",
    "load_build_request",
]

_EXPORTS = {
    "BuildRequest": ("config", "BuildRequest"),
    "BuildResult": ("build", "BuildResult"),
    "TorchConversionResult": ("conversion", "TorchConversionResult"),
    "FoundationResult": ("foundation", "FoundationResult"),
    "build_initial_model": ("build", "build_initial_model"),
    "convert_torch_model": ("conversion", "convert_torch_model"),
    "download_foundation_model": ("foundation", "download_foundation_model"),
    "get_mace_mp_names": ("foundation", "get_mace_mp_names"),
    "load_serialized_torch_model": ("conversion", "load_serialized_torch_model"),
    "save_initialized_model": ("build", "save_initialized_model"),
    "save_converted_model": ("conversion", "save_converted_model"),
    "save_foundation_model": ("foundation", "save_foundation_model"),
    "load_foundation_torch_model": ("foundation", "load_foundation_torch_model"),
    "load_config": ("config", "load_config"),
    "load_build_request": ("config", "load_build_request"),
}

_SUBMODULES = {"core", "jax", "torch"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - normal Python fallback
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .build import BuildResult, build_initial_model, save_initialized_model
    from .config import BuildRequest, load_build_request, load_config
    from .foundation import (
        FoundationResult,
        download_foundation_model,
        get_mace_mp_names,
        load_foundation_torch_model,
        save_foundation_model,
    )
    from .conversion import (
        TorchConversionResult,
        convert_torch_model,
        load_serialized_torch_model,
        save_converted_model,
    )
