from __future__ import annotations

from pathlib import Path

from .legacy_checkpoint import (
    SUPPORTED_FOUNDATION_SOURCES,
    get_mace_mp_names,
    resolve_foundation_checkpoint,
)
from .conversion import (
    TorchConversionResult,
    convert_torch_model,
    load_serialized_torch_model,
    save_converted_model,
)

FoundationResult = TorchConversionResult


def load_foundation_torch_model(
    *,
    source: str,
    model: str | None = None,
    device: str = "cpu",
    default_dtype: str | None = None,
):
    del default_dtype
    checkpoint = resolve_foundation_checkpoint(source=source, model=model)
    torch_model, _normalized = load_serialized_torch_model(checkpoint)
    return torch_model


def download_foundation_model(
    *,
    backend: str,
    source: str,
    model: str | None = None,
    head: str | None = None,
    device: str = "cpu",
    default_dtype: str | None = None,
) -> FoundationResult:
    del default_dtype
    checkpoint = resolve_foundation_checkpoint(source=source, model=model)
    torch_model, normalized = load_serialized_torch_model(checkpoint)
    return convert_torch_model(
        torch_model,
        backend=backend,
        head=head,
        device=device,
        config=normalized,
    )


def _default_output_path(
    *,
    backend: str,
    source: str,
    model: str | None,
) -> Path:
    model_tag = (model or "default").replace("/", "-")
    return Path(f"{source}-{model_tag}-{backend}")


def save_foundation_model(
    result: FoundationResult,
    output: str | Path | None = None,
    *,
    source: str | None = None,
    model: str | None = None,
) -> list[Path]:
    if output is None:
        output = _default_output_path(
            backend=result.backend,
            source=source or "foundation",
            model=model,
        )
    return save_converted_model(result, output)


__all__ = [
    "FoundationResult",
    "SUPPORTED_FOUNDATION_SOURCES",
    "download_foundation_model",
    "get_mace_mp_names",
    "load_foundation_torch_model",
    "save_foundation_model",
]
