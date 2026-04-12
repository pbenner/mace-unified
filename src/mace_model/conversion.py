"""Torch checkpoint conversion helpers for ``mace_model``.

This module is responsible for the user-facing conversion workflows:

* load serialized local or legacy Torch checkpoints
* normalize extracted model configs
* convert legacy Torch MACE models into the local Torch representation
* convert Torch models into local JAX bundles
* save converted models in backend-specific output layouts

The conversion paths intentionally keep the legacy compatibility details in one
place so the rest of the package can operate purely on local model objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flax import serialization

from mace_model.core.modules.native_symmetric_weights import (
    convert_native_symmetric_weights,
    torch_target_design_matrix,
)
from mace_model.torch.adapters.cuequivariance import CuEquivarianceConfig
from mace_model.build import (
    _jsonable,
    _normalize_model_class,
    _resolve_jax_output,
    _resolve_torch_output,
    _torch_kwargs_from_config,
)
from mace_model.legacy_checkpoint import load_legacy_torch_model
from mace_model.torch.model_utils import (
    extract_torch_model_config as _extract_torch_model_config,
    select_local_torch_model_head,
)


@dataclass(frozen=True)
class TorchConversionResult:
    """Result returned by :func:`convert_torch_model`."""

    backend: str
    model_class: str
    model: Any
    normalized_model_config: dict[str, Any]


def _extract_model_class(torch_model) -> str:
    model_class = torch_model.__class__.__name__
    if model_class not in {"MACE", "ScaleShiftMACE"}:
        raise ValueError(
            f"Unsupported Torch model class {model_class!r}. "
            "Expected 'MACE' or 'ScaleShiftMACE'."
        )
    return model_class


def select_torch_model_head(torch_model, head: str | None = None):
    """Return a single-head local Torch model view.

    Head selection is only supported for local Torch models because the local
    package controls that serialization and module surface directly.
    """
    if head is None:
        return torch_model
    if not _looks_like_local_torch_model(torch_model):
        raise NotImplementedError(
            "Head selection is only supported on local Torch models. "
            "Convert the legacy model to the local Torch format first."
        )
    return select_local_torch_model_head(torch_model, head_to_keep=head)


def extract_torch_model_config(torch_model) -> dict[str, Any]:
    """Extract a normalized Torch model config from a model instance."""
    config = _extract_torch_model_config(torch_model)
    if "error" in config:
        raise RuntimeError(f"Failed to extract Torch model config: {config['error']}")
    config["torch_model_class"] = _extract_model_class(torch_model)
    return config


def _normalize_gate_name(value: Any) -> Any:
    if value is None or not callable(value):
        return value
    name = getattr(value, "__name__", None)
    if not name:
        return str(value)
    normalized = name.lower()
    if normalized in {"silu", "relu", "tanh", "sigmoid", "softplus", "abs"}:
        return normalized
    if normalized == "swish":
        return "silu"
    return normalized


def _cue_config_to_dict(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    attrs = (
        "enabled",
        "layout",
        "group",
        "optimize_all",
        "optimize_linear",
        "optimize_channelwise",
        "optimize_symmetric",
        "optimize_fctp",
        "conv_fusion",
    )
    return {key: getattr(value, key) for key in attrs if hasattr(value, key)}


def normalize_extracted_torch_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw extracted Torch config into a JSON-safe config dict."""
    normalized = dict(config)

    for key in ("interaction_cls", "interaction_cls_first", "readout_cls"):
        value = normalized.get(key)
        if value is not None and hasattr(value, "__name__"):
            normalized[key] = value.__name__

    for key in ("hidden_irreps", "MLP_irreps", "edge_irreps"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = str(value)

    if normalized.get("atomic_numbers") is not None:
        normalized["atomic_numbers"] = [
            int(x) for x in list(normalized["atomic_numbers"])
        ]
    if normalized.get("atomic_energies") is not None:
        normalized["atomic_energies"] = np.asarray(
            normalized["atomic_energies"], dtype=np.float32
        ).tolist()

    for key in (
        "r_max",
        "avg_num_neighbors",
        "atomic_inter_scale",
        "atomic_inter_shift",
    ):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = float(np.asarray(value))

    normalized["cueq_config"] = _cue_config_to_dict(normalized.get("cueq_config"))
    normalized["gate"] = _normalize_gate_name(normalized.get("gate"))
    return normalized


def _map_upstream_to_local_torch_key(key: str) -> str:
    if key.startswith("node_embedding.linear."):
        return key.replace("node_embedding.linear.", "node_embedding.linear.linear.")
    if key.startswith("interactions."):
        if ".linear_up." in key:
            return key.replace(".linear_up.", ".linear_up.linear.")
        if ".linear." in key:
            return key.replace(".linear.", ".linear.linear.")
    if key.startswith("products.") and ".linear." in key:
        return key.replace(".linear.", ".linear.linear.")
    if key.startswith("readouts."):
        if ".linear_1." in key:
            return key.replace(".linear_1.", ".linear_1.linear.")
        if ".linear_2." in key:
            return key.replace(".linear_2.", ".linear_2.linear.")
        if ".linear." in key:
            return key.replace(".linear.", ".linear.linear.")
    if key.startswith("radial_embedding.bessel_fn."):
        return key.replace("radial_embedding.bessel_fn.", "radial_embedding.basis_fn.")
    return key


def _shapes_match_up_to_unsqueeze(a, b) -> bool:
    def drop_ones(shape):
        return tuple(dim for dim in tuple(shape) if dim != 1)

    return drop_ones(a) == drop_ones(b)


def _reshape_like(src: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
    try:
        return src.reshape(ref_shape)
    except RuntimeError:
        return src.clone().reshape(ref_shape)


def _torch_target_basis_kind(target_product) -> str | None:
    use_reduced_cg = getattr(target_product, "use_reduced_cg", None)
    if use_reduced_cg is None:
        return None
    # Local cue-backed Torch modules use the same canonical full-CG ordering as
    # the JAX path. Full-CG legacy imports therefore need the native->canonical
    # change of basis instead of a raw full-basis copy.
    return "reduced" if bool(use_reduced_cg) else "canonical_full"


def _transfer_upstream_symmetric_contractions(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    target_dict: dict[str, torch.Tensor],
) -> None:
    for layer_index, (source_product, target_product) in enumerate(
        zip(source_model.products, target_model.products)
    ):
        key = f"products.{layer_index}.symmetric_contractions.weight"
        target_weight = target_dict[key]
        converted = convert_native_symmetric_weights(
            source_product.symmetric_contractions,
            target_template=target_weight.detach().cpu().numpy(),
            target_design_matrix_fn=lambda basis_dim,
            inputs_np,
            module=target_product.symmetric_contractions: torch_target_design_matrix(
                module,
                basis_dim=basis_dim,
                inputs_np=inputs_np,
            ),
            target_basis_kind=_torch_target_basis_kind(target_product),
        )
        target_dict[key] = torch.tensor(
            converted,
            dtype=target_weight.dtype,
            device=target_weight.device,
        )


def _transfer_upstream_to_cueq_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
) -> None:
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    _transfer_upstream_symmetric_contractions(
        source_model,
        target_model,
        target_dict,
    )

    remaining_keys = set(source_dict.keys()) & set(target_dict.keys())
    remaining_keys = {
        key for key in remaining_keys if "symmetric_contraction" not in key
    }
    for key in remaining_keys:
        src = source_dict[key]
        tgt = target_dict[key]
        if src.shape == tgt.shape:
            target_dict[key] = src
        elif _shapes_match_up_to_unsqueeze(src.shape, tgt.shape):
            target_dict[key] = _reshape_like(src, tgt.shape)

    for layer_index in range(len(target_model.interactions)):
        target_model.interactions[
            layer_index
        ].avg_num_neighbors = source_model.interactions[layer_index].avg_num_neighbors

    target_model.load_state_dict(target_dict)


def _convert_upstream_torch_to_cueq_torch(torch_model, *, device: str = "cpu"):
    config = extract_torch_model_config(torch_model)
    config.pop("torch_model_class", None)
    config["cueq_config"] = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3",
        optimize_all=True,
        conv_fusion=(device == "cuda"),
    )

    # Upstream Torch MACE expects its own irreps class when we reinstantiate the
    # temporary cue-enabled donor model. Reuse the source model's runtime type
    # instead of importing the external e3nn package directly.
    upstream_module = type(torch_model).__module__
    if upstream_module.startswith("mace."):
        source_irreps = getattr(torch_model.products[0].linear, "irreps_out", None)
        source_irreps_type = type(source_irreps) if source_irreps is not None else None
        if source_irreps_type is None:
            raise RuntimeError(
                "Failed to determine the legacy Torch irreps type required to "
                "rebuild the temporary cue-backed donor model."
            )
        for key in ("hidden_irreps", "MLP_irreps", "edge_irreps"):
            value = config.get(key)
            if value is not None and not isinstance(value, source_irreps_type):
                config[key] = source_irreps_type(str(value))

    cueq_model = torch_model.__class__(**config).to(device)
    _transfer_upstream_to_cueq_weights(
        torch_model,
        cueq_model,
    )
    return cueq_model.eval()


def _ensure_local_torch_model(torch_model, *, model_class: str, device: str = "cpu"):
    if _looks_like_local_torch_model(torch_model):
        return torch_model.eval()
    del device
    local_model, _normalized = _convert_native_torch_to_local(torch_model, model_class)
    return local_model.eval()


def _looks_like_local_torch_model(torch_model) -> bool:
    try:
        state_keys = set(torch_model.state_dict().keys())
    except Exception:
        return False
    return "node_embedding.linear.linear.weight" in state_keys


def _infer_floating_dtype_from_state_dict(state_dict: dict[str, torch.Tensor]):
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            return value.dtype
    return torch.get_default_dtype()


def _instantiate_local_torch_model(
    model_class: str,
    kwargs: dict[str, Any],
    *,
    source_dtype: torch.dtype | None = None,
):
    local_model_class = _normalize_model_class("torch", model_class)
    previous_dtype = torch.get_default_dtype()
    if source_dtype in {torch.float32, torch.float64}:
        torch.set_default_dtype(source_dtype)
    try:
        return local_model_class(**kwargs).eval()
    finally:
        torch.set_default_dtype(previous_dtype)


def _convert_to_local_torch_from_cueq(torch_model, model_class: str):
    cueq_config = normalize_extracted_torch_config(
        extract_torch_model_config(torch_model)
    )
    cueq_config.pop("torch_model_class", None)
    kwargs, normalized = _torch_kwargs_from_config(model_class, cueq_config)
    try:
        source_dtype = next(torch_model.parameters()).dtype
    except StopIteration:
        source_dtype = torch.get_default_dtype()
    local_model = _instantiate_local_torch_model(
        model_class,
        kwargs,
        source_dtype=source_dtype,
    )

    source_state = torch_model.state_dict()
    target_state = local_model.state_dict()
    assigned = 0
    for key, value in source_state.items():
        mapped = _map_upstream_to_local_torch_key(key)
        if mapped not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[mapped].shape):
            continue
        target_state[mapped] = value
        assigned += 1

    local_model.load_state_dict(target_state)
    if assigned != len(source_state):
        raise RuntimeError(
            "Failed to transfer all Torch weights to the local Torch model "
            f"({assigned} of {len(source_state)} assigned)."
        )
    normalized["model_class"] = model_class
    return local_model, normalized


def _convert_native_torch_to_local(torch_model, model_class: str):
    native_config = normalize_extracted_torch_config(
        extract_torch_model_config(torch_model)
    )
    native_config.pop("torch_model_class", None)
    kwargs, normalized = _torch_kwargs_from_config(model_class, native_config)
    try:
        source_dtype = next(torch_model.parameters()).dtype
    except StopIteration:
        source_dtype = torch.get_default_dtype()
    local_model = _instantiate_local_torch_model(
        model_class,
        kwargs,
        source_dtype=source_dtype,
    )

    source_state = torch_model.state_dict()
    target_state = local_model.state_dict()
    for key, value in source_state.items():
        mapped = _map_upstream_to_local_torch_key(key)
        if mapped not in target_state or "symmetric_contraction" in key:
            continue
        target_value = target_state[mapped]
        if tuple(value.shape) == tuple(target_value.shape):
            target_state[mapped] = value
        elif _shapes_match_up_to_unsqueeze(value.shape, target_value.shape):
            target_state[mapped] = _reshape_like(value, target_value.shape)

    for layer_index, (source_product, target_product) in enumerate(
        zip(torch_model.products, local_model.products)
    ):
        key = f"products.{layer_index}.symmetric_contractions.weight"
        target_weight = target_state[key]
        converted = convert_native_symmetric_weights(
            source_product.symmetric_contractions,
            target_template=target_weight.detach().cpu().numpy(),
            target_design_matrix_fn=lambda basis_dim,
            inputs_np,
            module=target_product.symmetric_contractions: torch_target_design_matrix(
                module,
                basis_dim=basis_dim,
                inputs_np=inputs_np,
            ),
            target_basis_kind=_torch_target_basis_kind(target_product),
        )
        target_state[key] = torch.tensor(
            converted,
            dtype=target_weight.dtype,
            device=target_weight.device,
        )

    for layer_index in range(len(local_model.interactions)):
        local_model.interactions[
            layer_index
        ].avg_num_neighbors = torch_model.interactions[layer_index].avg_num_neighbors

    local_model.load_state_dict(target_state)
    normalized["model_class"] = model_class
    return local_model, normalized


def _build_local_torch_model_from_config(
    config: dict[str, Any],
    state_dict: dict[str, Any],
):
    model_config = dict(config)
    model_class = str(
        model_config.pop("model_class", model_config.pop("torch_model_class", "MACE"))
    )
    kwargs, normalized = _torch_kwargs_from_config(model_class, model_config)
    model = _instantiate_local_torch_model(
        model_class,
        kwargs,
        source_dtype=_infer_floating_dtype_from_state_dict(state_dict),
    )
    model.load_state_dict(state_dict)
    normalized["model_class"] = model_class
    return model, normalized


def load_serialized_torch_model(path_arg: str | Path):
    """Load a serialized local or legacy Torch model payload.

    Supported inputs are:

    * local Torch bundle directories containing ``config.json`` and
      ``state_dict.pt``
    * checkpoint files containing a local serialized payload
    * legacy pickled Torch model modules
    """
    path = Path(path_arg).expanduser().resolve()
    if path.is_dir():
        config_path = path / "config.json"
        state_path = path / "state_dict.pt"
        if not config_path.exists() or not state_path.exists():
            raise FileNotFoundError(
                "Torch model directory must contain config.json and state_dict.pt."
            )
        config = json.loads(config_path.read_text())
        state_dict = torch.load(state_path, map_location="cpu")
        return _build_local_torch_model_from_config(config, state_dict)

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        model = load_legacy_torch_model(path, map_location="cpu")
        return model, normalize_extracted_torch_config(
            extract_torch_model_config(model)
        )
    if isinstance(payload, torch.nn.Module):
        return payload.eval(), normalize_extracted_torch_config(
            extract_torch_model_config(payload.eval())
        )
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], torch.nn.Module):
            model = payload["model"].eval()
            return model, normalize_extracted_torch_config(
                extract_torch_model_config(model)
            )
        if "state_dict" in payload and "model_config" in payload:
            return _build_local_torch_model_from_config(
                payload["model_config"],
                payload["state_dict"],
            )
    raise ValueError(
        f"Unsupported Torch model payload at {path}. Expected a Torch module, "
        "a dict containing 'model', or a dict containing 'model_config' and 'state_dict'."
    )


def convert_torch_model(
    torch_model,
    *,
    backend: str,
    head: str | None = None,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> TorchConversionResult:
    """Convert a Torch model into the local Torch or JAX representation.

    Parameters
    ----------
    torch_model:
        Source Torch model, either already local or a legacy upstream model.
    backend:
        Conversion target, ``"torch"`` or ``"jax"``.
    head:
        Optional head name to keep when converting a multi-head local model.
    device:
        Device hint used for temporary Torch-side conversion helpers.
    config:
        Optional pre-extracted model config.  When omitted, the config is
        extracted from ``torch_model``.
    """
    backend = str(backend).strip().lower()
    if backend not in {"torch", "jax"}:
        raise ValueError(f"Unsupported backend {backend!r}; expected 'torch' or 'jax'.")

    torch_model = torch_model.eval()
    if config is None:
        config = extract_torch_model_config(torch_model)
    else:
        config = dict(config)
        if "torch_model_class" not in config and "model_class" in config:
            config["torch_model_class"] = config["model_class"]
    model_class = str(config["torch_model_class"])
    is_local_torch = _looks_like_local_torch_model(torch_model)

    if head is not None:
        if not is_local_torch:
            torch_model = _ensure_local_torch_model(
                torch_model,
                model_class=model_class,
                device=device,
            )
            is_local_torch = True
        torch_model = select_torch_model_head(torch_model, head=head)
        config = extract_torch_model_config(torch_model)
        model_class = str(config["torch_model_class"])

    if backend == "torch":
        if is_local_torch:
            normalized = normalize_extracted_torch_config(config)
            normalized["model_class"] = model_class
            return TorchConversionResult(
                backend="torch",
                model_class=model_class,
                model=torch_model,
                normalized_model_config=normalized,
            )

        local_model, normalized = _convert_native_torch_to_local(
            torch_model,
            model_class,
        )
        return TorchConversionResult(
            backend="torch",
            model_class=model_class,
            model=local_model,
            normalized_model_config=normalized,
        )

    normalized = normalize_extracted_torch_config(config)
    import_model = torch_model
    from mace_model.jax.cli.from_torch import (
        convert_model as convert_torch_to_jax,
    )

    try:
        jax_model, variables, _template_data = convert_torch_to_jax(
            import_model,
            normalized,
        )
    except NotImplementedError as exc:
        if "symmetric-contraction" not in str(exc).lower():
            raise
        import_model = _convert_upstream_torch_to_cueq_torch(torch_model, device=device)
        normalized = normalize_extracted_torch_config(
            extract_torch_model_config(import_model)
        )
        jax_model, variables, _template_data = convert_torch_to_jax(
            import_model,
            normalized,
        )
    normalized = _jsonable(normalized)
    normalized["model_class"] = model_class
    return TorchConversionResult(
        backend="jax",
        model_class=model_class,
        model=(jax_model, variables),
        normalized_model_config=normalized,
    )


def save_converted_model(
    result: TorchConversionResult,
    output: str | Path,
) -> list[Path]:
    """Save a converted model using the backend-specific bundle layout."""
    if result.backend == "torch":
        config_path, params_path = _resolve_torch_output(output)
        if config_path is None:
            payload = {
                "backend": "torch",
                "model_class": result.model_class,
                "model_config": result.normalized_model_config,
                "state_dict": result.model.state_dict(),
            }
            torch.save(payload, params_path)
            return [params_path]
        config_path.write_text(
            json.dumps(result.normalized_model_config, indent=2, sort_keys=True)
        )
        torch.save(result.model.state_dict(), params_path)
        return [config_path, params_path]

    config_path, params_path = _resolve_jax_output(output)
    _jax_model, variables = result.model
    config_path.write_text(
        json.dumps(result.normalized_model_config, indent=2, sort_keys=True)
    )
    params_path.write_bytes(serialization.to_bytes(variables))
    return [config_path, params_path]


__all__ = [
    "TorchConversionResult",
    "convert_torch_model",
    "extract_torch_model_config",
    "load_serialized_torch_model",
    "normalize_extracted_torch_config",
    "save_converted_model",
    "select_torch_model_head",
]
