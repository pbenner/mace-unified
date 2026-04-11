"""Public helpers for constructing and saving initial ``mace_model`` models.

This module is the small orchestration layer used by the config-driven CLI and
by tests.  It keeps backend-specific details out of the CLI entrypoints by:

* normalizing model-class names and config payloads
* instantiating either the local Torch or JAX model implementation
* converting config objects to JSON-safe summaries for serialization
* saving initialized model weights in the backend-specific bundle layout
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_model.torch.adapters.cuequivariance import (
    CuEquivarianceConfig as TorchCueConfig,
)
from mace_model.torch.adapters.cuequivariance import OEQConfig as TorchOEQConfig
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.blocks import (
    LinearDipolePolarReadoutBlock as TorchLinearDipolePolarReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    LinearDipoleReadoutBlock as TorchLinearDipoleReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    LinearReadoutBlock as TorchLinearReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    NonLinearBiasReadoutBlock as TorchNonLinearBiasReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    NonLinearDipolePolarReadoutBlock as TorchNonLinearDipolePolarReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    NonLinearDipoleReadoutBlock as TorchNonLinearDipoleReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    NonLinearReadoutBlock as TorchNonLinearReadoutBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticAttResidualInteractionBlock as TorchRealAgnosticAttResidualInteractionBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticDensityInteractionBlock as TorchRealAgnosticDensityInteractionBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock as TorchRealAgnosticDensityResidualInteractionBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticInteractionBlock as TorchRealAgnosticInteractionBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticResidualInteractionBlock as TorchRealAgnosticResidualInteractionBlock,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticResidualNonLinearInteractionBlock as TorchRealAgnosticResidualNonLinearInteractionBlock,
)
from mace_model.torch.modules.models import MACE as TorchMACE
from mace_model.torch.modules.models import ScaleShiftMACE as TorchScaleShiftMACE

from .config import BuildRequest

TORCH_INTERACTIONS = {
    cls.__name__: cls
    for cls in (
        TorchRealAgnosticInteractionBlock,
        TorchRealAgnosticResidualInteractionBlock,
        TorchRealAgnosticDensityInteractionBlock,
        TorchRealAgnosticDensityResidualInteractionBlock,
        TorchRealAgnosticAttResidualInteractionBlock,
        TorchRealAgnosticResidualNonLinearInteractionBlock,
    )
}

TORCH_READOUTS = {
    cls.__name__: cls
    for cls in (
        TorchNonLinearReadoutBlock,
        TorchNonLinearBiasReadoutBlock,
        TorchLinearReadoutBlock,
        TorchLinearDipoleReadoutBlock,
        TorchLinearDipolePolarReadoutBlock,
        TorchNonLinearDipoleReadoutBlock,
        TorchNonLinearDipolePolarReadoutBlock,
    )
}

TORCH_MODEL_CLASSES = {"MACE": TorchMACE, "ScaleShiftMACE": TorchScaleShiftMACE}


@dataclass(frozen=True)
class BuildResult:
    """Result returned by :func:`build_initial_model`.

    Attributes
    ----------
    request:
        Fully normalized build request used to instantiate the model.
    model:
        Backend-specific model instance.  This is either a local Torch module or
        a local JAX/NNX model.
    normalized_model_config:
        JSON-serializable version of the effective model configuration.
    """

    request: BuildRequest
    model: Any
    normalized_model_config: dict[str, Any]


@lru_cache(maxsize=1)
def _jax_runtime():
    from flax import nnx, serialization

    from mace_model.jax.modules.models import MACE as JaxMACE
    from mace_model.jax.modules.models import ScaleShiftMACE as JaxScaleShiftMACE
    from mace_model.jax.nnx_utils import state_to_pure_dict, state_to_serializable_dict
    from mace_model.jax.tools.model_builder import build_model as build_jax_model

    return {
        "nnx": nnx,
        "serialization": serialization,
        "state_to_pure_dict": state_to_pure_dict,
        "state_to_serializable_dict": state_to_serializable_dict,
        "build_model": build_jax_model,
        "model_classes": {"MACE": JaxMACE, "ScaleShiftMACE": JaxScaleShiftMACE},
    }


def _normalize_atomic_config(
    config: dict[str, Any],
) -> tuple[tuple[int, ...], np.ndarray]:
    atomic_numbers = tuple(int(z) for z in config.get("atomic_numbers", []))
    if not atomic_numbers:
        raise ValueError("Model config is missing atomic_numbers.")
    if "atomic_energies" not in config:
        raise ValueError("Model config is missing atomic_energies.")
    atomic_energies = np.asarray(config["atomic_energies"], dtype=np.float32)
    if atomic_energies.ndim == 1:
        expected = len(atomic_numbers)
        if int(atomic_energies.shape[0]) != expected:
            raise ValueError(
                "atomic_energies length does not match atomic_numbers "
                f"({atomic_energies.shape[0]} vs {expected})."
            )
    elif int(atomic_energies.shape[-1]) != len(atomic_numbers):
        raise ValueError(
            "atomic_energies last dimension does not match atomic_numbers "
            f"({atomic_energies.shape[-1]} vs {len(atomic_numbers)})."
        )
    return atomic_numbers, atomic_energies


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if value.__class__.__name__ == "Irreps":
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if inspect.isclass(value):
        return value.__name__
    if callable(value) and hasattr(value, "__name__"):
        return value.__name__
    if hasattr(value, "__dict__"):
        public_attrs = {
            key: val for key, val in vars(value).items() if not key.startswith("_")
        }
        if public_attrs:
            return _jsonable(public_attrs)
    return str(value)


def _resolve_torch_gate(gate: Any):
    if gate is None:
        return None
    if callable(gate):
        return gate
    name = str(gate).strip().lower()
    mapping = {
        "silu": torch.nn.functional.silu,
        "silu6": torch.nn.functional.silu,
        "swish": torch.nn.functional.silu,
        "relu": torch.nn.functional.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softplus": torch.nn.functional.softplus,
        "abs": torch.abs,
        "none": None,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported Torch gate {gate!r}.")
    return mapping[name]


def _as_torch_irreps(value: Any) -> o3.Irreps | None:
    if value is None:
        return None
    if value.__class__.__name__ == "Irreps":
        return value
    if isinstance(value, str):
        return o3.Irreps(value)
    if isinstance(value, int):
        return o3.Irreps(f"{value}x0e")
    return o3.Irreps(str(value))


def _normalize_model_class(backend: str, model_class: str):
    if backend == "torch":
        try:
            return TORCH_MODEL_CLASSES[model_class]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Torch model_class {model_class!r}. Expected one of "
                f"{sorted(TORCH_MODEL_CLASSES)}."
            ) from exc
    jax_model_classes = _jax_runtime()["model_classes"]
    try:
        return jax_model_classes[model_class]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported JAX model_class {model_class!r}. Expected one of "
            f"{sorted(jax_model_classes)}."
        ) from exc


def _torch_kwargs_from_config(
    model_class: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a normalized model config into Torch constructor kwargs."""
    model_cls = _normalize_model_class("torch", model_class)
    signature = inspect.signature(model_cls.__init__)
    allowed = set(signature.parameters) - {"self", "kwargs"}
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        allowed |= set(inspect.signature(TorchMACE.__init__).parameters) - {
            "self",
            "kwargs",
        }
    unsupported = sorted(set(config) - allowed)
    if unsupported:
        raise ValueError(
            "Unsupported Torch model config keys: " + ", ".join(unsupported)
        )

    atomic_numbers, atomic_energies = _normalize_atomic_config(config)
    kwargs = dict(config)
    kwargs["interaction_cls"] = TORCH_INTERACTIONS[str(config["interaction_cls"])]
    kwargs["interaction_cls_first"] = TORCH_INTERACTIONS[
        str(config["interaction_cls_first"])
    ]
    kwargs["hidden_irreps"] = _as_torch_irreps(config["hidden_irreps"])
    kwargs["MLP_irreps"] = _as_torch_irreps(config["MLP_irreps"])
    kwargs["gate"] = _resolve_torch_gate(config.get("gate"))
    kwargs["atomic_numbers"] = list(atomic_numbers)
    kwargs["atomic_energies"] = atomic_energies
    kwargs["num_elements"] = len(atomic_numbers)
    kwargs["cueq_config"] = (
        TorchCueConfig(**config["cueq_config"])
        if isinstance(config.get("cueq_config"), dict)
        else config.get("cueq_config")
    )
    kwargs["oeq_config"] = (
        TorchOEQConfig(**config["oeq_config"])
        if isinstance(config.get("oeq_config"), dict)
        else config.get("oeq_config")
    )
    kwargs["readout_cls"] = (
        TORCH_READOUTS[str(config["readout_cls"])]
        if config.get("readout_cls") is not None
        else TorchNonLinearReadoutBlock
    )
    if config.get("edge_irreps") is not None:
        kwargs["edge_irreps"] = _as_torch_irreps(config["edge_irreps"])
    if config.get("radial_MLP") is not None:
        kwargs["radial_MLP"] = tuple(int(v) for v in config["radial_MLP"])
    normalized = _jsonable(kwargs)
    normalized["model_class"] = model_class
    return kwargs, normalized


def _jax_config_from_request(
    model_class: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Normalize a build request into the config expected by the JAX builder."""
    normalized = dict(config)
    if model_class != "MACE":
        normalized["torch_model_class"] = model_class
    return normalized


def _count_array_leaves(tree: Any) -> int:
    if isinstance(tree, dict):
        return sum(_count_array_leaves(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_count_array_leaves(v) for v in tree)
    if hasattr(tree, "shape") and hasattr(tree, "dtype"):
        try:
            return int(np.asarray(tree).size)
        except Exception:
            return 0
    return 0


def _parameter_count(backend: str, model: Any) -> int:
    if backend == "torch":
        return int(sum(param.numel() for param in model.parameters()))
    runtime = _jax_runtime()
    _, state = runtime["nnx"].split(model)
    return _count_array_leaves(runtime["state_to_pure_dict"](state))


def build_initial_model(request: BuildRequest) -> BuildResult:
    """Instantiate a new local Torch or JAX model from a build request.

    The returned :class:`BuildResult` contains both the model object and a
    JSON-safe representation of the effective model configuration, which makes
    it suitable for later serialization with :func:`save_initialized_model`.
    """
    if request.backend == "torch":
        torch.manual_seed(request.seed)
        kwargs, normalized = _torch_kwargs_from_config(
            request.model_class, request.model_config
        )
        model_cls = _normalize_model_class("torch", request.model_class)
        model = model_cls(**kwargs)
        return BuildResult(
            request=request, model=model, normalized_model_config=normalized
        )

    runtime = _jax_runtime()
    normalized_config = _jax_config_from_request(
        request.model_class, request.model_config
    )
    model = runtime["build_model"](
        normalized_config,
        rngs=runtime["nnx"].Rngs(request.seed),
    )
    normalized = _jsonable(normalized_config)
    normalized["model_class"] = request.model_class
    return BuildResult(request=request, model=model, normalized_model_config=normalized)


def summarize_build(result: BuildResult) -> dict[str, Any]:
    """Return a compact human-readable summary of a completed build."""
    return {
        "backend": result.request.backend,
        "model_class": result.request.model_class,
        "parameters": _parameter_count(result.request.backend, result.model),
        "atomic_numbers": list(
            result.normalized_model_config.get("atomic_numbers", [])
        ),
        "num_interactions": result.normalized_model_config.get("num_interactions"),
    }


def _resolve_torch_output(path_arg: str | Path) -> tuple[Path | None, Path]:
    path = Path(path_arg).expanduser().resolve()
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return None, path
    path.mkdir(parents=True, exist_ok=True)
    return path / "config.json", path / "state_dict.pt"


def _resolve_jax_output(path_arg: str | Path) -> tuple[Path, Path]:
    path = Path(path_arg).expanduser().resolve()
    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, path.with_suffix(".msgpack")
    if path.suffix.lower() == ".msgpack":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.with_suffix(".json"), path
    path.mkdir(parents=True, exist_ok=True)
    return path / "config.json", path / "params.msgpack"


def save_initialized_model(result: BuildResult, output: str | Path) -> list[Path]:
    """Persist an initialized model in the backend-specific bundle format.

    Torch outputs are stored either as a directory containing ``config.json``
    plus ``state_dict.pt`` or as a single checkpoint payload.  JAX outputs are
    stored as ``config.json`` plus ``params.msgpack``.
    """
    if result.request.backend == "torch":
        config_path, params_path = _resolve_torch_output(output)
        if config_path is None:
            payload = {
                "backend": "torch",
                "model_class": result.request.model_class,
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

    runtime = _jax_runtime()
    config_path, params_path = _resolve_jax_output(output)
    _, state = runtime["nnx"].split(result.model)
    state_pure = runtime["state_to_serializable_dict"](state)
    config_path.write_text(
        json.dumps(result.normalized_model_config, indent=2, sort_keys=True)
    )
    params_path.write_bytes(runtime["serialization"].to_bytes(state_pure))
    return [config_path, params_path]


__all__ = [
    "BuildResult",
    "build_initial_model",
    "save_initialized_model",
    "summarize_build",
]
