from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python >= 3.11 path
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

DEFAULT_CONFIG_TOML = """backend = "torch"
model_class = "ScaleShiftMACE"
seed = 0
# output = "artifacts/init-model"

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

[model.cueq_config]
enabled = false
layout = "mul_ir"
group = "O3"
optimize_all = false
optimize_linear = false
optimize_channelwise = false
optimize_symmetric = false
optimize_fctp = false
conv_fusion = false

[torch.model]
use_edge_irreps_first = false
lammps_mliap = false

[torch.model.oeq_config]
enabled = false
optimize_all = false
optimize_channelwise = false
conv_fusion = "atomic"

[jax.model]
replace_symmetric_contraction = false
replacement_depth = 2
replacement_use_species_conditioning = true
attn_num_heads = 4
attn_head_dim = 16
attn_gate_mode = "scalar"
collapse_hidden_irreps = true
"""


@dataclass(frozen=True)
class BuildRequest:
    backend: str
    model_class: str
    seed: int
    output: str | None
    model_config: dict[str, Any]
    raw_config: dict[str, Any]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _normalize_backend(value: str | None) -> str:
    if value is None:
        raise ValueError("Config is missing 'backend'.")
    backend = str(value).strip().lower()
    if backend not in {"torch", "jax"}:
        raise ValueError(f"Unsupported backend {value!r}; expected 'torch' or 'jax'.")
    return backend


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        return tomllib.loads(config_path.read_text())
    if suffix == ".json":
        return json.loads(config_path.read_text())
    raise ValueError(
        f"Unsupported config format {config_path.suffix!r}; use .toml or .json."
    )


def load_build_request(
    path: str | Path,
    *,
    backend_override: str | None = None,
    output_override: str | None = None,
) -> BuildRequest:
    raw = load_config(path)
    backend = _normalize_backend(backend_override or raw.get("backend"))
    model_class = str(raw.get("model_class", "MACE"))
    seed = int(raw.get("seed", 0))
    output = output_override if output_override is not None else raw.get("output")

    base_model = raw.get("model")
    if not isinstance(base_model, dict):
        raise ValueError("Config must contain a [model] table or 'model' object.")

    backend_section = raw.get(backend, {})
    if backend_section is None:
        backend_section = {}
    if not isinstance(backend_section, dict):
        raise ValueError(f"Backend section {backend!r} must be a table/object.")

    backend_model = backend_section.get("model", {})
    if backend_model is None:
        backend_model = {}
    if not isinstance(backend_model, dict):
        raise ValueError(
            f"Backend-specific model section for {backend!r} must be a table/object."
        )

    merged_model = _deep_merge(base_model, backend_model)
    return BuildRequest(
        backend=backend,
        model_class=model_class,
        seed=seed,
        output=output,
        model_config=merged_model,
        raw_config=raw,
    )


__all__ = [
    "BuildRequest",
    "DEFAULT_CONFIG_TOML",
    "load_build_request",
    "load_config",
]
