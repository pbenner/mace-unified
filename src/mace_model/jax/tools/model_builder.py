"""Public helpers to build MACE-JAX models from serialized configs.

This module intentionally exposes the small config -> module/template surface
that downstream packages such as ``equitrain`` need. Training, data loading,
and batching stay outside ``mace-model``.
"""

from __future__ import annotations

from typing import Any

import cuequivariance as cue
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mace_model.core.data.neighborhood import get_neighborhood
from mace_model.core.data.utils import (
    AtomicNumberTable,
    Configuration,
    atomic_numbers_to_indices,
)
from mace_model.jax.adapters.e3nn import Irrep, Irreps
from mace_model.jax.adapters.nnx import resolve_gate_callable
from mace_model.jax.modules.blocks import (
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
)
from mace_model.jax.modules.models import MACE, ScaleShiftMACE
from mace_model.jax.adapters.cuequivariance import CuEquivarianceConfig


_INTERACTION_CLASSES = {
    cls.__name__: cls
    for cls in (
        RealAgnosticInteractionBlock,
        RealAgnosticResidualInteractionBlock,
        RealAgnosticDensityInteractionBlock,
        RealAgnosticDensityResidualInteractionBlock,
        RealAgnosticAttResidualInteractionBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
}

_READOUT_CLASSES = {
    cls.__name__: cls
    for cls in (
        NonLinearReadoutBlock,
        NonLinearBiasReadoutBlock,
        LinearReadoutBlock,
        LinearDipoleReadoutBlock,
        LinearDipolePolarReadoutBlock,
        NonLinearDipoleReadoutBlock,
        NonLinearDipolePolarReadoutBlock,
    )
}


def _normalize_atomic_config(
    config: dict[str, Any],
    *,
    dtype: np.dtype = np.float32,
) -> tuple[dict[str, Any], tuple[int, ...], np.ndarray]:
    atomic_numbers = tuple(int(z) for z in config.get("atomic_numbers", []))
    if not atomic_numbers:
        raise ValueError("Config is missing atomic_numbers.")

    if "atomic_energies" not in config:
        raise ValueError("Config is missing atomic_energies.")

    atomic_energies = np.asarray(config.get("atomic_energies"), dtype=dtype)
    if atomic_energies.size == 0:
        raise ValueError("Config has empty atomic_energies.")

    num_elements = len(atomic_numbers)
    if atomic_energies.ndim == 1:
        if atomic_energies.shape[0] != num_elements:
            raise ValueError(
                "atomic_energies length does not match atomic_numbers "
                f"({atomic_energies.shape[0]} vs {num_elements})."
            )
    else:
        last_dim = atomic_energies.shape[-1]
        if last_dim != num_elements:
            raise ValueError(
                "atomic_energies last dimension does not match atomic_numbers "
                f"({last_dim} vs {num_elements})."
            )

    normalized = dict(config)
    normalized["atomic_numbers"] = list(atomic_numbers)
    normalized["atomic_energies"] = atomic_energies.tolist()
    return normalized, atomic_numbers, atomic_energies


def _parse_parity(parity: Any) -> int:
    if parity is None:
        return 1
    if isinstance(parity, str):
        normalized = parity.strip().lower()
        if normalized in {"e", "even"}:
            return 1
        if normalized in {"o", "odd"}:
            return -1
    try:
        parity_int = int(parity)
    except (TypeError, ValueError):
        return 1
    return 1 if parity_int >= 0 else -1


def _as_l_parity(rep: Any):
    if hasattr(rep, "ir") and not hasattr(rep, "l"):
        try:
            rep = rep.ir
        except Exception:
            pass

    if hasattr(rep, "l") and hasattr(rep, "p"):
        try:
            return int(rep.l), _parse_parity(rep.p)
        except Exception:
            return None

    if isinstance(rep, (list, tuple)) and len(rep) == 2:
        try:
            return int(rep[0]), _parse_parity(rep[1])
        except Exception:
            return None

    return None


def _as_irrep_entry(entry: Any):
    if isinstance(entry, dict):
        mul = entry.get("mul") or entry.get("multiplicity") or entry.get("n")
        rep = entry.get("irrep") or entry.get("rep") or entry.get("l")
        parity = entry.get("p") or entry.get("parity")
        if isinstance(rep, dict):
            l_val = rep.get("l")
            parity = parity or rep.get("p") or rep.get("parity")
        else:
            l_val = rep
        if mul is None or l_val is None:
            return None
        return int(mul), (int(l_val), _parse_parity(parity))

    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and isinstance(entry[0], (int, np.integer)):
            mul = int(entry[0])
            rep = entry[1]
            l_parity = _as_l_parity(rep)
            if l_parity is not None:
                return mul, l_parity
            if isinstance(rep, dict):
                l_val = rep.get("l")
                parity = rep.get("p") or rep.get("parity")
                if l_val is None:
                    return None
                return mul, (int(l_val), _parse_parity(parity))
            if isinstance(rep, (int, np.integer)):
                return mul, (int(rep), 1)

    l_parity = _as_l_parity(entry)
    if l_parity is not None:
        return 1, l_parity
    return None


def _normalize_irreps(value: Any):
    if isinstance(value, dict):
        value = [value]

    if isinstance(value, (list, tuple)):
        if value and _as_irrep_entry(value) is not None:
            entries = [value]
        else:
            entries = value

        parsed = []
        for item in entries:
            entry = _as_irrep_entry(item)
            if entry is None:
                return None
            parsed.append(entry)
        return parsed

    return None


def _as_irreps(value: Any) -> Irreps:
    if isinstance(value, cue.Irreps):
        return value
    if isinstance(value, str):
        return Irreps(value)
    if isinstance(value, int):
        return Irreps(f"{value}x0e")
    normalized = _normalize_irreps(value)
    if normalized is not None:
        return Irreps(normalized)
    return Irreps(str(value))


def _interaction(name_or_cls: Any):
    name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
    try:
        return _INTERACTION_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported interaction class {name!r} in config.") from exc


def _readout(name_or_cls: Any):
    if name_or_cls is None:
        return NonLinearReadoutBlock
    name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
    return _READOUT_CLASSES.get(name, NonLinearReadoutBlock)


def _build_configuration(
    atomic_numbers: tuple[int, ...],
    r_max: float,
) -> Configuration:
    node_numbers = atomic_numbers if len(atomic_numbers) > 1 else atomic_numbers * 2
    num_atoms = len(node_numbers)
    spacing = max(min(float(r_max) * 0.4, 1.0), 0.5)
    positions = np.zeros((num_atoms, 3), dtype=float)
    for idx in range(num_atoms):
        positions[idx, 0] = spacing * idx
        positions[idx, 1] = spacing * (idx % 2)

    cell = np.eye(3, dtype=float) * (spacing * max(num_atoms, 1) * 4.0)
    return Configuration(
        atomic_numbers=np.asarray(node_numbers, dtype=int),
        positions=positions,
        properties={},
        property_weights={},
        cell=cell,
        pbc=(False, False, False),
        weight=1.0,
        config_type="Default",
        head="Default",
    )


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(indices.shape[0]), indices] = 1.0
    return encoded


def _configuration_to_jax_graph(
    configuration: Configuration,
    *,
    atomic_numbers: tuple[int, ...],
    r_max: float,
) -> dict[str, jnp.ndarray]:
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=np.asarray(configuration.positions, dtype=np.float32),
        cutoff=float(r_max),
        pbc=configuration.pbc,
        cell=configuration.cell,
    )
    z_table = AtomicNumberTable(atomic_numbers)
    species_index = atomic_numbers_to_indices(
        np.asarray(configuration.atomic_numbers, dtype=np.int32),
        z_table=z_table,
    )
    node_attrs = _one_hot(np.asarray(species_index, dtype=np.int32), len(z_table))
    num_nodes = int(node_attrs.shape[0])

    return {
        "positions": jnp.asarray(configuration.positions, dtype=jnp.float32),
        "node_attrs": jnp.asarray(node_attrs, dtype=jnp.float32),
        "edge_index": jnp.asarray(edge_index, dtype=jnp.int32),
        "shifts": jnp.asarray(shifts, dtype=jnp.float32),
        "unit_shifts": jnp.asarray(unit_shifts, dtype=jnp.float32),
        "cell": jnp.asarray(cell, dtype=jnp.float32),
        "batch": jnp.zeros((num_nodes,), dtype=jnp.int32),
        "ptr": jnp.asarray([0, num_nodes], dtype=jnp.int32),
        "head": jnp.asarray([0], dtype=jnp.int32),
    }


def _prepare_template_data(config: dict[str, Any]) -> dict[str, jnp.ndarray]:
    atomic_numbers = tuple(int(z) for z in config["atomic_numbers"])
    configuration = _build_configuration(atomic_numbers, float(config["r_max"]))
    return _configuration_to_jax_graph(
        configuration,
        atomic_numbers=atomic_numbers,
        r_max=float(config["r_max"]),
    )


def _build_cueq_config(
    config: dict[str, Any],
    cueq_config: CuEquivarianceConfig | None,
) -> CuEquivarianceConfig | None:
    if cueq_config is not None:
        return cueq_config
    raw = config.get("cueq_config")
    if isinstance(raw, CuEquivarianceConfig):
        return raw
    if isinstance(raw, dict):
        return CuEquivarianceConfig(**raw)
    if config.get("cue_conv_fusion"):
        return CuEquivarianceConfig(
            enabled=False,
            optimize_channelwise=True,
            conv_fusion=bool(config["cue_conv_fusion"]),
            layout="mul_ir",
        )
    return None


def _build_jax_model(
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
    rngs: nnx.Rngs | None = None,
):
    if rngs is None:
        rngs = nnx.Rngs(0)

    collapse_hidden_irreps = config.get("collapse_hidden_irreps", None)
    if collapse_hidden_irreps is None:
        try:
            num_interactions = int(config.get("num_interactions", 0))
        except Exception:
            num_interactions = 0
        if num_interactions == 1 and config.get("hidden_irreps") is not None:
            try:
                hidden_irreps = _as_irreps(config["hidden_irreps"])
                collapse_hidden_irreps = len(hidden_irreps) <= 1
            except Exception:
                collapse_hidden_irreps = None

    cueq_config_obj = _build_cueq_config(config, cueq_config)

    config, atomic_numbers, atomic_energies = _normalize_atomic_config(
        config,
        dtype=np.float32,
    )
    num_elements = len(atomic_numbers)

    common_kwargs = dict(
        r_max=config["r_max"],
        num_bessel=config["num_bessel"],
        num_polynomial_cutoff=config["num_polynomial_cutoff"],
        max_ell=config["max_ell"],
        interaction_cls=_interaction(config["interaction_cls"]),
        interaction_cls_first=_interaction(config["interaction_cls_first"]),
        num_interactions=config["num_interactions"],
        num_elements=num_elements,
        hidden_irreps=_as_irreps(config["hidden_irreps"]),
        MLP_irreps=_as_irreps(config["MLP_irreps"]),
        atomic_numbers=atomic_numbers,
        atomic_energies=atomic_energies,
        avg_num_neighbors=float(config["avg_num_neighbors"]),
        correlation=config["correlation"],
        radial_type=config.get("radial_type", "bessel"),
        pair_repulsion=config.get("pair_repulsion", False),
        distance_transform=config.get("distance_transform", None),
        embedding_specs=config.get("embedding_specs"),
        use_so3=config.get("use_so3", False),
        use_reduced_cg=config.get("use_reduced_cg", True),
        use_agnostic_product=config.get("use_agnostic_product", False),
        replace_symmetric_contraction=config.get(
            "replace_symmetric_contraction", False
        ),
        replacement_hidden_irreps=(
            _as_irreps(config["replacement_hidden_irreps"])
            if config.get("replacement_hidden_irreps") is not None
            else None
        ),
        replacement_depth=int(config.get("replacement_depth", 2)),
        replacement_use_species_conditioning=bool(
            config.get("replacement_use_species_conditioning", True)
        ),
        attn_num_heads=int(config.get("attn_num_heads", 4)),
        attn_head_dim=int(config.get("attn_head_dim", 16)),
        attn_gate_mode=str(config.get("attn_gate_mode", "scalar")),
        use_last_readout_only=config.get("use_last_readout_only", False),
        use_embedding_readout=config.get("use_embedding_readout", False),
        collapse_hidden_irreps=(
            True if collapse_hidden_irreps is None else bool(collapse_hidden_irreps)
        ),
        readout_cls=_readout(config.get("readout_cls", None)),
        readout_use_higher_irrep_invariants=bool(
            config.get("readout_use_higher_irrep_invariants", False)
        ),
        readout_invariant_eps=float(config.get("readout_invariant_eps", 1e-12)),
        gate=resolve_gate_callable(config.get("gate", None)),
        cueq_config=cueq_config_obj,
    )

    if config.get("normalize2mom_consts") is not None:
        common_kwargs["normalize2mom_consts"] = {
            str(key): float(value)
            for key, value in config["normalize2mom_consts"].items()
        }

    if config.get("radial_MLP") is not None:
        common_kwargs["radial_MLP"] = tuple(int(x) for x in config["radial_MLP"])

    if config.get("edge_irreps") is not None:
        common_kwargs["edge_irreps"] = _as_irreps(config["edge_irreps"])

    if config.get("apply_cutoff") is not None:
        common_kwargs["apply_cutoff"] = bool(config["apply_cutoff"])

    torch_class = config.get("torch_model_class", "MACE")
    if torch_class == "ScaleShiftMACE" or "atomic_inter_scale" in config:
        return ScaleShiftMACE(
            atomic_inter_scale=np.asarray(config.get("atomic_inter_scale", 1.0)),
            atomic_inter_shift=np.asarray(config.get("atomic_inter_shift", 0.0)),
            rngs=rngs,
            **common_kwargs,
        )
    return MACE(rngs=rngs, **common_kwargs)


def coerce_irreps(value: Any) -> Irreps:
    """Return a cue-backed irreps object from a serialized or backend value."""

    return _as_irreps(value)


def normalize_atomic_config(
    config: dict[str, Any],
    *,
    dtype: np.dtype = np.float32,
) -> tuple[dict[str, Any], tuple[int, ...], np.ndarray]:
    """Validate and normalize the atomic-number/energy portion of a model config."""

    return _normalize_atomic_config(config, dtype=dtype)


def prepare_template_data(config: dict[str, Any]) -> dict[str, jnp.ndarray]:
    """Build a small template graph in JAX array form for model initialization."""

    return _prepare_template_data(config)


def build_model(
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
    rngs: nnx.Rngs | None = None,
):
    """Construct a local ``mace-model`` JAX MACE module from serialized config."""

    return _build_jax_model(config, cueq_config=cueq_config, rngs=rngs)


__all__ = [
    "build_model",
    "coerce_irreps",
    "normalize_atomic_config",
    "prepare_template_data",
    "_as_irreps",
    "_build_jax_model",
    "_normalize_atomic_config",
    "_prepare_template_data",
]
