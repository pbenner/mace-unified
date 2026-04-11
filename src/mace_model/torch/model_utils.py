from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .adapters.e3nn import o3


def extract_radial_mlp(model: torch.nn.Module) -> list[int]:
    try:
        return list(model.interactions[0].conv_tp_weights.hs[1:-1])
    except AttributeError:
        try:
            return [
                int(
                    model.interactions[0]
                    .conv_tp_weights.net[k]
                    .__dict__["normalized_shape"][0]
                )
                for k in range(1, len(model.interactions[0].conv_tp_weights.net), 3)
            ]
        except AttributeError:
            return []


def extract_torch_model_config(model: torch.nn.Module) -> dict[str, Any]:
    model_class = model.__class__.__name__
    if model_class not in {"MACE", "ScaleShiftMACE"}:
        return {"error": f"Model is not a supported Torch MACE model: {model_class}"}

    def radial_to_name(radial_type: str) -> str:
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    def radial_to_transform(radial) -> str | None:
        if not hasattr(radial, "distance_transform"):
            return None
        transform_name = radial.distance_transform.__class__.__name__
        if transform_name == "AgnesiTransform":
            return "Agnesi"
        if transform_name == "SoftTransform":
            return "Soft"
        return transform_name

    scale = model.scale_shift.scale
    shift = model.scale_shift.shift
    if hasattr(model, "heads") and model.heads is not None:
        heads = list(model.heads)
    else:
        heads = ["Default"]
    num_interactions = int(model.num_interactions.item())
    model_mlp_irreps = (
        o3.Irreps(str(model.readouts[-1].hidden_irreps))
        if num_interactions > 1
        else o3.Irreps("1x0e")
    )
    try:
        correlation = (
            len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = model.products[0].symmetric_contractions.contraction_degree

    gate = None
    if num_interactions > 1:
        acts = getattr(model.readouts[-1].non_linearity, "_modules", {})
        act0 = acts.get("acts", [None])[0] if "acts" in acts else None
        gate = getattr(act0, "f", None)

    basis_module = getattr(model.radial_embedding, "bessel_fn", None)
    if basis_module is None:
        basis_module = getattr(model.radial_embedding, "basis_fn", None)
    if basis_module is None:
        raise AttributeError(
            "Unsupported radial embedding: expected 'bessel_fn' or 'basis_fn'."
        )

    config = {
        "r_max": float(model.r_max.item()),
        "num_bessel": len(basis_module.bessel_weights),
        "num_polynomial_cutoff": int(model.radial_embedding.cutoff_fn.p.item()),
        "max_ell": int(model.spherical_harmonics._lmax),  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": num_interactions,
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "edge_irreps": getattr(model, "edge_irreps", None),
        "MLP_irreps": (
            o3.Irreps(f"{model_mlp_irreps.count((0, 1)) // len(heads)}x0e")
            if num_interactions > 1
            else o3.Irreps("1x0e")
        ),
        "gate": gate,
        "use_reduced_cg": getattr(model, "use_reduced_cg", False),
        "use_so3": getattr(model, "use_so3", False),
        "use_edge_irreps_first": getattr(model, "use_edge_irreps_first", False),
        "use_agnostic_product": getattr(model, "use_agnostic_product", False),
        "use_last_readout_only": getattr(model, "use_last_readout_only", False),
        "use_embedding_readout": hasattr(model, "embedding_readout"),
        "readout_cls": model.readouts[-1].__class__,
        "cueq_config": getattr(model, "cueq_config", None),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.detach()
        .cpu()
        .numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": list(model.atomic_numbers),
        "correlation": correlation,
        "radial_type": radial_to_name(basis_module.__class__.__name__),
        "embedding_specs": getattr(model, "embedding_specs", None),
        "apply_cutoff": getattr(model, "apply_cutoff", True),
        "radial_MLP": extract_radial_mlp(model),
        "pair_repulsion": hasattr(model, "pair_repulsion_fn"),
        "distance_transform": radial_to_transform(model.radial_embedding),
        "atomic_inter_scale": scale.detach().cpu().numpy(),
        "atomic_inter_shift": shift.detach().cpu().numpy(),
        "heads": heads,
        "torch_model_class": model_class,
    }
    return config


def select_local_torch_model_head(
    model: torch.nn.Module,
    head_to_keep: str | None = None,
) -> torch.nn.Module:
    if not hasattr(model, "heads") or len(model.heads) <= 1:
        raise ValueError("Model must be a multihead model with more than one head")

    if head_to_keep is None:
        try:
            head_idx = next(
                i for i, head in enumerate(model.heads) if head != "pt_head"
            )
        except StopIteration as exc:
            raise ValueError("No non-PT head found in model") from exc
    else:
        try:
            head_idx = model.heads.index(head_to_keep)
        except ValueError as exc:
            raise ValueError(f"Head {head_to_keep} not found in model") from exc

    model_config = extract_torch_model_config(model)
    if "error" in model_config:
        raise RuntimeError(model_config["error"])
    model_config["heads"] = [model.heads[head_idx]]
    model_config["atomic_energies"] = (
        model.atomic_energies_fn.atomic_energies[head_idx : head_idx + 1]
        .detach()
        .cpu()
        .numpy()
    )
    model_config["atomic_inter_scale"] = model.scale_shift.scale[head_idx].item()
    model_config["atomic_inter_shift"] = model.scale_shift.shift[head_idx].item()
    mlp_count_irreps = model_config["MLP_irreps"].count((0, 1))

    model_class = model.__class__
    model_config.pop("torch_model_class", None)
    new_model = model_class(**model_config)
    target_state = new_model.state_dict()
    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        selected = None
        if "atomic_energies" in name:
            selected = param[head_idx : head_idx + 1]
        elif name.endswith("scale") or name.endswith("shift"):
            selected = param[head_idx : head_idx + 1]
        elif "embedding_readout.linear" in name and name.endswith("weight"):
            selected = param.reshape(-1, len(model.heads))[:, head_idx].flatten()
        elif "readouts" in name:
            channels_per_head = (
                param.shape[0] // len(model.heads) if param.ndim > 0 else 0
            )
            start_idx = head_idx * channels_per_head
            end_idx = start_idx + channels_per_head

            if ".linear_mid." in name and name.endswith("weight"):
                selected = param.reshape(
                    len(model.heads),
                    mlp_count_irreps,
                    len(model.heads),
                    mlp_count_irreps,
                )[head_idx, :, head_idx, :].flatten() / (len(model.heads) ** 0.5)
            elif ".linear_mid." in name and name.endswith("bias"):
                if param.shape == torch.Size([0]):
                    continue
                selected = param.reshape(len(model.heads), mlp_count_irreps)[
                    head_idx, :
                ].flatten()
            elif ".linear_2." in name and name.endswith("weight"):
                selected = param.reshape(len(model.heads), -1, len(model.heads))[
                    head_idx, :, head_idx
                ].flatten() / (len(model.heads) ** 0.5)
            elif ".linear_2." in name and name.endswith("bias"):
                if param.shape == torch.Size([0]):
                    continue
                selected = param[head_idx].flatten()
            elif ".linear_1." in name and name.endswith("weight"):
                selected = param.reshape(-1, len(model.heads), mlp_count_irreps)[
                    :, head_idx, :
                ].flatten()
            elif ".linear_1." in name and name.endswith("bias"):
                if param.shape == torch.Size([0]):
                    continue
                selected = param.reshape(len(model.heads), mlp_count_irreps)[
                    head_idx, :
                ].flatten()
            elif ".linear." in name and name.endswith("weight"):
                selected = param.reshape(-1, len(model.heads))[:, head_idx].flatten()
            else:
                if channels_per_head:
                    selected = param[start_idx:end_idx]
                else:
                    selected = param
        else:
            selected = param

        target = target_state.get(name)
        if (
            target is not None
            and hasattr(selected, "shape")
            and tuple(selected.shape) != tuple(target.shape)
            and int(selected.numel()) == int(target.numel())
        ):
            selected = selected.reshape(target.shape)
        new_state_dict[name] = selected

    new_model.load_state_dict(new_state_dict, strict=False)
    return new_model.eval()


__all__ = [
    "extract_radial_mlp",
    "extract_torch_model_config",
    "select_local_torch_model_head",
]
