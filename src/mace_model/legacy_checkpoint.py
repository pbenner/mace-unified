from __future__ import annotations

import contextlib
import importlib
import os
import pkgutil
import sys
import types
import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import Iterator

import torch
from torch.serialization import add_safe_globals

from mace_model.torch.adapters.e3nn import (
    get_optimization_defaults,
    set_optimization_defaults,
)
from mace_model.torch.adapters.e3nn import Irreps as make_irreps
from mace_model.torch.adapters.e3nn.math import normalize2mom

MACE_MP_URLS = {
    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
    "small-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "medium-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "small-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model",
    "medium-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "large-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "medium-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    "medium-mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "small-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace-matpes-pbe-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace-matpes-r2scan-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "mh-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
}
MACE_OFF_URLS = {
    "small": "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true",
    "medium": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
    "large": "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true",
}
MACE_OMOL_URLS = {
    "extra_large": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model",
}
ANICC_DEFAULT_URL = "https://github.com/ACEsuit/mace/raw/main/mace/calculators/foundations_models/ani500k_large_CC.model"
SUPPORTED_FOUNDATION_SOURCES = ("mp", "off", "anicc", "omol")

_IRREPS_TYPE = type(make_irreps("0e"))
_MUL_IRREP_TYPE = type(next(iter(make_irreps("1x0e"))))
_IRREP_TYPE = type(next(iter(make_irreps("1x1o"))).ir)


class _LegacyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - safety guard
        raise NotImplementedError(
            "Legacy checkpoint stubs are only intended for deserialization."
        )


class _LegacyCodeGenMixin:
    pass


def get_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mace"


def get_mace_mp_names() -> list[str | None]:
    return [None] + list(MACE_MP_URLS.keys())


def _download_if_needed(url: str, cached_path: Path) -> Path:
    if cached_path.exists():
        return cached_path
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cached_path)
    return cached_path


def resolve_foundation_checkpoint(source: str, model: str | None = None) -> Path:
    source = str(source).strip().lower()
    if source not in SUPPORTED_FOUNDATION_SOURCES:
        raise ValueError(
            "Unknown foundation source. Supported values are "
            + ", ".join(repr(value) for value in SUPPORTED_FOUNDATION_SOURCES)
            + "."
        )

    if source == "mp":
        if model is None:
            model = "medium-mpa-0"
        if model.startswith("https:"):
            url = model
        elif Path(model).exists():
            return Path(model).expanduser().resolve()
        else:
            try:
                url = MACE_MP_URLS[model]
            except KeyError as exc:
                raise ValueError(f"Unsupported MACE-MP model {model!r}.") from exc
    elif source == "off":
        if model is None:
            model = "medium"
        if model.startswith("https:"):
            url = model
        elif Path(model).exists():
            return Path(model).expanduser().resolve()
        else:
            try:
                url = MACE_OFF_URLS[model]
            except KeyError as exc:
                raise ValueError(f"Unsupported MACE-OFF model {model!r}.") from exc
    elif source == "omol":
        if model is None:
            model = "extra_large"
        if model.startswith("https:"):
            url = model
        elif Path(model).exists():
            return Path(model).expanduser().resolve()
        else:
            try:
                url = MACE_OMOL_URLS[model]
            except KeyError as exc:
                raise ValueError(f"Unsupported MACE-OMOL model {model!r}.") from exc
    else:
        if model is None:
            url = ANICC_DEFAULT_URL
        elif model.startswith("https:"):
            url = model
        elif Path(model).exists():
            return Path(model).expanduser().resolve()
        else:
            raise ValueError(
                "ANI-CC expects a local path, a direct URL, or no model argument."
            )

    filename = os.path.basename(url).split("?")[0]
    return _download_if_needed(url, get_cache_dir() / filename)


def _legacy_module_type(name: str, module_name: str) -> type[_LegacyModule]:
    return type(name, (_LegacyModule,), {"__module__": module_name})


def _build_legacy_imports() -> dict[str, types.ModuleType]:
    linear_instruction = namedtuple(
        "Instruction",
        "i_in i_out path_shape path_weight",
    )
    linear_instruction.__module__ = "e3nn.o3._linear"
    tp_instruction = namedtuple(
        "Instruction",
        "i_in1 i_in2 i_out connection_mode has_weight path_weight path_shape",
    )
    tp_instruction.__module__ = "e3nn.o3._tensor_product._instruction"

    modules: dict[str, types.ModuleType] = {}

    def add_module(name: str, *, package: bool = False) -> types.ModuleType:
        module = types.ModuleType(name)
        if package:
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        return module

    e3nn = add_module("e3nn", package=True)
    e3nn_o3 = add_module("e3nn.o3", package=True)
    e3nn_o3_irreps = add_module("e3nn.o3._irreps")
    e3nn_o3_linear = add_module("e3nn.o3._linear")
    e3nn_o3_sh = add_module("e3nn.o3._spherical_harmonics")
    e3nn_o3_tp = add_module("e3nn.o3._tensor_product", package=True)
    e3nn_o3_tp_inst = add_module("e3nn.o3._tensor_product._instruction")
    e3nn_o3_tp_main = add_module("e3nn.o3._tensor_product._tensor_product")
    e3nn_o3_tp_sub = add_module("e3nn.o3._tensor_product._sub")
    e3nn_nn = add_module("e3nn.nn", package=True)
    e3nn_nn_fc = add_module("e3nn.nn._fc")
    e3nn_nn_activation = add_module("e3nn.nn._activation")
    e3nn_math = add_module("e3nn.math", package=True)
    e3nn_math_norm = add_module("e3nn.math._normalize_activation")
    e3nn_util = add_module("e3nn.util", package=True)
    e3nn_util_jit = add_module("e3nn.util.jit")
    e3nn_util_codegen = add_module("e3nn.util.codegen")

    e3nn.get_optimization_defaults = get_optimization_defaults
    e3nn.set_optimization_defaults = set_optimization_defaults
    e3nn.o3 = e3nn_o3
    e3nn.nn = e3nn_nn
    e3nn.math = e3nn_math
    e3nn.util = e3nn_util

    e3nn_o3.Irrep = _IRREP_TYPE
    e3nn_o3.Irreps = _IRREPS_TYPE
    e3nn_o3.Linear = _legacy_module_type("Linear", "e3nn.o3._linear")
    e3nn_o3.TensorProduct = _legacy_module_type(
        "TensorProduct",
        "e3nn.o3._tensor_product._tensor_product",
    )
    e3nn_o3.FullyConnectedTensorProduct = _legacy_module_type(
        "FullyConnectedTensorProduct",
        "e3nn.o3._tensor_product._sub",
    )
    e3nn_o3.SphericalHarmonics = _legacy_module_type(
        "SphericalHarmonics",
        "e3nn.o3._spherical_harmonics",
    )

    e3nn_o3_irreps.Irrep = _IRREP_TYPE
    e3nn_o3_irreps.Irreps = _IRREPS_TYPE
    e3nn_o3_irreps._MulIr = _MUL_IRREP_TYPE

    e3nn_o3_linear.Linear = e3nn_o3.Linear
    e3nn_o3_linear.Instruction = linear_instruction
    e3nn_o3_sh.SphericalHarmonics = e3nn_o3.SphericalHarmonics
    e3nn_o3_tp_inst.Instruction = tp_instruction
    e3nn_o3_tp_main.TensorProduct = e3nn_o3.TensorProduct
    e3nn_o3_tp_sub.FullyConnectedTensorProduct = e3nn_o3.FullyConnectedTensorProduct

    e3nn_nn_fc.FullyConnectedNet = _legacy_module_type(
        "FullyConnectedNet",
        "e3nn.nn._fc",
    )
    e3nn_nn_fc._Layer = _legacy_module_type("_Layer", "e3nn.nn._fc")
    e3nn_nn_activation.Activation = _legacy_module_type(
        "Activation",
        "e3nn.nn._activation",
    )
    e3nn_nn.FullyConnectedNet = e3nn_nn_fc.FullyConnectedNet
    e3nn_nn._Layer = e3nn_nn_fc._Layer
    e3nn_nn.Activation = e3nn_nn_activation.Activation

    e3nn_math.normalize2mom = normalize2mom
    e3nn_math_norm.normalize2mom = normalize2mom

    def compile_mode(_mode: str):
        def decorator(obj):
            return obj

        return decorator

    e3nn_util.compile_mode = compile_mode
    e3nn_util_jit.compile_mode = compile_mode
    e3nn_util_codegen.CodeGenMixin = _LegacyCodeGenMixin

    mace = add_module("mace", package=True)
    mace_modules = add_module("mace.modules", package=True)
    mace_blocks = add_module("mace.modules.blocks")
    mace_models = add_module("mace.modules.models")
    mace_radial = add_module("mace.modules.radial")
    mace_irreps_tools = add_module("mace.modules.irreps_tools")
    mace_symmetric = add_module("mace.modules.symmetric_contraction")

    mace.modules = mace_modules
    legacy_block_names = (
        "AtomicEnergiesBlock",
        "EquivariantProductBasisBlock",
        "LinearNodeEmbeddingBlock",
        "LinearReadoutBlock",
        "NonLinearReadoutBlock",
        "RadialEmbeddingBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
        "ScaleShiftBlock",
    )
    for name in legacy_block_names:
        setattr(mace_blocks, name, _legacy_module_type(name, "mace.modules.blocks"))

    mace_models.ScaleShiftMACE = _legacy_module_type(
        "ScaleShiftMACE",
        "mace.modules.models",
    )
    mace_models.MACE = _legacy_module_type("MACE", "mace.modules.models")

    for name in ("AgnesiTransform", "BesselBasis", "PolynomialCutoff", "ZBLBasis"):
        setattr(mace_radial, name, _legacy_module_type(name, "mace.modules.radial"))

    mace_irreps_tools.reshape_irreps = _legacy_module_type(
        "reshape_irreps",
        "mace.modules.irreps_tools",
    )
    mace_symmetric.SymmetricContraction = _legacy_module_type(
        "SymmetricContraction",
        "mace.modules.symmetric_contraction",
    )
    mace_symmetric.Contraction = _legacy_module_type(
        "Contraction",
        "mace.modules.symmetric_contraction",
    )

    return modules


def _build_local_namespace_aliases() -> dict[str, types.ModuleType]:
    aliases: dict[str, types.ModuleType] = {}
    package_roots = (
        ("mace_torch", "mace_model.torch"),
        ("mace_core", "mace_model.core"),
    )
    for legacy_root, current_root in package_roots:
        root_module = importlib.import_module(current_root)
        aliases[legacy_root] = root_module
        for module_info in pkgutil.walk_packages(
            root_module.__path__,
            prefix=f"{current_root}.",
        ):
            current_name = module_info.name
            legacy_name = current_name.replace(current_root, legacy_root, 1)
            aliases[legacy_name] = importlib.import_module(current_name)
    return aliases


@contextlib.contextmanager
def legacy_checkpoint_imports() -> Iterator[None]:
    temp_modules = _build_legacy_imports()
    temp_modules.update(_build_local_namespace_aliases())
    previous = {name: sys.modules.get(name) for name in temp_modules}
    sys.modules.update(temp_modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_legacy_torch_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
):
    add_safe_globals([slice])
    with legacy_checkpoint_imports():
        model = torch.load(
            Path(path).expanduser().resolve(),
            map_location=map_location,
            weights_only=False,
        )
    if isinstance(model, torch.nn.Module):
        return model.eval()
    raise ValueError(f"Legacy checkpoint at {path} did not contain a Torch module.")


__all__ = [
    "SUPPORTED_FOUNDATION_SOURCES",
    "MACE_MP_URLS",
    "get_cache_dir",
    "get_mace_mp_names",
    "legacy_checkpoint_imports",
    "load_legacy_torch_model",
    "resolve_foundation_checkpoint",
]
