from __future__ import annotations

import importlib.util
import numpy as np
import pytest
import sys
import torch
import types
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from mace_model.torch.modules.radial import BesselBasis as TorchBesselBasis
from mace_model.torch.modules.radial import RadialMLP as TorchRadialMLP
from mace_model.torch.modules.radial import ZBLBasis as TorchZBLBasis

_LOCAL_JAX_RADIAL = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mace_model"
    / "jax"
    / "modules"
    / "radial.py"
)
_LOCAL_JAX_ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mace_model"
    / "jax"
    / "adapters"
    / "nnx"
    / "torch.py"
)
_LOCAL_JAX_ROOT = _LOCAL_JAX_RADIAL.parent.parent
_LOCAL_JAX_MODULES = _LOCAL_JAX_RADIAL.parent
_ALIAS_ROOT = "mace_local_jax_radial"
_ALIAS_MODULES = f"{_ALIAS_ROOT}.modules"


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if _ALIAS_ROOT not in sys.modules:
    root_pkg = types.ModuleType(_ALIAS_ROOT)
    root_pkg.__path__ = [str(_LOCAL_JAX_ROOT)]  # type: ignore[attr-defined]
    sys.modules[_ALIAS_ROOT] = root_pkg

if _ALIAS_MODULES not in sys.modules:
    modules_pkg = types.ModuleType(_ALIAS_MODULES)
    modules_pkg.__path__ = [str(_LOCAL_JAX_MODULES)]  # type: ignore[attr-defined]
    sys.modules[_ALIAS_MODULES] = modules_pkg

_LOCAL_JAX_MODULE = _load_local_module(f"{_ALIAS_MODULES}.radial", _LOCAL_JAX_RADIAL)
_LOCAL_ADAPTER_MODULE = _load_local_module(
    f"{_ALIAS_ROOT}.nnx_torch_adapter",
    _LOCAL_JAX_ADAPTER,
)
init_from_torch = _LOCAL_ADAPTER_MODULE.init_from_torch

JaxBesselBasis = _LOCAL_JAX_MODULE.BesselBasis
JaxRadialMLP = _LOCAL_JAX_MODULE.RadialMLP
JaxZBLBasis = _LOCAL_JAX_MODULE.ZBLBasis


def _to_numpy(value):
    return np.asarray(value.array if hasattr(value, "array") else value)


def test_torch_and_jax_bessel_match_after_weight_transfer():
    rng = np.random.default_rng(2)
    torch_model = TorchBesselBasis(r_max=5.0, num_basis=6, trainable=True).float()
    jax_model = JaxBesselBasis(
        r_max=5.0,
        num_basis=6,
        trainable=True,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.uniform(0.1, 4.9, size=(8, 1)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_radial_mlp_match_after_weight_transfer():
    rng = np.random.default_rng(5)
    channels = [4, 8, 16]
    torch_model = TorchRadialMLP(channels).float()
    jax_model = JaxRadialMLP(channels, rngs=nnx.Rngs(0))
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(7, channels[0])).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_zbl_match_after_weight_transfer():
    x_np = np.linspace(0.4, 2.0, 8, dtype=np.float32)[:, None]
    node_attrs_np = np.eye(3, dtype=np.float32)[np.array([0, 1, 2, 0, 1])]
    edge_index_np = np.array(
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 2, 3, 4]],
        dtype=np.int32,
    )
    atomic_numbers_np = np.array([1, 6, 8], dtype=np.int32)

    torch_model = TorchZBLBasis(p=6).float()
    jax_model = JaxZBLBasis(p=6, rngs=nnx.Rngs(0))
    jax_model, _ = init_from_torch(jax_model, torch_model)

    out_torch = torch_model(
        torch.tensor(x_np),
        torch.tensor(node_attrs_np),
        torch.tensor(edge_index_np),
        torch.tensor(atomic_numbers_np),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(x_np),
        jnp.asarray(node_attrs_np),
        jnp.asarray(edge_index_np),
        jnp.asarray(atomic_numbers_np),
    )
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
