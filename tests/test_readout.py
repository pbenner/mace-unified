from __future__ import annotations

import importlib.util
import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
import torch
import types
from pathlib import Path

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from flax import nnx
from mace_model.jax.adapters.e3nn import Irreps
from mace_model.torch.adapters.e3nn import o3
from mace_model.jax.modules.blocks import NonLinearReadoutBlock as JaxReferenceReadout

from mace_model.torch.modules.blocks import NonLinearReadoutBlock as TorchLocalReadout

_LOCAL_JAX_BLOCKS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mace_model"
    / "jax"
    / "modules"
    / "blocks.py"
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
_LOCAL_JAX_BACKENDS = _LOCAL_JAX_BLOCKS.with_name("backends.py")
_LOCAL_JAX_ROOT = _LOCAL_JAX_BLOCKS.parent.parent
_LOCAL_JAX_MODULES = _LOCAL_JAX_BLOCKS.parent
_ALIAS_ROOT = "mace_local_jax"
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

_load_local_module(
    f"{_ALIAS_MODULES}.backends",
    _LOCAL_JAX_BACKENDS,
)
_LOCAL_JAX_MODULE = _load_local_module(
    f"{_ALIAS_MODULES}.blocks",
    _LOCAL_JAX_BLOCKS,
)
_LOCAL_ADAPTER_MODULE = _load_local_module(
    f"{_ALIAS_ROOT}.nnx_torch_adapter",
    _LOCAL_JAX_ADAPTER,
)
init_from_torch = _LOCAL_ADAPTER_MODULE.init_from_torch

JaxLocalReadout = _LOCAL_JAX_MODULE.NonLinearReadoutBlock


def _to_numpy(x):
    return np.asarray(x.array if hasattr(x, "array") else x)


def _make_jax_input(array: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(array)


def _apply_with_layout(graphdef, state, x_ir, *, heads=None):
    with cue.assume(cue.O3, cue.mul_ir):
        return graphdef.apply(state)(x_ir, heads=heads)


@pytest.mark.parametrize("num_heads", [1, 2])
def test_jax_matches_local_jax_reference(num_heads: int):
    rng = np.random.default_rng(1)

    irreps_in = Irreps("8x0e")
    mlp_irreps = Irreps("6x0e")
    irrep_out = Irreps("2x0e")

    donor_torch = TorchLocalReadout(
        irreps_in=o3.Irreps(str(irreps_in)),
        MLP_irreps=o3.Irreps(str(mlp_irreps)),
        gate=torch.nn.functional.silu,
        irrep_out=o3.Irreps(str(irrep_out)),
        num_heads=num_heads,
    ).float()

    ref_jax = JaxReferenceReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        irrep_out=irrep_out,
        num_heads=num_heads,
        rngs=nnx.Rngs(0),
    )
    local_jax = JaxLocalReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        irrep_out=irrep_out,
        num_heads=num_heads,
        rngs=nnx.Rngs(0),
    )

    ref_jax, _ = init_from_torch(ref_jax, donor_torch)
    local_jax, _ = init_from_torch(local_jax, donor_torch)

    x_np = rng.normal(size=(5, irreps_in.dim)).astype(np.float32)
    x_jax = _make_jax_input(x_np)
    heads = None
    if num_heads > 1:
        heads = jnp.asarray(
            rng.integers(0, num_heads, size=(x_np.shape[0],)),
            dtype=jnp.int32,
        )

    graph_ref, state_ref = nnx.split(ref_jax)
    graph_uni, state_uni = nnx.split(local_jax)

    out_ref, _ = _apply_with_layout(graph_ref, state_ref, x_jax, heads=heads)
    out_uni, _ = _apply_with_layout(graph_uni, state_uni, x_jax, heads=heads)

    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_match_after_weight_transfer():
    rng = np.random.default_rng(2)

    irreps_in_torch = o3.Irreps("8x0e")
    mlp_irreps_torch = o3.Irreps("6x0e")
    irrep_out_torch = o3.Irreps("2x0e")

    irreps_in_jax = Irreps(str(irreps_in_torch))
    mlp_irreps_jax = Irreps(str(mlp_irreps_torch))
    irrep_out_jax = Irreps(str(irrep_out_torch))

    num_heads = 2

    local_torch = TorchLocalReadout(
        irreps_in=irreps_in_torch,
        MLP_irreps=mlp_irreps_torch,
        gate=torch.nn.functional.silu,
        irrep_out=irrep_out_torch,
        num_heads=num_heads,
    ).float()
    local_jax = JaxLocalReadout(
        irreps_in=irreps_in_jax,
        MLP_irreps=mlp_irreps_jax,
        gate=jax.nn.silu,
        irrep_out=irrep_out_jax,
        num_heads=num_heads,
        rngs=nnx.Rngs(0),
    )
    local_jax, _ = init_from_torch(local_jax, local_torch)

    x_np = rng.normal(size=(7, irreps_in_torch.dim)).astype(np.float32)
    heads_np = rng.integers(0, num_heads, size=(x_np.shape[0],))

    out_torch = local_torch(
        torch.tensor(x_np),
        heads=torch.tensor(heads_np, dtype=torch.int64),
    )

    x_jax = _make_jax_input(x_np)
    graphdef, state = nnx.split(local_jax)
    out_jax, _ = _apply_with_layout(
        graphdef,
        state,
        x_jax,
        heads=jnp.asarray(heads_np, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
