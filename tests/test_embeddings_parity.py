from __future__ import annotations

import importlib.util
import jax.numpy as jnp
import numpy as np
import pytest
import sys
import torch
import types
from pathlib import Path
from flax import nnx

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from mace_model.torch.modules.embeddings import GenericJointEmbedding as TorchEmbedding

_LOCAL_JAX_EMBEDDINGS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mace_model"
    / "jax"
    / "modules"
    / "embeddings.py"
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
_LOCAL_JAX_BACKENDS = _LOCAL_JAX_EMBEDDINGS.with_name("backends.py")
_LOCAL_JAX_ROOT = _LOCAL_JAX_EMBEDDINGS.parent.parent
_LOCAL_JAX_MODULES = _LOCAL_JAX_EMBEDDINGS.parent
_ALIAS_ROOT = "mace_local_jax_embeddings"
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

_load_local_module(f"{_ALIAS_MODULES}.backends", _LOCAL_JAX_BACKENDS)
_LOCAL_JAX_MODULE = _load_local_module(
    f"{_ALIAS_MODULES}.embeddings",
    _LOCAL_JAX_EMBEDDINGS,
)
_LOCAL_ADAPTER_MODULE = _load_local_module(
    f"{_ALIAS_ROOT}.nnx_torch_adapter",
    _LOCAL_JAX_ADAPTER,
)
init_from_torch = _LOCAL_ADAPTER_MODULE.init_from_torch
JaxEmbedding = _LOCAL_JAX_MODULE.GenericJointEmbedding


def _embedding_specs() -> dict[str, dict[str, object]]:
    return {
        "cat_node": {
            "type": "categorical",
            "per": "node",
            "emb_dim": 4,
            "num_classes": 5,
        },
        "cat_graph": {
            "type": "categorical",
            "per": "graph",
            "emb_dim": 2,
            "num_classes": 3,
        },
        "cont_node": {
            "type": "continuous",
            "per": "node",
            "emb_dim": 3,
            "in_dim": 2,
            "use_bias": True,
        },
        "cont_graph": {
            "type": "continuous",
            "per": "graph",
            "emb_dim": 2,
            "in_dim": 1,
            "use_bias": False,
        },
    }


def _make_inputs():
    rng = np.random.default_rng(0)
    num_nodes = 6
    num_graphs = 2
    batch_np = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    specs = _embedding_specs()
    features_np = {
        "cat_node": rng.integers(
            0, specs["cat_node"]["num_classes"], size=(num_nodes, 1)
        ),
        "cat_graph": rng.integers(
            0, specs["cat_graph"]["num_classes"], size=(num_graphs,)
        ),
        "cont_node": rng.standard_normal(
            (num_nodes, specs["cont_node"]["in_dim"]),
            dtype=np.float32,
        ),
        "cont_graph": rng.standard_normal(num_graphs, dtype=np.float32),
    }
    return batch_np, features_np


def _torch_features(features_np):
    return {
        "cat_node": torch.tensor(features_np["cat_node"], dtype=torch.long),
        "cat_graph": torch.tensor(features_np["cat_graph"], dtype=torch.long),
        "cont_node": torch.tensor(features_np["cont_node"], dtype=torch.float32),
        "cont_graph": torch.tensor(features_np["cont_graph"], dtype=torch.float32),
    }


def _jax_features(features_np):
    return {
        "cat_node": jnp.asarray(features_np["cat_node"], dtype=jnp.int32),
        "cat_graph": jnp.asarray(features_np["cat_graph"], dtype=jnp.int32),
        "cont_node": jnp.asarray(features_np["cont_node"], dtype=jnp.float32),
        "cont_graph": jnp.asarray(features_np["cont_graph"], dtype=jnp.float32),
    }


def _to_numpy(value):
    return np.asarray(value.array if hasattr(value, "array") else value)


def test_torch_and_jax_embedding_match_after_weight_transfer():
    specs = _embedding_specs()
    batch_np, features_np = _make_inputs()

    torch_model = TorchEmbedding(
        base_dim=5,
        embedding_specs=specs,
        out_dim=7,
    ).float()
    jax_model = JaxEmbedding(
        base_dim=5,
        embedding_specs=specs,
        out_dim=7,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    out_torch = torch_model(
        torch.tensor(batch_np, dtype=torch.long),
        _torch_features(features_np),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(batch_np, dtype=jnp.int32),
        _jax_features(features_np),
    )

    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
