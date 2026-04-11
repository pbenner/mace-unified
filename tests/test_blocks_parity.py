from __future__ import annotations

import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys
import torch
import types
from pathlib import Path
from torch.serialization import add_safe_globals

add_safe_globals([slice])

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from flax import nnx
from mace_model.jax.adapters.e3nn import Irreps, IrrepsArray
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.adapters.e3nn import o3 as torch_reference_o3
from mace_model.jax.modules.blocks import (
    AtomicEnergiesBlock as JaxReferenceAtomicEnergies,
)
from mace_model.jax.modules.blocks import (
    EquivariantProductBasisBlock as JaxReferenceEquivariantProductBasis,
)
from mace_model.jax.modules.backends import _ReshapeIrreps as jax_reshape_irreps
from mace_model.jax.modules.blocks import (
    LinearDipolePolarReadoutBlock as JaxReferenceLinearDipolePolarReadout,
)
from mace_model.jax.modules.blocks import (
    LinearDipoleReadoutBlock as JaxReferenceLinearDipoleReadout,
)
from mace_model.jax.modules.blocks import (
    LinearNodeEmbeddingBlock as JaxReferenceLinearNodeEmbedding,
)
from mace_model.jax.modules.blocks import (
    LinearReadoutBlock as JaxReferenceLinearReadout,
)
from mace_model.jax.modules.blocks import (
    NonLinearDipolePolarReadoutBlock as JaxReferenceNonLinearDipolePolarReadout,
)
from mace_model.jax.modules.blocks import (
    NonLinearBiasReadoutBlock as JaxReferenceNonLinearBiasReadout,
)
from mace_model.jax.modules.blocks import (
    RadialEmbeddingBlock as JaxReferenceRadialEmbedding,
)
from mace_model.jax.modules.blocks import ScaleShiftBlock as JaxReferenceScaleShift
from mace_model.torch.modules.blocks import (
    AtomicEnergiesBlock as TorchLocalAtomicEnergies,
)
from mace_model.torch.modules.blocks import (
    EquivariantProductBasisBlock as TorchLocalEquivariantProductBasis,
)
from mace_model.torch.modules.blocks import (
    LinearDipolePolarReadoutBlock as TorchLocalLinearDipolePolarReadout,
)
from mace_model.torch.modules.blocks import (
    LinearDipoleReadoutBlock as TorchLocalLinearDipoleReadout,
)
from mace_model.torch.modules.blocks import (
    LinearNodeEmbeddingBlock as TorchLocalLinearNodeEmbedding,
)
from mace_model.torch.modules.blocks import (
    LinearReadoutBlock as TorchLocalLinearReadout,
)
from mace_model.torch.modules.blocks import (
    NonLinearDipolePolarReadoutBlock as TorchLocalNonLinearDipolePolarReadout,
)
from mace_model.torch.modules.blocks import (
    NonLinearDipoleReadoutBlock as TorchLocalNonLinearDipoleReadout,
)
from mace_model.torch.modules.blocks import (
    NonLinearBiasReadoutBlock as TorchLocalNonLinearBiasReadout,
)
from mace_model.torch.modules.blocks import (
    RadialEmbeddingBlock as TorchLocalRadialEmbedding,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticAttResidualInteractionBlock as TorchLocalRealAgnosticAttResidualInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticDensityInteractionBlock as TorchLocalRealAgnosticDensityInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock as TorchLocalRealAgnosticDensityResidualInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticInteractionBlock as TorchLocalRealAgnosticInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticResidualInteractionBlock as TorchLocalRealAgnosticResidualInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticResidualNonLinearInteractionBlock as TorchLocalRealAgnosticResidualNonLinearInteraction,
)
from mace_model.torch.modules.blocks import ScaleShiftBlock as TorchLocalScaleShift
from mace_model.torch.modules.backends import _ReshapeIrreps as torch_reshape_irreps

TorchReferenceAtomicEnergies = TorchLocalAtomicEnergies
TorchReferenceEquivariantProductBasis = TorchLocalEquivariantProductBasis
TorchReferenceLinearDipolePolarReadout = TorchLocalLinearDipolePolarReadout
TorchReferenceLinearDipoleReadout = TorchLocalLinearDipoleReadout
TorchReferenceLinearNodeEmbedding = TorchLocalLinearNodeEmbedding
TorchReferenceLinearReadout = TorchLocalLinearReadout
TorchReferenceNonLinearDipolePolarReadout = TorchLocalNonLinearDipolePolarReadout
TorchReferenceNonLinearDipoleReadout = TorchLocalNonLinearDipoleReadout
TorchReferenceNonLinearBiasReadout = TorchLocalNonLinearBiasReadout
TorchReferenceRadialEmbedding = TorchLocalRadialEmbedding
TorchReferenceRealAgnosticAttResidualInteraction = (
    TorchLocalRealAgnosticAttResidualInteraction
)
TorchReferenceRealAgnosticDensityInteraction = TorchLocalRealAgnosticDensityInteraction
TorchReferenceRealAgnosticDensityResidualInteraction = (
    TorchLocalRealAgnosticDensityResidualInteraction
)
TorchReferenceRealAgnosticInteraction = TorchLocalRealAgnosticInteraction
TorchReferenceRealAgnosticResidualInteraction = (
    TorchLocalRealAgnosticResidualInteraction
)
TorchReferenceRealAgnosticResidualNonLinearInteraction = (
    TorchLocalRealAgnosticResidualNonLinearInteraction
)
TorchReferenceScaleShift = TorchLocalScaleShift

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
_ALIAS_ROOT = "mace_local_jax_blocks_parity"
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

JaxLocalLinearNodeEmbedding = _LOCAL_JAX_MODULE.LinearNodeEmbeddingBlock
JaxLocalLinearReadout = _LOCAL_JAX_MODULE.LinearReadoutBlock
JaxLocalLinearDipoleReadout = _LOCAL_JAX_MODULE.LinearDipoleReadoutBlock
JaxLocalNonLinearDipoleReadout = _LOCAL_JAX_MODULE.NonLinearDipoleReadoutBlock
JaxLocalLinearDipolePolarReadout = _LOCAL_JAX_MODULE.LinearDipolePolarReadoutBlock
JaxLocalNonLinearDipolePolarReadout = _LOCAL_JAX_MODULE.NonLinearDipolePolarReadoutBlock
JaxLocalNonLinearBiasReadout = _LOCAL_JAX_MODULE.NonLinearBiasReadoutBlock
JaxLocalAtomicEnergies = _LOCAL_JAX_MODULE.AtomicEnergiesBlock
JaxLocalRadialEmbedding = _LOCAL_JAX_MODULE.RadialEmbeddingBlock
JaxLocalScaleShift = _LOCAL_JAX_MODULE.ScaleShiftBlock
JaxLocalEquivariantProductBasis = _LOCAL_JAX_MODULE.EquivariantProductBasisBlock
JaxLocalRealAgnosticInteraction = _LOCAL_JAX_MODULE.RealAgnosticInteractionBlock
JaxLocalRealAgnosticResidualInteraction = (
    _LOCAL_JAX_MODULE.RealAgnosticResidualInteractionBlock
)
JaxLocalRealAgnosticDensityInteraction = (
    _LOCAL_JAX_MODULE.RealAgnosticDensityInteractionBlock
)
JaxLocalRealAgnosticDensityResidualInteraction = (
    _LOCAL_JAX_MODULE.RealAgnosticDensityResidualInteractionBlock
)
JaxLocalRealAgnosticAttResidualInteraction = (
    _LOCAL_JAX_MODULE.RealAgnosticAttResidualInteractionBlock
)
JaxLocalRealAgnosticResidualNonLinearInteraction = (
    _LOCAL_JAX_MODULE.RealAgnosticResidualNonLinearInteractionBlock
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x.array if hasattr(x, "array") else x)


def _to_reference_irreps(irreps):
    return torch_reference_o3.Irreps(str(irreps))


def _as_irreps_array(irreps: Irreps, array: jnp.ndarray) -> IrrepsArray:
    if hasattr(IrrepsArray, "from_array"):
        return IrrepsArray.from_array(irreps, array)
    return IrrepsArray(irreps, array)


def _assert_optional_allclose(a, b, *, rtol=1e-6, atol=1e-6):
    if a is None or b is None:
        assert a is None and b is None
        return
    np.testing.assert_allclose(_to_numpy(a), _to_numpy(b), rtol=rtol, atol=atol)


def _make_radial_inputs(
    rng: np.random.Generator,
    *,
    n_nodes: int = 6,
    n_edges: int = 10,
    num_elements: int = 4,
):
    edge_lengths = rng.uniform(0.2, 2.5, size=(n_edges, 1)).astype(np.float32)
    node_species = rng.integers(0, num_elements, size=(n_nodes,))
    node_attrs = np.eye(num_elements, dtype=np.float32)[node_species]
    edge_index = np.vstack(
        [
            rng.integers(0, n_nodes, size=(n_edges,)),
            rng.integers(0, n_nodes, size=(n_edges,)),
        ]
    ).astype(np.int64)
    atomic_numbers = np.asarray([1, 6, 8, 14][:num_elements], dtype=np.int64)
    node_attrs_index = node_species.astype(np.int32)
    return edge_lengths, node_attrs, edge_index, atomic_numbers, node_attrs_index


def _make_equivariant_product_inputs(
    rng: np.random.Generator,
    *,
    n_nodes: int,
    node_dim: int,
    target_dim: int,
    num_elements: int,
    use_sc: bool,
):
    node_feats = rng.normal(size=(n_nodes, node_dim)).astype(np.float32)
    node_species = rng.integers(0, num_elements, size=(n_nodes,))
    node_attrs = np.eye(num_elements, dtype=np.float32)[node_species]
    sc = None
    if use_sc:
        sc = rng.normal(size=(n_nodes, target_dim)).astype(np.float32)
    return node_feats, node_attrs, node_species.astype(np.int32), sc


def _make_interaction_inputs(
    rng: np.random.Generator,
    *,
    n_nodes: int,
    n_edges: int,
    num_elements: int,
    node_feat_dim: int,
    edge_attr_dim: int,
    edge_feat_dim: int,
):
    node_species = rng.integers(0, num_elements, size=(n_nodes,))
    node_attrs = np.eye(num_elements, dtype=np.float32)[node_species]
    node_feats = rng.normal(size=(n_nodes, node_feat_dim)).astype(np.float32)
    edge_attrs = rng.normal(size=(n_edges, edge_attr_dim)).astype(np.float32)
    edge_feats = rng.normal(size=(n_edges, edge_feat_dim)).astype(np.float32)
    edge_index = np.vstack(
        [
            rng.integers(0, n_nodes, size=(n_edges,)),
            rng.integers(0, n_nodes, size=(n_edges,)),
        ]
    ).astype(np.int64)
    return node_attrs, node_feats, edge_attrs, edge_feats, edge_index


RADIAL_CASES = [
    ("bessel", "None", True),
    ("bessel", "None", False),
    ("gaussian", "Agnesi", True),
    ("chebyshev", "Soft", True),
]

TORCH_INTERACTION_CASES = [
    (
        TorchReferenceRealAgnosticInteraction,
        TorchLocalRealAgnosticInteraction,
    ),
    (
        TorchReferenceRealAgnosticResidualInteraction,
        TorchLocalRealAgnosticResidualInteraction,
    ),
    (
        TorchReferenceRealAgnosticDensityInteraction,
        TorchLocalRealAgnosticDensityInteraction,
    ),
    (
        TorchReferenceRealAgnosticDensityResidualInteraction,
        TorchLocalRealAgnosticDensityResidualInteraction,
    ),
    (
        TorchReferenceRealAgnosticAttResidualInteraction,
        TorchLocalRealAgnosticAttResidualInteraction,
    ),
    (
        TorchReferenceRealAgnosticResidualNonLinearInteraction,
        TorchLocalRealAgnosticResidualNonLinearInteraction,
    ),
]

CROSS_BACKEND_JAX_INTERACTION_CASES = [
    (
        TorchLocalRealAgnosticInteraction,
        JaxLocalRealAgnosticInteraction,
    ),
    (
        TorchLocalRealAgnosticResidualInteraction,
        JaxLocalRealAgnosticResidualInteraction,
    ),
    (
        TorchLocalRealAgnosticDensityInteraction,
        JaxLocalRealAgnosticDensityInteraction,
    ),
    (
        TorchLocalRealAgnosticDensityResidualInteraction,
        JaxLocalRealAgnosticDensityResidualInteraction,
    ),
    (
        TorchLocalRealAgnosticAttResidualInteraction,
        JaxLocalRealAgnosticAttResidualInteraction,
    ),
    (
        TorchLocalRealAgnosticResidualNonLinearInteraction,
        JaxLocalRealAgnosticResidualNonLinearInteraction,
    ),
]


@pytest.mark.parametrize("radial_type,distance_transform,apply_cutoff", RADIAL_CASES)
def test_torch_radial_embedding_matches_reference(
    radial_type: str, distance_transform: str, apply_cutoff: bool
):
    rng = np.random.default_rng(80)
    edge_lengths, node_attrs, edge_index, atomic_numbers, _ = _make_radial_inputs(rng)

    reference = TorchReferenceRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
    ).float()
    local = TorchLocalRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
    ).float()

    out_ref = reference(
        torch.tensor(edge_lengths),
        torch.tensor(node_attrs),
        torch.tensor(edge_index, dtype=torch.int64),
        torch.tensor(atomic_numbers, dtype=torch.int64),
    )
    out_uni = local(
        torch.tensor(edge_lengths),
        torch.tensor(node_attrs),
        torch.tensor(edge_index, dtype=torch.int64),
        torch.tensor(atomic_numbers, dtype=torch.int64),
    )

    np.testing.assert_allclose(
        out_uni[0].detach().cpu().numpy(),
        out_ref[0].detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_optional_allclose(out_uni[1], out_ref[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("radial_type,distance_transform,apply_cutoff", RADIAL_CASES)
def test_jax_radial_embedding_matches_reference(
    radial_type: str, distance_transform: str, apply_cutoff: bool
):
    rng = np.random.default_rng(81)
    edge_lengths, node_attrs, edge_index, atomic_numbers, node_attrs_index = (
        _make_radial_inputs(rng)
    )

    reference = JaxReferenceRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
        rngs=nnx.Rngs(0),
    )

    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(
        jnp.asarray(edge_lengths),
        jnp.asarray(node_attrs),
        jnp.asarray(edge_index, dtype=jnp.int32),
        jnp.asarray(atomic_numbers, dtype=jnp.int32),
        node_attrs_index=jnp.asarray(node_attrs_index, dtype=jnp.int32),
    )
    out_uni, _ = graph_uni.apply(state_uni)(
        jnp.asarray(edge_lengths),
        jnp.asarray(node_attrs),
        jnp.asarray(edge_index, dtype=jnp.int32),
        jnp.asarray(atomic_numbers, dtype=jnp.int32),
        node_attrs_index=jnp.asarray(node_attrs_index, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        _to_numpy(out_uni[0]),
        _to_numpy(out_ref[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_optional_allclose(out_uni[1], out_ref[1], rtol=1e-6, atol=1e-6)


def test_torch_and_jax_radial_embedding_match():
    rng = np.random.default_rng(82)
    edge_lengths, node_attrs, edge_index, atomic_numbers, node_attrs_index = (
        _make_radial_inputs(rng)
    )

    torch_model = TorchLocalRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type="bessel",
        distance_transform="None",
        apply_cutoff=False,
    ).float()
    jax_model = JaxLocalRadialEmbedding(
        r_max=3.0,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type="bessel",
        distance_transform="None",
        apply_cutoff=False,
        rngs=nnx.Rngs(0),
    )

    out_torch = torch_model(
        torch.tensor(edge_lengths),
        torch.tensor(node_attrs),
        torch.tensor(edge_index, dtype=torch.int64),
        torch.tensor(atomic_numbers, dtype=torch.int64),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(edge_lengths),
        jnp.asarray(node_attrs),
        jnp.asarray(edge_index, dtype=jnp.int32),
        jnp.asarray(atomic_numbers, dtype=jnp.int32),
        node_attrs_index=jnp.asarray(node_attrs_index, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        _to_numpy(out_jax[0]),
        out_torch[0].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_optional_allclose(out_jax[1], out_torch[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("multi_head", [False, True])
def test_torch_scale_shift_matches_reference(multi_head: bool):
    rng = np.random.default_rng(83)
    x = torch.tensor(rng.normal(size=(9,)).astype(np.float32))
    if multi_head:
        scale = np.asarray([0.5, 1.5], dtype=np.float32)
        shift = np.asarray([-0.2, 0.3], dtype=np.float32)
        head = torch.tensor(rng.integers(0, 2, size=(x.shape[0],)), dtype=torch.int64)
    else:
        scale = 0.75
        shift = -0.1
        head = torch.zeros((x.shape[0],), dtype=torch.int64)

    reference = TorchReferenceScaleShift(scale=scale, shift=shift).float()
    local = TorchLocalScaleShift(scale=scale, shift=shift).float()
    local.load_state_dict(reference.state_dict(), strict=True)

    out_ref = reference(x, head)
    out_uni = local(x, head)
    np.testing.assert_allclose(
        out_uni.detach().cpu().numpy(),
        out_ref.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("multi_head", [False, True])
def test_jax_scale_shift_matches_reference(multi_head: bool):
    rng = np.random.default_rng(84)
    x_np = rng.normal(size=(9,)).astype(np.float32)
    if multi_head:
        scale = np.asarray([0.5, 1.5], dtype=np.float32)
        shift = np.asarray([-0.2, 0.3], dtype=np.float32)
        head_np = rng.integers(0, 2, size=(x_np.shape[0],)).astype(np.int32)
    else:
        scale = 0.75
        shift = -0.1
        head_np = np.zeros((x_np.shape[0],), dtype=np.int32)

    reference = JaxReferenceScaleShift(scale=scale, shift=shift)
    local = JaxLocalScaleShift(scale=scale, shift=shift)

    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(jnp.asarray(x_np), jnp.asarray(head_np))
    out_uni, _ = graph_uni.apply(state_uni)(jnp.asarray(x_np), jnp.asarray(head_np))
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_scale_shift_match():
    rng = np.random.default_rng(85)
    x_np = rng.normal(size=(11,)).astype(np.float32)
    head_np = rng.integers(0, 2, size=(x_np.shape[0],)).astype(np.int64)
    scale = np.asarray([0.5, 1.5], dtype=np.float32)
    shift = np.asarray([-0.2, 0.3], dtype=np.float32)

    torch_model = TorchLocalScaleShift(scale=scale, shift=shift)
    jax_model = JaxLocalScaleShift(scale=scale, shift=shift)

    out_torch = torch_model(
        torch.tensor(x_np),
        torch.tensor(head_np, dtype=torch.int64),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(x_np),
        jnp.asarray(head_np, dtype=jnp.int32),
    )
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_equivariant_product_basis_match_after_weight_transfer():
    rng = np.random.default_rng(88)
    node_irreps_torch = o3.Irreps("4x0e + 4x1o")
    target_irreps_torch = o3.Irreps("4x0e + 4x1o")
    node_irreps_jax = Irreps(str(node_irreps_torch))
    target_irreps_jax = Irreps(str(target_irreps_torch))

    node_feats, node_attrs, node_attrs_index, sc = _make_equivariant_product_inputs(
        rng,
        n_nodes=8,
        node_dim=node_irreps_torch.dim,
        target_dim=target_irreps_torch.dim,
        num_elements=4,
        use_sc=True,
    )

    torch_model = TorchLocalEquivariantProductBasis(
        node_feats_irreps=node_irreps_torch,
        target_irreps=target_irreps_torch,
        correlation=2,
        use_sc=True,
        num_elements=4,
        use_agnostic_product=False,
        use_reduced_cg=None,
        cueq_config=None,
        oeq_config=None,
    ).float()
    jax_model = JaxLocalEquivariantProductBasis(
        node_feats_irreps=node_irreps_jax,
        target_irreps=target_irreps_jax,
        correlation=2,
        use_sc=True,
        num_elements=4,
        use_agnostic_product=False,
        use_reduced_cg=None,
        cueq_config=None,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    out_torch = torch_model(
        torch_reshape_irreps(_to_reference_irreps(node_irreps_torch))(
            torch.tensor(node_feats)
        ),
        torch.tensor(sc),
        torch.tensor(node_attrs),
    )

    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jax_reshape_irreps(node_irreps_jax)(jnp.asarray(node_feats)),
        jnp.asarray(sc),
        jnp.asarray(node_attrs),
        node_attrs_index=jnp.asarray(node_attrs_index, dtype=jnp.int32),
    )
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "torch_local_cls,jax_local_cls",
    CROSS_BACKEND_JAX_INTERACTION_CASES,
    ids=[torch_cls.__name__ for torch_cls, _ in CROSS_BACKEND_JAX_INTERACTION_CASES],
)
def test_torch_and_jax_interaction_blocks_match_after_weight_transfer(
    torch_local_cls,
    jax_local_cls,
):
    rng = np.random.default_rng(89)
    num_elements = 4
    mul = 2
    node_attrs_irreps_torch = o3.Irreps(f"{num_elements}x0e")
    node_feats_irreps_torch = o3.Irreps(f"{mul}x0e + {mul}x1o")
    edge_attrs_irreps_torch = o3.Irreps("1x0e + 1x1o + 1x2e")
    edge_feats_irreps_torch = o3.Irreps("5x0e")
    target_irreps_torch = o3.Irreps(f"{mul}x0e + {mul}x1o")
    hidden_irreps_torch = o3.Irreps(f"{mul}x0e + {mul}x1o")

    node_attrs_irreps_jax = Irreps(str(node_attrs_irreps_torch))
    node_feats_irreps_jax = Irreps(str(node_feats_irreps_torch))
    edge_attrs_irreps_jax = Irreps(str(edge_attrs_irreps_torch))
    edge_feats_irreps_jax = Irreps(str(edge_feats_irreps_torch))
    target_irreps_jax = Irreps(str(target_irreps_torch))
    hidden_irreps_jax = Irreps(str(hidden_irreps_torch))

    node_attrs, node_feats, edge_attrs, edge_feats, edge_index = (
        _make_interaction_inputs(
            rng,
            n_nodes=6,
            n_edges=14,
            num_elements=num_elements,
            node_feat_dim=node_feats_irreps_torch.dim,
            edge_attr_dim=edge_attrs_irreps_torch.dim,
            edge_feat_dim=edge_feats_irreps_torch.dim,
        )
    )

    torch_model = torch_local_cls(
        node_attrs_irreps=node_attrs_irreps_torch,
        node_feats_irreps=node_feats_irreps_torch,
        edge_attrs_irreps=edge_attrs_irreps_torch,
        edge_feats_irreps=edge_feats_irreps_torch,
        target_irreps=target_irreps_torch,
        hidden_irreps=hidden_irreps_torch,
        avg_num_neighbors=3.0,
        edge_irreps=None,
        radial_MLP=[16, 16],
        cueq_config=None,
        oeq_config=None,
    ).float()
    jax_model = jax_local_cls(
        node_attrs_irreps=node_attrs_irreps_jax,
        node_feats_irreps=node_feats_irreps_jax,
        edge_attrs_irreps=edge_attrs_irreps_jax,
        edge_feats_irreps=edge_feats_irreps_jax,
        target_irreps=target_irreps_jax,
        hidden_irreps=hidden_irreps_jax,
        avg_num_neighbors=3.0,
        edge_irreps=None,
        radial_MLP=[16, 16],
        cueq_config=None,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    out_torch = torch_model(
        torch.tensor(node_attrs),
        torch.tensor(node_feats),
        torch.tensor(edge_attrs),
        torch.tensor(edge_feats),
        torch.tensor(edge_index, dtype=torch.int64),
    )
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        jnp.asarray(node_attrs),
        jnp.asarray(node_feats),
        jnp.asarray(edge_attrs),
        jnp.asarray(edge_feats),
        jnp.asarray(edge_index, dtype=jnp.int32),
    )

    if not isinstance(out_torch, tuple):
        out_torch = (out_torch,)
        out_jax = (out_jax,)
    assert isinstance(out_jax, tuple)
    assert len(out_jax) == len(out_torch)

    for jax_item, torch_item in zip(out_jax, out_torch):
        _assert_optional_allclose(jax_item, torch_item, rtol=1e-6, atol=1e-6)


def test_jax_linear_node_embedding_matches_reference():
    rng = np.random.default_rng(91)
    irreps_in = Irreps("8x0e")
    irreps_out = Irreps("6x0e")

    donor_torch = TorchReferenceLinearNodeEmbedding(
        irreps_in=_to_reference_irreps(irreps_in),
        irreps_out=_to_reference_irreps(irreps_out),
    ).float()
    reference = JaxReferenceLinearNodeEmbedding(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalLinearNodeEmbedding(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(6, irreps_in.dim)).astype(np.float32)
    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(jnp.asarray(x_np))
    out_uni, _ = graph_uni.apply(state_uni)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_linear_node_embedding_match_after_weight_transfer():
    rng = np.random.default_rng(92)
    irreps_in_torch = o3.Irreps("8x0e")
    irreps_out_torch = o3.Irreps("6x0e")
    irreps_in_jax = Irreps(str(irreps_in_torch))
    irreps_out_jax = Irreps(str(irreps_out_torch))

    torch_model = TorchLocalLinearNodeEmbedding(
        irreps_in=irreps_in_torch,
        irreps_out=irreps_out_torch,
    ).float()
    jax_model = JaxLocalLinearNodeEmbedding(
        irreps_in=irreps_in_jax,
        irreps_out=irreps_out_jax,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(7, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("dipole_only", [False, True])
def test_jax_linear_dipole_readout_matches_reference(dipole_only: bool):
    rng = np.random.default_rng(94)
    irreps_in = Irreps("8x0e + 4x1o")

    donor_torch = TorchReferenceLinearDipoleReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        dipole_only=dipole_only,
    ).float()
    reference = JaxReferenceLinearDipoleReadout(
        irreps_in=irreps_in,
        dipole_only=dipole_only,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalLinearDipoleReadout(
        irreps_in=irreps_in,
        dipole_only=dipole_only,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(5, irreps_in.dim)).astype(np.float32)
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x_ir)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_linear_dipole_readout_match_after_weight_transfer():
    rng = np.random.default_rng(95)
    irreps_in_torch = o3.Irreps("8x0e + 4x1o")
    irreps_in_jax = Irreps(str(irreps_in_torch))

    torch_model = TorchLocalLinearDipoleReadout(
        irreps_in=irreps_in_torch,
        dipole_only=False,
    ).float()
    jax_model = JaxLocalLinearDipoleReadout(
        irreps_in=irreps_in_jax,
        dipole_only=False,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(6, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("dipole_only", [False, True])
def test_jax_non_linear_dipole_readout_matches_torch_after_weight_transfer(
    dipole_only: bool,
):
    rng = np.random.default_rng(97)
    irreps_in = Irreps("8x0e + 4x1o")
    if dipole_only:
        mlp_irreps = Irreps("4x1o")
    else:
        mlp_irreps = Irreps("4x0e + 4x1o")

    donor_torch = TorchReferenceNonLinearDipoleReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        MLP_irreps=_to_reference_irreps(mlp_irreps),
        gate=torch.nn.functional.silu,
        dipole_only=dipole_only,
    ).float()
    local = JaxLocalNonLinearDipoleReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        dipole_only=dipole_only,
        rngs=nnx.Rngs(0),
    )
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(5, irreps_in.dim)).astype(np.float32)
    out_torch = donor_torch(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    graph_uni, state_uni = nnx.split(local)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_non_linear_dipole_readout_match_after_weight_transfer():
    rng = np.random.default_rng(98)
    irreps_in_torch = o3.Irreps("8x0e + 4x1o")
    mlp_irreps_torch = o3.Irreps("4x0e + 4x1o")
    irreps_in_jax = Irreps(str(irreps_in_torch))
    mlp_irreps_jax = Irreps(str(mlp_irreps_torch))

    torch_model = TorchLocalNonLinearDipoleReadout(
        irreps_in=irreps_in_torch,
        MLP_irreps=mlp_irreps_torch,
        gate=torch.nn.functional.silu,
        dipole_only=False,
    ).float()
    jax_model = JaxLocalNonLinearDipoleReadout(
        irreps_in=irreps_in_jax,
        MLP_irreps=mlp_irreps_jax,
        gate=jax.nn.silu,
        dipole_only=False,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(6, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_linear_dipole_polar_readout_matches_reference():
    rng = np.random.default_rng(100)
    irreps_in = Irreps("8x0e + 4x1o + 2x2e")

    donor_torch = TorchReferenceLinearDipolePolarReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        use_polarizability=True,
    ).float()
    reference = JaxReferenceLinearDipolePolarReadout(
        irreps_in=irreps_in,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalLinearDipolePolarReadout(
        irreps_in=irreps_in,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(4, irreps_in.dim)).astype(np.float32)
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x_ir)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_linear_dipole_polar_readout_match_after_weight_transfer():
    rng = np.random.default_rng(110)
    irreps_in_torch = o3.Irreps("8x0e + 4x1o + 2x2e")
    irreps_in_jax = Irreps(str(irreps_in_torch))

    torch_model = TorchLocalLinearDipolePolarReadout(
        irreps_in=irreps_in_torch,
        use_polarizability=True,
    ).float()
    jax_model = JaxLocalLinearDipolePolarReadout(
        irreps_in=irreps_in_jax,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(4, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_non_linear_dipole_polar_readout_matches_reference():
    rng = np.random.default_rng(112)
    irreps_in = Irreps("8x0e + 4x1o + 2x2e")
    mlp_irreps = Irreps("4x0e + 4x1o")

    donor_torch = TorchReferenceNonLinearDipolePolarReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        MLP_irreps=_to_reference_irreps(mlp_irreps),
        gate=torch.nn.functional.silu,
        use_polarizability=True,
    ).float()
    reference = JaxReferenceNonLinearDipolePolarReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalNonLinearDipolePolarReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(4, irreps_in.dim)).astype(np.float32)
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x_ir)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_non_linear_dipole_polar_readout_match_after_weight_transfer():
    rng = np.random.default_rng(113)
    irreps_in_torch = o3.Irreps("8x0e + 4x1o + 2x2e")
    mlp_irreps_torch = o3.Irreps("4x0e + 4x1o")
    irreps_in_jax = Irreps(str(irreps_in_torch))
    mlp_irreps_jax = Irreps(str(mlp_irreps_torch))

    torch_model = TorchLocalNonLinearDipolePolarReadout(
        irreps_in=irreps_in_torch,
        MLP_irreps=mlp_irreps_torch,
        gate=torch.nn.functional.silu,
        use_polarizability=True,
    ).float()
    jax_model = JaxLocalNonLinearDipolePolarReadout(
        irreps_in=irreps_in_jax,
        MLP_irreps=mlp_irreps_jax,
        gate=jax.nn.silu,
        use_polarizability=True,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(4, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("with_heads", [False, True])
def test_jax_linear_readout_matches_reference(with_heads: bool):
    rng = np.random.default_rng(102)
    irreps_in = Irreps("8x0e")
    irrep_out = Irreps("2x0e")

    donor_torch = TorchReferenceLinearReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        irrep_out=_to_reference_irreps(irrep_out),
    ).float()
    reference = JaxReferenceLinearReadout(
        irreps_in=irreps_in,
        irrep_out=irrep_out,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalLinearReadout(
        irreps_in=irreps_in,
        irrep_out=irrep_out,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(6, irreps_in.dim)).astype(np.float32)
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    heads = None
    if with_heads:
        heads = jnp.asarray(rng.integers(0, 2, size=(x_np.shape[0],)), dtype=jnp.int32)

    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x_ir, heads=heads)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir, heads=heads)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_linear_readout_match_after_weight_transfer():
    rng = np.random.default_rng(103)
    irreps_in_torch = o3.Irreps("8x0e")
    irrep_out_torch = o3.Irreps("2x0e")

    irreps_in_jax = Irreps(str(irreps_in_torch))
    irrep_out_jax = Irreps(str(irrep_out_torch))

    torch_model = TorchLocalLinearReadout(
        irreps_in=irreps_in_torch,
        irrep_out=irrep_out_torch,
    ).float()
    jax_model = JaxLocalLinearReadout(
        irreps_in=irreps_in_jax,
        irrep_out=irrep_out_jax,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(7, irreps_in_torch.dim)).astype(np.float32)
    heads_np = rng.integers(0, 2, size=(x_np.shape[0],))

    out_torch = torch_model(
        torch.tensor(x_np),
        heads=torch.tensor(heads_np, dtype=torch.int64),
    )
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(
        x_ir,
        heads=jnp.asarray(heads_np, dtype=jnp.int32),
    )
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("num_heads", [1, 2])
def test_jax_non_linear_bias_readout_matches_reference(num_heads: int):
    rng = np.random.default_rng(105)
    irreps_in = Irreps("8x0e")
    mlp_irreps = Irreps("6x0e")
    irrep_out = Irreps("2x0e")

    donor_torch = TorchReferenceNonLinearBiasReadout(
        irreps_in=_to_reference_irreps(irreps_in),
        MLP_irreps=_to_reference_irreps(mlp_irreps),
        gate=torch.nn.functional.silu,
        irrep_out=_to_reference_irreps(irrep_out),
        num_heads=num_heads,
    ).float()

    reference = JaxReferenceNonLinearBiasReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        irrep_out=irrep_out,
        num_heads=num_heads,
        rngs=nnx.Rngs(0),
    )
    local = JaxLocalNonLinearBiasReadout(
        irreps_in=irreps_in,
        MLP_irreps=mlp_irreps,
        gate=jax.nn.silu,
        irrep_out=irrep_out,
        num_heads=num_heads,
        rngs=nnx.Rngs(0),
    )
    reference, _ = init_from_torch(reference, donor_torch)
    local, _ = init_from_torch(local, donor_torch)

    x_np = rng.normal(size=(5, irreps_in.dim)).astype(np.float32)
    x_ir = _as_irreps_array(irreps_in, jnp.asarray(x_np))
    heads = None
    if num_heads > 1:
        heads = jnp.asarray(
            rng.integers(0, num_heads, size=(x_np.shape[0],)),
            dtype=jnp.int32,
        )

    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x_ir, heads=heads)
    out_uni, _ = graph_uni.apply(state_uni)(x_ir, heads=heads)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_non_linear_bias_readout_match_for_single_head():
    rng = np.random.default_rng(106)
    irreps_in_torch = o3.Irreps("8x0e")
    mlp_irreps_torch = o3.Irreps("6x0e")
    irrep_out_torch = o3.Irreps("2x0e")

    irreps_in_jax = Irreps(str(irreps_in_torch))
    mlp_irreps_jax = Irreps(str(mlp_irreps_torch))
    irrep_out_jax = Irreps(str(irrep_out_torch))

    torch_model = TorchLocalNonLinearBiasReadout(
        irreps_in=irreps_in_torch,
        MLP_irreps=mlp_irreps_torch,
        gate=torch.nn.functional.silu,
        irrep_out=irrep_out_torch,
        num_heads=1,
    ).float()
    jax_model = JaxLocalNonLinearBiasReadout(
        irreps_in=irreps_in_jax,
        MLP_irreps=mlp_irreps_jax,
        gate=jax.nn.silu,
        irrep_out=irrep_out_jax,
        num_heads=1,
        rngs=nnx.Rngs(0),
    )
    jax_model, _ = init_from_torch(jax_model, torch_model)

    x_np = rng.normal(size=(7, irreps_in_torch.dim)).astype(np.float32)
    out_torch = torch_model(torch.tensor(x_np))
    x_ir = _as_irreps_array(irreps_in_jax, jnp.asarray(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(x_ir)
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_atomic_energies_matches_reference():
    rng = np.random.default_rng(107)
    atomic_energies = rng.normal(size=(2, 4)).astype(np.float64)
    x = torch.tensor(rng.normal(size=(6, 4)).astype(np.float32))

    reference = TorchReferenceAtomicEnergies(atomic_energies=atomic_energies).float()
    local = TorchLocalAtomicEnergies(atomic_energies=atomic_energies).float()
    out_ref = reference(x)
    out_local = local(x)
    np.testing.assert_allclose(
        out_local.detach().cpu().numpy(),
        out_ref.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_atomic_energies_matches_reference():
    rng = np.random.default_rng(108)
    atomic_energies = rng.normal(size=(2, 4)).astype(np.float32)
    x_np = rng.normal(size=(6, 4)).astype(np.float32)
    x = jnp.asarray(x_np)

    reference = JaxReferenceAtomicEnergies(atomic_energies_init=atomic_energies)
    local = JaxLocalAtomicEnergies(atomic_energies_init=atomic_energies)

    graph_ref, state_ref = nnx.split(reference)
    graph_uni, state_uni = nnx.split(local)
    out_ref, _ = graph_ref.apply(state_ref)(x)
    out_uni, _ = graph_uni.apply(state_uni)(x)
    np.testing.assert_allclose(
        _to_numpy(out_uni),
        _to_numpy(out_ref),
        rtol=1e-6,
        atol=1e-6,
    )


def test_torch_and_jax_atomic_energies_match():
    rng = np.random.default_rng(109)
    atomic_energies = rng.normal(size=(2, 4)).astype(np.float32)
    x_np = rng.normal(size=(6, 4)).astype(np.float32)

    torch_model = TorchLocalAtomicEnergies(atomic_energies=atomic_energies)
    jax_model = JaxLocalAtomicEnergies(atomic_energies_init=atomic_energies)

    out_torch = torch_model(torch.tensor(x_np))
    graphdef, state = nnx.split(jax_model)
    out_jax, _ = graphdef.apply(state)(jnp.asarray(x_np))
    np.testing.assert_allclose(
        _to_numpy(out_jax),
        out_torch.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
