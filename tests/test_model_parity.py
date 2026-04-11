from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ase.build import bulk
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
from mace_model.core.data.neighborhood import get_neighborhood
from mace_model.core.data.utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    config_from_atoms,
)
from mace_model.jax.adapters.e3nn import Irreps
from mace_model.jax.modules.utils import prepare_graph as prepare_graph_jax
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.utils import prepare_graph as prepare_graph_torch
from mace_model.torch.modules.blocks import (
    RealAgnosticInteractionBlock as TorchLocalRealAgnosticInteraction,
)
from mace_model.torch.modules.blocks import (
    RealAgnosticResidualInteractionBlock as TorchLocalRealAgnosticResidualInteraction,
)
from mace_model.torch.modules.models import ScaleShiftMACE as TorchLocalScaleShiftMACE

_LOCAL_JAX_BLOCKS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mace_model"
    / "jax"
    / "modules"
    / "blocks.py"
)
_LOCAL_JAX_MODELS = _LOCAL_JAX_BLOCKS.with_name("models.py")
_LOCAL_JAX_BACKENDS = _LOCAL_JAX_BLOCKS.with_name("backends.py")
_LOCAL_JAX_ROOT = _LOCAL_JAX_BLOCKS.parent.parent
_LOCAL_JAX_MODULES = _LOCAL_JAX_BLOCKS.parent
_ALIAS_ROOT = "mace_local_jax_model_parity"
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
_LOCAL_JAX_BLOCKS_MODULE = _load_local_module(
    f"{_ALIAS_MODULES}.blocks",
    _LOCAL_JAX_BLOCKS,
)
_LOCAL_JAX_MODELS_MODULE = _load_local_module(
    f"{_ALIAS_MODULES}.models",
    _LOCAL_JAX_MODELS,
)

JaxLocalRealAgnosticInteraction = _LOCAL_JAX_BLOCKS_MODULE.RealAgnosticInteractionBlock
JaxLocalRealAgnosticResidualInteraction = (
    _LOCAL_JAX_BLOCKS_MODULE.RealAgnosticResidualInteractionBlock
)
JaxLocalScaleShiftMACE = _LOCAL_JAX_MODELS_MODULE.ScaleShiftMACE


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "array"):
        return np.asarray(value.array)
    return np.asarray(value)


def _assert_max_abs_diff(actual, expected, *, max_abs: float) -> None:
    actual_np = _to_numpy(actual)
    expected_np = _to_numpy(expected)
    if actual_np.shape != expected_np.shape:
        actual_np = _align_to_reference(actual_np, expected_np)
    diff = np.abs(actual_np - expected_np)
    observed = float(diff.max()) if diff.size else 0.0
    assert observed < max_abs, f"max |Δ|={observed:.6f}, limit={max_abs:.6f}"


def _align_to_reference(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    if candidate.shape == reference.shape:
        return candidate
    if candidate.size != reference.size or candidate.ndim != reference.ndim:
        raise AssertionError(
            "Unable to align arrays with shapes "
            f"{candidate.shape} and {reference.shape}."
        )
    for perm in itertools.permutations(range(candidate.ndim)):
        permuted_shape = tuple(candidate.shape[idx] for idx in perm)
        if permuted_shape == reference.shape:
            return np.transpose(candidate, perm)
    raise AssertionError(
        "Unable to find a permutation aligning shapes "
        f"{candidate.shape} and {reference.shape}."
    )


def _state_to_pure_dict(state: nnx.State) -> dict[str, object]:
    def _extract(value):
        if isinstance(value, nnx.Variable):
            return value.get_value()
        return value

    return nnx.to_pure_dict(state, extract_fn=_extract)


def _init_model_from_torch(module: nnx.Module, torch_module):
    graphdef, state = nnx.split(module)
    pure = _state_to_pure_dict(state)
    updated = module.__class__.import_from_torch(torch_module, pure)
    if updated is not None:
        updated.pop("_normalize2mom_consts_var", None)
        nnx.replace_by_pure_dict(state, updated)
        module = nnx.merge(graphdef, state)
    return module, state


def _make_structures():
    rng = np.random.default_rng(0)
    structures = []
    repeats = [(1, 1, 1), (1, 1, 2)]
    strains = [
        np.zeros((3, 3)),
        np.array(
            [
                [0.0, 0.01, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.0, -0.005],
            ],
            dtype=np.float64,
        ),
    ]

    for idx, (repeat, strain) in enumerate(zip(repeats, strains, strict=False)):
        atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(repeat)
        atoms.positions += (0.03 + 0.01 * idx) * rng.normal(size=atoms.positions.shape)
        if np.any(strain):
            deformation = np.identity(3) + strain
            atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
        atoms.wrap()
        structures.append(atoms)

    return structures


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(indices.shape[0]), indices] = 1.0
    return encoded


def _graph_from_atoms(
    atoms, *, z_table: AtomicNumberTable, r_max: float
) -> dict[str, np.ndarray]:
    config = config_from_atoms(atoms)
    config.pbc = [bool(x) for x in config.pbc]
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=np.asarray(config.positions, dtype=np.float32),
        cutoff=float(r_max),
        pbc=config.pbc,
        cell=config.cell,
    )
    species_index = atomic_numbers_to_indices(
        np.asarray(config.atomic_numbers, dtype=np.int32),
        z_table=z_table,
    )
    num_nodes = int(len(config.atomic_numbers))
    return {
        "positions": np.asarray(config.positions, dtype=np.float32),
        "node_attrs": _one_hot(np.asarray(species_index, dtype=np.int32), len(z_table)),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "shifts": np.asarray(shifts, dtype=np.float32),
        "unit_shifts": np.asarray(unit_shifts, dtype=np.float32),
        "cell": np.asarray(cell, dtype=np.float32),
        "head": np.asarray(0, dtype=np.int64),
        "num_nodes": num_nodes,
    }


def _make_batch(r_max: float) -> dict[str, torch.Tensor]:
    z_table = AtomicNumberTable([11, 17])
    graphs = [
        _graph_from_atoms(atoms, z_table=z_table, r_max=r_max)
        for atoms in _make_structures()
    ]
    positions = []
    node_attrs = []
    edge_index = []
    shifts = []
    unit_shifts = []
    cells = []
    batch = []
    ptr = [0]
    heads = []
    node_offset = 0

    for graph_index, graph in enumerate(graphs):
        num_nodes = int(graph["num_nodes"])
        positions.append(graph["positions"])
        node_attrs.append(graph["node_attrs"])
        edge_index.append(graph["edge_index"] + node_offset)
        shifts.append(graph["shifts"])
        unit_shifts.append(graph["unit_shifts"])
        cells.append(graph["cell"])
        batch.append(np.full((num_nodes,), graph_index, dtype=np.int64))
        ptr.append(ptr[-1] + num_nodes)
        heads.append(int(graph["head"]))
        node_offset += num_nodes

    return {
        "positions": torch.tensor(
            np.concatenate(positions, axis=0),
            dtype=torch.get_default_dtype(),
        ),
        "node_attrs": torch.tensor(
            np.concatenate(node_attrs, axis=0),
            dtype=torch.get_default_dtype(),
        ),
        "edge_index": torch.tensor(
            np.concatenate(edge_index, axis=1), dtype=torch.long
        ),
        "shifts": torch.tensor(
            np.concatenate(shifts, axis=0),
            dtype=torch.get_default_dtype(),
        ),
        "unit_shifts": torch.tensor(
            np.concatenate(unit_shifts, axis=0),
            dtype=torch.get_default_dtype(),
        ),
        "cell": torch.tensor(np.stack(cells, axis=0), dtype=torch.get_default_dtype()),
        "batch": torch.tensor(np.concatenate(batch, axis=0), dtype=torch.long),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "head": torch.tensor(heads, dtype=torch.long),
    }


def _clone_torch_data(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in batch.items()}


def _to_jax_data(data: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    converted = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            converted[key] = value
            continue
        array = value.detach().cpu().numpy()
        if key in {"edge_index", "batch", "ptr", "head"}:
            converted[key] = jnp.asarray(array, dtype=jnp.int32)
        else:
            converted[key] = jnp.asarray(array)
    return converted


def _estimate_avg_num_neighbors(data: dict[str, torch.Tensor]) -> float:
    n_nodes = int(data["positions"].shape[0])
    n_edges = int(data["edge_index"].shape[1])
    return float(n_edges / max(n_nodes, 1))


def _make_torch_model(avg_num_neighbors: float) -> TorchLocalScaleShiftMACE:
    torch.manual_seed(0)
    model = TorchLocalScaleShiftMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls_first=TorchLocalRealAgnosticInteraction,
        interaction_cls=TorchLocalRealAgnosticResidualInteraction,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.asarray([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=[11, 17],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=False,
        apply_cutoff=True,
        use_reduced_cg=True,
        use_so3=False,
        use_agnostic_product=False,
        use_last_readout_only=False,
        use_embedding_readout=False,
        distance_transform="None",
        edge_irreps=None,
        radial_MLP=[16],
        radial_type="bessel",
        heads=None,
        cueq_config=None,
        embedding_specs=None,
        atomic_inter_scale=0.75,
        atomic_inter_shift=-0.1,
    ).float()
    model.eval()
    return model


def _make_jax_model(avg_num_neighbors: float) -> JaxLocalScaleShiftMACE:
    return JaxLocalScaleShiftMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls_first=JaxLocalRealAgnosticInteraction,
        interaction_cls=JaxLocalRealAgnosticResidualInteraction,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=Irreps("16x0e + 16x1o"),
        MLP_irreps=Irreps("8x0e"),
        atomic_energies=np.asarray([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=(11, 17),
        correlation=2,
        gate=jax.nn.silu,
        pair_repulsion=False,
        apply_cutoff=True,
        use_reduced_cg=True,
        use_so3=False,
        use_agnostic_product=False,
        use_last_readout_only=False,
        use_embedding_readout=False,
        distance_transform="None",
        edge_irreps=None,
        radial_MLP=[16],
        radial_type="bessel",
        heads=None,
        cueq_config=None,
        embedding_specs=None,
        atomic_inter_scale=0.75,
        atomic_inter_shift=-0.1,
        rngs=nnx.Rngs(0),
    )


def _collect_torch_intermediates(
    model: TorchLocalScaleShiftMACE,
    data: dict[str, torch.Tensor],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    torch_data = {key: value.clone() for key, value in data.items()}
    ctx = prepare_graph_torch(
        torch_data,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        lammps_mliap=False,
    )
    node_attrs = torch_data["node_attrs"]
    edge_index = torch_data["edge_index"]

    node_feats = model.node_embedding(node_attrs)
    edge_attrs = model.spherical_harmonics(ctx.vectors)
    edge_feats, cutoff = model.radial_embedding(
        ctx.lengths,
        node_attrs,
        edge_index,
        model.atomic_numbers,
    )

    interaction_outputs = []
    product_outputs = []
    for idx, (interaction, product) in enumerate(
        zip(model.interactions, model.products, strict=False)
    ):
        node_feats, sc = interaction(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
            first_layer=(idx == 0),
        )
        interaction_outputs.append(_to_numpy(node_feats))
        node_feats = product(
            node_feats=node_feats,
            sc=sc,
            node_attrs=node_attrs,
        )
        product_outputs.append(_to_numpy(node_feats))

    return interaction_outputs, product_outputs


def _collect_jax_intermediates(
    model,
    data: dict[str, jnp.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ctx = prepare_graph_jax(data)
    node_attrs = data["node_attrs"]
    edge_index = data["edge_index"]
    node_attrs_index = model._resolve_node_attrs_index(data, node_attrs)

    node_feats = model.node_embedding(node_attrs)
    edge_attrs = model._make_edge_attrs(ctx.vectors)
    edge_feats, cutoff = model.radial_embedding(
        ctx.lengths,
        node_attrs,
        edge_index,
        model._atomic_numbers,
        node_attrs_index=node_attrs_index,
    )

    interaction_outputs = []
    product_outputs = []
    for idx, (interaction, product) in enumerate(
        zip(model.interactions, model.products, strict=False)
    ):
        node_feats, sc = interaction(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
            lammps_class=None,
            lammps_natoms=(0, 0),
            first_layer=(idx == 0),
        )
        interaction_outputs.append(_to_numpy(node_feats))
        node_feats = product(
            node_feats=node_feats,
            sc=sc,
            node_attrs=node_attrs,
            node_attrs_index=node_attrs_index,
        )
        product_outputs.append(_to_numpy(node_feats))

    return interaction_outputs, product_outputs


class TestModelParity:
    @classmethod
    def setup_class(cls):
        cls.batch = _make_batch(r_max=4.5)
        cls.torch_data = _clone_torch_data(cls.batch)
        cls.jax_data = _to_jax_data(cls.torch_data)
        cls.avg_num_neighbors = _estimate_avg_num_neighbors(cls.torch_data)

        cls.torch_model = _make_torch_model(cls.avg_num_neighbors)
        cls.jax_model = _make_jax_model(cls.avg_num_neighbors)
        cls.jax_model, _ = _init_model_from_torch(cls.jax_model, cls.torch_model)

        torch_outputs = cls.torch_model(
            _clone_torch_data(cls.batch),
            compute_force=True,
            compute_stress=True,
            compute_node_feats=True,
        )
        graphdef, state = nnx.split(cls.jax_model)
        jax_outputs, _ = graphdef.apply(state)(
            cls.jax_data,
            compute_force=True,
            compute_stress=True,
            compute_node_feats=True,
        )

        cls.torch_outputs = {
            key: _to_numpy(value) for key, value in torch_outputs.items()
        }
        cls.jax_outputs = {
            key: _to_numpy(value) if value is not None else None
            for key, value in jax_outputs.items()
        }

        cls.torch_interactions, cls.torch_products = _collect_torch_intermediates(
            cls.torch_model,
            _clone_torch_data(cls.batch),
        )
        cls.jax_interactions, cls.jax_products = _collect_jax_intermediates(
            cls.jax_model,
            cls.jax_data,
        )

    def test_torch_and_jax_model_outputs_match(self):
        cls = self.__class__

        _assert_max_abs_diff(
            cls.jax_outputs["energy"],
            cls.torch_outputs["energy"],
            max_abs=2e-2,
        )
        _assert_max_abs_diff(
            cls.jax_outputs["forces"],
            cls.torch_outputs["forces"],
            max_abs=5e-2,
        )
        _assert_max_abs_diff(
            cls.jax_outputs["stress"],
            cls.torch_outputs["stress"],
            max_abs=5e-2,
        )
        _assert_max_abs_diff(
            cls.jax_outputs["node_energy"],
            cls.torch_outputs["node_energy"],
            max_abs=2e-2,
        )
        _assert_max_abs_diff(
            cls.jax_outputs["interaction_energy"],
            cls.torch_outputs["interaction_energy"],
            max_abs=2e-2,
        )

        _assert_max_abs_diff(
            cls.jax_outputs["node_feats"],
            cls.torch_outputs["node_feats"],
            max_abs=5e-2,
        )

    def test_torch_and_jax_intermediate_blocks_match(self):
        cls = self.__class__

        aligned_interactions = [
            _align_to_reference(jax_block, torch_block)
            for jax_block, torch_block in zip(
                cls.jax_interactions,
                cls.torch_interactions,
                strict=False,
            )
        ]
        aligned_products = [
            _align_to_reference(jax_block, torch_block)
            for jax_block, torch_block in zip(
                cls.jax_products,
                cls.torch_products,
                strict=False,
            )
        ]

        for jax_block, torch_block in zip(
            aligned_interactions,
            cls.torch_interactions,
            strict=False,
        ):
            _assert_max_abs_diff(
                jax_block,
                torch_block,
                max_abs=5e-2,
            )

        for jax_block, torch_block in zip(
            aligned_products,
            cls.torch_products,
            strict=False,
        ):
            _assert_max_abs_diff(
                jax_block,
                torch_block,
                max_abs=5e-2,
            )

        torch_concat = np.concatenate(cls.torch_products, axis=-1)
        jax_concat = np.concatenate(cls.jax_products, axis=-1)
        jax_concat = _align_to_reference(jax_concat, torch_concat)

        _assert_max_abs_diff(
            torch_concat,
            cls.torch_outputs["node_feats"],
            max_abs=1e-6,
        )
        _assert_max_abs_diff(
            jax_concat,
            _align_to_reference(cls.jax_outputs["node_feats"], torch_concat),
            max_abs=1e-6,
        )
