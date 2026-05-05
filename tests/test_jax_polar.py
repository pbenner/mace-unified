from __future__ import annotations

import math

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest
import torch

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f'cuequivariance_jax is unavailable in this environment: {exc}',
        allow_module_level=True,
    )

from flax import nnx

from mace_model.build import build_initial_model
from mace_model.config import BuildRequest
from mace_model.conversion import (
    extract_torch_model_config,
    normalize_extracted_torch_config,
)
from mace_model.jax.adapters.e3nn import Irreps
from mace_model.jax.modules.blocks import RealAgnosticInteractionBlock
from mace_model.jax.modules.field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    SparseUvuTensorProduct,
    instructions_for_sparse_tp,
)
from mace_model.jax.modules.polar import PolarMACE
from mace_model.jax.modules.polar_longrange import (
    GTOElectrostaticEnergy as JaxGTOElectrostaticEnergy,
)
from mace_model.jax.modules.polar_longrange import (
    GTOElectrostaticFeatures as JaxGTOElectrostaticFeatures,
)
from mace_model.jax.modules.polar_longrange import (
    compute_k_vectors_flat as jax_compute_k_vectors_flat,
)
from mace_model.jax.tools.torch_import import convert_torch_to_jax
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.blocks import (
    RealAgnosticInteractionBlock as TorchRealAgnosticInteractionBlock,
)
from mace_model.torch.modules.models import PolarMACE as TorchPolarMACE
from mace_model.torch.modules.polar_longrange import (
    GTOElectrostaticEnergy as TorchGTOElectrostaticEnergy,
)
from mace_model.torch.modules.polar_longrange import (
    GTOElectrostaticFeatures as TorchGTOElectrostaticFeatures,
)
from mace_model.torch.modules.polar_longrange import (
    compute_k_vectors_flat as torch_compute_k_vectors_flat,
)


def _polar_config() -> dict[str, object]:
    return {
        'r_max': 4.5,
        'num_bessel': 4,
        'num_polynomial_cutoff': 3,
        'max_ell': 1,
        'interaction_cls': 'RealAgnosticInteractionBlock',
        'interaction_cls_first': 'RealAgnosticInteractionBlock',
        'num_interactions': 1,
        'hidden_irreps': '4x0e + 4x1o',
        'MLP_irreps': '4x0e',
        'atomic_numbers': [11, 17],
        'atomic_energies': [-1.25, -2.0],
        'avg_num_neighbors': 6.0,
        'correlation': 2,
        'gate': 'silu',
        'pair_repulsion': False,
        'distance_transform': 'None',
        'radial_type': 'bessel',
        'atomic_inter_scale': 1.0,
        'atomic_inter_shift': 0.0,
    }


def _make_polar_model() -> PolarMACE:
    return PolarMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=Irreps('4x0e + 4x1o'),
        MLP_irreps=Irreps('4x0e'),
        atomic_energies=np.array([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=6.0,
        atomic_numbers=(11, 17),
        correlation=2,
        gate=jnn.silu,
        pair_repulsion=False,
        distance_transform='None',
        radial_type='bessel',
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        rngs=nnx.Rngs(0),
    )


def _make_torch_polar_model() -> TorchPolarMACE:
    torch.manual_seed(0)
    return TorchPolarMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=TorchRealAgnosticInteractionBlock,
        interaction_cls_first=TorchRealAgnosticInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps('4x0e + 4x1o'),
        MLP_irreps=o3.Irreps('4x0e'),
        atomic_energies=np.array([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=6.0,
        atomic_numbers=[11, 17],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=False,
        distance_transform='None',
        radial_type='bessel',
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).eval()


def _make_polar_data() -> dict[str, jnp.ndarray]:
    cell = 10.0 * jnp.eye(3, dtype=jnp.float32)[None, :, :]
    return {
        'positions': jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
        'node_attrs': jnp.eye(2, dtype=jnp.float32),
        'edge_index': jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
        'shifts': jnp.zeros((2, 3), dtype=jnp.float32),
        'unit_shifts': jnp.zeros((2, 3), dtype=jnp.float32),
        'cell': cell,
        'batch': jnp.zeros((2,), dtype=jnp.int32),
        'ptr': jnp.asarray([0, 2], dtype=jnp.int32),
        'pbc': jnp.asarray([[False, False, False]]),
        'rcell': jnp.linalg.inv(cell),
        'volume': jnp.abs(jnp.linalg.det(cell)),
        'total_charge': jnp.zeros((1,), dtype=jnp.float32),
        'total_spin': jnp.ones((1,), dtype=jnp.float32),
        'fermi_level': jnp.zeros((1,), dtype=jnp.float32),
        'external_field': jnp.zeros((1, 3), dtype=jnp.float32),
    }


def _make_torch_polar_data() -> dict[str, torch.Tensor]:
    dtype = torch.get_default_dtype()
    cell = 10.0 * torch.eye(3, dtype=dtype).unsqueeze(0)
    return {
        'positions': torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        'node_attrs': torch.eye(2, dtype=dtype),
        'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        'shifts': torch.zeros((2, 3), dtype=dtype),
        'unit_shifts': torch.zeros((2, 3), dtype=dtype),
        'cell': cell,
        'batch': torch.zeros(2, dtype=torch.long),
        'ptr': torch.tensor([0, 2], dtype=torch.long),
        'pbc': torch.tensor([[False, False, False]]),
        'rcell': torch.linalg.inv(cell),
        'volume': torch.linalg.det(cell).abs(),
        'total_charge': torch.zeros(1, dtype=dtype),
        'total_spin': torch.ones(1, dtype=dtype),
        'fermi_level': torch.zeros(1, dtype=dtype),
        'external_field': torch.zeros((1, 3), dtype=dtype),
    }


def _torch_data_to_jax(data: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    converted = {}
    for key, value in data.items():
        array = value.detach().cpu().numpy()
        if key in {'edge_index', 'batch', 'ptr'}:
            converted[key] = jnp.asarray(array, dtype=jnp.int32)
        else:
            converted[key] = jnp.asarray(array)
    return converted


def test_jax_polar_mace_constructs_and_runs_local_longrange_forward():
    model = _make_polar_model()

    assert model.keep_last_layer_irreps is True
    assert len(model.lr_source_maps) == 1
    assert len(model.field_dependent_charges_maps) == 1
    assert model.potential_irreps.dim == 2 * model.field_irreps.dim

    output = model(_make_polar_data(), compute_force=False, compute_node_feats=False)
    assert output['energy'].shape == (1,)
    assert output['forces'] is None
    assert output['node_feats'] is None
    assert output['density_coefficients'].shape == (2, model.charges_irreps.dim // 2)

    force_output = model(
        _make_polar_data(), compute_force=True, compute_node_feats=False
    )
    assert force_output['energy'].shape == (1,)
    assert force_output['forces'].shape == (2, 3)


def test_jax_polar_mace_jits_realspace_energy_and_forces():
    model = _make_polar_model()
    data = _make_polar_data()

    def energy_fn(graph):
        return model(
            graph,
            compute_force=False,
            compute_node_feats=False,
            longrange_mode='realspace',
        )['energy']

    def forces_fn(graph):
        return model(
            graph,
            compute_force=True,
            compute_node_feats=False,
            longrange_mode='realspace',
        )['forces']

    energy = jax.jit(energy_fn)(data)
    forces = jax.jit(forces_fn)(data)

    assert energy.shape == (1,)
    assert forces.shape == (2, 3)
    assert jnp.isfinite(energy).all()
    assert jnp.isfinite(forces).all()


def test_jax_polar_mace_jits_pbc_energy_and_forces_with_precomputed_kspace():
    model = _make_polar_model()
    data = dict(_make_polar_data())
    data['pbc'] = jnp.asarray([[True, True, True]])
    k_vectors, k_norm2, k_vector_batch, k0_mask = jax_compute_k_vectors_flat(
        model.kspace_cutoff,
        data['cell'].reshape(-1, 3, 3),
        data['rcell'].reshape(-1, 3, 3),
    )
    data.update(
        {
            'k_vectors': k_vectors,
            'k_norm2': k_norm2,
            'k_vector_batch': k_vector_batch,
            'k0_mask': k0_mask,
        }
    )

    def energy_fn(graph):
        return model(
            graph,
            compute_force=False,
            compute_node_feats=False,
            longrange_mode='pbc',
        )['energy']

    def forces_fn(graph):
        return model(
            graph,
            compute_force=True,
            compute_node_feats=False,
            longrange_mode='pbc',
        )['forces']

    energy = jax.jit(energy_fn)(data)
    forces = jax.jit(forces_fn)(data)

    assert energy.shape == (1,)
    assert forces.shape == (2, 3)
    assert jnp.isfinite(energy).all()
    assert jnp.isfinite(forces).all()


def test_torch_polar_imports_to_jax_and_matches_forward_outputs():
    torch_model = _make_torch_polar_model()
    config = normalize_extracted_torch_config(extract_torch_model_config(torch_model))
    jax_model, _variables, _template_data = convert_torch_to_jax(torch_model, config)
    torch_data = _make_torch_polar_data()

    with torch.no_grad():
        expected = torch_model(
            {key: value.clone() for key, value in torch_data.items()},
            compute_force=False,
            compute_node_feats=False,
        )
    actual = jax_model(
        _torch_data_to_jax(torch_data),
        compute_force=False,
        compute_node_feats=False,
    )

    for key in (
        'energy',
        'interaction_energy',
        'electrostatic_energy',
        'electron_energy',
        'dipole',
        'density_coefficients',
    ):
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            expected[key].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )


def test_build_initial_model_accepts_jax_polar_mace():
    result = build_initial_model(
        BuildRequest(
            backend='jax',
            model_class='PolarMACE',
            seed=0,
            output=None,
            model_config=_polar_config(),
            raw_config={},
        )
    )

    assert isinstance(result.model, PolarMACE)
    assert result.normalized_model_config['model_class'] == 'PolarMACE'


def test_jax_polar_field_source_blocks_have_expected_shapes():
    irreps = Irreps('2x0e')
    all_node_feats = jnp.asarray(
        np.random.default_rng(0).normal(size=(2, 3, irreps.dim)),
        dtype=jnp.float32,
    )

    mixer = MultiLayerFeatureMixer(
        node_feats_irreps=irreps,
        num_interactions=2,
        rngs=nnx.Rngs(0),
    )
    mixed = mixer(all_node_feats)
    assert mixed.shape == (3, irreps.dim)

    source = EnvironmentDependentSpinSourceBlock(
        irreps_in=irreps,
        max_l=1,
        rngs=nnx.Rngs(0),
    )
    multipoles = source(mixed)
    expected_irreps = 2 * Irreps.spherical_harmonics(1)
    assert multipoles.shape == (3, 1, expected_irreps.dim)


def test_jax_sparse_uvu_tensor_product_uses_cue_scalar_contraction():
    irreps = Irreps('2x0e + 2x1o')
    out_irreps = Irreps('2x0e')
    instructions = instructions_for_sparse_tp(irreps, irreps, out_irreps)
    sparse = SparseUvuTensorProduct(
        irreps_in1=irreps,
        irreps_in2=irreps,
        irreps_out=out_irreps,
        instructions=instructions,
        rngs=nnx.Rngs(0),
    )
    sparse.weight[...] = jnp.ones_like(jnp.asarray(sparse.weight))

    x1 = jnp.asarray([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    x2 = jnp.asarray([[0.0, 0.0, 2.0, 1.0, 0.0, -1.0, 1.0, 2.0]])
    x1_vectors = x1[:, 2:].reshape(1, 2, 3)
    x2_vectors = x2[:, 2:].reshape(1, 2, 3)
    expected = 0.5 * jnp.einsum('bud,bvd->bu', x1_vectors, x2_vectors)
    expected = expected / math.sqrt(3.0)

    np.testing.assert_allclose(
        np.asarray(sparse(x1, x2)),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_local_longrange_realspace_l0_matches_torch_and_jax():
    positions_np = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32)
    batch_np = np.zeros((2,), dtype=np.int64)
    source_np = np.asarray([[0.25], [-0.1]], dtype=np.float32)
    pbc_np = np.asarray([[False, False, False]])
    volume_np = np.asarray([1000.0], dtype=np.float32)

    torch_features = TorchGTOElectrostaticFeatures(
        density_max_l=0,
        density_smearing_width=1.0,
        feature_max_l=0,
        feature_smearing_widths=[1.0],
        include_self_interaction=False,
        kspace_cutoff=4.5,
    )
    torch_energy = TorchGTOElectrostaticEnergy(
        density_max_l=0,
        density_smearing_width=1.0,
        kspace_cutoff=4.5,
        include_self_interaction=False,
    )
    torch_positions = torch.as_tensor(positions_np)
    torch_batch = torch.as_tensor(batch_np)
    torch_source = torch.as_tensor(source_np)
    torch_cache = torch_features.precompute_geometry(
        k_vectors=torch.zeros((1, 3)),
        k_norm2=torch.zeros((1,)),
        k_vector_batch=torch.zeros((1,), dtype=torch.long),
        k0_mask=torch.ones((1,)),
        node_positions=torch_positions,
        batch=torch_batch,
        volume=torch.as_tensor(volume_np),
        pbc=torch.as_tensor(pbc_np),
    )
    expected_features = torch_features.forward_dynamic(
        torch_cache,
        torch_source[:, None, :],
        torch.as_tensor(pbc_np),
    )
    expected_energy = torch_energy(
        k_vectors=torch.zeros((1, 3)),
        k_norm2=torch.zeros((1,)),
        k_vector_batch=torch.zeros((1,), dtype=torch.long),
        k0_mask=torch.ones((1,)),
        source_feats=torch_source,
        node_positions=torch_positions,
        batch=torch_batch,
        volume=torch.as_tensor(volume_np),
        pbc=torch.as_tensor(pbc_np),
    )

    jax_features = JaxGTOElectrostaticFeatures(
        density_max_l=0,
        density_smearing_width=1.0,
        feature_max_l=0,
        feature_smearing_widths=[1.0],
        include_self_interaction=False,
        kspace_cutoff=4.5,
    )
    jax_energy = JaxGTOElectrostaticEnergy(
        density_max_l=0,
        density_smearing_width=1.0,
        kspace_cutoff=4.5,
        include_self_interaction=False,
    )
    jax_cache = jax_features.precompute_geometry(
        k_vectors=jnp.zeros((1, 3), dtype=jnp.float32),
        k_norm2=jnp.zeros((1,), dtype=jnp.float32),
        k_vector_batch=jnp.zeros((1,), dtype=jnp.int32),
        k0_mask=jnp.ones((1,), dtype=jnp.float32),
        node_positions=jnp.asarray(positions_np),
        batch=jnp.asarray(batch_np, dtype=jnp.int32),
        volume=jnp.asarray(volume_np),
        pbc=jnp.asarray(pbc_np),
    )
    actual_features = jax_features.forward_dynamic(
        jax_cache,
        jnp.asarray(source_np)[:, None, :],
        jnp.asarray(pbc_np),
    )
    actual_energy = jax_energy(
        k_vectors=jnp.zeros((1, 3), dtype=jnp.float32),
        k_norm2=jnp.zeros((1,), dtype=jnp.float32),
        k_vector_batch=jnp.zeros((1,), dtype=jnp.int32),
        k0_mask=jnp.ones((1,), dtype=jnp.float32),
        source_feats=jnp.asarray(source_np),
        node_positions=jnp.asarray(positions_np),
        batch=jnp.asarray(batch_np, dtype=jnp.int32),
        volume=jnp.asarray(volume_np),
        pbc=jnp.asarray(pbc_np),
    )

    np.testing.assert_allclose(
        np.asarray(actual_features),
        expected_features.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_energy),
        expected_energy.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_longrange_pbc_mode_matches_torch_and_jax():
    positions_np = np.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.1, 1.1, 0.2]],
        dtype=np.float32,
    )
    batch_np = np.zeros((3,), dtype=np.int64)
    source_np = np.asarray(
        [
            [0.25, 0.03, -0.02, 0.01],
            [-0.1, 0.01, 0.04, -0.03],
            [0.05, -0.02, 0.01, 0.02],
        ],
        dtype=np.float32,
    )
    pbc_np = np.asarray([[True, True, True]])
    volume_np = np.asarray([1000.0], dtype=np.float32)

    torch_features = TorchGTOElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0],
        include_self_interaction=False,
        kspace_cutoff=4.5,
    )
    torch_energy = TorchGTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=1.0,
        kspace_cutoff=4.5,
        include_self_interaction=False,
    )
    torch_positions = torch.as_tensor(positions_np)
    torch_batch = torch.as_tensor(batch_np)
    torch_source = torch.as_tensor(source_np)
    torch_cell = 10.0 * torch.eye(3).unsqueeze(0)
    torch_k_vectors, torch_k_norm2, torch_k_batch, torch_k0_mask = (
        torch_compute_k_vectors_flat(
            4.5,
            torch_cell,
            torch.linalg.inv(torch_cell),
        )
    )
    torch_cache = torch_features.precompute_geometry(
        k_vectors=torch_k_vectors,
        k_norm2=torch_k_norm2,
        k_vector_batch=torch_k_batch,
        k0_mask=torch_k0_mask,
        node_positions=torch_positions,
        batch=torch_batch,
        volume=torch.as_tensor(volume_np),
        pbc=torch.as_tensor(pbc_np),
    )
    expected_features = torch_features.forward_dynamic(
        torch_cache,
        torch_source[:, None, :],
        torch.as_tensor(pbc_np),
    )
    expected_energy = torch_energy(
        k_vectors=torch_k_vectors,
        k_norm2=torch_k_norm2,
        k_vector_batch=torch_k_batch,
        k0_mask=torch_k0_mask,
        source_feats=torch_source,
        node_positions=torch_positions,
        batch=torch_batch,
        volume=torch.as_tensor(volume_np),
        pbc=torch.as_tensor(pbc_np),
    )

    jax_features = JaxGTOElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0],
        include_self_interaction=False,
        kspace_cutoff=4.5,
    )
    jax_energy = JaxGTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=1.0,
        kspace_cutoff=4.5,
        include_self_interaction=False,
    )
    jax_cell = 10.0 * jnp.eye(3, dtype=jnp.float32)[None, :, :]
    jax_k_vectors, jax_k_norm2, jax_k_batch, jax_k0_mask = jax_compute_k_vectors_flat(
        4.5,
        jax_cell,
        jnp.linalg.inv(jax_cell),
    )
    jax_cache = jax_features.precompute_geometry(
        k_vectors=jax_k_vectors,
        k_norm2=jax_k_norm2,
        k_vector_batch=jax_k_batch,
        k0_mask=jax_k0_mask,
        node_positions=jnp.asarray(positions_np),
        batch=jnp.asarray(batch_np, dtype=jnp.int32),
        volume=jnp.asarray(volume_np),
        pbc=jnp.asarray(pbc_np),
        mode='pbc',
    )
    actual_features = jax_features.forward_dynamic(
        jax_cache,
        jnp.asarray(source_np)[:, None, :],
        jnp.asarray(pbc_np),
    )
    actual_energy = jax_energy(
        k_vectors=jax_k_vectors,
        k_norm2=jax_k_norm2,
        k_vector_batch=jax_k_batch,
        k0_mask=jax_k0_mask,
        source_feats=jnp.asarray(source_np),
        node_positions=jnp.asarray(positions_np),
        batch=jnp.asarray(batch_np, dtype=jnp.int32),
        volume=jnp.asarray(volume_np),
        pbc=jnp.asarray(pbc_np),
        mode='pbc',
    )

    np.testing.assert_allclose(
        np.asarray(actual_features),
        expected_features.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_energy),
        expected_energy.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
