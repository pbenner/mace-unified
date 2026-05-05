from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mace_model.conversion import convert_torch_model, load_serialized_torch_model
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.modules.blocks import RealAgnosticInteractionBlock
from mace_model.torch.modules.field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    SparseUvuTensorProduct,
    instructions_for_sparse_tp,
)
from mace_model.torch.modules.models import PolarMACE

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore:`torch\\.jit\\.script` is deprecated.*:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:__array_wrap__ must accept context and return_scalar arguments.*:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*:UserWarning'
    ),
]


def _make_polar_model(*, pair_repulsion: bool = False) -> PolarMACE:
    return PolarMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps('4x0e + 4x1o'),
        MLP_irreps=o3.Irreps('4x0e'),
        atomic_energies=np.array([-1.25, -2.0], dtype=np.float32),
        avg_num_neighbors=6.0,
        atomic_numbers=[11, 17],
        correlation=2,
        gate=torch.nn.functional.silu,
        pair_repulsion=pair_repulsion,
        distance_transform='None',
        radial_type='bessel',
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _make_polar_data() -> dict[str, torch.Tensor]:
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


def _clone_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in data.items()}


def _make_probe_data_for_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    dtype = torch.get_default_dtype()
    atomic_numbers = [int(value) for value in model.atomic_numbers]
    z_to_index = {
        atomic_number: index for index, atomic_number in enumerate(atomic_numbers)
    }
    species = [8, 1, 1]
    node_attrs = torch.zeros((len(species), len(atomic_numbers)), dtype=dtype)
    for node_index, atomic_number in enumerate(species):
        node_attrs[node_index, z_to_index[atomic_number]] = 1.0

    senders = []
    receivers = []
    for sender in range(len(species)):
        for receiver in range(len(species)):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    cell = 20.0 * torch.eye(3, dtype=dtype).unsqueeze(0)
    return {
        'positions': torch.tensor(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]],
            dtype=dtype,
        ),
        'node_attrs': node_attrs,
        'edge_index': edge_index,
        'shifts': torch.zeros((edge_index.shape[1], 3), dtype=dtype),
        'unit_shifts': torch.zeros((edge_index.shape[1], 3), dtype=dtype),
        'cell': cell,
        'batch': torch.zeros(len(species), dtype=torch.long),
        'ptr': torch.tensor([0, len(species)], dtype=torch.long),
        'pbc': torch.tensor([[False, False, False]]),
        'rcell': torch.linalg.inv(cell),
        'volume': torch.linalg.det(cell).abs(),
        'total_charge': torch.zeros(1, dtype=dtype),
        'total_spin': torch.ones(1, dtype=dtype),
        'fermi_level': torch.zeros(1, dtype=dtype),
        'external_field': torch.zeros((1, 3), dtype=dtype),
    }


class _ConstantPairEnergy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        del lengths, edge_index, atomic_numbers
        return node_attrs.new_full((node_attrs.shape[0],), self.value)


def test_polar_mace_constructs_with_local_longrange_backend():
    model = _make_polar_model()

    assert model.keep_last_layer_irreps is True
    assert len(model.lr_source_maps) == 1
    assert len(model.field_dependent_charges_maps) == 1
    assert model.potential_irreps.dim == 2 * model.field_irreps.dim


def test_polar_mace_forward_accepts_explicit_pbc_mode():
    model = _make_polar_model().eval()
    data = _make_polar_data()
    data['pbc'] = torch.tensor([[True, True, True]])

    with torch.no_grad():
        output = model(
            data,
            compute_force=False,
            compute_node_feats=False,
            longrange_mode='pbc',
        )

    assert torch.isfinite(output['energy']).all()
    assert output['energy'].shape == (1,)
    assert output['node_feats'] is None


def test_cached_foundation_polar_checkpoint_converts_and_runs_with_local_backend():
    checkpoint = Path.home() / '.cache' / 'mace' / 'MACE-POLAR-1-S.model'
    if not checkpoint.exists():
        pytest.skip('Cached MACE-POLAR-1-S checkpoint is not available.')

    legacy_model, normalized = load_serialized_torch_model(checkpoint)
    result = convert_torch_model(legacy_model, backend='torch', config=normalized)
    model = result.model.eval()

    assert result.model_class == 'PolarMACE'
    assert isinstance(model, PolarMACE)
    assert model.atomic_multipoles_max_l == 1
    assert model.field_feature_max_l == 1

    with torch.no_grad():
        output = model(
            _make_probe_data_for_model(model),
            compute_force=False,
            compute_node_feats=False,
        )

    assert torch.isfinite(output['energy']).all()
    assert output['energy'].shape == (1,)
    assert output['dipole'].shape == (1, 3)
    assert output['density_coefficients'].shape == (3, 4)


def test_polar_mace_forward_uses_pair_repulsion_like_scale_shift_mace():
    torch.manual_seed(0)
    model = _make_polar_model(pair_repulsion=True)
    data = _make_polar_data()

    model.pair_repulsion_fn = _ConstantPairEnergy(0.0)
    without_pair = model(
        _clone_data(data),
        compute_force=False,
        compute_node_feats=False,
    )

    model.pair_repulsion_fn = _ConstantPairEnergy(0.25)
    with_pair = model(
        _clone_data(data),
        compute_force=False,
        compute_node_feats=False,
    )

    torch.testing.assert_close(
        with_pair['interaction_energy'] - without_pair['interaction_energy'],
        torch.tensor([0.5], dtype=torch.get_default_dtype()),
    )
    assert with_pair['node_feats'] is None


def test_polar_field_source_blocks_have_expected_shapes():
    irreps = o3.Irreps('2x0e')
    all_node_feats = torch.randn(2, 3, irreps.dim)

    mixer = MultiLayerFeatureMixer(node_feats_irreps=irreps, num_interactions=2)
    mixed = mixer(all_node_feats)
    assert mixed.shape == (3, irreps.dim)

    source = EnvironmentDependentSpinSourceBlock(irreps_in=irreps, max_l=1)
    multipoles = source(mixed)
    expected_irreps = 2 * o3.Irreps.spherical_harmonics(1)
    assert multipoles.shape == (3, 1, expected_irreps.dim)


def test_sparse_uvu_tensor_product_uses_cue_scalar_contraction():
    irreps = o3.Irreps('2x0e + 2x1o')
    out_irreps = o3.Irreps('2x0e')
    instructions = instructions_for_sparse_tp(irreps, irreps, out_irreps)
    sparse = SparseUvuTensorProduct(
        irreps_in1=irreps,
        irreps_in2=irreps,
        irreps_out=out_irreps,
        instructions=instructions,
    )
    assert all('_cg_' not in name for name in sparse.state_dict())
    sparse.weight.data.fill_(1.0)

    x1 = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    x2 = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, -1.0, 1.0, 2.0]])
    x1_vectors = x1[:, 2:].view(1, 2, 3)
    x2_vectors = x2[:, 2:].view(1, 2, 3)
    expected = 0.5 * torch.einsum('bud,bvd->bu', x1_vectors, x2_vectors)
    expected = expected / math.sqrt(3.0)

    torch.testing.assert_close(sparse(x1, x2), expected)
