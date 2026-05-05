from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.constants import pi

from mace_model.core.modules.polar_longrange import (
    CUBIC_MADELUNG,
    FIELD_CONSTANT,
    cl_sigma_np,
    expanded_l_indices_np,
    external_field_matrix_np,
    normalization_denominator_np,
    output_permutation_np,
    phase_factors_np,
    self_interaction_constants_np,
)
from mace_model.torch.adapters.e3nn import o3
from mace_model.torch.tools.scatter import scatter_sum

CORRECTION_MODE_PBC = 0
CORRECTION_MODE_MOLECULE = 1
CORRECTION_MODE_SLAB = 2
CORRECTION_MODE_MIXED = 3


def compute_k_vectors_flat(
    cutoff: float | torch.Tensor,
    cell_vectors: torch.Tensor,
    r_cell_vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cutoff_value = torch.as_tensor(
        cutoff, dtype=cell_vectors.dtype, device=cell_vectors.device
    )
    norms = torch.norm(cell_vectors, dim=-1)
    normed_lattice_vectors = cell_vectors / norms.unsqueeze(-1)
    dot_products = torch.einsum('bij,bij->bi', r_cell_vectors, normed_lattice_vectors)
    max_ns = torch.ceil(cutoff_value * torch.pow(dot_products, -1)).type(torch.int64)
    max_max_ns = torch.max(max_ns, dim=0).values
    n1max, n2max, n3max = (int(max_max_ns[0]), int(max_max_ns[1]), int(max_max_ns[2]))

    origin = torch.cartesian_prod(
        torch.arange(0, 1, device=cell_vectors.device),
        torch.arange(0, 1, device=cell_vectors.device),
        torch.arange(0, 1, device=cell_vectors.device),
    ).to(dtype=r_cell_vectors.dtype)
    open_half_line = torch.cartesian_prod(
        torch.arange(0, 1, device=cell_vectors.device),
        torch.arange(0, 1, device=cell_vectors.device),
        torch.arange(1, n3max, device=cell_vectors.device),
    ).to(dtype=r_cell_vectors.dtype)
    open_half_plane = torch.cartesian_prod(
        torch.arange(0, 1, device=cell_vectors.device),
        torch.arange(1, n2max, device=cell_vectors.device),
        torch.arange(-n3max, n3max, device=cell_vectors.device),
    ).to(dtype=r_cell_vectors.dtype)
    open_half_sphere = torch.cartesian_prod(
        torch.arange(1, n1max, device=cell_vectors.device),
        torch.arange(-n2max, n2max, device=cell_vectors.device),
        torch.arange(-n3max, n3max, device=cell_vectors.device),
    ).to(dtype=r_cell_vectors.dtype)
    kvecs = torch.cat(
        (origin, open_half_line, open_half_plane, open_half_sphere), dim=0
    )

    k_vectors = torch.einsum('ni,bij->bnj', kvecs, r_cell_vectors)
    k_norm2 = torch.einsum('bni,bni->bn', k_vectors, k_vectors)
    mask = k_norm2.le(cutoff_value * cutoff_value)

    k_vectors_flat = []
    k_norm2_flat = []
    k_batch_flat = []
    k0_mask_flat = []
    for graph_i in range(k_vectors.shape[0]):
        graph_mask = mask[graph_i]
        graph_k = k_vectors[graph_i][graph_mask]
        graph_k_norm2 = k_norm2[graph_i][graph_mask]
        graph_k0_mask = torch.zeros_like(graph_k_norm2)
        graph_k0_mask[0] = 1.0
        k_vectors_flat.append(graph_k)
        k_norm2_flat.append(graph_k_norm2)
        k_batch_flat.append(
            torch.full(
                (graph_k.shape[0],),
                graph_i,
                dtype=torch.long,
                device=cell_vectors.device,
            )
        )
        k0_mask_flat.append(graph_k0_mask)

    return (
        torch.cat(k_vectors_flat, dim=0),
        torch.cat(k_norm2_flat, dim=0),
        torch.cat(k_batch_flat, dim=0),
        torch.cat(k0_mask_flat, dim=0),
    )


class RadialIntegralDirect(torch.nn.Module):
    def __init__(self, sigmas: Sequence[float], max_l: int) -> None:
        super().__init__()
        if max_l > 1:
            raise NotImplementedError('RadialIntegralDirect only supports max_l <= 1.')
        sigmas_t = torch.as_tensor(sigmas, dtype=torch.get_default_dtype())
        pref_const = (4 * pi) * np.sqrt(pi / 2.0)
        self.num_sigma = len(sigmas)
        self.max_l = int(max_l)
        self.register_buffer('sigma2', sigmas_t * sigmas_t)
        self.register_buffer('pref0', pref_const * sigmas_t**3)
        if self.max_l == 1:
            self.register_buffer('pref1', pref_const * sigmas_t**5)

    def forward(self, k_mods: torch.Tensor) -> torch.Tensor:
        k2 = k_mods * k_mods
        exp_term = torch.exp(-0.5 * k2.unsqueeze(-1) * self.sigma2)
        if self.max_l == 0:
            return (self.pref0 * exp_term).unsqueeze(-1)

        out = torch.empty(
            (*k_mods.shape, self.num_sigma, 2),
            dtype=k_mods.dtype,
            device=k_mods.device,
        )
        out[..., 0] = self.pref0 * exp_term
        out[..., 1] = self.pref1 * k_mods.unsqueeze(-1) * exp_term
        return out


class GTOBasis(torch.nn.Module):
    def __init__(
        self,
        max_l: int,
        sigmas: Sequence[float],
        kspace_cutoff: float,
        normalize: str,
    ) -> None:
        super().__init__()
        if normalize not in {'none', 'multipoles', 'receiver'}:
            raise ValueError(
                "normalize must be one of 'multipoles', 'none', or 'receiver'"
            )
        if len(sigmas) == 0:
            raise ValueError('sigmas must contain at least one value')
        self.max_l = int(max_l)
        self.sigmas = [float(sigma) for sigma in sigmas]
        self.normalize = normalize
        self.kspace_cutoff = float(kspace_cutoff)
        self.radial_spline = RadialIntegralDirect(self.sigmas, self.max_l)
        self.spherical_harmonics = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(self.max_l),
            normalize=True,
            normalization='integral',
        )
        cl_inverse = normalization_denominator_np(self.sigmas, self.max_l, normalize)
        real_phase, imag_phase = phase_factors_np(self.max_l)
        self.register_buffer(
            'cl_scale',
            torch.as_tensor(1.0 / cl_inverse, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            'expanded_l_indices',
            torch.as_tensor(expanded_l_indices_np(self.max_l), dtype=torch.long),
        )
        self.register_buffer(
            'real_phase_factors',
            torch.as_tensor(real_phase, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            'imag_phase_factors',
            torch.as_tensor(imag_phase, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            'permute_indices', torch.tensor([1, 2, 0], dtype=torch.long)
        )

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> torch.Tensor:
        k_moduli = torch.sqrt(torch.clamp_min(k_norm2, 0.0))
        k_moduli = torch.where(k0_mask > 0.0, torch.zeros_like(k_moduli), k_moduli)
        k_vectors_e3nn = torch.index_select(k_vectors, -1, self.permute_indices)
        yklm = self.spherical_harmonics(k_vectors_e3nn)
        fnlk = self.radial_spline(k_moduli) * self.cl_scale.to(k_moduli.dtype)
        expanded_fnlk = torch.index_select(fnlk, -1, self.expanded_l_indices)
        xnlk = expanded_fnlk * yklm.unsqueeze(-2)
        return torch.stack(
            (
                xnlk * self.real_phase_factors.to(dtype=xnlk.dtype),
                xnlk * self.imag_phase_factors.to(dtype=xnlk.dtype),
            ),
            dim=-1,
        )


class GTOSelfInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        l_source: int,
        sigma_source: float,
        l_receive: int,
        sigmas_receive: Sequence[float],
        normalize_source: str,
        normalize_receive: str,
    ) -> None:
        super().__init__()
        sh_irreps = o3.Irreps.spherical_harmonics(l_receive)
        self.features_irreps = (sh_irreps * len(sigmas_receive)).sort()[0].simplify()
        constants, indices, non_zero_terms = self_interaction_constants_np(
            l_source=l_source,
            sigma_source=float(sigma_source),
            l_receive=l_receive,
            sigmas_receive=sigmas_receive,
            normalize_source=normalize_source,
            normalize_receive=normalize_receive,
        )
        self.non_zero_terms = int(non_zero_terms)
        self.register_buffer(
            'overlap_constants',
            torch.as_tensor(constants, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            'select_indices', torch.as_tensor(indices, dtype=torch.long)
        )

    def forward(self, charge_density: torch.Tensor) -> torch.Tensor:
        qs_expanded = torch.index_select(
            charge_density, dim=-1, index=self.select_indices
        )
        features = torch.zeros(
            (charge_density.shape[0], self.features_irreps.dim),
            device=charge_density.device,
            dtype=charge_density.dtype,
        )
        features[..., : self.non_zero_terms] = (
            self.overlap_constants.to(charge_density.dtype) * qs_expanded
        )
        return features


class DisplacedGTOExternalFieldBlock(torch.nn.Module):
    def __init__(
        self,
        l_receive: int,
        sigmas_receive: Sequence[float],
        normalize_receive: str,
    ) -> None:
        super().__init__()
        self.projections_dim = (int(l_receive) + 1) ** 2 * len(sigmas_receive)
        self.register_buffer(
            'matrix',
            torch.as_tensor(
                external_field_matrix_np(l_receive, sigmas_receive, normalize_receive),
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        node_fields = torch.index_select(field, 0, batch).clone()
        node_fields[:, 0] = node_fields[:, 0] + torch.einsum(
            'bi,bi->b',
            positions,
            node_fields[:, 1:],
        )
        node_fields = node_fields[:, [0, 3, 1, 2]]
        return torch.einsum('pf,nf->np', self.matrix.to(node_fields.dtype), node_fields)


class GTOInternalFieldtoFeaturesBlock(DisplacedGTOExternalFieldBlock):
    def forward(
        self,
        batch: torch.Tensor,
        positions: torch.Tensor,
        node_fields: torch.Tensor,
    ) -> torch.Tensor:
        del batch, positions
        node_fields = node_fields[:, [0, 3, 1, 2]]
        return torch.einsum('pf,nf->np', self.matrix.to(node_fields.dtype), node_fields)


@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor,
    duplicates_per_node: int,
) -> torch.Tensor:
    batch = batch.long()
    orig = torch.arange(batch.shape[0], device=batch.device)
    batch2 = batch.repeat_interleave(duplicates_per_node)
    orig2 = orig.repeat_interleave(duplicates_per_node)
    num_graphs = int(batch2.max().item()) + 1 if batch2.numel() else 0
    edges = []
    for graph_i in range(num_graphs):
        mask = batch2 == graph_i
        nodes = mask.nonzero(as_tuple=False).view(-1)
        if nodes.numel() <= 1:
            continue
        num_nodes = nodes.shape[0]
        row = nodes.view(-1, 1).expand(-1, num_nodes).reshape(-1)
        col = nodes.view(1, -1).expand(num_nodes, -1).reshape(-1)
        orig_row = orig2[mask].view(-1, 1).expand(-1, num_nodes).reshape(-1)
        orig_col = orig2[mask].view(1, -1).expand(num_nodes, -1).reshape(-1)
        edges.append(
            torch.stack([row[orig_row != orig_col], col[orig_row != orig_col]], dim=0)
        )
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(edges, dim=1)


def charges_energy_from_graph(
    charges: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    density_smearing_width: float,
) -> torch.Tensor:
    sender, receiver = edge_index
    r_ij = positions[receiver] - positions[sender]
    d_ij = torch.linalg.norm(r_ij, dim=-1)
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    if edge_energy.numel() == 0:
        return torch.zeros(num_graphs, dtype=charges.dtype, device=charges.device)
    node_energies = scatter_sum(src=edge_energy, index=receiver, dim=-1)
    return scatter_sum(src=node_energies, index=batch, dim=-1, dim_size=num_graphs)


def charges_features_from_graph(
    charges: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    total_width_factors: torch.Tensor,
) -> torch.Tensor:
    num_nodes = positions.shape[0]
    sender, receiver = edge_index
    r_ij = positions[sender] - positions[receiver]
    d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)
    features = torch.zeros(
        (num_nodes, total_width_factors.shape[-1]),
        dtype=positions.dtype,
        device=positions.device,
    )
    features.index_add_(0, receiver, charges[sender].unsqueeze(-1) * smooth_reciprocal)
    return FIELD_CONSTANT * features / (4 * pi)


class RealSpaceFiniteDiffereneEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset: float = 0.02,
    ) -> None:
        super().__init__()
        if density_max_l > 1:
            raise ValueError(
                'RealSpaceFiniteDiffereneEnergy only supports l=0 and l=1.'
            )
        self.density_max_l = int(density_max_l)
        self.density_smearing_width = float(density_smearing_width)
        self.include_self_interaction = bool(include_self_interaction)
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            'multipoles',
            'multipoles',
        )
        self.offset = float(offset)
        self.register_buffer(
            'x', torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'y', torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'z', torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0:
            edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
            energy = charges_energy_from_graph(
                source_feats.squeeze(-1),
                positions,
                edge_index,
                batch,
                self.density_smearing_width,
            )
        else:
            extended_positions = positions.repeat_interleave(4, dim=0)
            extended_positions[1::4] += self.x.to(positions.dtype)
            extended_positions[2::4] += self.y.to(positions.dtype)
            extended_positions[3::4] += self.z.to(positions.dtype)
            extended_batch = batch.repeat_interleave(4)
            charges = torch.zeros_like(extended_positions[:, 0])
            charges[1::4] = source_feats[:, 3] / self.offset
            charges[2::4] = source_feats[:, 1] / self.offset
            charges[3::4] = source_feats[:, 2] / self.offset
            charges[0::4] = source_feats[:, 0] - (
                charges[1::4] + charges[2::4] + charges[3::4]
            )
            edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)
            energy = charges_energy_from_graph(
                charges,
                extended_positions,
                edge_index,
                extended_batch,
                self.density_smearing_width,
            )

        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)
            node_energies = torch.einsum('nb,nb->n', source_feats, self_fields)
            energy = energy + 0.5 * scatter_sum(src=node_energies, index=batch, dim=-1)
        return energy


class RealSpaceFiniteDifferenceElectrostaticFeatures(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: Sequence[float],
        include_self_interaction: bool = False,
        integral_normalization: str = 'receiver',
        offset: float = 0.1,
    ) -> None:
        super().__init__()
        self.density_max_l = int(density_max_l)
        self.projection_max_l = int(projection_max_l)
        self.include_self_interaction = bool(include_self_interaction)
        self.num_radial = len(projection_smearing_widths)
        self.offset = float(offset)
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            'multipoles',
            integral_normalization,
        )
        total_width_factors = torch.pow(
            (
                float(density_smearing_width) ** 2
                + torch.tensor(
                    projection_smearing_widths, dtype=torch.get_default_dtype()
                )
                ** 2
            )
            / 2,
            0.5,
        )
        self.register_buffer('total_width_factors', total_width_factors)
        self.register_buffer(
            'x', torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'y', torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'z', torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )
        l0_factors = [
            cl_sigma_np(0, sigma, integral_normalization)
            / cl_sigma_np(0, sigma, 'multipoles')
            for sigma in projection_smearing_widths
        ]
        l1_factors = [
            3**0.5
            * float(sigma) ** 2
            * (
                cl_sigma_np(1, sigma, integral_normalization)
                / cl_sigma_np(0, sigma, 'multipoles')
            )
            / self.offset
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            'l0_factors', torch.tensor(l0_factors, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'l1_factors', torch.tensor(l1_factors, dtype=torch.get_default_dtype())
        )

    def _density_0_feats_0(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
        feats = charges_features_from_graph(
            charges=source_feats[:, 0],
            positions=positions,
            edge_index=edge_index,
            total_width_factors=self.total_width_factors.to(positions.dtype).unsqueeze(
                0
            ),
        )
        return self.l0_factors.to(feats.dtype) * feats

    def _density_1_feats_1(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x.to(positions.dtype)
        extended_positions[2::4] += self.y.to(positions.dtype)
        extended_positions[3::4] += self.z.to(positions.dtype)
        charges = torch.zeros_like(extended_positions[:, 0])
        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )
        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)
        scalar_features = charges_features_from_graph(
            charges=charges,
            positions=extended_positions,
            edge_index=edge_index,
            total_width_factors=self.total_width_factors.to(positions.dtype).unsqueeze(
                0
            ),
        )
        all_features = torch.zeros(
            batch.shape[0],
            4 * self.num_radial,
            dtype=positions.dtype,
            device=positions.device,
        )
        all_features[:, : self.num_radial] = (
            self.l0_factors.to(positions.dtype) * scalar_features[0::4]
        )
        all_features[:, self.num_radial :: 3] = self.l1_factors.to(positions.dtype) * (
            scalar_features[2::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors.to(
            positions.dtype
        ) * (scalar_features[3::4] - scalar_features[0::4])
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors.to(
            positions.dtype
        ) * (scalar_features[1::4] - scalar_features[0::4])
        return all_features

    def forward(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        source_lm = source_feats.squeeze(-2)
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self._density_0_feats_0(source_lm, node_positions, batch)
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded = torch.zeros(
                (source_feats.shape[0], 4),
                dtype=source_feats.dtype,
                device=source_feats.device,
            )
            padded[:, 0] = source_feats[:, 0, 0]
            features = self._density_1_feats_1(padded, node_positions, batch)
        elif self.density_max_l == 1 and self.projection_max_l == 0:
            features = self._density_1_feats_1(source_lm, node_positions, batch)[
                :, : self.num_radial
            ]
        else:
            features = self._density_1_feats_1(source_lm, node_positions, batch)

        self_interaction_terms = self.self_interaction(source_lm)
        if self.include_self_interaction:
            features = features + self_interaction_terms
        return features, self_interaction_terms, None


def assemble_fourier_series_batch(
    source_feats: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    n_nodes = source_feats.shape[0]
    n_sigma = density_basis_fs.shape[1]
    m_dim = density_basis_fs.shape[2]
    sm_dim = n_sigma * m_dim
    coeff_2d = source_feats.reshape(n_nodes, sm_dim)
    coeff_cos = torch.matmul(cosines, coeff_2d)
    coeff_sin = torch.matmul(sines, coeff_2d)
    density_basis_r = density_basis_fs[..., 0].reshape(
        density_basis_fs.shape[0], sm_dim
    )
    density_basis_i = density_basis_fs[..., 1].reshape(
        density_basis_fs.shape[0], sm_dim
    )
    rho_real = (density_basis_r * coeff_cos).sum(dim=-1) + (
        density_basis_i * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i * coeff_cos).sum(dim=-1) - (
        density_basis_r * coeff_sin
    ).sum(dim=-1)
    return (
        (2 * pi) ** 3
        * torch.stack([rho_real, rho_imag], dim=-1)
        / volume_per_k.unsqueeze(-1)
    )


def apply_coulomb_kernel_batch(
    k_norm2: torch.Tensor,
    density: torch.Tensor,
    k_factor_coulomb: torch.Tensor | None = None,
) -> torch.Tensor:
    if k_factor_coulomb is None:
        k_factor_coulomb = torch.where(
            k_norm2 == 0, torch.zeros_like(k_norm2), 1.0 / k_norm2
        )
    return (
        FIELD_CONSTANT
        * density
        * k_factor_coulomb.reshape(-1, *([1] * (density.dim() - 1)))
    )


def project_to_features_batch(
    potential: torch.Tensor,
    feature_basis_fs: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: torch.Tensor | None = None,
) -> torch.Tensor:
    n_k = feature_basis_fs.shape[0]
    n_sigma = feature_basis_fs.shape[1]
    m_dim = feature_basis_fs.shape[2]
    sm_dim = n_sigma * m_dim
    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)
    a_terms = (
        potential[:, 0].unsqueeze(-1) * proj_basis_r
        + potential[:, 1].unsqueeze(-1) * proj_basis_i
    )
    b_terms = (
        potential[:, 0].unsqueeze(-1) * proj_basis_i
        - potential[:, 1].unsqueeze(-1) * proj_basis_r
    )
    if k_factor_proj is not None:
        a_terms = a_terms * k_factor_proj.unsqueeze(-1)
        b_terms = b_terms * k_factor_proj.unsqueeze(-1)
    proj_total = 2.0 * (
        torch.matmul(a_terms.t(), cosines) + torch.matmul(b_terms.t(), sines)
    )
    return proj_total.t().reshape(cosines.shape[1], n_sigma, m_dim) / (2 * pi) ** 3


def energy_product_batch(
    density: torch.Tensor,
    potential: torch.Tensor,
    volume: torch.Tensor,
    k_vector_batch: torch.Tensor,
) -> torch.Tensor:
    per_k = 2.0 * torch.sum(density * potential, dim=-1)
    energy_k = torch.zeros(int(volume.shape[0]), dtype=per_k.dtype, device=per_k.device)
    energy_k.index_add_(0, k_vector_batch, per_k)
    return 0.5 * volume.reshape(-1) * energy_k / (2 * pi) ** 6


def _get_total_dipole_z(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_dipole_z = scatter_sum(
        src=node_positions[:, 2] * source_feats[:, 0], index=batch, dim=0
    )
    if source_feats.shape[-1] > 1:
        total_dipole_p = scatter_sum(src=source_feats[:, 1:4], index=batch, dim=0)
        total_dipole_z = total_dipole_z + total_dipole_p[:, 1]
    return total_dipole_z


def slab_dipole_correction_energy(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    return FIELD_CONSTANT / (4 * pi) * 2 * pi * total_dipole_z**2 / volumes


def slab_dipole_correction_node_fields(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    total_field_z = FIELD_CONSTANT * total_dipole_z / volumes
    spread_total_field_z = torch.index_select(total_field_z, 0, batch)
    node_fields = torch.zeros(
        (node_positions.shape[0], 4),
        dtype=node_positions.dtype,
        device=node_positions.device,
    )
    node_fields[:, 0] = spread_total_field_z * node_positions[:, 2]
    node_fields[:, 3] = spread_total_field_z
    return node_fields


class CorrectivePotentialBlock(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        quadrupole_feature_corrections: bool = False,
    ) -> None:
        super().__init__()
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = int(density_max_l)
        self.include_quadrupole_corrections = bool(quadrupole_feature_corrections)

    def forward(
        self,
        charge_coefficients: torch.Tensor,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
        q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)
        total_dipole = scatter_sum(src=q_r, index=batch, dim=0)
        r_squared = torch.sum(torch.square(positions), dim=-1)
        quadrupole = scatter_sum(
            src=r_squared * charge_coefficients[:, 0], index=batch, dim=0
        )

        if self.density_max_l > 0:
            local_dipoles = charge_coefficients[..., [3, 1, 2]]
            total_dipole = total_dipole + scatter_sum(
                src=local_dipoles, index=batch, dim=0
            )
            quadrupole = quadrupole + 2 * scatter_sum(
                src=torch.einsum('bi,bi->b', positions, local_dipoles),
                index=batch,
                dim=0,
            )

        spread_dipoles = torch.index_select(total_dipole, 0, batch)
        spread_total_charge = torch.index_select(total_charge, 0, batch)
        spread_volumes = torch.index_select(volumes, 0, batch)
        spread_total_quadrupole = torch.index_select(quadrupole, 0, batch)

        node_fields = torch.zeros(
            (positions.shape[0], 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        l_values = torch.pow(volumes, 0.333333)
        delta_v_0 = CUBIC_MADELUNG * self.const * total_charge / l_values
        node_delta_v = torch.index_select(delta_v_0, 0, batch)
        node_delta_v = node_delta_v - (
            self.const * 2 * pi * spread_total_charge * r_squared / (3 * spread_volumes)
        )
        node_delta_v = node_delta_v + (
            self.const
            * 4
            * pi
            * torch.einsum('bi,bi->b', spread_dipoles, positions)
            / (3 * spread_volumes)
        )
        node_delta_v = node_delta_v - (
            self.const * 2 * pi * spread_total_quadrupole / (3 * spread_volumes)
        )
        node_fields[:, 0] = node_delta_v

        quantity_a = spread_dipoles - spread_total_charge.unsqueeze(-1) * positions
        node_fields[:, 1:] = (
            4 * pi * self.const * quantity_a / (3 * spread_volumes.unsqueeze(-1))
        )
        return node_fields


class MonopoleDipoleCorrectionBlock(torch.nn.Module):
    def __init__(self, density_max_l: int) -> None:
        super().__init__()
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = int(density_max_l)

    def forward(
        self,
        charge_coefficients: torch.Tensor,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
        q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)
        total_dipole = scatter_sum(src=q_r, index=batch, dim=0)
        r_squared = torch.sum(torch.square(positions), dim=-1)
        quadrupole = scatter_sum(
            src=r_squared * charge_coefficients[:, 0], index=batch, dim=0
        )
        if self.density_max_l > 0:
            local_dipoles = charge_coefficients[..., [3, 1, 2]]
            total_dipole = total_dipole + scatter_sum(
                src=local_dipoles, index=batch, dim=0
            )
            quadrupole = quadrupole + 2 * scatter_sum(
                src=torch.einsum('bi,bi->b', positions, local_dipoles),
                index=batch,
                dim=0,
            )
        delta_e = (
            0.5
            * CUBIC_MADELUNG
            * self.const
            * torch.square(total_charge)
            / torch.pow(volumes, 0.3333)
        )
        delta_e = delta_e + 2 * self.const * pi * torch.sum(
            torch.square(total_dipole), dim=-1
        ) / (3 * volumes)
        return delta_e - 2 * self.const * pi * total_charge * quadrupole / (3 * volumes)


class NonPeriodicFeatureCorrections(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        projection_max_l: int,
        projection_smearing_widths: Sequence[float],
        integral_normalization: str = 'receiver',
    ) -> None:
        super().__init__()
        self.self_field = CorrectivePotentialBlock(
            density_max_l=density_max_l,
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=projection_max_l,
            sigmas_receive=projection_smearing_widths,
            normalize_receive=integral_normalization,
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        correction_mode: int | None = None,
        correction_node_masks: dict | None = None,
    ) -> torch.Tensor:
        source_lm = (
            source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        )
        if correction_mode is None:
            pbc_bool = pbc.to(dtype=torch.bool)
            is_pbc_graph = pbc_bool.all(dim=1)
            is_molecule_graph = (~pbc_bool).all(dim=1)
            is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
            if is_pbc_graph.all():
                correction_mode = CORRECTION_MODE_PBC
            elif is_molecule_graph.all():
                correction_mode = CORRECTION_MODE_MOLECULE
            elif is_slab_graph.all():
                correction_mode = CORRECTION_MODE_SLAB
            else:
                correction_mode = CORRECTION_MODE_MIXED
                correction_node_masks = {
                    'is_molecule_node': torch.index_select(is_molecule_graph, 0, batch),
                    'is_slab_node': torch.index_select(is_slab_graph, 0, batch),
                }

        if correction_mode == CORRECTION_MODE_PBC:
            return node_positions.new_zeros(
                (node_positions.shape[0], self.displaced_interactions.projections_dim)
            )
        if correction_mode == CORRECTION_MODE_MOLECULE:
            node_fields = self.self_field(source_lm, node_positions, volumes, batch)
            return self.displaced_interactions(batch, node_positions, node_fields)
        if correction_mode == CORRECTION_MODE_SLAB:
            node_fields = slab_dipole_correction_node_fields(
                source_lm,
                node_positions,
                volumes,
                batch,
            )
            return self.displaced_interactions(batch, node_positions, node_fields)

        if correction_node_masks is None:
            pbc_bool = pbc.to(dtype=torch.bool)
            is_molecule_graph = (~pbc_bool).all(dim=1)
            is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
            correction_node_masks = {
                'is_molecule_node': torch.index_select(is_molecule_graph, 0, batch),
                'is_slab_node': torch.index_select(is_slab_graph, 0, batch),
            }
        node_fields_molecule = self.self_field(
            source_lm, node_positions, volumes, batch
        )
        node_fields_slab = slab_dipole_correction_node_fields(
            source_lm,
            node_positions,
            volumes,
            batch,
        )
        node_fields = torch.zeros_like(node_fields_molecule)
        is_molecule = correction_node_masks['is_molecule_node']
        is_slab = correction_node_masks['is_slab_node']
        node_fields[is_molecule] = node_fields_molecule[is_molecule]
        node_fields[is_slab] = node_fields_slab[is_slab]
        return self.displaced_interactions(batch, node_positions, node_fields)


class GTOElectrostaticFeatures(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: Sequence[float],
        include_self_interaction: bool,
        kspace_cutoff: float,
        quadrupole_feature_corrections: bool = False,
        integral_normalization: str = 'receiver',
    ) -> None:
        super().__init__()
        del quadrupole_feature_corrections
        self.density_basis = GTOBasis(
            density_max_l, [density_smearing_width], kspace_cutoff, 'multipoles'
        )
        self.feature_basis = GTOBasis(
            feature_max_l,
            feature_smearing_widths,
            kspace_cutoff,
            integral_normalization,
        )
        self.include_self_interaction = bool(include_self_interaction)
        self.self_interaction_terms = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            feature_max_l,
            feature_smearing_widths,
            'multipoles',
            integral_normalization,
        )
        self.realspace_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l,
            density_smearing_width,
            feature_max_l,
            feature_smearing_widths,
            include_self_interaction,
            integral_normalization,
        )
        self.non_periodic_correction_terms = NonPeriodicFeatureCorrections(
            density_max_l,
            feature_max_l,
            feature_smearing_widths,
            integral_normalization,
        )
        self.register_buffer(
            'output_permutation',
            torch.as_tensor(
                output_permutation_np(feature_max_l, len(feature_smearing_widths)),
                dtype=torch.long,
            ),
        )

    @staticmethod
    def _build_correction_cache(pbc: torch.Tensor, batch: torch.Tensor) -> dict:
        pbc_bool = pbc.to(dtype=torch.bool)
        is_pbc_graph = pbc_bool.all(dim=1)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])

        if is_pbc_graph.all():
            return {'correction_mode': CORRECTION_MODE_PBC}
        if is_molecule_graph.all():
            return {'correction_mode': CORRECTION_MODE_MOLECULE}
        if is_slab_graph.all():
            return {'correction_mode': CORRECTION_MODE_SLAB}

        return {
            'correction_mode': CORRECTION_MODE_MIXED,
            'correction_node_masks': {
                'is_molecule_node': torch.index_select(is_molecule_graph, 0, batch),
                'is_slab_node': torch.index_select(is_slab_graph, 0, batch),
            },
        }

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> dict:
        if torch.any(pbc) or force_pbc_evaluator:
            inner_products = torch.matmul(k_vectors, node_positions.t())
            mask_f = (k_vector_batch[:, None] == batch[None, :]).to(
                inner_products.dtype
            )
            k0_mask_bool = k0_mask > 0.0
            k_factor_coulomb = torch.zeros_like(k_norm2)
            k_factor_coulomb[~k0_mask_bool] = 1.0 / k_norm2[~k0_mask_bool]
            k_factor_proj = torch.ones_like(k_norm2)
            k_factor_proj[k0_mask_bool] = 0.5
            correction_cache = self._build_correction_cache(pbc=pbc, batch=batch)
            return {
                'mode': 'pbc',
                'k_norm2': k_norm2,
                'k0_mask': k0_mask,
                'volume_per_k': volume.reshape(-1)[k_vector_batch],
                'k_factor_coulomb': k_factor_coulomb,
                'k_factor_proj': k_factor_proj,
                'volumes': volume.reshape(-1),
                'batch': batch,
                'node_positions': node_positions,
                'pbc': pbc,
                'cosines': torch.cos(inner_products) * mask_f,
                'sines': torch.sin(inner_products) * mask_f,
                'density_basis_fs': self.density_basis(k_vectors, k_norm2, k0_mask),
                'feature_basis_fs': self.feature_basis(k_vectors, k_norm2, k0_mask),
                **correction_cache,
            }
        return {'mode': 'realspace', 'node_positions': node_positions, 'batch': batch}

    def forward_dynamic(
        self,
        cache: dict,
        source_feats: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        del pbc
        if cache.get('mode') == 'realspace':
            features, _, _ = self.realspace_features(
                source_feats=source_feats,
                node_positions=cache['node_positions'],
                batch=cache['batch'],
            )
            return features
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache['cosines'],
            sines=cache['sines'],
            density_basis_fs=cache['density_basis_fs'],
            volume_per_k=cache['volume_per_k'],
        )
        potential = apply_coulomb_kernel_batch(
            k_norm2=cache['k_norm2'],
            density=density,
            k_factor_coulomb=cache['k_factor_coulomb'],
        )
        features_si = project_to_features_batch(
            potential=potential,
            feature_basis_fs=cache['feature_basis_fs'],
            cosines=cache['cosines'],
            sines=cache['sines'],
            k_factor_proj=cache['k_factor_proj'],
        )
        features_flat = features_si.reshape(features_si.shape[0], -1)
        features_flat = torch.index_select(
            features_flat, dim=-1, index=self.output_permutation
        )
        correction_mode = cache.get('correction_mode', CORRECTION_MODE_MIXED)
        if correction_mode != CORRECTION_MODE_PBC:
            correction_terms = self.non_periodic_correction_terms(
                source_feats=source_feats,
                node_positions=cache['node_positions'],
                batch=cache['batch'],
                volumes=cache['volumes'],
                pbc=cache['pbc'],
                correction_mode=correction_mode,
                correction_node_masks=cache.get('correction_node_masks'),
            )
        source_lm = (
            source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        )
        if not self.include_self_interaction:
            features_flat = features_flat - self.self_interaction_terms(source_lm)
        if correction_mode != CORRECTION_MODE_PBC:
            features_flat = features_flat + correction_terms
        return features_flat


class GTOElectrostaticEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        include_self_interaction: bool = False,
        include_pbc_corrections: bool = True,
    ) -> None:
        super().__init__()
        self.include_self_interaction = bool(include_self_interaction)
        self.include_pbc_corrections = bool(include_pbc_corrections)
        self.density_basis = GTOBasis(
            density_max_l, [density_smearing_width], kspace_cutoff, 'multipoles'
        )
        self.self_interaction_terms = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            'multipoles',
            'multipoles',
        )
        self.realspace_energy = RealSpaceFiniteDiffereneEnergy(
            density_max_l,
            density_smearing_width,
            include_self_interaction,
        )
        self.monopole_dipole_correction = MonopoleDipoleCorrectionBlock(density_max_l)

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> torch.Tensor:
        if not (torch.any(pbc) or force_pbc_evaluator):
            source_lm = (
                source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
            )
            return self.realspace_energy(source_lm, node_positions, batch)

        inner_products = torch.matmul(k_vectors, node_positions.t())
        mask_f = (k_vector_batch[:, None] == batch[None, :]).to(inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f
        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume.reshape(-1)[k_vector_batch],
        )
        k_factor = torch.zeros_like(k_norm2)
        k_factor[k0_mask <= 0.0] = 1.0 / k_norm2[k0_mask <= 0.0]
        potential = apply_coulomb_kernel_batch(k_norm2, density, k_factor)
        energy = energy_product_batch(density, potential, volume, k_vector_batch)
        source_lm = (
            source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        )
        if not self.include_self_interaction:
            self_fields = self.self_interaction_terms(source_lm)
            node_energies = torch.einsum('nb,nb->n', source_lm, self_fields)
            energy = energy - 0.5 * scatter_sum(
                src=node_energies, index=batch, dim=-1, dim_size=volume.shape[0]
            )
        if self.include_pbc_corrections:
            molecule_correction = self.monopole_dipole_correction(
                source_lm, node_positions, volume, batch
            )
            slab_correction = slab_dipole_correction_energy(
                source_lm, node_positions, volume, batch
            )
            slab = torch.tensor(
                [False, False, True], dtype=torch.bool, device=pbc.device
            )
            is_molecule = torch.all(torch.logical_not(pbc), dim=1)
            is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
            energy = energy + torch.where(
                is_molecule, molecule_correction, torch.zeros_like(energy)
            )
            energy = energy + torch.where(
                is_slab, slab_correction, torch.zeros_like(energy)
            )
        return energy


__all__ = [
    'DisplacedGTOExternalFieldBlock',
    'GTOElectrostaticEnergy',
    'GTOElectrostaticFeatures',
    'compute_k_vectors_flat',
]
