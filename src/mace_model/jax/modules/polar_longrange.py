from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
from flax import nnx
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
from mace_model.jax.adapters.e3nn import Irreps
from mace_model.jax.adapters.e3nn.o3 import SphericalHarmonics
from mace_model.jax.tools.dtype import default_dtype
from mace_model.jax.tools.scatter import scatter_sum

CORRECTION_MODE_PBC = 0
CORRECTION_MODE_MOLECULE = 1
CORRECTION_MODE_SLAB = 2
CORRECTION_MODE_MIXED = 3


def _cartesian_prod(*values: jnp.ndarray) -> jnp.ndarray:
    if any(int(value.shape[0]) == 0 for value in values):
        return jnp.zeros((0, len(values)), dtype=values[0].dtype)
    grids = jnp.meshgrid(*values, indexing='ij')
    return jnp.stack([grid.reshape(-1) for grid in grids], axis=-1)


def compute_k_vectors_flat(
    cutoff: float | jnp.ndarray,
    cell_vectors: jnp.ndarray,
    r_cell_vectors: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cutoff_value = jnp.asarray(cutoff, dtype=cell_vectors.dtype)
    norms = jnp.linalg.norm(cell_vectors, axis=-1)
    normed_lattice_vectors = cell_vectors / jnp.expand_dims(norms, axis=-1)
    dot_products = jnp.einsum('bij,bij->bi', r_cell_vectors, normed_lattice_vectors)
    max_ns = jnp.ceil(cutoff_value * jnp.power(dot_products, -1)).astype(jnp.int32)
    max_max_ns = jnp.max(max_ns, axis=0)
    n1max, n2max, n3max = (int(max_max_ns[0]), int(max_max_ns[1]), int(max_max_ns[2]))

    dtype = r_cell_vectors.dtype
    origin = _cartesian_prod(
        jnp.arange(0, 1, dtype=dtype),
        jnp.arange(0, 1, dtype=dtype),
        jnp.arange(0, 1, dtype=dtype),
    )
    open_half_line = _cartesian_prod(
        jnp.arange(0, 1, dtype=dtype),
        jnp.arange(0, 1, dtype=dtype),
        jnp.arange(1, n3max, dtype=dtype),
    )
    open_half_plane = _cartesian_prod(
        jnp.arange(0, 1, dtype=dtype),
        jnp.arange(1, n2max, dtype=dtype),
        jnp.arange(-n3max, n3max, dtype=dtype),
    )
    open_half_sphere = _cartesian_prod(
        jnp.arange(1, n1max, dtype=dtype),
        jnp.arange(-n2max, n2max, dtype=dtype),
        jnp.arange(-n3max, n3max, dtype=dtype),
    )
    kvecs = jnp.concatenate(
        (origin, open_half_line, open_half_plane, open_half_sphere),
        axis=0,
    )
    k_vectors = jnp.einsum('ni,bij->bnj', kvecs, r_cell_vectors)
    k_norm2 = jnp.einsum('bni,bni->bn', k_vectors, k_vectors)
    mask = k_norm2 <= cutoff_value * cutoff_value

    k_vectors_flat = []
    k_norm2_flat = []
    k_batch_flat = []
    k0_mask_flat = []
    for graph_i in range(int(k_vectors.shape[0])):
        graph_mask = np.asarray(mask[graph_i])
        graph_k = k_vectors[graph_i][graph_mask]
        graph_norm2 = k_norm2[graph_i][graph_mask]
        graph_k0_mask = jnp.zeros_like(graph_norm2).at[0].set(1.0)
        k_vectors_flat.append(graph_k)
        k_norm2_flat.append(graph_norm2)
        k_batch_flat.append(jnp.full((graph_k.shape[0],), graph_i, dtype=jnp.int32))
        k0_mask_flat.append(graph_k0_mask)

    return (
        jnp.concatenate(k_vectors_flat, axis=0),
        jnp.concatenate(k_norm2_flat, axis=0),
        jnp.concatenate(k_batch_flat, axis=0),
        jnp.concatenate(k0_mask_flat, axis=0),
    )


class RadialIntegralDirect(nnx.Module):
    def __init__(self, sigmas: Sequence[float], max_l: int) -> None:
        if max_l > 1:
            raise NotImplementedError('RadialIntegralDirect only supports max_l <= 1.')
        sigmas_array = jnp.asarray(sigmas, dtype=default_dtype())
        pref_const = (4 * pi) * np.sqrt(pi / 2.0)
        self.num_sigma = len(sigmas)
        self.max_l = int(max_l)
        self.sigma2 = sigmas_array * sigmas_array
        self.pref0 = pref_const * sigmas_array**3
        self.pref1 = pref_const * sigmas_array**5 if self.max_l == 1 else None

    def __call__(self, k_mods: jnp.ndarray) -> jnp.ndarray:
        k2 = k_mods * k_mods
        exp_term = jnp.exp(-0.5 * jnp.expand_dims(k2, axis=-1) * self.sigma2)
        if self.max_l == 0:
            return jnp.expand_dims(self.pref0 * exp_term, axis=-1)
        pref1 = jnp.asarray(self.pref1, dtype=k_mods.dtype)
        return jnp.stack(
            (
                self.pref0.astype(k_mods.dtype) * exp_term,
                pref1 * jnp.expand_dims(k_mods, axis=-1) * exp_term,
            ),
            axis=-1,
        )


class GTOBasis(nnx.Module):
    def __init__(
        self,
        max_l: int,
        sigmas: Sequence[float],
        kspace_cutoff: float,
        normalize: str,
    ) -> None:
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
        self.spherical_harmonics = SphericalHarmonics(
            Irreps.spherical_harmonics(self.max_l),
            normalize=True,
            normalization='integral',
            layout_str='mul_ir',
        )
        cl_inverse = normalization_denominator_np(self.sigmas, self.max_l, normalize)
        real_phase, imag_phase = phase_factors_np(self.max_l)
        self.cl_scale = jnp.asarray(1.0 / cl_inverse, dtype=default_dtype())
        self.expanded_l_indices = jnp.asarray(
            expanded_l_indices_np(self.max_l),
            dtype=jnp.int32,
        )
        self.real_phase_factors = jnp.asarray(real_phase, dtype=default_dtype())
        self.imag_phase_factors = jnp.asarray(imag_phase, dtype=default_dtype())
        self.permute_indices = jnp.asarray([1, 2, 0], dtype=jnp.int32)

    def __call__(
        self,
        k_vectors: jnp.ndarray,
        k_norm2: jnp.ndarray,
        k0_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        k_moduli = jnp.sqrt(jnp.clip(k_norm2, min=0.0))
        k_moduli = jnp.where(k0_mask > 0.0, jnp.zeros_like(k_moduli), k_moduli)
        yklm = self.spherical_harmonics(k_vectors[:, self.permute_indices])
        fnlk = self.radial_spline(k_moduli) * self.cl_scale.astype(k_moduli.dtype)
        expanded_fnlk = jnp.take(fnlk, self.expanded_l_indices, axis=-1)
        xnlk = expanded_fnlk * jnp.expand_dims(yklm, axis=-2)
        return jnp.stack(
            (
                xnlk * self.real_phase_factors.astype(xnlk.dtype),
                xnlk * self.imag_phase_factors.astype(xnlk.dtype),
            ),
            axis=-1,
        )


class GTOSelfInteractionBlock(nnx.Module):
    def __init__(
        self,
        l_source: int,
        sigma_source: float,
        l_receive: int,
        sigmas_receive: Sequence[float],
        normalize_source: str,
        normalize_receive: str,
    ) -> None:
        sh_irreps = Irreps.spherical_harmonics(l_receive)
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
        self.overlap_constants = jnp.asarray(constants, dtype=default_dtype())
        self.select_indices = jnp.asarray(indices, dtype=jnp.int32)

    def __call__(self, charge_density: jnp.ndarray) -> jnp.ndarray:
        qs_expanded = jnp.take(charge_density, self.select_indices, axis=-1)
        features = jnp.zeros(
            (charge_density.shape[0], self.features_irreps.dim),
            dtype=charge_density.dtype,
        )
        values = self.overlap_constants.astype(charge_density.dtype) * qs_expanded
        return features.at[..., : self.non_zero_terms].set(values)


class DisplacedGTOExternalFieldBlock(nnx.Module):
    def __init__(
        self,
        l_receive: int,
        sigmas_receive: Sequence[float],
        normalize_receive: str,
    ) -> None:
        self.projections_dim = (int(l_receive) + 1) ** 2 * len(sigmas_receive)
        self.matrix = jnp.asarray(
            external_field_matrix_np(l_receive, sigmas_receive, normalize_receive),
            dtype=default_dtype(),
        )

    def __call__(
        self,
        batch: jnp.ndarray,
        positions: jnp.ndarray,
        field: jnp.ndarray,
    ) -> jnp.ndarray:
        node_fields = field[batch]
        node_fields = node_fields.at[:, 0].set(
            node_fields[:, 0] + jnp.einsum('bi,bi->b', positions, node_fields[:, 1:])
        )
        node_fields = node_fields[:, jnp.asarray([0, 3, 1, 2], dtype=jnp.int32)]
        return jnp.einsum(
            'pf,nf->np', self.matrix.astype(node_fields.dtype), node_fields
        )


class GTOInternalFieldtoFeaturesBlock(DisplacedGTOExternalFieldBlock):
    def __call__(
        self,
        batch: jnp.ndarray,
        positions: jnp.ndarray,
        node_fields: jnp.ndarray,
    ) -> jnp.ndarray:
        del batch, positions
        node_fields = node_fields[:, jnp.asarray([0, 3, 1, 2], dtype=jnp.int32)]
        return jnp.einsum(
            'pf,nf->np', self.matrix.astype(node_fields.dtype), node_fields
        )


def batch_complete_graph_excluding_self_duplicates_vector(
    batch: jnp.ndarray,
    duplicates_per_node: int,
) -> jnp.ndarray:
    batch = batch.astype(jnp.int32)
    orig = jnp.arange(batch.shape[0], dtype=jnp.int32)
    batch2 = jnp.repeat(batch, duplicates_per_node)
    orig2 = jnp.repeat(orig, duplicates_per_node)
    num_graphs = int(jnp.max(batch2)) + 1 if batch2.size else 0
    edges = []
    for graph_i in range(num_graphs):
        nodes = jnp.nonzero(batch2 == graph_i, size=batch2.shape[0])[0]
        nodes = nodes[batch2[nodes] == graph_i]
        if nodes.shape[0] <= 1:
            continue
        num_nodes = nodes.shape[0]
        row = jnp.broadcast_to(nodes[:, None], (num_nodes, num_nodes)).reshape(-1)
        col = jnp.broadcast_to(nodes[None, :], (num_nodes, num_nodes)).reshape(-1)
        orig_row = jnp.broadcast_to(
            orig2[nodes][:, None], (num_nodes, num_nodes)
        ).reshape(-1)
        orig_col = jnp.broadcast_to(
            orig2[nodes][None, :], (num_nodes, num_nodes)
        ).reshape(-1)
        keep = orig_row != orig_col
        edges.append(jnp.stack([row[keep], col[keep]], axis=0))
    if not edges:
        return jnp.zeros((2, 0), dtype=jnp.int32)
    return jnp.concatenate(edges, axis=1)


def charges_energy_from_graph(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    edge_index: jnp.ndarray,
    batch: jnp.ndarray,
    density_smearing_width: float,
) -> jnp.ndarray:
    sender, receiver = edge_index
    r_ij = positions[receiver] - positions[sender]
    d_ij = jnp.linalg.norm(r_ij, axis=-1)
    smooth_reciprocal = jsp_special.erf(d_ij * 0.5 / density_smearing_width) / (
        jnp.abs(d_ij) + 1e-6
    )
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    num_graphs = int(jnp.max(batch)) + 1 if batch.size else 0
    if edge_energy.size == 0:
        return jnp.zeros((num_graphs,), dtype=charges.dtype)
    node_energies = scatter_sum(src=edge_energy, index=receiver, dim=-1)
    return scatter_sum(src=node_energies, index=batch, dim=-1, dim_size=num_graphs)


def charges_energy_from_pairs(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    batch: jnp.ndarray,
    density_smearing_width: float,
    *,
    original_nodes: jnp.ndarray | None = None,
    num_graphs: int | None = None,
) -> jnp.ndarray:
    if original_nodes is None:
        original_nodes = jnp.arange(positions.shape[0], dtype=jnp.int32)
    if num_graphs is None:
        num_graphs = int(batch.shape[0])

    same_graph = batch[:, None] == batch[None, :]
    different_node = original_nodes[:, None] != original_nodes[None, :]
    active_pair = same_graph & different_node
    r_ij = positions[None, :, :] - positions[:, None, :]
    safe_r_ij = jnp.where(
        jnp.expand_dims(active_pair, axis=-1),
        r_ij,
        jnp.ones_like(r_ij),
    )
    d_ij = jnp.linalg.norm(safe_r_ij, axis=-1)
    mask = active_pair.astype(positions.dtype)
    smooth_reciprocal = jsp_special.erf(d_ij * 0.5 / density_smearing_width) / (
        d_ij + 1e-6
    )
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[:, None]
        * charges[None, :]
        * mask
        / (4 * pi)
    )
    node_energies = jnp.sum(edge_energy, axis=0)
    return scatter_sum(src=node_energies, index=batch, dim=-1, dim_size=num_graphs)


def charges_features_from_graph(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    edge_index: jnp.ndarray,
    total_width_factors: jnp.ndarray,
) -> jnp.ndarray:
    sender, receiver = edge_index
    r_ij = positions[sender] - positions[receiver]
    d_ij = jnp.linalg.norm(r_ij, axis=-1, keepdims=True)
    smooth_reciprocal = jsp_special.erf(0.5 * d_ij / total_width_factors) / (
        d_ij + 1e-6
    )
    updates = charges[sender, None] * smooth_reciprocal
    features = jnp.zeros(
        (positions.shape[0], total_width_factors.shape[-1]),
        dtype=positions.dtype,
    )
    return FIELD_CONSTANT * features.at[receiver].add(updates) / (4 * pi)


def charges_features_from_pairs(
    charges: jnp.ndarray,
    positions: jnp.ndarray,
    batch: jnp.ndarray,
    total_width_factors: jnp.ndarray,
    *,
    original_nodes: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if original_nodes is None:
        original_nodes = jnp.arange(positions.shape[0], dtype=jnp.int32)
    same_graph = batch[:, None] == batch[None, :]
    different_node = original_nodes[:, None] != original_nodes[None, :]
    active_pair = same_graph & different_node
    r_ij = positions[:, None, :] - positions[None, :, :]
    safe_r_ij = jnp.where(
        jnp.expand_dims(active_pair, axis=-1),
        r_ij,
        jnp.ones_like(r_ij),
    )
    d_ij = jnp.linalg.norm(safe_r_ij, axis=-1, keepdims=True)
    mask = jnp.expand_dims(active_pair.astype(positions.dtype), axis=-1)
    smooth_reciprocal = jsp_special.erf(0.5 * d_ij / total_width_factors) / (
        d_ij + 1e-6
    )
    updates = charges[:, None, None] * smooth_reciprocal * mask
    features = jnp.sum(updates, axis=0)
    return FIELD_CONSTANT * features / (4 * pi)


class RealSpaceFiniteDiffereneEnergy(nnx.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset: float = 0.02,
    ) -> None:
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
        self.x = jnp.asarray([offset, 0.0, 0.0], dtype=default_dtype())
        self.y = jnp.asarray([0.0, offset, 0.0], dtype=default_dtype())
        self.z = jnp.asarray([0.0, 0.0, offset], dtype=default_dtype())

    def __call__(
        self,
        source_feats: jnp.ndarray,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
        num_graphs: int | None = None,
    ) -> jnp.ndarray:
        if num_graphs is None:
            num_graphs = int(batch.shape[0])
        if self.density_max_l == 0:
            energy = charges_energy_from_pairs(
                source_feats.squeeze(-1),
                positions,
                batch,
                self.density_smearing_width,
                num_graphs=num_graphs,
            )
        else:
            extended_positions = jnp.repeat(positions, 4, axis=0)
            extended_positions = extended_positions.at[1::4].add(
                self.x.astype(positions.dtype)
            )
            extended_positions = extended_positions.at[2::4].add(
                self.y.astype(positions.dtype)
            )
            extended_positions = extended_positions.at[3::4].add(
                self.z.astype(positions.dtype)
            )
            extended_batch = jnp.repeat(batch, 4)
            charges = jnp.zeros((extended_positions.shape[0],), dtype=positions.dtype)
            charges = charges.at[1::4].set(source_feats[:, 3] / self.offset)
            charges = charges.at[2::4].set(source_feats[:, 1] / self.offset)
            charges = charges.at[3::4].set(source_feats[:, 2] / self.offset)
            charges = charges.at[0::4].set(
                source_feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
            )
            energy = charges_energy_from_pairs(
                charges,
                extended_positions,
                extended_batch,
                self.density_smearing_width,
                original_nodes=jnp.repeat(
                    jnp.arange(positions.shape[0], dtype=jnp.int32),
                    4,
                ),
                num_graphs=num_graphs,
            )
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)
            node_energies = jnp.einsum('nb,nb->n', source_feats, self_fields)
            energy = energy + 0.5 * scatter_sum(
                src=node_energies,
                index=batch,
                dim=-1,
                dim_size=num_graphs,
            )
        return energy


class RealSpaceFiniteDifferenceElectrostaticFeatures(nnx.Module):
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
        widths = jnp.asarray(projection_smearing_widths, dtype=default_dtype())
        self.total_width_factors = jnp.sqrt(
            (float(density_smearing_width) ** 2 + widths**2) / 2
        )
        self.x = jnp.asarray([offset, 0.0, 0.0], dtype=default_dtype())
        self.y = jnp.asarray([0.0, offset, 0.0], dtype=default_dtype())
        self.z = jnp.asarray([0.0, 0.0, offset], dtype=default_dtype())
        self.l0_factors = jnp.asarray(
            [
                cl_sigma_np(0, sigma, integral_normalization)
                / cl_sigma_np(0, sigma, 'multipoles')
                for sigma in projection_smearing_widths
            ],
            dtype=default_dtype(),
        )
        self.l1_factors = jnp.asarray(
            [
                3**0.5
                * float(sigma) ** 2
                * (
                    cl_sigma_np(1, sigma, integral_normalization)
                    / cl_sigma_np(0, sigma, 'multipoles')
                )
                / self.offset
                for sigma in projection_smearing_widths
            ],
            dtype=default_dtype(),
        )

    def _density_0_feats_0(
        self,
        source_feats: jnp.ndarray,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        feats = charges_features_from_pairs(
            charges=source_feats[:, 0],
            positions=positions,
            batch=batch,
            total_width_factors=jnp.expand_dims(
                self.total_width_factors.astype(positions.dtype),
                axis=0,
            ),
        )
        return self.l0_factors.astype(feats.dtype) * feats

    def _density_1_feats_1(
        self,
        source_feats: jnp.ndarray,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        extended_positions = jnp.repeat(positions, 4, axis=0)
        extended_positions = extended_positions.at[1::4].add(
            self.x.astype(positions.dtype)
        )
        extended_positions = extended_positions.at[2::4].add(
            self.y.astype(positions.dtype)
        )
        extended_positions = extended_positions.at[3::4].add(
            self.z.astype(positions.dtype)
        )
        charges = jnp.zeros((extended_positions.shape[0],), dtype=positions.dtype)
        charges = charges.at[1::4].set(source_feats[:, 3] / self.offset)
        charges = charges.at[2::4].set(source_feats[:, 1] / self.offset)
        charges = charges.at[3::4].set(source_feats[:, 2] / self.offset)
        charges = charges.at[0::4].set(
            source_feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
        )
        scalar_features = charges_features_from_pairs(
            charges=charges,
            positions=extended_positions,
            batch=jnp.repeat(batch, 4),
            total_width_factors=jnp.expand_dims(
                self.total_width_factors.astype(positions.dtype),
                axis=0,
            ),
            original_nodes=jnp.repeat(
                jnp.arange(positions.shape[0], dtype=jnp.int32),
                4,
            ),
        )
        all_features = jnp.zeros(
            (batch.shape[0], 4 * self.num_radial),
            dtype=positions.dtype,
        )
        all_features = all_features.at[:, : self.num_radial].set(
            self.l0_factors.astype(positions.dtype) * scalar_features[0::4]
        )
        all_features = all_features.at[:, self.num_radial :: 3].set(
            self.l1_factors.astype(positions.dtype)
            * (scalar_features[2::4] - scalar_features[0::4])
        )
        all_features = all_features.at[:, self.num_radial + 1 :: 3].set(
            self.l1_factors.astype(positions.dtype)
            * (scalar_features[3::4] - scalar_features[0::4])
        )
        all_features = all_features.at[:, self.num_radial + 2 :: 3].set(
            self.l1_factors.astype(positions.dtype)
            * (scalar_features[1::4] - scalar_features[0::4])
        )
        return all_features

    def __call__(
        self,
        source_feats: jnp.ndarray,
        node_positions: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        source_lm = jnp.squeeze(source_feats, axis=-2)
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self._density_0_feats_0(source_lm, node_positions, batch)
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded = jnp.zeros((source_feats.shape[0], 4), dtype=source_feats.dtype)
            padded = padded.at[:, 0].set(source_feats[:, 0, 0])
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
    source_feats: jnp.ndarray,
    cosines: jnp.ndarray,
    sines: jnp.ndarray,
    density_basis_fs: jnp.ndarray,
    volume_per_k: jnp.ndarray,
) -> jnp.ndarray:
    n_nodes = source_feats.shape[0]
    n_sigma = density_basis_fs.shape[1]
    m_dim = density_basis_fs.shape[2]
    sm_dim = n_sigma * m_dim
    coeff_2d = source_feats.reshape(n_nodes, sm_dim)
    coeff_cos = jnp.matmul(cosines, coeff_2d)
    coeff_sin = jnp.matmul(sines, coeff_2d)
    density_basis_r = density_basis_fs[..., 0].reshape(
        density_basis_fs.shape[0], sm_dim
    )
    density_basis_i = density_basis_fs[..., 1].reshape(
        density_basis_fs.shape[0], sm_dim
    )
    rho_real = jnp.sum(density_basis_r * coeff_cos, axis=-1) + jnp.sum(
        density_basis_i * coeff_sin,
        axis=-1,
    )
    rho_imag = jnp.sum(density_basis_i * coeff_cos, axis=-1) - jnp.sum(
        density_basis_r * coeff_sin,
        axis=-1,
    )
    return (
        (2 * pi) ** 3
        * jnp.stack([rho_real, rho_imag], axis=-1)
        / jnp.expand_dims(volume_per_k, axis=-1)
    )


def apply_coulomb_kernel_batch(
    k_norm2: jnp.ndarray,
    density: jnp.ndarray,
    k_factor_coulomb: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if k_factor_coulomb is None:
        k_factor_coulomb = jnp.where(
            k_norm2 == 0, jnp.zeros_like(k_norm2), 1.0 / k_norm2
        )
    return (
        FIELD_CONSTANT
        * density
        * k_factor_coulomb.reshape((-1,) + (1,) * (density.ndim - 1))
    )


def project_to_features_batch(
    potential: jnp.ndarray,
    feature_basis_fs: jnp.ndarray,
    cosines: jnp.ndarray,
    sines: jnp.ndarray,
    k_factor_proj: jnp.ndarray | None = None,
) -> jnp.ndarray:
    n_k = feature_basis_fs.shape[0]
    n_sigma = feature_basis_fs.shape[1]
    m_dim = feature_basis_fs.shape[2]
    sm_dim = n_sigma * m_dim
    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)
    a_terms = (
        jnp.expand_dims(potential[:, 0], axis=-1) * proj_basis_r
        + jnp.expand_dims(
            potential[:, 1],
            axis=-1,
        )
        * proj_basis_i
    )
    b_terms = (
        jnp.expand_dims(potential[:, 0], axis=-1) * proj_basis_i
        - jnp.expand_dims(
            potential[:, 1],
            axis=-1,
        )
        * proj_basis_r
    )
    if k_factor_proj is not None:
        a_terms = a_terms * jnp.expand_dims(k_factor_proj, axis=-1)
        b_terms = b_terms * jnp.expand_dims(k_factor_proj, axis=-1)
    proj_total = 2.0 * (jnp.matmul(a_terms.T, cosines) + jnp.matmul(b_terms.T, sines))
    return proj_total.T.reshape(cosines.shape[1], n_sigma, m_dim) / (2 * pi) ** 3


def energy_product_batch(
    density: jnp.ndarray,
    potential: jnp.ndarray,
    volume: jnp.ndarray,
    k_vector_batch: jnp.ndarray,
) -> jnp.ndarray:
    per_k = 2.0 * jnp.sum(density * potential, axis=-1)
    energy_k = scatter_sum(
        src=per_k,
        index=k_vector_batch,
        dim=-1,
        dim_size=int(volume.shape[0]),
    )
    return 0.5 * volume.reshape(-1) * energy_k / (2 * pi) ** 6


def _get_total_dipole_z(
    source_feats: jnp.ndarray,
    node_positions: jnp.ndarray,
    batch: jnp.ndarray,
    num_graphs: int,
) -> jnp.ndarray:
    total_dipole_z = scatter_sum(
        src=node_positions[:, 2] * source_feats[:, 0],
        index=batch,
        dim=0,
        dim_size=num_graphs,
    )
    if source_feats.shape[-1] > 1:
        total_dipole_p = scatter_sum(
            src=source_feats[:, 1:4],
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        total_dipole_z = total_dipole_z + total_dipole_p[:, 1]
    return total_dipole_z


def slab_dipole_correction_energy(
    source_feats: jnp.ndarray,
    node_positions: jnp.ndarray,
    volumes: jnp.ndarray,
    batch: jnp.ndarray,
) -> jnp.ndarray:
    total_dipole_z = _get_total_dipole_z(
        source_feats,
        node_positions,
        batch,
        num_graphs=volumes.shape[0],
    )
    return FIELD_CONSTANT / (4 * pi) * 2 * pi * total_dipole_z**2 / volumes


def slab_dipole_correction_node_fields(
    source_feats: jnp.ndarray,
    node_positions: jnp.ndarray,
    volumes: jnp.ndarray,
    batch: jnp.ndarray,
) -> jnp.ndarray:
    total_dipole_z = _get_total_dipole_z(
        source_feats,
        node_positions,
        batch,
        num_graphs=volumes.shape[0],
    )
    total_field_z = FIELD_CONSTANT * total_dipole_z / volumes
    spread_total_field_z = jnp.take(total_field_z, batch, axis=0)
    node_fields = jnp.zeros((node_positions.shape[0], 4), dtype=node_positions.dtype)
    node_fields = node_fields.at[:, 0].set(spread_total_field_z * node_positions[:, 2])
    return node_fields.at[:, 3].set(spread_total_field_z)


class CorrectivePotentialBlock(nnx.Module):
    def __init__(
        self,
        density_max_l: int,
        quadrupole_feature_corrections: bool = False,
    ) -> None:
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = int(density_max_l)
        self.include_quadrupole_corrections = bool(quadrupole_feature_corrections)

    def __call__(
        self,
        charge_coefficients: jnp.ndarray,
        positions: jnp.ndarray,
        volumes: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        num_graphs = volumes.shape[0]
        total_charge = scatter_sum(
            src=charge_coefficients[:, 0],
            index=batch,
            dim=-1,
            dim_size=num_graphs,
        )
        total_dipole = scatter_sum(
            src=positions * jnp.expand_dims(charge_coefficients[:, 0], axis=-1),
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        r_squared = jnp.sum(jnp.square(positions), axis=-1)
        quadrupole = scatter_sum(
            src=r_squared * charge_coefficients[:, 0],
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )

        if self.density_max_l > 0:
            local_dipoles = charge_coefficients[
                ..., jnp.asarray([3, 1, 2], dtype=jnp.int32)
            ]
            total_dipole = total_dipole + scatter_sum(
                src=local_dipoles,
                index=batch,
                dim=0,
                dim_size=num_graphs,
            )
            quadrupole = quadrupole + 2 * scatter_sum(
                src=jnp.einsum('bi,bi->b', positions, local_dipoles),
                index=batch,
                dim=0,
                dim_size=num_graphs,
            )

        spread_dipoles = jnp.take(total_dipole, batch, axis=0)
        spread_total_charge = jnp.take(total_charge, batch, axis=0)
        spread_volumes = jnp.take(volumes, batch, axis=0)
        spread_total_quadrupole = jnp.take(quadrupole, batch, axis=0)

        l_values = jnp.power(volumes, 0.333333)
        delta_v_0 = CUBIC_MADELUNG * self.const * total_charge / l_values
        node_delta_v = jnp.take(delta_v_0, batch, axis=0)
        node_delta_v = node_delta_v - (
            self.const * 2 * pi * spread_total_charge * r_squared / (3 * spread_volumes)
        )
        node_delta_v = node_delta_v + (
            self.const
            * 4
            * pi
            * jnp.einsum('bi,bi->b', spread_dipoles, positions)
            / (3 * spread_volumes)
        )
        node_delta_v = node_delta_v - (
            self.const * 2 * pi * spread_total_quadrupole / (3 * spread_volumes)
        )

        quantity_a = (
            spread_dipoles - jnp.expand_dims(spread_total_charge, axis=-1) * positions
        )
        node_fields = jnp.zeros((positions.shape[0], 4), dtype=positions.dtype)
        node_fields = node_fields.at[:, 0].set(node_delta_v)
        return node_fields.at[:, 1:].set(
            4
            * pi
            * self.const
            * quantity_a
            / (3 * jnp.expand_dims(spread_volumes, axis=-1))
        )


class MonopoleDipoleCorrectionBlock(nnx.Module):
    def __init__(self, density_max_l: int) -> None:
        self.const = FIELD_CONSTANT / (4 * pi)
        self.density_max_l = int(density_max_l)

    def __call__(
        self,
        charge_coefficients: jnp.ndarray,
        positions: jnp.ndarray,
        volumes: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        num_graphs = volumes.shape[0]
        total_charge = scatter_sum(
            src=charge_coefficients[:, 0],
            index=batch,
            dim=-1,
            dim_size=num_graphs,
        )
        total_dipole = scatter_sum(
            src=positions * jnp.expand_dims(charge_coefficients[:, 0], axis=-1),
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        r_squared = jnp.sum(jnp.square(positions), axis=-1)
        quadrupole = scatter_sum(
            src=r_squared * charge_coefficients[:, 0],
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        if self.density_max_l > 0:
            local_dipoles = charge_coefficients[
                ..., jnp.asarray([3, 1, 2], dtype=jnp.int32)
            ]
            total_dipole = total_dipole + scatter_sum(
                src=local_dipoles,
                index=batch,
                dim=0,
                dim_size=num_graphs,
            )
            quadrupole = quadrupole + 2 * scatter_sum(
                src=jnp.einsum('bi,bi->b', positions, local_dipoles),
                index=batch,
                dim=0,
                dim_size=num_graphs,
            )
        delta_e = (
            0.5
            * CUBIC_MADELUNG
            * self.const
            * jnp.square(total_charge)
            / jnp.power(volumes, 0.3333)
        )
        delta_e = delta_e + 2 * self.const * pi * jnp.sum(
            jnp.square(total_dipole), axis=-1
        ) / (3 * volumes)
        return delta_e - 2 * self.const * pi * total_charge * quadrupole / (3 * volumes)


class NonPeriodicFeatureCorrections(nnx.Module):
    def __init__(
        self,
        density_max_l: int,
        projection_max_l: int,
        projection_smearing_widths: Sequence[float],
        integral_normalization: str = 'receiver',
    ) -> None:
        self.self_field = CorrectivePotentialBlock(
            density_max_l=density_max_l,
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=projection_max_l,
            sigmas_receive=projection_smearing_widths,
            normalize_receive=integral_normalization,
        )

    def __call__(
        self,
        source_feats: jnp.ndarray,
        node_positions: jnp.ndarray,
        batch: jnp.ndarray,
        volumes: jnp.ndarray,
        pbc: jnp.ndarray,
        correction_mode: int | None = None,
        correction_node_masks: dict | None = None,
    ) -> jnp.ndarray:
        source_lm = (
            jnp.squeeze(source_feats, axis=-2)
            if source_feats.ndim == 3
            else source_feats
        )
        if correction_mode is None:
            correction_mode = CORRECTION_MODE_MIXED

        if correction_mode == CORRECTION_MODE_PBC:
            return jnp.zeros(
                (node_positions.shape[0], self.displaced_interactions.projections_dim),
                dtype=node_positions.dtype,
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
            pbc_bool = pbc.astype(bool)
            is_molecule_graph = jnp.all(jnp.logical_not(pbc_bool), axis=1)
            is_slab_graph = (
                pbc_bool[:, 0] & pbc_bool[:, 1] & jnp.logical_not(pbc_bool[:, 2])
            )
            correction_node_masks = {
                'is_molecule_node': jnp.take(is_molecule_graph, batch, axis=0),
                'is_slab_node': jnp.take(is_slab_graph, batch, axis=0),
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
        is_molecule = jnp.expand_dims(
            correction_node_masks['is_molecule_node'], axis=-1
        )
        is_slab = jnp.expand_dims(correction_node_masks['is_slab_node'], axis=-1)
        node_fields = jnp.where(
            is_molecule, node_fields_molecule, jnp.zeros_like(node_fields_molecule)
        )
        node_fields = jnp.where(is_slab, node_fields_slab, node_fields)
        return self.displaced_interactions(batch, node_positions, node_fields)


class GTOElectrostaticFeatures(nnx.Module):
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
        del quadrupole_feature_corrections
        self.density_basis = GTOBasis(
            density_max_l,
            [density_smearing_width],
            kspace_cutoff,
            'multipoles',
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
        self.output_permutation = jnp.asarray(
            output_permutation_np(feature_max_l, len(feature_smearing_widths)),
            dtype=jnp.int32,
        )

    def precompute_geometry(
        self,
        k_vectors: jnp.ndarray,
        k_norm2: jnp.ndarray,
        k_vector_batch: jnp.ndarray,
        k0_mask: jnp.ndarray,
        node_positions: jnp.ndarray,
        batch: jnp.ndarray,
        volume: jnp.ndarray,
        pbc: jnp.ndarray,
        mode: str = 'realspace',
    ) -> dict:
        if mode not in {'realspace', 'pbc'}:
            raise ValueError(f'Unsupported electrostatic feature mode {mode!r}.')
        if mode == 'realspace':
            return {
                'mode': 'realspace',
                'node_positions': node_positions,
                'batch': batch,
            }

        inner_products = jnp.matmul(k_vectors, node_positions.T)
        mask_f = (k_vector_batch[:, None] == batch[None, :]).astype(
            inner_products.dtype
        )
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = jnp.where(
            k0_mask_bool, jnp.zeros_like(k_norm2), 1.0 / k_norm2
        )
        k_factor_proj = jnp.where(
            k0_mask_bool, 0.5 * jnp.ones_like(k_norm2), jnp.ones_like(k_norm2)
        )
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
            'cosines': jnp.cos(inner_products) * mask_f,
            'sines': jnp.sin(inner_products) * mask_f,
            'density_basis_fs': self.density_basis(k_vectors, k_norm2, k0_mask),
            'feature_basis_fs': self.feature_basis(k_vectors, k_norm2, k0_mask),
            'correction_mode': CORRECTION_MODE_PBC,
        }

    def forward_dynamic(
        self,
        cache: dict,
        source_feats: jnp.ndarray,
        pbc: jnp.ndarray,
    ) -> jnp.ndarray:
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
        features_flat = jnp.take(features_flat, self.output_permutation, axis=-1)
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
            jnp.squeeze(source_feats, axis=-2)
            if source_feats.ndim == 3
            else source_feats
        )
        if not self.include_self_interaction:
            features_flat = features_flat - self.self_interaction_terms(source_lm)
        if correction_mode != CORRECTION_MODE_PBC:
            features_flat = features_flat + correction_terms
        return features_flat


class GTOElectrostaticEnergy(nnx.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        include_self_interaction: bool = False,
        include_pbc_corrections: bool = True,
    ) -> None:
        self.include_self_interaction = bool(include_self_interaction)
        self.include_pbc_corrections = bool(include_pbc_corrections)
        self.density_basis = GTOBasis(
            density_max_l,
            [density_smearing_width],
            kspace_cutoff,
            'multipoles',
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

    def __call__(
        self,
        k_vectors: jnp.ndarray,
        k_norm2: jnp.ndarray,
        k_vector_batch: jnp.ndarray,
        k0_mask: jnp.ndarray,
        source_feats: jnp.ndarray,
        node_positions: jnp.ndarray,
        batch: jnp.ndarray,
        volume: jnp.ndarray,
        pbc: jnp.ndarray,
        mode: str = 'realspace',
    ) -> jnp.ndarray:
        source_lm = (
            jnp.squeeze(source_feats, axis=-2)
            if source_feats.ndim == 3
            else source_feats
        )
        if mode not in {'realspace', 'pbc'}:
            raise ValueError(f'Unsupported electrostatic energy mode {mode!r}.')
        if mode == 'realspace':
            return self.realspace_energy(
                source_lm,
                node_positions,
                batch,
                num_graphs=int(volume.shape[0]),
            )

        inner_products = jnp.matmul(k_vectors, node_positions.T)
        mask_f = (k_vector_batch[:, None] == batch[None, :]).astype(
            inner_products.dtype
        )
        cosines = jnp.cos(inner_products) * mask_f
        sines = jnp.sin(inner_products) * mask_f
        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume.reshape(-1)[k_vector_batch],
        )
        k_factor = jnp.where(k0_mask > 0.0, jnp.zeros_like(k_norm2), 1.0 / k_norm2)
        potential = apply_coulomb_kernel_batch(k_norm2, density, k_factor)
        energy = energy_product_batch(density, potential, volume, k_vector_batch)
        if not self.include_self_interaction:
            self_fields = self.self_interaction_terms(source_lm)
            node_energies = jnp.einsum('nb,nb->n', source_lm, self_fields)
            energy = energy - 0.5 * scatter_sum(
                src=node_energies,
                index=batch,
                dim=-1,
                dim_size=volume.shape[0],
            )
        if self.include_pbc_corrections and mode != 'pbc':
            molecule_correction = self.monopole_dipole_correction(
                source_lm,
                node_positions,
                volume,
                batch,
            )
            slab_correction = slab_dipole_correction_energy(
                source_lm,
                node_positions,
                volume,
                batch,
            )
            slab = jnp.asarray([False, False, True], dtype=bool)
            is_molecule = jnp.all(jnp.logical_not(pbc), axis=1)
            is_slab = jnp.all(jnp.logical_xor(slab, pbc), axis=1)
            energy = energy + jnp.where(
                is_molecule, molecule_correction, jnp.zeros_like(energy)
            )
            energy = energy + jnp.where(
                is_slab, slab_correction, jnp.zeros_like(energy)
            )
        return energy


__all__ = [
    'DisplacedGTOExternalFieldBlock',
    'GTOElectrostaticEnergy',
    'GTOElectrostaticFeatures',
    'compute_k_vectors_flat',
]
