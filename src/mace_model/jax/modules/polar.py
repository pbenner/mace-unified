from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from flax import nnx

from mace_model.core.modules.models import PolarMACEModel
from mace_model.core.modules.polar import (
    estimate_gto_basis_kspace_cutoff,
    expand_field_feature_norms,
    layout_from_cueq_config,
)
from mace_model.jax.adapters.e3nn import Irrep, Irreps
from mace_model.jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_model.jax.modules.utils import add_output_interface, prepare_graph
from mace_model.jax.tools.scatter import scatter_sum

from ..tools.dtype import default_dtype
from .backends import JAX_BACKEND
from .blocks import NonLinearBiasReadoutBlock, NonLinearReadoutBlock
from .field_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding,
    EnvironmentDependentSpinSourceBlock,
    MLPNonLinearity,
    MultiLayerFeatureMixer,
    NoNonLinearity,
    field_readout_blocks,
    field_update_blocks,
)
from .models import ScaleShiftMACE, _apply_lammps_exchange
from .polar_longrange import (
    DisplacedGTOExternalFieldBlock,
    GTOElectrostaticEnergy,
    GTOElectrostaticFeatures,
    compute_k_vectors_flat,
)


def _permute_to_e3nn_convention(value: jnp.ndarray) -> jnp.ndarray:
    return value[..., jnp.asarray([1, 2, 0], dtype=jnp.int32)]


def _scatter_mean(
    *,
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
) -> jnp.ndarray:
    summed = scatter_sum(src=src, index=index, dim=0, dim_size=dim_size)
    counts = scatter_sum(
        src=jnp.ones((src.shape[0],), dtype=src.dtype),
        index=index,
        dim=0,
        dim_size=dim_size,
    )
    return summed / jnp.expand_dims(jnp.clip(counts, min=1), axis=-1)


def _compute_total_charge_dipole_permuted(
    density_coefficients: jnp.ndarray,
    positions: jnp.ndarray,
    batch: jnp.ndarray,
    num_graphs: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dipole = scatter_sum(
        src=positions * density_coefficients[:, :1],
        index=batch,
        dim=0,
        dim_size=num_graphs,
    )
    if density_coefficients.shape[1] > 1:
        dipole_p = scatter_sum(
            src=density_coefficients[..., 1:4],
            index=batch,
            dim=0,
            dim_size=num_graphs,
        )
        dipole = dipole + dipole_p[..., jnp.asarray([2, 0, 1], dtype=jnp.int32)]

    total_charge = scatter_sum(
        src=density_coefficients[:, 0],
        index=batch,
        dim=-1,
        dim_size=num_graphs,
    )
    return total_charge, dipole


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class PolarMACE(PolarMACEModel, ScaleShiftMACE):
    """JAX-side PolarMACE module with local GTO electrostatic descriptors."""

    def __init__(
        self,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: Sequence[float] = (1.0,),
        num_recursion_steps: int = 1,
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        field_feature_norms: Sequence[float] | None = None,
        field_norm_factor: float | None = 0.02,
        fixedpoint_update_config: dict[str, Any] | None = None,
        field_readout_config: dict[str, Any] | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        constructor_args = self.require_polar_mace_kwargs(
            kwargs,
            make_irreps=Irreps,
        )
        hidden_irreps = constructor_args.hidden_irreps
        mlp_irreps = constructor_args.mlp_irreps
        gate = constructor_args.gate
        avg_num_neighbors = constructor_args.avg_num_neighbors
        num_interactions = constructor_args.num_interactions
        num_elements = constructor_args.num_elements

        kwargs = self.prepare_polar_base_kwargs(
            kwargs,
            readout_cls=NonLinearReadoutBlock,
        )
        kwargs.pop('keep_last_layer_irreps', None)
        kwargs['collapse_hidden_irreps'] = False
        oeq_config = kwargs.pop('oeq_config', None)
        cueq_config = kwargs.get('cueq_config')

        super().__init__(rngs=rngs, **kwargs)

        self.initialize_polar_common_attributes(
            kspace_cutoff_factor=kspace_cutoff_factor,
            atomic_multipoles_max_l=atomic_multipoles_max_l,
            atomic_multipoles_smearing_width=atomic_multipoles_smearing_width,
            field_feature_max_l=field_feature_max_l,
            field_feature_widths=field_feature_widths,
            num_recursion_steps=num_recursion_steps,
            field_si=field_si,
            include_electrostatic_self_interaction=include_electrostatic_self_interaction,
            add_local_electron_energy=add_local_electron_energy,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
            return_electrostatic_potentials=return_electrostatic_potentials,
            field_feature_norms=field_feature_norms,
            field_norm_factor=field_norm_factor,
            fixedpoint_update_config=fixedpoint_update_config,
            field_readout_config=field_readout_config,
        )

        kspace_cutoff = self.kspace_cutoff_factor * estimate_gto_basis_kspace_cutoff(
            [self.atomic_multipoles_smearing_width] + self.field_feature_widths,
            max(self.atomic_multipoles_max_l, self.field_feature_max_l),
        )
        self.kspace_cutoff = jnp.asarray(kspace_cutoff, dtype=default_dtype())

        self.field_feature_norms = jnp.asarray(
            expand_field_feature_norms(
                field_feature_norms=self._field_feature_norms,
                field_feature_widths=self.field_feature_widths,
                field_feature_max_l=self.field_feature_max_l,
            ),
            dtype=default_dtype(),
        )

        self.lr_source_maps = nnx.List(
            [
                EnvironmentDependentSpinSourceBlock(
                    irreps_in=hidden_irreps,
                    max_l=self.atomic_multipoles_max_l,
                    cueq_config=cueq_config,
                    rngs=rngs,
                )
                for _ in range(num_interactions)
            ]
        )

        polar_irreps = self.make_polar_irreps_layout(
            make_irreps=Irreps,
            make_irrep=Irrep,
            hidden_irreps=hidden_irreps,
            atomic_multipoles_max_l=self.atomic_multipoles_max_l,
            field_feature_max_l=self.field_feature_max_l,
            field_feature_widths=self.field_feature_widths,
            num_elements=num_elements,
            radial_embedding_out_dim=self.radial_embedding.out_dim,
        )
        self.charges_irreps = polar_irreps.charges_irreps
        self.field_irreps = polar_irreps.field_irreps
        self.potential_irreps = polar_irreps.potential_irreps
        self.from_ell_max_field_update = polar_irreps.from_ell_max_field_update

        layout_str = layout_from_cueq_config(cueq_config)
        self._charges_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )
        self._charges_from_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.charges_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )
        self._field_from_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.field_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )

        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden_irreps,
            mlp_irreps.simplify(),
            gate,
            Irreps('2x0e'),
            cueq_config=None,
            rngs=rngs,
        )
        self._fukui_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=hidden_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )

        field_update_cls, update_config = self.resolve_field_update_config(
            self._fixedpoint_update_config,
            field_update_registry=field_update_blocks,
            potential_embedding_registry={
                'AgnosticChargeBiasedLinearPotentialEmbedding': (
                    AgnosticChargeBiasedLinearPotentialEmbedding
                ),
            },
            nonlinearity_registry={
                'MLPNonLinearity': MLPNonLinearity,
                'NoNonLinearity': NoNonLinearity,
            },
            default_potential_embedding_cls=(
                AgnosticChargeBiasedLinearPotentialEmbedding
            ),
        )
        self.field_dependent_charges_maps = nnx.List(
            [
                field_update_cls(
                    node_attrs_irreps=polar_irreps.node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=polar_irreps.field_update_sh_irreps,
                    edge_feats_irreps=polar_irreps.edge_feats_irreps,
                    target_irreps=polar_irreps.field_interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    potential_irreps=self.potential_irreps,
                    charges_irreps=self.charges_irreps,
                    num_elements=num_elements,
                    field_norm_factor=self.field_norm_factor,
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                    rngs=rngs,
                    **update_config,
                )
                for _ in range(self.num_recursion_steps)
            ]
        )

        field_readout_cls, readout_config = self.resolve_field_readout_config(
            self._field_readout_config,
            field_readout_registry=field_readout_blocks,
        )
        self.local_electron_energy = field_readout_cls(
            node_attrs_irreps=polar_irreps.node_attr_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=polar_irreps.field_update_sh_irreps,
            edge_feats_irreps=polar_irreps.edge_feats_irreps,
            target_irreps=polar_irreps.field_interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            potential_irreps=self.potential_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            rngs=rngs,
            **readout_config,
        )

        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps,
            num_interactions=num_interactions,
            cueq_config=cueq_config,
            rngs=rngs,
        )
        self.electric_potential_descriptor = GTOElectrostaticFeatures(
            density_max_l=self.atomic_multipoles_max_l,
            density_smearing_width=self.atomic_multipoles_smearing_width,
            feature_max_l=self.field_feature_max_l,
            feature_smearing_widths=self.field_feature_widths,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=self.field_si,
            quadrupole_feature_corrections=self.quadrupole_feature_corrections,
            integral_normalization='receiver',
        )
        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            self.field_feature_max_l,
            self.field_feature_widths,
            'receiver',
        )
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=self.atomic_multipoles_max_l,
            density_smearing_width=self.atomic_multipoles_smearing_width,
            kspace_cutoff=float(kspace_cutoff),
            include_self_interaction=self.include_electrostatic_self_interaction,
        )

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool = False,
        lammps_class: Any | None = None,
        compute_node_feats: bool = True,
        longrange_mode: str = 'realspace',
        fermi_level: jnp.ndarray | None = None,
        external_field: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray | None]:
        if longrange_mode not in {'realspace', 'pbc'}:
            raise ValueError(
                f'Unsupported PolarMACE longrange_mode {longrange_mode!r}.'
            )
        ctx = prepare_graph(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=data.get('lammps_class', lammps_class),
        )
        lammps_class, lammps_natoms, n_real = self.resolve_lammps_runtime(ctx)
        batch = jnp.asarray(data['batch'], dtype=jnp.int32)
        edge_index = jnp.asarray(data['edge_index'], dtype=jnp.int32)
        node_attrs = jnp.asarray(data['node_attrs'])
        node_attrs_index = self._resolve_node_attrs_index(data, node_attrs)

        if fermi_level is None:
            fermi_level = jnp.asarray(data['fermi_level'], dtype=ctx.vectors.dtype)
        if external_field is None:
            external_field = jnp.asarray(
                data['external_field'], dtype=ctx.vectors.dtype
            )
        external_potential = jnp.concatenate(
            (
                jnp.expand_dims(jnp.zeros_like(fermi_level), axis=-1),
                external_field,
            ),
            axis=-1,
        )

        node_e0 = self.atomic_energies_fn(node_attrs)[
            ctx.num_atoms_arange, ctx.node_heads
        ]
        e0 = scatter_sum(
            src=node_e0,
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        ).astype(ctx.vectors.dtype)

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(_permute_to_e3nn_convention(ctx.vectors))
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths,
            node_attrs,
            edge_index,
            self._atomic_numbers,
            node_attrs_index=node_attrs_index,
        )
        if self.pair_repulsion:
            pair_node_energy = self.pair_repulsion_fn(
                ctx.lengths,
                node_attrs,
                edge_index,
                self._atomic_numbers,
                node_attrs_index=node_attrs_index,
            )
            if lammps_class is not None:
                pair_node_energy = pair_node_energy[:n_real]
        else:
            pair_node_energy = jnp.zeros_like(node_e0)

        apply_embedding = self.make_apply_embedding(
            data=data,
            batch=batch,
            num_graphs=ctx.num_graphs,
            indices_are_sorted=True,
        )
        node_feats, e0 = apply_embedding(node_feats, e0, ctx.node_heads)

        node_es_list: list[jnp.ndarray] = [pair_node_energy]
        node_feats_list: list[jnp.ndarray] = []
        spin_charge_density = jnp.zeros(
            (batch.shape[0], self.charges_irreps.dim),
            dtype=ctx.vectors.dtype,
        )

        for layer_index, (interaction, product, lr_source) in enumerate(
            zip(self.interactions, self.products, self.lr_source_maps)
        ):
            if lammps_class is not None and layer_index > 0:
                node_feats = _apply_lammps_exchange(
                    node_feats,
                    lammps_class,
                    lammps_natoms,
                )
            node_attrs_slice = node_attrs
            node_attrs_index_slice = node_attrs_index
            if lammps_class is not None and layer_index > 0:
                node_attrs_slice = node_attrs_slice[:n_real]
                if node_attrs_index_slice is not None:
                    node_attrs_index_slice = node_attrs_index_slice[:n_real]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(layer_index == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if lammps_class is not None and layer_index == 0:
                node_attrs_slice = node_attrs_slice[:n_real]
                if node_attrs_index_slice is not None:
                    node_attrs_index_slice = node_attrs_index_slice[:n_real]
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
                node_attrs_index=node_attrs_index_slice,
            )
            if lammps_class is not None:
                node_feats = node_feats[:n_real]
            node_feats_list.append(node_feats)

            feat_idx = (
                -1
                if len(self.readouts) == 1
                else min(layer_index, len(self.readouts) - 1)
            )
            node_es = self.readouts[feat_idx](node_feats, ctx.node_heads)[
                ctx.num_atoms_arange, ctx.node_heads
            ]
            node_es_list.append(node_es)
            spin_charge_density = spin_charge_density + jnp.squeeze(
                lr_source(node_feats),
                axis=-2,
            )

        node_feats_out = (
            jnp.concatenate(node_feats_list, axis=-1) if compute_node_feats else None
        )
        node_inter_es = jnp.sum(jnp.stack(node_es_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, ctx.node_heads)
        inter_e = scatter_sum(
            src=node_inter_es,
            index=batch,
            dim=-1,
            dim_size=ctx.num_graphs,
        )

        pbc = jnp.asarray(data['pbc']).reshape(-1, 3)
        if longrange_mode == 'realspace':
            k_vectors = jnp.zeros((1, 3), dtype=ctx.vectors.dtype)
            kv_norms_squared = jnp.zeros((1,), dtype=ctx.vectors.dtype)
            k_vectors_batch = jnp.zeros((1,), dtype=jnp.int32)
            k_vectors_0mask = jnp.ones((1,), dtype=ctx.vectors.dtype)
        elif {
            'k_vectors',
            'k_norm2',
            'k_vector_batch',
            'k0_mask',
        }.issubset(data):
            k_vectors = jnp.asarray(data['k_vectors'], dtype=ctx.vectors.dtype)
            kv_norms_squared = jnp.asarray(data['k_norm2'], dtype=ctx.vectors.dtype)
            k_vectors_batch = jnp.asarray(data['k_vector_batch'], dtype=jnp.int32)
            k_vectors_0mask = jnp.asarray(data['k0_mask'], dtype=ctx.vectors.dtype)
        else:
            k_vectors, kv_norms_squared, k_vectors_batch, k_vectors_0mask = (
                compute_k_vectors_flat(
                    self.kspace_cutoff,
                    ctx.cell.reshape(-1, 3, 3),
                    jnp.asarray(data['rcell'], dtype=ctx.vectors.dtype).reshape(
                        -1,
                        3,
                        3,
                    ),
                )
            )
        field_feature_cache = self.electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            node_positions=ctx.positions,
            batch=batch,
            volume=jnp.asarray(data['volume'], dtype=ctx.vectors.dtype),
            pbc=pbc,
            mode=longrange_mode,
        )

        features_mixed = self.layer_feature_mixer(jnp.stack(node_feats_list, axis=0))
        spin_charge_density = spin_charge_density.reshape(
            spin_charge_density.shape[0],
            2,
            -1,
        )
        fukui_sources = self.fukui_source_map(self._fukui_to_mul_ir(node_feats))
        fukui_norm = scatter_sum(
            src=fukui_sources.astype(ctx.vectors.dtype),
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        )[batch].astype(ctx.vectors.dtype)
        fukui_norm = jnp.where(fukui_norm == 0, jnp.ones_like(fukui_norm), fukui_norm)
        fukui_sources = fukui_sources / fukui_norm
        total_charge_data = jnp.asarray(data['total_charge'], dtype=ctx.vectors.dtype)
        total_spin_data = jnp.asarray(data['total_spin'], dtype=ctx.vectors.dtype)
        q_plus_spin = (total_charge_data + (total_spin_data - 1))[batch]
        q_minus_spin = (total_charge_data - (total_spin_data - 1))[batch]
        pred_total_charges_0 = scatter_sum(
            src=spin_charge_density[:, :, 0].astype(ctx.vectors.dtype),
            index=batch,
            dim=0,
            dim_size=ctx.num_graphs,
        )[batch].astype(ctx.vectors.dtype)
        spin_charge_density = spin_charge_density.at[:, 0, 0].add(
            fukui_sources[:, 0] * ((q_plus_spin / 2) - pred_total_charges_0[:, 0])
        )
        spin_charge_density = spin_charge_density.at[:, 1, 0].add(
            fukui_sources[:, 1] * ((q_minus_spin / 2) - pred_total_charges_0[:, 1])
        )

        potential_features = jnp.zeros(
            (batch.shape[0], self.potential_irreps.dim),
            dtype=ctx.vectors.dtype,
        )
        field_independent_spin_charge_density = spin_charge_density
        electrostatic_potentials = None

        for recursion_index in range(self.num_recursion_steps):
            source_feats_alpha = self._charges_to_mul_ir(spin_charge_density[:, 0, :])
            source_feats_beta = self._charges_to_mul_ir(spin_charge_density[:, 1, :])
            field_feats_alpha = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=jnp.expand_dims(source_feats_alpha, axis=-2),
                pbc=pbc,
            )
            field_feats_beta = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=jnp.expand_dims(source_feats_beta, axis=-2),
                pbc=pbc,
            )
            field_feats_alpha = self._field_from_mul_ir(field_feats_alpha)
            field_feats_beta = self._field_from_mul_ir(field_feats_beta)
            barycenter = _scatter_mean(
                src=ctx.positions.astype(ctx.positions.dtype),
                index=batch,
                dim_size=ctx.num_graphs,
            ).astype(ctx.positions.dtype)
            half_external_field = 0.5 * self.external_field_contribution(
                batch,
                ctx.positions - barycenter[batch, :],
                external_potential,
            )
            field_feats_alpha = (
                field_feats_alpha + half_external_field
            ) / self.field_feature_norms
            field_feats_beta = (
                field_feats_beta + half_external_field
            ) / self.field_feature_norms
            potential_features = jnp.concatenate(
                (field_feats_alpha, field_feats_beta),
                axis=-1,
            )
            charge_sources_out = self.field_dependent_charges_maps[recursion_index](
                node_attrs=node_attrs,
                node_feats=features_mixed,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=edge_index,
                potential_features=potential_features,
                local_charges=spin_charge_density.reshape(
                    spin_charge_density.shape[0],
                    -1,
                ),
            )

            current_fukui_sources = charge_sources_out[:, -2:]
            charge_sources = charge_sources_out[:, :-2]
            spin_charge_density = spin_charge_density + charge_sources.reshape(
                spin_charge_density.shape[0],
                2,
                -1,
            )
            fukui_norm2 = scatter_sum(
                src=current_fukui_sources.astype(ctx.vectors.dtype),
                index=batch,
                dim=0,
                dim_size=ctx.num_graphs,
            )[batch].astype(ctx.vectors.dtype)
            fukui_norm2 = jnp.where(
                fukui_norm2 == 0,
                jnp.ones_like(fukui_norm2),
                fukui_norm2,
            )
            current_fukui_sources = current_fukui_sources / fukui_norm2
            pred_total_charges = scatter_sum(
                src=spin_charge_density[:, :, 0].astype(ctx.vectors.dtype),
                index=batch,
                dim=0,
                dim_size=ctx.num_graphs,
            )[batch].astype(ctx.vectors.dtype)
            spin_charge_density = spin_charge_density.at[:, 0, 0].add(
                current_fukui_sources[:, 0]
                * ((q_plus_spin / 2) - pred_total_charges[:, 0])
            )
            spin_charge_density = spin_charge_density.at[:, 1, 0].add(
                current_fukui_sources[:, 1]
                * ((q_minus_spin / 2) - pred_total_charges[:, 1])
            )

        total_energy = e0 + inter_e
        local_q_e = self.local_electron_energy(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
            edge_feats=edge_feats,
            edge_index=edge_index,
            field_feats=potential_features,
            charges_0=field_independent_spin_charge_density.reshape(
                field_independent_spin_charge_density.shape[0],
                -1,
            ),
            charges_induced=spin_charge_density.reshape(
                spin_charge_density.shape[0],
                -1,
            ),
        )
        electron_energy = scatter_sum(
            src=local_q_e,
            index=batch,
            dim=-1,
            dim_size=ctx.num_graphs,
        )
        if self.add_local_electron_energy:
            total_energy = total_energy + electron_energy
        else:
            electron_energy = jnp.zeros_like(electron_energy)

        charge_density = jnp.sum(spin_charge_density, axis=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        charge_density_mul_ir = self._charges_to_mul_ir(charge_density)
        spin_density_mul_ir = self._charges_to_mul_ir(spin_density)
        spin_charge_density_mul_ir = jnp.stack(
            [
                self._charges_to_mul_ir(spin_charge_density[:, 0, :]),
                self._charges_to_mul_ir(spin_charge_density[:, 1, :]),
            ],
            axis=1,
        )
        total_charge, total_dipole = _compute_total_charge_dipole_permuted(
            charge_density_mul_ir,
            ctx.positions,
            batch,
            ctx.num_graphs,
        )
        electrostatic_energy = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            source_feats=charge_density_mul_ir,
            node_positions=ctx.positions,
            batch=batch,
            volume=jnp.asarray(data['volume'], dtype=ctx.vectors.dtype),
            pbc=pbc,
            mode=longrange_mode,
        )
        total_energy = (
            total_energy
            + electrostatic_energy
            + jnp.sum(external_potential[:, 1:] * total_dipole, axis=-1)
        )

        return {
            'energy': total_energy,
            'node_energy': node_e0.astype(ctx.vectors.dtype)
            + node_inter_es.astype(ctx.vectors.dtype),
            'interaction_energy': inter_e,
            'node_feats': node_feats_out,
            'density_coefficients': charge_density_mul_ir,
            'spin_density': spin_density_mul_ir,
            'charges_history': jnp.expand_dims(spin_charge_density_mul_ir, axis=-1),
            'fermi_level': external_potential[:, 0],
            'external_field': external_potential[:, 1:],
            'charges': charge_density_mul_ir[:, 0],
            'spins': spin_density_mul_ir[:, 0],
            'dipole': total_dipole,
            'total_charge': total_charge,
            'electrostatic_energy': electrostatic_energy,
            'electron_energy': electron_energy,
            'electrostatic_potentials': electrostatic_potentials,
            'spin_charge_density': spin_charge_density_mul_ir,
            'displacement': ctx.displacement,
        }


__all__ = ['PolarMACE']
