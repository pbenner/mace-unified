from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from .utils import (
    get_atomic_virials_stresses,
    get_outputs,
    prepare_graph,
)
from mace_model.core.modules.backends import use_backend
from mace_model.core.modules.models import MACEModel

from mace_model.torch.adapters.e3nn import o3

from .backends import TORCH_BACKEND
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .embeddings import GenericJointEmbedding
from .radial import ZBLBasis


@use_backend(TORCH_BACKEND)
class MACE(MACEModel, torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
        correlation: int | Sequence[int],
        gate: Callable | None,
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = "None",
        edge_irreps: o3.Irreps | None = None,
        use_edge_irreps_first: bool = False,
        radial_MLP: Sequence[int] | None = None,
        radial_type: str | None = "bessel",
        heads: Sequence[str] | None = None,
        cueq_config: Any = None,
        embedding_specs: dict[str, Any] | None = None,
        oeq_config: Any = None,
        lammps_mliap: bool | None = False,
        readout_cls: type[NonLinearReadoutBlock] = NonLinearReadoutBlock,
        readout_use_higher_irrep_invariants: bool = False,
        readout_invariant_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        self.initialize_mace_common_attributes(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            mlp_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            correlation=correlation,
            gate=gate,
            pair_repulsion=pair_repulsion,
            apply_cutoff=apply_cutoff,
            use_reduced_cg=use_reduced_cg,
            use_so3=use_so3,
            use_agnostic_product=use_agnostic_product,
            use_last_readout_only=use_last_readout_only,
            use_embedding_readout=use_embedding_readout,
            distance_transform=distance_transform,
            edge_irreps=edge_irreps,
            radial_mlp=radial_MLP,
            radial_type=radial_type,
            heads=heads,
            cueq_config=cueq_config,
            embedding_specs=embedding_specs,
            readout_cls=readout_cls,
            readout_use_higher_irrep_invariants=readout_use_higher_irrep_invariants,
            readout_invariant_eps=readout_invariant_eps,
            mlp_attr_name="mlp_irreps",
            radial_mlp_attr_name="radial_mlp",
            keep_r_max_attr=False,
            extra_attrs={
                "use_edge_irreps_first": bool(use_edge_irreps_first),
                "oeq_config": oeq_config,
                "lammps_mliap": lammps_mliap,
            },
        )

        self._init_model_graph()

    def _build_node_embedding(
        self,
        attr_irreps: o3.Irreps,
        feat_irreps: o3.Irreps,
    ) -> LinearNodeEmbeddingBlock:
        return LinearNodeEmbeddingBlock(
            irreps_in=attr_irreps,
            irreps_out=feat_irreps,
            cueq_config=self.cueq_config,
        )

    def _build_joint_embedding(self, embedding_dim: int) -> GenericJointEmbedding:
        return GenericJointEmbedding(
            base_dim=embedding_dim,
            embedding_specs=self._embedding_specs,
            out_dim=embedding_dim,
        )

    def _build_embedding_readout(self, readout_irreps: o3.Irreps) -> LinearReadoutBlock:
        return LinearReadoutBlock(
            readout_irreps,
            self.make_head_output_irreps(len(self._heads), o3.Irreps),
            self.cueq_config,
            self.oeq_config,
        )

    def _build_radial_embedding(self) -> RadialEmbeddingBlock:
        return RadialEmbeddingBlock(
            r_max=self.r_max_value,
            num_bessel=self.num_bessel,
            num_polynomial_cutoff=self.num_polynomial_cutoff,
            radial_type=self.radial_type or "bessel",
            distance_transform=self.distance_transform,
            apply_cutoff=self.apply_cutoff,
        )

    def _build_pair_repulsion(self) -> ZBLBasis:
        return ZBLBasis(p=self.num_polynomial_cutoff)

    def _build_atomic_energies(self) -> AtomicEnergiesBlock:
        return AtomicEnergiesBlock(self.atomic_energies)

    def _shared_oeq_kwargs(self) -> dict[str, Any]:
        return {"oeq_config": self.oeq_config}

    def _interaction_kwargs(self, *, edge_irreps: o3.Irreps | None) -> dict[str, Any]:
        kwargs = self._shared_oeq_kwargs()
        kwargs["edge_irreps"] = edge_irreps
        return kwargs

    def _pair_node_energy(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        atomic_numbers: torch.Tensor,
        *,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.pair_repulsion_fn(
            lengths,
            node_attrs,
            edge_index,
            atomic_numbers,
        )

    @staticmethod
    def _zeros_scale_shift(
        _pair_node_energy: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(energy)

    @staticmethod
    def _trim_pair_node_energy(
        pair_node_energy: torch.Tensor,
        *,
        n_real: int | None,
    ) -> torch.Tensor:
        return pair_node_energy if n_real is None else pair_node_energy[:n_real]

    def _init_model_graph(self) -> None:
        num_interactions = int(self.num_interactions.item())
        hidden_irreps = self.coerce_irreps(self.hidden_irreps, o3.Irreps)
        mlp_irreps = self.coerce_irreps(self.mlp_irreps, o3.Irreps)

        (
            node_attr_irreps,
            node_feats_irreps,
            sh_irreps,
            interaction_irreps,
            interaction_irreps_first,
            hidden_irreps_out_first,
        ) = self.initialize_layout(
            heads=self.heads,
            correlation=self.correlation,
            num_interactions=num_interactions,
            num_elements=self.num_elements,
            hidden_irreps=hidden_irreps,
            max_ell=self.max_ell,
            use_so3=self.use_so3,
            collapse_hidden_irreps=True,
            make_irreps=o3.Irreps,
            make_irrep=o3.Irrep,
        )
        (
            num_heads,
            radial_mlp,
            edge_feats_irreps,
            readout_output_irreps,
            make_hidden_irreps_out,
        ) = self.initialize_energy_modules(
            node_attr_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            hidden_irreps=hidden_irreps,
            num_interactions=num_interactions,
            make_irreps=o3.Irreps,
            scalar_irrep=o3.Irrep(0, 1),
            embedding_specs=self.embedding_specs,
            radial_mlp=self.radial_mlp,
            use_embedding_readout=self.use_embedding_readout,
            build_node_embedding=self._build_node_embedding,
            build_joint_embedding=self._build_joint_embedding,
            build_embedding_readout=self._build_embedding_readout,
            build_radial_embedding=self._build_radial_embedding,
            build_pair_repulsion=self._build_pair_repulsion,
            build_atomic_energies=self._build_atomic_energies,
        )

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        edge_irreps_first = None
        if self.use_edge_irreps_first and self.edge_irreps is not None:
            edge_irreps_first = o3.Irreps(
                f"{self.edge_irreps.count(o3.Irrep(0, 1))}x0e"
            )

        self.interactions, self.products, self.readouts = (
            self.build_standard_energy_stack(
                num_interactions=num_interactions,
                hidden_irreps=hidden_irreps,
                hidden_irreps_out_first=hidden_irreps_out_first,
                use_last_readout_only=self.use_last_readout_only,
                make_hidden_irreps_out=make_hidden_irreps_out,
                collection_factory=torch.nn.ModuleList,
                node_attr_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                sh_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                interaction_irreps_first=interaction_irreps_first,
                interaction_irreps=interaction_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                radial_mlp=radial_mlp,
                cueq_config=self.cueq_config,
                interaction_cls_first=self.interaction_cls_first,
                interaction_cls=self.interaction_cls,
                interaction_first_extra_kwargs=self._interaction_kwargs(
                    edge_irreps=edge_irreps_first
                ),
                interaction_extra_kwargs=self._interaction_kwargs(
                    edge_irreps=self.edge_irreps
                ),
                product_cls=EquivariantProductBasisBlock,
                num_elements=self.num_elements,
                correlation=self._correlation,
                first_product_use_sc=self.is_residual_interaction(
                    self.interaction_cls_first
                ),
                use_reduced_cg=self.use_reduced_cg,
                use_agnostic_product=self.use_agnostic_product,
                product_extra_kwargs=self._shared_oeq_kwargs(),
                linear_readout_cls=LinearReadoutBlock,
                readout_output_irreps=readout_output_irreps,
                linear_readout_extra_kwargs=self._shared_oeq_kwargs(),
                readout_cls=self.readout_cls,
                mlp_irreps=mlp_irreps,
                gate=self.gate,
                num_heads=num_heads,
                readout_use_higher_irrep_invariants=self.readout_use_higher_irrep_invariants,
                readout_invariant_eps=self.readout_invariant_eps,
                final_readout_extra_kwargs=self._shared_oeq_kwargs(),
            )
        )

    def _compute_atomic_stress_terms(
        self,
        *,
        compute_atomic_stresses: bool,
        edge_forces: torch.Tensor | None,
        data: dict[str, torch.Tensor],
        vectors: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if compute_atomic_stresses and edge_forces is not None:
            return get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return None, None

    def _forward_core(
        self,
        data: dict[str, torch.Tensor],
        *,
        lammps_mliap: bool,
        compute_virials: bool,
        compute_stress: bool,
        compute_displacement: bool,
        compute_node_feats: bool,
        use_scale_shift: bool,
    ) -> tuple[dict[str, Any], Any]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        lammps_class, lammps_natoms, _ = self.resolve_lammps_runtime(ctx)
        is_lammps = ctx.is_lammps
        n_real = lammps_natoms[0] if is_lammps else None
        batch = data["batch"]
        edge_index = data["edge_index"]
        node_attrs = data["node_attrs"]

        def _pair_node_energy(
            lengths_value: torch.Tensor,
            node_attrs_value: torch.Tensor,
            atomic_numbers_value: torch.Tensor,
        ) -> torch.Tensor:
            return self._pair_node_energy(
                lengths_value,
                node_attrs_value,
                atomic_numbers_value,
                edge_index=edge_index,
            )

        def _trim_pair_node_energy(pair_node_energy: torch.Tensor) -> torch.Tensor:
            return self._trim_pair_node_energy(pair_node_energy, n_real=n_real)

        apply_embedding = self.make_apply_embedding(
            data=data,
            batch=batch,
            num_graphs=ctx.num_graphs,
        )
        make_pair_terms = self.make_pair_terms(
            batch=batch,
            num_graphs=ctx.num_graphs,
            use_scale_shift=use_scale_shift,
            pair_node_energy_fn=_pair_node_energy,
            zero_node_energy=torch.zeros_like,
            zero_graph_energy=torch.zeros_like,
            zeros_scale_shift=self._zeros_scale_shift,
            trim_pair_node_energy=_trim_pair_node_energy,
        )
        readout_energy_from_node, interaction_energy_from_node = (
            self.make_graph_energy_reducers(
                batch=batch,
                num_graphs=ctx.num_graphs,
            )
        )

        def _cast_graph_energy(
            graph_energy: torch.Tensor, vecs: torch.Tensor
        ) -> torch.Tensor:
            return graph_energy.to(vecs.dtype)

        def _make_edge_attrs(vecs: torch.Tensor) -> torch.Tensor:
            return self.spherical_harmonics(vecs)

        def _make_edge_feats(
            lengths_value: torch.Tensor,
            node_attrs_value: torch.Tensor,
            atomic_numbers_value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            return self.radial_embedding(
                lengths_value,
                node_attrs_value,
                edge_index,
                atomic_numbers_value,
            )

        def _run_interaction(
            layer_index: int,
            interaction: Any,
            feats: torch.Tensor,
            edge_attrs_value: torch.Tensor,
            edge_feats_value: torch.Tensor,
            cutoff_value: torch.Tensor | None,
        ):
            node_attrs_slice = node_attrs
            if is_lammps and layer_index > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]

            feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=feats,
                edge_attrs=edge_attrs_value,
                edge_feats=edge_feats_value,
                edge_index=edge_index,
                cutoff=cutoff_value,
                first_layer=(layer_index == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and layer_index == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            return feats, (sc, node_attrs_slice)

        def _run_product(
            layer_index: int,
            product: Any,
            feats: torch.Tensor,
            interaction_state: tuple[Any, torch.Tensor],
        ) -> torch.Tensor:
            del layer_index
            sc, node_attrs_slice = interaction_state
            return product(
                node_feats=feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )

        def _finalize_node_energy(
            node_e0_value: torch.Tensor, node_inter_es_value: torch.Tensor
        ) -> torch.Tensor:
            return node_e0_value.clone().double() + node_inter_es_value.clone().double()

        core = self.forward_energy_core_from_context(
            ctx=ctx,
            node_attrs=node_attrs,
            atomic_numbers=self.atomic_numbers,
            compute_node_feats=compute_node_feats,
            cast_graph_energy=_cast_graph_energy,
            make_edge_attrs=_make_edge_attrs,
            make_edge_feats=_make_edge_feats,
            make_pair_terms=make_pair_terms,
            apply_embedding=apply_embedding,
            run_interaction=_run_interaction,
            run_product=_run_product,
            readout_energy_from_node=readout_energy_from_node,
            interaction_energy_from_node=interaction_energy_from_node,
            stack=self.backend_stack,
            reduce_sum=self.backend_sum,
            concat=self.backend_cat,
            use_scale_shift=use_scale_shift,
            scale_shift=self.scale_shift if use_scale_shift else None,
            finalize_node_energy=_finalize_node_energy if use_scale_shift else None,
        )
        return core, ctx

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        compute_node_feats: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        return self._forward_with_observables(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            compute_atomic_stresses=compute_atomic_stresses,
            lammps_mliap=lammps_mliap,
            compute_node_feats=compute_node_feats,
            use_scale_shift=False,
        )

    def _forward_with_observables(
        self,
        data: dict[str, torch.Tensor],
        *,
        training: bool,
        compute_force: bool,
        compute_virials: bool,
        compute_stress: bool,
        compute_displacement: bool,
        compute_hessian: bool,
        compute_edge_forces: bool,
        compute_atomic_stresses: bool,
        lammps_mliap: bool,
        compute_node_feats: bool,
        use_scale_shift: bool,
    ) -> dict[str, torch.Tensor | None]:
        core, ctx = self._forward_core(
            data,
            lammps_mliap=lammps_mliap,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_node_feats=compute_node_feats,
            use_scale_shift=use_scale_shift,
        )
        energy_for_derivatives = (
            core["interaction_energy"] if use_scale_shift else core["total_energy"]
        )
        edge_forces_flag = compute_edge_forces or (
            use_scale_shift and compute_atomic_stresses
        )

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=energy_for_derivatives,
            positions=ctx.positions,
            displacement=ctx.displacement,
            vectors=ctx.vectors,
            cell=ctx.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=edge_forces_flag,
        )

        atomic_virials, atomic_stresses = self._compute_atomic_stress_terms(
            compute_atomic_stresses=compute_atomic_stresses,
            edge_forces=edge_forces,
            data=data,
            vectors=ctx.vectors,
            positions=ctx.positions,
            cell=ctx.cell,
        )

        return self.merge_output_fields(
            self.make_energy_output_from_core(core=core, ctx=ctx),
            forces=forces,
            edge_forces=edge_forces,
            virials=virials,
            stress=stress,
            atomic_virials=atomic_virials,
            atomic_stresses=atomic_stresses,
            hessian=hessian,
        )


class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        compute_node_feats: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        return self._forward_with_observables(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            compute_atomic_stresses=compute_atomic_stresses,
            lammps_mliap=lammps_mliap,
            compute_node_feats=compute_node_feats,
            use_scale_shift=True,
        )


class _UnavailableReferenceModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "This auxiliary Torch model has not been ported to mace-model yet."
        )


class AtomicDipolesMACE(_UnavailableReferenceModel):
    pass


class AtomicDielectricMACE(_UnavailableReferenceModel):
    pass


class EnergyDipolesMACE(_UnavailableReferenceModel):
    pass


__all__ = [
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
]
