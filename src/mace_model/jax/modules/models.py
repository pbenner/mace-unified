from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from mace_model.core.modules.backends import use_backend
from mace_model.core.modules.models import MACEModel

from mace_model.jax.adapters.cuequivariance.ir_dict import mul_ir_to_ir_dict
from mace_model.jax.adapters.e3nn import Irrep, Irreps
from mace_model.jax.adapters.e3nn.math import (
    estimate_normalize2mom_const,
    register_normalize2mom_const,
)
from mace_model.jax.adapters.e3nn.o3 import SphericalHarmonics
from mace_model.jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_model.jax.modules.utils import add_output_interface, prepare_graph
from mace_model.jax.nnx_config import ConfigVar
from mace_model.jax.tools.lammps_exchange import (
    forward_exchange as lammps_forward_exchange,
)

from ..tools.dtype import default_dtype
from .backends import JAX_BACKEND
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .embeddings import GenericJointEmbedding
from .radial import ZBLBasis


def _apply_lammps_exchange(
    node_feats: jnp.ndarray,
    lammps_class: Any | None,
    lammps_natoms: tuple[int, int],
) -> jnp.ndarray:
    if lammps_class is None:
        return node_feats

    n_pad = int(lammps_natoms[1])
    if n_pad <= 0:
        return node_feats

    pad = jnp.zeros((n_pad, node_feats.shape[1]), dtype=node_feats.dtype)
    padded = jnp.concatenate((node_feats, pad), axis=0)
    exchanged = lammps_forward_exchange(padded, lammps_class)
    return exchanged


def _prepare_normalize2mom_consts(
    consts: dict[str, float] | None,
) -> dict[str, float]:
    if consts is None:
        silu_value = estimate_normalize2mom_const("silu")
        consts = {"silu": silu_value, "swish": silu_value}
    else:
        consts = dict(consts)
        if "silu" not in consts:
            silu_value = estimate_normalize2mom_const("silu")
            consts["silu"] = silu_value
            consts.setdefault("swish", silu_value)
        if "swish" not in consts:
            consts["swish"] = consts["silu"]
    cleaned: dict[str, float] = {}
    for key, val in consts.items():
        try:
            scalar_val = float(np.asarray(val))
        except Exception as exc:
            raise ValueError(
                f"normalize2mom_consts for {key} must be a concrete float."
            ) from exc
        register_normalize2mom_const(key, scalar_val)
        cleaned[key] = scalar_val
    return cleaned


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class MACE(MACEModel, nnx.Module):
    def __init__(
        self,
        *,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[Any],
        interaction_cls_first: type[Any],
        atomic_energies: np.ndarray,
        atomic_numbers: tuple[int, ...],
        num_interactions: int = 3,
        num_elements: int = 1,
        hidden_irreps: Irreps = Irreps("1x0e"),
        MLP_irreps: Irreps = Irreps("1x0e"),
        avg_num_neighbors: float = 1.0,
        correlation: int | Sequence[int] = 1,
        gate: Callable | None = None,
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        replace_symmetric_contraction: bool = False,
        replacement_hidden_irreps: Irreps | None = None,
        replacement_depth: int = 2,
        replacement_use_species_conditioning: bool = True,
        attn_num_heads: int = 4,
        attn_head_dim: int = 16,
        attn_gate_mode: str = "scalar",
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        collapse_hidden_irreps: bool = True,
        distance_transform: str = "None",
        edge_irreps: Irreps | None = None,
        radial_MLP: Sequence[int] | None = None,
        radial_type: str = "bessel",
        heads: Sequence[str] | None = None,
        cueq_config: dict[str, Any] | None = None,
        embedding_specs: dict[str, Any] | None = None,
        readout_cls: type[NonLinearReadoutBlock] = NonLinearReadoutBlock,
        readout_use_higher_irrep_invariants: bool = False,
        readout_invariant_eps: float = 1e-12,
        normalize2mom_consts: dict[str, float] | None = None,
        rngs: nnx.Rngs,
    ) -> None:
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
            mlp_attr_name="MLP_irreps",
            radial_mlp_attr_name="radial_MLP",
            keep_r_max_attr=True,
            extra_attrs={
                "replace_symmetric_contraction": bool(replace_symmetric_contraction),
                "replacement_hidden_irreps": replacement_hidden_irreps,
                "replacement_depth": max(1, int(replacement_depth)),
                "replacement_use_species_conditioning": bool(
                    replacement_use_species_conditioning
                ),
                "attn_num_heads": max(1, int(attn_num_heads)),
                "attn_head_dim": max(1, int(attn_head_dim)),
                "attn_gate_mode": str(attn_gate_mode),
                "collapse_hidden_irreps": bool(collapse_hidden_irreps),
            },
        )

        self.atomic_numbers = tuple(atomic_numbers)
        self.num_interactions = int(num_interactions)

        self._init_model_graph(normalize2mom_consts=normalize2mom_consts, rngs=rngs)

    def _build_node_embedding(
        self,
        attr_irreps: Irreps,
        feat_irreps: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> LinearNodeEmbeddingBlock:
        return LinearNodeEmbeddingBlock(
            irreps_in=attr_irreps,
            irreps_out=feat_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def _build_joint_embedding(
        self,
        embedding_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> GenericJointEmbedding:
        return GenericJointEmbedding(
            base_dim=embedding_dim,
            embedding_specs=self._embedding_specs,
            out_dim=embedding_dim,
            rngs=rngs,
        )

    def _build_embedding_readout(
        self,
        readout_irreps: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> LinearReadoutBlock:
        return LinearReadoutBlock(
            readout_irreps,
            self.make_head_output_irreps(len(self._heads), Irreps),
            self.cueq_config,
            rngs=rngs,
        )

    def _build_radial_embedding(self, *, rngs: nnx.Rngs) -> RadialEmbeddingBlock:
        return RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=self.num_bessel,
            num_polynomial_cutoff=self.num_polynomial_cutoff,
            radial_type=self.radial_type,
            distance_transform=self.distance_transform,
            apply_cutoff=self.apply_cutoff,
            rngs=rngs,
        )

    def _build_pair_repulsion(self) -> ZBLBasis:
        return ZBLBasis(p=self.num_polynomial_cutoff)

    def _build_atomic_energies(self, *, rngs: nnx.Rngs) -> AtomicEnergiesBlock:
        return AtomicEnergiesBlock(self._atomic_energies, rngs=rngs)

    def _interaction_kwargs(
        self,
        *,
        rngs: nnx.Rngs,
        edge_irreps: Irreps | None = None,
        include_edge_irreps: bool = False,
    ) -> dict[str, Any]:
        kwargs = {
            "attn_num_heads": self.attn_num_heads,
            "attn_head_dim": self.attn_head_dim,
            "attn_gate_mode": self.attn_gate_mode,
            "rngs": rngs,
        }
        if include_edge_irreps:
            kwargs["edge_irreps"] = edge_irreps
        return kwargs

    def _product_kwargs(self, *, rngs: nnx.Rngs) -> dict[str, Any]:
        return {
            "replace_symmetric_contraction": self.replace_symmetric_contraction,
            "replacement_hidden_irreps": self._replacement_hidden_irreps,
            "replacement_depth": self.replacement_depth,
            "replacement_use_species_conditioning": self.replacement_use_species_conditioning,
            "rngs": rngs,
        }

    @staticmethod
    def _module_kwargs(*, rngs: nnx.Rngs) -> dict[str, nnx.Rngs]:
        return {"rngs": rngs}

    def _pair_node_energy(
        self,
        lengths: jnp.ndarray,
        node_attrs: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        *,
        edge_index: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None,
    ) -> jnp.ndarray:
        return self.pair_repulsion_fn(
            lengths,
            node_attrs,
            edge_index,
            atomic_numbers,
            node_attrs_index=node_attrs_index,
        )

    @staticmethod
    def _zeros_scale_shift(
        pair_node_energy: jnp.ndarray,
        _energy: jnp.ndarray,
        *,
        num_graphs: int,
    ) -> jnp.ndarray:
        return jnp.zeros((num_graphs,), dtype=pair_node_energy.dtype)

    @staticmethod
    def _trim_pair_node_energy(
        pair_node_energy: jnp.ndarray,
        *,
        n_real: int | None,
    ) -> jnp.ndarray:
        return pair_node_energy if n_real is None else pair_node_energy[:n_real]

    def _init_model_graph(
        self,
        *,
        normalize2mom_consts: dict[str, float] | None,
        rngs: nnx.Rngs,
    ) -> None:
        self._atomic_numbers = jnp.asarray(self.atomic_numbers, dtype=jnp.int32)
        self._atomic_energies = jnp.asarray(self.atomic_energies, dtype=default_dtype())

        hidden_irreps = self.coerce_irreps(self.hidden_irreps, Irreps)
        mlp_irreps = self.coerce_irreps(self.MLP_irreps, Irreps)
        self._hidden_irreps = hidden_irreps
        self._mlp_irreps = mlp_irreps
        self._replacement_hidden_irreps = self.coerce_optional_irreps(
            self.replacement_hidden_irreps, Irreps
        )

        consts = _prepare_normalize2mom_consts(normalize2mom_consts)
        self._normalize2mom_consts = consts
        dtype = default_dtype()
        const_arrays = {
            key: jnp.asarray(val, dtype=dtype) for key, val in consts.items()
        }
        self._normalize2mom_consts_var = ConfigVar(const_arrays)
        build_node_embedding = partial(self._build_node_embedding, rngs=rngs)
        build_joint_embedding = partial(self._build_joint_embedding, rngs=rngs)
        build_embedding_readout = partial(self._build_embedding_readout, rngs=rngs)
        build_radial_embedding = partial(self._build_radial_embedding, rngs=rngs)
        build_atomic_energies = partial(self._build_atomic_energies, rngs=rngs)

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
            num_interactions=self.num_interactions,
            num_elements=self.num_elements,
            hidden_irreps=hidden_irreps,
            max_ell=self.max_ell,
            use_so3=self.use_so3,
            collapse_hidden_irreps=self.collapse_hidden_irreps,
            make_irreps=Irreps,
            make_irrep=Irrep,
        )
        (
            num_heads,
            radial_mlp,
            edge_feats_irreps,
            readout_output_irreps,
            _make_hidden_irreps_out,
        ) = self.initialize_energy_modules(
            node_attr_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            hidden_irreps=hidden_irreps,
            num_interactions=self.num_interactions,
            make_irreps=Irreps,
            scalar_irrep=Irrep(0, 1),
            embedding_specs=self.embedding_specs,
            radial_mlp=self.radial_MLP,
            use_embedding_readout=self.use_embedding_readout,
            build_node_embedding=build_node_embedding,
            build_joint_embedding=build_joint_embedding,
            build_embedding_readout=build_embedding_readout,
            build_radial_embedding=build_radial_embedding,
            build_pair_repulsion=self._build_pair_repulsion,
            build_atomic_energies=build_atomic_energies,
        )

        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
            layout_str=getattr(self.cueq_config, "layout_str", "mul_ir")
            if self.cueq_config is not None
            else "mul_ir",
        )
        self.edge_attrs_irreps = sh_irreps

        self.interactions, self.products, self.readouts = (
            self.build_standard_energy_stack(
                num_interactions=self.num_interactions,
                hidden_irreps=hidden_irreps,
                hidden_irreps_out_first=hidden_irreps_out_first,
                use_last_readout_only=self.use_last_readout_only,
                make_hidden_irreps_out=_make_hidden_irreps_out,
                collection_factory=nnx.List,
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
                interaction_first_extra_kwargs=self._interaction_kwargs(rngs=rngs),
                interaction_extra_kwargs=self._interaction_kwargs(
                    rngs=rngs,
                    edge_irreps=self.edge_irreps,
                    include_edge_irreps=True,
                ),
                product_cls=EquivariantProductBasisBlock,
                num_elements=self.num_elements,
                correlation=self._correlation,
                first_product_use_sc=self.is_residual_interaction(
                    self.interaction_cls_first
                ),
                use_reduced_cg=self.use_reduced_cg,
                use_agnostic_product=self.use_agnostic_product,
                product_extra_kwargs=self._product_kwargs(rngs=rngs),
                linear_readout_cls=LinearReadoutBlock,
                readout_output_irreps=readout_output_irreps,
                linear_readout_extra_kwargs=self._module_kwargs(rngs=rngs),
                readout_cls=self.readout_cls,
                mlp_irreps=mlp_irreps,
                gate=self.gate,
                num_heads=num_heads,
                readout_use_higher_irrep_invariants=self.readout_use_higher_irrep_invariants,
                readout_invariant_eps=self.readout_invariant_eps,
                final_readout_extra_kwargs=self._module_kwargs(rngs=rngs),
            )
        )

    def _resolve_node_attrs_index(
        self,
        data: dict[str, jnp.ndarray],
        node_attrs: jnp.ndarray,
    ) -> jnp.ndarray | None:
        need_node_attrs_index = self.pair_repulsion or self.distance_transform in {
            "Agnesi",
            "Soft",
        }
        if self.cueq_config is not None and getattr(self.cueq_config, "enabled", False):
            need_node_attrs_index = need_node_attrs_index or bool(
                getattr(self.cueq_config, "optimize_all", False)
                or getattr(self.cueq_config, "optimize_symmetric", False)
            )
        node_attrs_index = data.get("node_attrs_index")
        if node_attrs_index is None:
            node_attrs_index = data.get("node_type")
        if node_attrs_index is None:
            node_attrs_index = data.get("species")
        if node_attrs_index is not None and getattr(node_attrs_index, "ndim", 1) != 1:
            node_attrs_index = None
        if node_attrs_index is None and need_node_attrs_index:
            node_attrs_index = jnp.argmax(node_attrs, axis=1)
        if node_attrs_index is not None:
            node_attrs_index = jnp.asarray(node_attrs_index, dtype=jnp.int32)
        return node_attrs_index

    def _make_edge_attrs(self, vectors: jnp.ndarray) -> jnp.ndarray:
        edge_attrs = self.spherical_harmonics(vectors)
        if self.cueq_config is None:
            return edge_attrs

        layout_str = getattr(self.cueq_config, "layout_str", "mul_ir")
        if layout_str != "ir_mul" or not getattr(self.cueq_config, "enabled", False):
            return edge_attrs

        group = getattr(self.cueq_config, "group", None)
        if group is None:
            return mul_ir_to_ir_dict(
                self.edge_attrs_irreps,
                edge_attrs,
                layout_str="ir_mul",
            )
        return mul_ir_to_ir_dict(
            self.edge_attrs_irreps,
            edge_attrs,
            group=group,
            layout_str="ir_mul",
        )

    def _forward_core(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool,
        lammps_class: Any | None,
        compute_node_feats: bool,
        use_scale_shift: bool,
    ) -> tuple[dict[str, Any], Any]:
        ctx = prepare_graph(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=data.get("lammps_class", lammps_class),
        )
        lammps_class, lammps_natoms, n_real = self.resolve_lammps_runtime(ctx)
        batch = data["batch"]
        edge_index = data["edge_index"]
        node_attrs = data["node_attrs"]
        node_attrs_index = self._resolve_node_attrs_index(data, node_attrs)
        pair_node_energy_fn = partial(
            self._pair_node_energy,
            edge_index=edge_index,
            node_attrs_index=node_attrs_index,
        )
        zeros_scale_shift = partial(self._zeros_scale_shift, num_graphs=ctx.num_graphs)
        trim_pair_node_energy = partial(self._trim_pair_node_energy, n_real=n_real)
        apply_embedding = self.make_apply_embedding(
            data=data,
            batch=batch,
            num_graphs=ctx.num_graphs,
            indices_are_sorted=True,
        )
        make_pair_terms = self.make_pair_terms(
            batch=batch,
            num_graphs=ctx.num_graphs,
            use_scale_shift=use_scale_shift,
            pair_node_energy_fn=pair_node_energy_fn,
            zero_node_energy=jnp.zeros_like,
            zero_graph_energy=jnp.zeros_like,
            zeros_scale_shift=zeros_scale_shift,
            trim_pair_node_energy=trim_pair_node_energy,
            indices_are_sorted=True,
        )
        readout_energy_from_node, interaction_energy_from_node = (
            self.make_graph_energy_reducers(
                batch=batch,
                num_graphs=ctx.num_graphs,
                indices_are_sorted=True,
            )
        )

        def _cast_graph_energy(
            graph_energy: jnp.ndarray, vecs: jnp.ndarray
        ) -> jnp.ndarray:
            return graph_energy.astype(vecs.dtype)

        def _make_edge_attrs(vecs: jnp.ndarray) -> jnp.ndarray:
            return self._make_edge_attrs(vecs)

        def _make_edge_feats(
            lengths_value: jnp.ndarray,
            node_attrs_value: jnp.ndarray,
            atomic_numbers_value: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            return self.radial_embedding(
                lengths_value,
                node_attrs_value,
                edge_index,
                atomic_numbers_value,
                node_attrs_index=node_attrs_index,
            )

        def _run_interaction(
            layer_index: int,
            interaction: Any,
            feats: jnp.ndarray,
            edge_attrs_value: jnp.ndarray,
            edge_feats_value: jnp.ndarray,
            cutoff_value: jnp.ndarray,
        ):
            if lammps_class is not None and layer_index > 0:
                feats = _apply_lammps_exchange(feats, lammps_class, lammps_natoms)

            node_attrs_slice = node_attrs
            node_attrs_index_slice = node_attrs_index
            if lammps_class is not None and layer_index > 0:
                node_attrs_slice = node_attrs_slice[:n_real]
                if node_attrs_index_slice is not None:
                    node_attrs_index_slice = node_attrs_index_slice[:n_real]

            feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=feats,
                edge_attrs=edge_attrs_value,
                edge_feats=edge_feats_value,
                edge_index=edge_index,
                cutoff=cutoff_value,
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
                first_layer=(layer_index == 0),
            )
            if lammps_class is not None and layer_index == 0:
                node_attrs_slice = node_attrs_slice[:n_real]
                if node_attrs_index_slice is not None:
                    node_attrs_index_slice = node_attrs_index_slice[:n_real]
            return feats, (sc, node_attrs_slice, node_attrs_index_slice)

        def _run_product(
            layer_index: int,
            product: Any,
            feats: jnp.ndarray,
            interaction_state: tuple[Any, jnp.ndarray, jnp.ndarray | None],
        ) -> jnp.ndarray:
            del layer_index
            sc, node_attrs_slice, node_attrs_index_slice = interaction_state
            feats = product(
                node_feats=feats,
                sc=sc,
                node_attrs=node_attrs_slice,
                node_attrs_index=node_attrs_index_slice,
            )
            if lammps_class is not None:
                feats = feats[:n_real]
            return feats

        core = self.forward_energy_core_from_context(
            ctx=ctx,
            node_attrs=node_attrs,
            atomic_numbers=self._atomic_numbers,
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
        )
        return core, ctx

    def _forward_energy_output(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool,
        lammps_class: Any | None,
        compute_node_feats: bool,
        use_scale_shift: bool,
    ) -> dict[str, jnp.ndarray | None]:
        core, ctx = self._forward_core(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=lammps_class,
            compute_node_feats=compute_node_feats,
            use_scale_shift=use_scale_shift,
        )
        return self.make_energy_output_from_core(core=core, ctx=ctx)

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool = False,
        lammps_class: Any | None = None,
        compute_node_feats: bool = True,
    ) -> dict[str, jnp.ndarray | None]:
        return self._forward_energy_output(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=lammps_class,
            compute_node_feats=compute_node_feats,
            use_scale_shift=False,
        )


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        *,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        self.atomic_inter_scale = atomic_inter_scale
        self.atomic_inter_shift = atomic_inter_shift
        super().__init__(rngs=rngs, **kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=self.atomic_inter_scale,
            shift=self.atomic_inter_shift,
        )

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool = False,
        lammps_class: Any | None = None,
        compute_node_feats: bool = True,
    ) -> dict[str, jnp.ndarray | None]:
        return self._forward_energy_output(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=lammps_class,
            compute_node_feats=compute_node_feats,
            use_scale_shift=True,
        )


__all__ = ["MACE", "ScaleShiftMACE"]
