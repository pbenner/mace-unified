"""Shared forward-pass and reduction helpers for unified MACE models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .backends import ModelBackend, _require_backend


class MACEModelForward:
    """Backend-agnostic forward-pass helpers shared by Torch and JAX models."""

    BACKEND: ModelBackend

    def backend_scatter_sum(
        self,
        *,
        src: Any,
        index: Any,
        dim: int,
        dim_size: int,
        indices_are_sorted: bool = False,
    ) -> Any:
        backend = _require_backend(self, "MACEModel")
        scatter_sum = backend.require("scatter_sum")
        return scatter_sum(
            src=src,
            index=index,
            dim=dim,
            dim_size=dim_size,
            indices_are_sorted=indices_are_sorted,
        )

    def backend_stack(self, values: Sequence[Any], dim: int) -> Any:
        backend = _require_backend(self, "MACEModel")
        stack = backend.require("stack")
        return stack(values, dim=dim)

    def backend_sum(self, value: Any, dim: int) -> Any:
        backend = _require_backend(self, "MACEModel")
        reduce_sum = backend.require("sum")
        return reduce_sum(value, dim=dim)

    def backend_cat(self, values: Sequence[Any], dim: int) -> Any:
        backend = _require_backend(self, "MACEModel")
        cat = backend.require("cat")
        return cat(values, dim=dim)

    def scatter_to_graph(
        self,
        *,
        values: Any,
        batch: Any,
        num_graphs: int,
        dim: int,
        indices_are_sorted: bool = False,
    ) -> Any:
        """Reduce node- or edge-aligned values to graph-aligned values."""
        return self.backend_scatter_sum(
            src=values,
            index=batch,
            dim=dim,
            dim_size=num_graphs,
            indices_are_sorted=indices_are_sorted,
        )

    @staticmethod
    def make_energy_output(
        *,
        total_energy: Any,
        node_energy: Any,
        contributions: Any,
        node_feats: Any,
        interaction_energy: Any,
        displacement: Any,
        lammps_natoms: Any,
    ) -> dict[str, Any]:
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "node_feats": node_feats,
            "interaction_energy": interaction_energy,
            "displacement": displacement,
            "lammps_natoms": lammps_natoms,
        }

    def make_energy_output_from_core(
        self, *, core: Mapping[str, Any], ctx: Any
    ) -> dict[str, Any]:
        """Convert the shared forward-core state into the public output payload."""
        return self.make_energy_output(
            total_energy=core["total_energy"],
            node_energy=core["node_energy"],
            contributions=core["contributions"],
            node_feats=core["node_feats_out"],
            interaction_energy=core["interaction_energy"],
            displacement=ctx.displacement,
            lammps_natoms=ctx.interaction_kwargs.lammps_natoms,
        )

    @staticmethod
    def merge_output_fields(
        base_output: Mapping[str, Any], **extra_fields: Any
    ) -> dict[str, Any]:
        merged = dict(base_output)
        merged.update(extra_fields)
        return merged

    @staticmethod
    def readout_feature_index(readout_index: int, num_readouts: int) -> int:
        return -1 if num_readouts == 1 else readout_index

    @staticmethod
    def resolve_lammps_runtime(ctx: Any) -> tuple[Any, tuple[int, int], int]:
        interaction_kwargs = ctx.interaction_kwargs
        lammps_class = interaction_kwargs.lammps_class
        lammps_natoms = interaction_kwargs.lammps_natoms
        n_real = int(ctx.num_atoms_arange.shape[0])
        if lammps_class is not None:
            n_real = int(lammps_natoms[0])
        return lammps_class, lammps_natoms, n_real

    @staticmethod
    def pair_energy_from_node(
        *,
        use_scale_shift: bool,
        has_pair_repulsion: bool,
        pair_node_energy: Any,
        e0: Any,
        scatter_pair_energy: Any,
        zeros_like_e0: Any,
        zeros_scale_shift: Any,
    ) -> Any:
        """Reduce pair node energies according to the current output mode."""
        if use_scale_shift:
            return zeros_scale_shift(pair_node_energy, e0)
        if has_pair_repulsion:
            return scatter_pair_energy(pair_node_energy)
        return zeros_like_e0(e0)

    def apply_optional_embedding(
        self,
        *,
        data: Any,
        node_feats: Any,
        e0: Any,
        node_heads: Any,
        scatter_node_energy: Any,
    ) -> tuple[Any, Any]:
        """Add optional auxiliary feature embeddings to node features and energy."""
        if not getattr(self, "_embedding_specs", None):
            return node_feats, e0

        joint_embedding = getattr(self, "joint_embedding", None)
        if joint_embedding is None:
            return node_feats, e0

        embedding_features = {name: data[name] for name in self._embedding_names}
        node_feats = node_feats + joint_embedding(data["batch"], embedding_features)
        if self.use_embedding_readout:
            embedding_readout = getattr(self, "embedding_readout", None)
            if embedding_readout is not None:
                embedding_node_energy = embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                e0 = e0 + scatter_node_energy(embedding_node_energy)
        return node_feats, e0

    def make_apply_embedding(
        self,
        *,
        data: Any,
        batch: Any,
        num_graphs: int,
        indices_are_sorted: bool = False,
    ) -> Any:
        """Return a closure that applies optional embeddings within a forward pass."""

        def _apply_embedding(
            node_feats: Any, e0: Any, node_heads: Any
        ) -> tuple[Any, Any]:
            return self.apply_optional_embedding(
                data=data,
                node_feats=node_feats,
                e0=e0,
                node_heads=node_heads,
                scatter_node_energy=lambda node_es: self.scatter_to_graph(
                    values=node_es,
                    batch=batch,
                    num_graphs=num_graphs,
                    dim=0,
                    indices_are_sorted=indices_are_sorted,
                ),
            )

        return _apply_embedding

    def make_pair_terms(
        self,
        *,
        batch: Any,
        num_graphs: int,
        use_scale_shift: bool,
        pair_node_energy_fn: Any,
        zero_node_energy: Any,
        zero_graph_energy: Any,
        zeros_scale_shift: Any,
        trim_pair_node_energy: Any | None = None,
        indices_are_sorted: bool = False,
    ) -> Any:
        """Return a closure that computes pair node energies and graph energies."""

        def _make_pair_terms(
            lengths: Any,
            node_attrs: Any,
            node_e0: Any,
            e0: Any,
            atomic_numbers: Any,
        ) -> tuple[Any, Any]:
            if self.pair_repulsion:
                pair_node_energy = pair_node_energy_fn(
                    lengths, node_attrs, atomic_numbers
                )
                if trim_pair_node_energy is not None:
                    pair_node_energy = trim_pair_node_energy(pair_node_energy)
            else:
                pair_node_energy = zero_node_energy(node_e0)
            pair_energy = self.pair_energy_from_node(
                use_scale_shift=use_scale_shift,
                has_pair_repulsion=self.pair_repulsion,
                pair_node_energy=pair_node_energy,
                e0=e0,
                scatter_pair_energy=lambda pair: self.scatter_to_graph(
                    values=pair,
                    batch=batch,
                    num_graphs=num_graphs,
                    dim=-1,
                    indices_are_sorted=indices_are_sorted,
                ),
                zeros_like_e0=zero_graph_energy,
                zeros_scale_shift=zeros_scale_shift,
            )
            return pair_node_energy, pair_energy

        return _make_pair_terms

    def make_graph_energy_reducers(
        self,
        *,
        batch: Any,
        num_graphs: int,
        indices_are_sorted: bool = False,
    ) -> tuple[Any, Any]:
        """Return closures that reduce node energies to graph energies."""

        def _readout_energy_from_node(node_es: Any) -> Any:
            return self.scatter_to_graph(
                values=node_es,
                batch=batch,
                num_graphs=num_graphs,
                dim=0,
                indices_are_sorted=indices_are_sorted,
            )

        def _interaction_energy_from_node(node_es: Any) -> Any:
            return self.scatter_to_graph(
                values=node_es,
                batch=batch,
                num_graphs=num_graphs,
                dim=-1,
                indices_are_sorted=indices_are_sorted,
            )

        return _readout_energy_from_node, _interaction_energy_from_node

    @staticmethod
    def run_interaction_stack(
        *,
        interactions: Any,
        products: Any,
        node_feats: Any,
        run_interaction: Any,
        run_product: Any,
    ) -> list[Any]:
        """Execute the interaction/product stack and collect intermediate node features."""
        node_feats_list: list[Any] = []
        for layer_index, (interaction, product) in enumerate(
            zip(interactions, products)
        ):
            node_feats, sc = run_interaction(layer_index, interaction, node_feats)
            node_feats = run_product(layer_index, product, node_feats, sc)
            node_feats_list.append(node_feats)
        return node_feats_list

    def apply_readouts(
        self,
        *,
        readouts: Any,
        node_feats_list: list[Any],
        node_heads: Any,
        num_atoms_arange: Any,
        handle_node_energy: Any,
    ) -> None:
        """Apply each readout and pass the resulting node energies to a callback."""
        num_readouts = len(readouts)
        for readout_index, readout in enumerate(readouts):
            feat_idx = self.readout_feature_index(readout_index, num_readouts)
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            handle_node_energy(node_es)

    def forward_energy_core(
        self,
        *,
        node_attrs: Any,
        num_atoms_arange: Any,
        node_heads: Any,
        vectors: Any,
        lengths: Any,
        atomic_numbers: Any,
        compute_node_feats: bool,
        cast_graph_energy: Any,
        make_edge_attrs: Any,
        make_edge_feats: Any,
        make_pair_terms: Any,
        apply_embedding: Any,
        run_interaction: Any,
        run_product: Any,
        readout_energy_from_node: Any,
        interaction_energy_from_node: Any,
        stack: Any,
        reduce_sum: Any,
        concat: Any,
        use_scale_shift: bool,
        scale_shift: Any = None,
        finalize_node_energy: Any = None,
    ) -> dict[str, Any]:
        """Run the backend-agnostic MACE energy forward pass."""
        node_e0 = self.atomic_energies_fn(node_attrs)[num_atoms_arange, node_heads]
        e0 = readout_energy_from_node(node_e0)
        e0 = cast_graph_energy(e0, vectors)

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = make_edge_attrs(vectors)
        edge_feats, cutoff = make_edge_feats(lengths, node_attrs, atomic_numbers)
        pair_node_energy, pair_energy = make_pair_terms(
            lengths, node_attrs, node_e0, e0, atomic_numbers
        )
        node_feats, e0 = apply_embedding(node_feats, e0, node_heads)

        node_feats_list = self.run_interaction_stack(
            interactions=self.interactions,
            products=self.products,
            node_feats=node_feats,
            run_interaction=lambda layer_index, interaction, feats: run_interaction(
                layer_index, interaction, feats, edge_attrs, edge_feats, cutoff
            ),
            run_product=run_product,
        )
        node_feats_out = None
        if compute_node_feats:
            node_feats_out = (
                concat(node_feats_list, dim=-1) if node_feats_list else node_feats
            )

        if use_scale_shift:
            node_energies_list = [pair_node_energy]
            self.apply_readouts(
                readouts=self.readouts,
                node_feats_list=node_feats_list,
                node_heads=node_heads,
                num_atoms_arange=num_atoms_arange,
                handle_node_energy=node_energies_list.append,
            )
            node_inter_es = reduce_sum(stack(node_energies_list, dim=0), dim=0)
            node_inter_es = scale_shift(node_inter_es, node_heads)
            inter_e = interaction_energy_from_node(node_inter_es)
            total_energy = e0 + inter_e
            if finalize_node_energy is None:
                node_energy = node_e0 + node_inter_es
            else:
                node_energy = finalize_node_energy(node_e0, node_inter_es)
            contributions = stack((e0, inter_e), dim=-1)
            interaction_energy = inter_e
            energies: list[Any] = []
        else:
            energies = [e0, pair_energy]
            node_energies_list = [node_e0, pair_node_energy]

            def _handle_node_energy(node_es: Any) -> None:
                energies.append(readout_energy_from_node(node_es))
                node_energies_list.append(node_es)

            self.apply_readouts(
                readouts=self.readouts,
                node_feats_list=node_feats_list,
                node_heads=node_heads,
                num_atoms_arange=num_atoms_arange,
                handle_node_energy=_handle_node_energy,
            )
            contributions = stack(energies, dim=-1)
            total_energy = reduce_sum(contributions, dim=-1)
            node_energy = reduce_sum(stack(node_energies_list, dim=-1), dim=-1)
            interaction_energy = total_energy - e0
            node_inter_es = None
            inter_e = None

        return {
            "node_e0": node_e0,
            "e0": e0,
            "node_feats": node_feats,
            "node_feats_list": node_feats_list,
            "node_feats_out": node_feats_out,
            "pair_node_energy": pair_node_energy,
            "pair_energy": pair_energy,
            "contributions": contributions,
            "total_energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": interaction_energy,
            "node_inter_es": node_inter_es,
            "inter_e": inter_e,
            "energies": energies,
        }

    def forward_energy_core_from_context(
        self,
        *,
        ctx: Any,
        node_attrs: Any,
        atomic_numbers: Any,
        compute_node_feats: bool,
        cast_graph_energy: Any,
        make_edge_attrs: Any,
        make_edge_feats: Any,
        make_pair_terms: Any,
        apply_embedding: Any,
        run_interaction: Any,
        run_product: Any,
        readout_energy_from_node: Any,
        interaction_energy_from_node: Any,
        stack: Any,
        reduce_sum: Any,
        concat: Any,
        use_scale_shift: bool,
        scale_shift: Any = None,
        finalize_node_energy: Any = None,
    ) -> dict[str, Any]:
        """Run `forward_energy_core` using vectors and indexing from a prepared context."""
        return self.forward_energy_core(
            node_attrs=node_attrs,
            num_atoms_arange=ctx.num_atoms_arange,
            node_heads=ctx.node_heads,
            vectors=ctx.vectors,
            lengths=ctx.lengths,
            atomic_numbers=atomic_numbers,
            compute_node_feats=compute_node_feats,
            cast_graph_energy=cast_graph_energy,
            make_edge_attrs=make_edge_attrs,
            make_edge_feats=make_edge_feats,
            make_pair_terms=make_pair_terms,
            apply_embedding=apply_embedding,
            run_interaction=run_interaction,
            run_product=run_product,
            readout_energy_from_node=readout_energy_from_node,
            interaction_energy_from_node=interaction_energy_from_node,
            stack=stack,
            reduce_sum=reduce_sum,
            concat=concat,
            use_scale_shift=use_scale_shift,
            scale_shift=scale_shift,
            finalize_node_energy=finalize_node_energy,
        )


__all__ = ["MACEModelForward"]
