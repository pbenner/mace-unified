"""Shared stack-construction helpers for unified MACE models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class MACEModelAssembly:
    """Backend-agnostic block and readout assembly helpers for shared MACE models."""

    @staticmethod
    def construct_interaction_block(
        *,
        interaction_cls: type[Any],
        node_attrs_irreps: Any,
        node_feats_irreps: Any,
        edge_attrs_irreps: Any,
        edge_feats_irreps: Any,
        target_irreps: Any,
        hidden_irreps: Any,
        avg_num_neighbors: float,
        radial_mlp: Sequence[int],
        cueq_config: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate one interaction block with normalized shared kwargs."""
        kwargs: dict[str, Any] = {
            "node_attrs_irreps": node_attrs_irreps,
            "node_feats_irreps": node_feats_irreps,
            "edge_attrs_irreps": edge_attrs_irreps,
            "edge_feats_irreps": edge_feats_irreps,
            "target_irreps": target_irreps,
            "hidden_irreps": hidden_irreps,
            "avg_num_neighbors": avg_num_neighbors,
            "radial_MLP": radial_mlp,
            "cueq_config": cueq_config,
        }
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return interaction_cls(**kwargs)

    @staticmethod
    def construct_product_block(
        *,
        product_cls: type[Any],
        node_feats_irreps: Any,
        target_irreps: Any,
        correlation: int,
        num_elements: int,
        use_sc: bool,
        cueq_config: Any,
        use_reduced_cg: bool,
        use_agnostic_product: bool,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate one equivariant product block with shared defaults."""
        kwargs: dict[str, Any] = {
            "node_feats_irreps": node_feats_irreps,
            "target_irreps": target_irreps,
            "correlation": correlation,
            "num_elements": num_elements,
            "use_sc": use_sc,
            "cueq_config": cueq_config,
            "use_reduced_cg": use_reduced_cg,
            "use_agnostic_product": use_agnostic_product,
        }
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return product_cls(**kwargs)

    @staticmethod
    def construct_linear_readout_block(
        *,
        readout_cls: type[Any],
        readout_irreps: Any,
        readout_output_irreps: Any,
        cueq_config: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate a linear readout block."""
        kwargs: dict[str, Any] = {
            "irreps_in": readout_irreps,
            "irrep_out": readout_output_irreps,
            "cueq_config": cueq_config,
        }
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return readout_cls(**kwargs)

    def construct_final_readout_block(
        self,
        *,
        readout_cls: type[Any],
        hidden_irreps_out: Any,
        mlp_irreps: Any,
        gate: Any,
        readout_output_irreps: Any,
        num_heads: int,
        cueq_config: Any,
        use_higher_irrep_invariants: bool,
        invariant_eps: float,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate the final non-linear readout block."""
        kwargs: dict[str, Any] = {
            "irreps_in": hidden_irreps_out,
            "MLP_irreps": (num_heads * mlp_irreps).simplify(),
            "gate": gate,
            "irrep_out": readout_output_irreps,
            "num_heads": num_heads,
            "cueq_config": cueq_config,
        }
        kwargs.update(
            self.maybe_readout_kwargs(
                readout_cls,
                use_higher_irrep_invariants=use_higher_irrep_invariants,
                invariant_eps=invariant_eps,
            )
        )
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return readout_cls(**kwargs)

    def build_standard_energy_stack(
        self,
        *,
        num_interactions: int,
        hidden_irreps: Any,
        hidden_irreps_out_first: Any,
        use_last_readout_only: bool,
        make_hidden_irreps_out: Any,
        collection_factory: Any,
        node_attr_irreps: Any,
        node_feats_irreps: Any,
        sh_irreps: Any,
        edge_feats_irreps: Any,
        interaction_irreps_first: Any,
        interaction_irreps: Any,
        avg_num_neighbors: float,
        radial_mlp: Sequence[int],
        cueq_config: Any,
        interaction_cls_first: type[Any],
        interaction_cls: type[Any],
        interaction_first_extra_kwargs: Mapping[str, Any] | None = None,
        interaction_extra_kwargs: Mapping[str, Any] | None = None,
        product_cls: type[Any],
        num_elements: int,
        correlation: Sequence[int],
        first_product_use_sc: bool,
        use_reduced_cg: bool,
        use_agnostic_product: bool,
        product_extra_kwargs: Mapping[str, Any] | None = None,
        linear_readout_cls: type[Any],
        readout_output_irreps: Any,
        linear_readout_extra_kwargs: Mapping[str, Any] | None = None,
        readout_cls: type[Any],
        mlp_irreps: Any,
        gate: Any,
        num_heads: int,
        readout_use_higher_irrep_invariants: bool,
        readout_invariant_eps: float,
        final_readout_extra_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any, Any]:
        """Build the standard interaction, product, and readout stack."""

        def _build_first_interaction(hidden_irreps_out_value: Any) -> Any:
            return self.construct_interaction_block(
                interaction_cls=interaction_cls_first,
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps_first,
                hidden_irreps=hidden_irreps_out_value,
                avg_num_neighbors=avg_num_neighbors,
                radial_mlp=radial_mlp,
                cueq_config=cueq_config,
                extra_kwargs=interaction_first_extra_kwargs,
            )

        def _build_interaction(layer_index: int, hidden_irreps_out_value: Any) -> Any:
            del layer_index
            return self.construct_interaction_block(
                interaction_cls=interaction_cls,
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out_value,
                avg_num_neighbors=avg_num_neighbors,
                radial_mlp=radial_mlp,
                cueq_config=cueq_config,
                extra_kwargs=interaction_extra_kwargs,
            )

        def _build_product_block(
            *,
            node_feats_irreps_value: Any,
            target_irreps_value: Any,
            correlation_value: int,
            use_sc_value: bool,
        ) -> Any:
            return self.construct_product_block(
                product_cls=product_cls,
                node_feats_irreps=node_feats_irreps_value,
                target_irreps=target_irreps_value,
                correlation=correlation_value,
                num_elements=num_elements,
                use_sc=use_sc_value,
                cueq_config=cueq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
                extra_kwargs=product_extra_kwargs,
            )

        def _build_first_product(
            target_irreps: Any, hidden_irreps_out_value: Any
        ) -> Any:
            return _build_product_block(
                node_feats_irreps_value=target_irreps,
                target_irreps_value=hidden_irreps_out_value,
                correlation_value=correlation[0],
                use_sc_value=first_product_use_sc,
            )

        def _build_product(layer_index: int, hidden_irreps_out_value: Any) -> Any:
            return _build_product_block(
                node_feats_irreps_value=interaction_irreps,
                target_irreps_value=hidden_irreps_out_value,
                correlation_value=correlation[layer_index + 1],
                use_sc_value=True,
            )

        def _build_linear_readout(readout_irreps: Any) -> Any:
            return self.construct_linear_readout_block(
                readout_cls=linear_readout_cls,
                readout_irreps=readout_irreps,
                readout_output_irreps=readout_output_irreps,
                cueq_config=cueq_config,
                extra_kwargs=linear_readout_extra_kwargs,
            )

        def _build_final_readout(hidden_irreps_out_value: Any) -> Any:
            return self.construct_final_readout_block(
                readout_cls=readout_cls,
                hidden_irreps_out=hidden_irreps_out_value,
                mlp_irreps=mlp_irreps,
                gate=gate,
                readout_output_irreps=readout_output_irreps,
                num_heads=num_heads,
                cueq_config=cueq_config,
                use_higher_irrep_invariants=readout_use_higher_irrep_invariants,
                invariant_eps=readout_invariant_eps,
                extra_kwargs=final_readout_extra_kwargs,
            )

        return self.build_interactions_products_readouts(
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            hidden_irreps_out_first=hidden_irreps_out_first,
            use_last_readout_only=use_last_readout_only,
            make_hidden_irreps_out=make_hidden_irreps_out,
            build_first_interaction=_build_first_interaction,
            build_interaction=_build_interaction,
            build_first_product=_build_first_product,
            build_product=_build_product,
            build_linear_readout=_build_linear_readout,
            build_final_readout=_build_final_readout,
            collection_factory=collection_factory,
        )

    def build_interactions_products_readouts(
        self,
        *,
        num_interactions: int,
        hidden_irreps: Any,
        hidden_irreps_out_first: Any,
        use_last_readout_only: bool,
        make_hidden_irreps_out: Any,
        build_first_interaction: Any,
        build_interaction: Any,
        build_first_product: Any,
        build_product: Any,
        build_linear_readout: Any,
        build_final_readout: Any,
        collection_factory: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Assemble the interaction, product, and readout module collections."""
        interactions = []
        products = []
        readouts = []

        first_interaction = build_first_interaction(hidden_irreps_out_first)
        interactions.append(first_interaction)
        products.append(
            build_first_product(
                first_interaction.target_irreps, hidden_irreps_out_first
            )
        )

        if not use_last_readout_only:
            readouts.append(build_linear_readout(hidden_irreps_out_first))

        for layer_index in range(num_interactions - 1):
            hidden_irreps_out = make_hidden_irreps_out(layer_index)
            interaction = build_interaction(layer_index, hidden_irreps_out)
            interactions.append(interaction)
            products.append(build_product(layer_index, hidden_irreps_out))
            if layer_index == num_interactions - 2:
                readouts.append(build_final_readout(hidden_irreps_out))
            elif not use_last_readout_only:
                readouts.append(build_linear_readout(hidden_irreps))

        if collection_factory is not None:
            return (
                collection_factory(interactions),
                collection_factory(products),
                collection_factory(readouts),
            )
        return interactions, products, readouts


__all__ = ["MACEModelAssembly"]
