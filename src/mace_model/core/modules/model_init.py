"""Shared constructor and layout helpers for backend-agnostic MACE models.

The logic in this module deals with the large amount of bookkeeping that is
common to both the local Torch and JAX model implementations: attribute
normalization, irreps-layout construction, optional embedding configuration,
and readout/output metadata setup.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any


class MACEModelInit:
    """Constructor, irreps-layout, and module-init helpers for shared MACE models."""

    def initialize_mace_common_attributes(
        self,
        *,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[Any],
        interaction_cls_first: type[Any],
        num_elements: int,
        hidden_irreps: Any,
        mlp_irreps: Any,
        atomic_energies: Any,
        avg_num_neighbors: float,
        correlation: int | Sequence[int],
        gate: Any,
        pair_repulsion: bool,
        apply_cutoff: bool,
        use_reduced_cg: bool,
        use_so3: bool,
        use_agnostic_product: bool,
        use_last_readout_only: bool,
        use_embedding_readout: bool,
        distance_transform: str,
        edge_irreps: Any,
        radial_mlp: Sequence[int] | None,
        radial_type: Any,
        heads: Sequence[str] | None,
        cueq_config: Any,
        embedding_specs: Mapping[str, Any] | None,
        readout_cls: type[Any],
        readout_use_higher_irrep_invariants: bool,
        readout_invariant_eps: float,
        mlp_attr_name: str,
        radial_mlp_attr_name: str,
        keep_r_max_attr: bool,
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the constructor arguments shared by Torch and JAX MACE models."""
        attrs: dict[str, Any] = {
            "r_max_value": float(r_max),
            "num_bessel": int(num_bessel),
            "num_polynomial_cutoff": int(num_polynomial_cutoff),
            "max_ell": int(max_ell),
            "interaction_cls": interaction_cls,
            "interaction_cls_first": interaction_cls_first,
            "num_elements": int(num_elements),
            "hidden_irreps": hidden_irreps,
            mlp_attr_name: mlp_irreps,
            "atomic_energies": atomic_energies,
            "avg_num_neighbors": float(avg_num_neighbors),
            "correlation": correlation,
            "gate": gate,
            "pair_repulsion": bool(pair_repulsion),
            "apply_cutoff": bool(apply_cutoff),
            "use_reduced_cg": bool(use_reduced_cg),
            "use_so3": bool(use_so3),
            "use_agnostic_product": bool(use_agnostic_product),
            "use_last_readout_only": bool(use_last_readout_only),
            "use_embedding_readout": bool(use_embedding_readout),
            "distance_transform": str(distance_transform),
            "edge_irreps": edge_irreps,
            radial_mlp_attr_name: radial_mlp,
            "radial_type": radial_type,
            "heads": heads,
            "cueq_config": cueq_config,
            "embedding_specs": embedding_specs,
            "readout_cls": readout_cls,
            "readout_use_higher_irrep_invariants": bool(
                readout_use_higher_irrep_invariants
            ),
            "readout_invariant_eps": float(readout_invariant_eps),
        }
        if keep_r_max_attr:
            attrs["r_max"] = r_max
        if extra_attrs is not None:
            attrs.update(dict(extra_attrs))
        for name, value in attrs.items():
            setattr(self, name, value)

    @staticmethod
    def as_heads(heads: Sequence[str] | None) -> tuple[str, ...]:
        if heads is None:
            return ("Default",)
        return tuple(heads)

    @staticmethod
    def as_correlation_tuple(
        correlation: int | Sequence[int], num_interactions: int
    ) -> tuple[int, ...]:
        if isinstance(correlation, int):
            return tuple([correlation] * int(num_interactions))
        values = tuple(int(value) for value in correlation)
        if len(values) != int(num_interactions):
            raise ValueError(
                "Length of correlation list must match num_interactions "
                f"(expected {num_interactions}, got {len(values)})"
            )
        return values

    @staticmethod
    def is_residual_interaction(interaction_cls_first: type[Any]) -> bool:
        return "Residual" in getattr(
            interaction_cls_first, "__name__", str(interaction_cls_first)
        )

    @staticmethod
    def make_parity_mixed_irreps(max_ell: int, make_irreps: Any) -> Any:
        repr_str = "+".join([f"1x{ell}e+1x{ell}o" for ell in range(max_ell + 1)])
        return make_irreps(repr_str)

    @staticmethod
    def maybe_readout_kwargs(
        readout_cls: type[Any],
        *,
        use_higher_irrep_invariants: bool,
        invariant_eps: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(readout_cls.__init__)
        except (TypeError, ValueError):
            return kwargs

        if "use_higher_irrep_invariants" in signature.parameters:
            kwargs["use_higher_irrep_invariants"] = bool(use_higher_irrep_invariants)
        if "invariant_eps" in signature.parameters:
            kwargs["invariant_eps"] = float(invariant_eps)
        return kwargs

    @staticmethod
    def resolve_embedding_specs(
        embedding_specs: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        specs = dict(embedding_specs or {})
        return specs, tuple(specs.keys())

    @staticmethod
    def resolve_radial_mlp(radial_mlp: Sequence[int] | None) -> list[int]:
        if radial_mlp is None:
            return [64, 64, 64]
        return [int(value) for value in radial_mlp]

    @staticmethod
    def make_head_output_irreps(num_heads: int, make_irreps: Any) -> Any:
        return make_irreps(f"{int(num_heads)}x0e")

    @staticmethod
    def coerce_irreps(value: Any, make_irreps: Any) -> Any:
        return make_irreps(value)

    @staticmethod
    def coerce_optional_irreps(value: Any | None, make_irreps: Any) -> Any | None:
        if value is None:
            return None
        return make_irreps(value)

    def initialize_layout(
        self,
        *,
        heads: Sequence[str] | None,
        correlation: int | Sequence[int],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: Any,
        max_ell: int,
        use_so3: bool,
        collapse_hidden_irreps: bool,
        make_irreps: Any,
        make_irrep: Any,
        collapse_hidden_irreps_value: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Resolve the model irreps layout and cache shared head metadata."""
        if collapse_hidden_irreps_value is None:

            def collapse_hidden_irreps_value(irreps):
                return make_irreps(str(irreps[0]))

        self._heads = self.as_heads(heads)
        self._correlation = self.as_correlation_tuple(
            correlation, int(num_interactions)
        )
        layout = self.build_irreps_layout(
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            max_ell=max_ell,
            use_so3=use_so3,
            num_interactions=num_interactions,
            collapse_hidden_irreps=collapse_hidden_irreps,
            make_irreps=make_irreps,
            make_irrep=make_irrep,
            collapse_hidden_irreps_value=collapse_hidden_irreps_value,
        )
        return (
            layout["node_attr_irreps"],
            layout["node_feats_irreps"],
            layout["sh_irreps"],
            layout["interaction_irreps"],
            layout["interaction_irreps_first"],
            layout["hidden_irreps_out_first"],
        )

    def initialize_embeddings(
        self,
        *,
        embedding_specs: Mapping[str, Any] | None,
        radial_mlp: Sequence[int] | None,
        use_embedding_readout: bool,
        node_feats_irreps: Any,
        scalar_irrep: Any,
        build_joint_embedding: Any,
        build_embedding_readout: Any,
    ) -> list[int]:
        """Initialize optional auxiliary-feature embeddings and radial MLP widths."""
        self._embedding_specs, self._embedding_names = self.resolve_embedding_specs(
            embedding_specs
        )
        radial_mlp_values = self.resolve_radial_mlp(radial_mlp)
        if not self._embedding_specs:
            return radial_mlp_values

        embedding_dim = node_feats_irreps.count(scalar_irrep)
        self.joint_embedding = build_joint_embedding(embedding_dim)
        if use_embedding_readout:
            self.embedding_readout = build_embedding_readout(node_feats_irreps)
        return radial_mlp_values

    def initialize_readout(
        self,
        *,
        num_heads: int,
        make_irreps: Any,
        hidden_irreps: Any,
        num_interactions: int,
        collapse_hidden_irreps_value: Any | None = None,
    ) -> tuple[Any, Any]:
        """Prepare the readout irreps and the hidden-irrep scheduling callback."""
        if collapse_hidden_irreps_value is None:

            def collapse_hidden_irreps_value(irreps):
                return make_irreps(str(irreps[0]))

        return (
            self.make_head_output_irreps(num_heads, make_irreps),
            self.make_hidden_irreps_out_factory(
                hidden_irreps=hidden_irreps,
                num_interactions=num_interactions,
                collapse_hidden_irreps_value=collapse_hidden_irreps_value,
            ),
        )

    def initialize_energy_modules(
        self,
        *,
        node_attr_irreps: Any,
        node_feats_irreps: Any,
        hidden_irreps: Any,
        num_interactions: int,
        make_irreps: Any,
        scalar_irrep: Any,
        embedding_specs: Mapping[str, Any] | None,
        radial_mlp: Sequence[int] | None,
        use_embedding_readout: bool,
        build_node_embedding: Any,
        build_joint_embedding: Any,
        build_embedding_readout: Any,
        build_radial_embedding: Any,
        build_pair_repulsion: Any,
        build_atomic_energies: Any,
    ) -> tuple[int, Sequence[int], Any, Any, Any]:
        """Build the common energy-model modules shared by both backends."""
        num_heads = len(self._heads)
        self.node_embedding = build_node_embedding(node_attr_irreps, node_feats_irreps)
        radial_mlp_values = self.initialize_embeddings(
            embedding_specs=embedding_specs,
            radial_mlp=radial_mlp,
            use_embedding_readout=use_embedding_readout,
            node_feats_irreps=node_feats_irreps,
            scalar_irrep=scalar_irrep,
            build_joint_embedding=build_joint_embedding,
            build_embedding_readout=build_embedding_readout,
        )
        self.radial_embedding = build_radial_embedding()
        edge_feats_irreps = make_irreps(f"{self.radial_embedding.out_dim}x0e")
        if self.pair_repulsion:
            self.pair_repulsion_fn = build_pair_repulsion()
        self.atomic_energies_fn = build_atomic_energies()
        readout_output_irreps, make_hidden_irreps_out = self.initialize_readout(
            num_heads=num_heads,
            make_irreps=make_irreps,
            hidden_irreps=hidden_irreps,
            num_interactions=num_interactions,
        )
        return (
            num_heads,
            radial_mlp_values,
            edge_feats_irreps,
            readout_output_irreps,
            make_hidden_irreps_out,
        )

    def build_irreps_layout(
        self,
        *,
        num_elements: int,
        hidden_irreps: Any,
        max_ell: int,
        use_so3: bool,
        num_interactions: int,
        collapse_hidden_irreps: bool,
        make_irreps: Any,
        make_irrep: Any,
        collapse_hidden_irreps_value: Any,
    ) -> dict[str, Any]:
        """Derive the irreps used throughout the model from the hidden layout."""
        node_attr_irreps = make_irreps([(int(num_elements), (0, 1))])
        scalar_mul = hidden_irreps.count(make_irrep(0, 1))
        node_feats_irreps = make_irreps([(scalar_mul, (0, 1))])

        if not use_so3:
            sh_irreps = make_irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = make_irreps.spherical_harmonics(max_ell, p=1)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(make_irrep(0, -1)) > 0:
            sh_irreps_inter = self.make_parity_mixed_irreps(max_ell, make_irreps)

        interaction_irreps = (sh_irreps_inter * scalar_mul).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * scalar_mul).sort()[0].simplify()

        hidden_irreps_out_first = hidden_irreps
        if collapse_hidden_irreps and int(num_interactions) == 1:
            hidden_irreps_out_first = collapse_hidden_irreps_value(hidden_irreps)

        return {
            "node_attr_irreps": node_attr_irreps,
            "node_feats_irreps": node_feats_irreps,
            "sh_irreps": sh_irreps,
            "interaction_irreps": interaction_irreps,
            "interaction_irreps_first": interaction_irreps_first,
            "hidden_irreps_out_first": hidden_irreps_out_first,
        }

    @staticmethod
    def make_hidden_irreps_out_factory(
        *,
        hidden_irreps: Any,
        num_interactions: int,
        collapse_hidden_irreps_value: Any,
    ) -> Any:
        def _make_hidden_irreps_out(layer_index: int) -> Any:
            if layer_index == int(num_interactions) - 2:
                return collapse_hidden_irreps_value(hidden_irreps)
            return hidden_irreps

        return _make_hidden_irreps_out


__all__ = ["MACEModelInit"]
