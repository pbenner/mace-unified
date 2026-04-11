"""Shared feature-embedding layers used by both Torch and JAX frontends."""

from __future__ import annotations

from typing import Any

from .backends import ModelBackend, _require_backend


class GenericJointEmbedding:
    """
    Shared joint feature embedding for auxiliary per-node or per-graph features.

    The frontend-specific subclasses provide the actual embedding modules
    through the bound backend. This class only owns the feature-spec parsing and
    the runtime logic that combines multiple feature embeddings into a single
    projection added to the node representation stream.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        base_dim: int,
        embedding_specs: dict[str, Any] | None,
        out_dim: int | None = None,
        rngs: Any = None,
    ) -> None:
        """Build the per-feature embedders and the final projection module."""
        backend = _require_backend(self, "GenericJointEmbedding")
        make_joint_embedders = backend.require("make_joint_embedders")
        make_joint_projection = backend.require("make_joint_projection")

        self.base_dim = int(base_dim)
        self.embedding_specs = embedding_specs
        self.out_dim = out_dim

        if self.embedding_specs is None:
            raise ValueError("embedding_specs must be provided for joint embedding.")

        self.specs = {name: dict(spec) for name, spec in self.embedding_specs.items()}
        if not self.specs:
            raise ValueError("embedding_specs must contain at least one feature.")

        self.feature_names = tuple(self.specs.keys())
        self._out_dim = int(self.out_dim or self.base_dim)
        self.total_dim = int(sum(spec["emb_dim"] for spec in self.specs.values()))

        self.embedders = make_joint_embedders(
            specs=self.specs,
            feature_names=self.feature_names,
            rngs=rngs,
        )
        self.project = make_joint_projection(
            total_dim=self.total_dim,
            out_dim=self._out_dim,
            rngs=rngs,
        )

    def forward(self, batch: Any, features: dict[str, Any]) -> Any:
        """Embed all configured features and project them to the target width."""
        backend = self.BACKEND
        cat = backend.require("cat")
        make_joint_categorical_indices = backend.require(
            "make_joint_categorical_indices"
        )

        embs = []
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f"Missing feature {name!r} required by joint embedding.")

            spec = self.specs[name]
            feat = features[name]

            per = spec.get("per", "node")
            if per == "graph":
                feat = feat[batch][..., None]
            elif per != "node":
                raise ValueError(f"Unknown 'per' value {per!r} for feature {name!r}.")

            feature_type = spec.get("type")
            if feature_type == "categorical":
                feat = make_joint_categorical_indices(
                    feat=feat,
                    offset=spec.get("offset", 0),
                )
            elif feature_type != "continuous":
                raise ValueError(f"Unknown feature type {feature_type!r}")

            embs.append(self.embedders[name](feat))

        if not embs:
            raise ValueError("No embeddings constructed; check embedding_specs input.")

        x = cat(embs, dim=-1)
        return self.project(x)
