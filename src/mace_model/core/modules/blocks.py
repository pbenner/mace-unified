"""Shared block implementations used by both Torch and JAX MACE frontends.

The classes in this module encode the common architectural structure of MACE
building blocks. Backend-specific subclasses provide tensor primitives and
module factories through `ModelBackend`.
"""

from __future__ import annotations

from typing import Any, Callable

from .backends import ModelBackend, _require_backend
from .irreps_utils import _build_gated_irreps


class LinearNodeEmbeddingBlock:
    """
    Shared base class inherited by both Torch and JAX node embedding wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        irreps_out: Any,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "LinearNodeEmbeddingBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")

        self.irreps_in = make_irreps(irreps_in)
        self.irreps_out = make_irreps(irreps_out)
        self.cueq_config = cueq_config
        self.linear = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, node_attrs: Any) -> Any:
        return self.linear(node_attrs)


class LinearReadoutBlock:
    """
    Shared base class inherited by both Torch and JAX linear readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        irrep_out: Any,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "LinearReadoutBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")

        self.irreps_in = make_irreps(irreps_in)
        self.irrep_out = make_irreps(irrep_out)
        self.cueq_config = cueq_config
        self.linear = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any, heads: Any = None) -> Any:
        del heads
        return self.linear(x)


class NonLinearReadoutBlock:
    """
    Shared base class inherited by both Torch and JAX readout wrappers.

    Subclasses provide a class-level BACKEND operation bundle.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        mlp_irreps: Any,
        gate: Callable | None,
        irrep_out: Any,
        num_heads: int,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "NonLinearReadoutBlock")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_activation = backend.require("make_activation")

        self.irreps_in = make_irreps(irreps_in)
        self.MLP_irreps = make_irreps(mlp_irreps)
        self.gate = gate
        self.irrep_out = make_irreps(irrep_out)
        self.num_heads = int(num_heads)
        self.cueq_config = cueq_config

        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity = make_activation(
            hidden_irreps=self.hidden_irreps,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.linear_2 = make_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any, heads: Any = None) -> Any:
        backend = self.BACKEND
        mask_head = backend.require("mask_head")
        x = self.linear_1(x)
        x = self.non_linearity(x)
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


class NonLinearBiasReadoutBlock:
    """
    Shared base class for bias-enabled non-linear readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        mlp_irreps: Any,
        gate: Callable | None,
        irrep_out: Any,
        num_heads: int,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "NonLinearBiasReadoutBlock")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_bias_linear = backend.require("make_bias_linear")
        make_activation = backend.require("make_activation")

        self.irreps_in = make_irreps(irreps_in)
        self.MLP_irreps = make_irreps(mlp_irreps)
        self.gate = gate
        self.irrep_out = make_irreps(irrep_out)
        self.num_heads = int(num_heads)
        self.cueq_config = cueq_config

        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity_1 = make_activation(
            hidden_irreps=self.hidden_irreps,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.linear_mid = make_bias_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity_2 = make_activation(
            hidden_irreps=self.hidden_irreps,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.linear_2 = make_bias_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any, heads: Any = None) -> Any:
        backend = self.BACKEND
        mask_head = backend.require("mask_head")
        mask_head_stage1 = getattr(backend, "mask_head_stage1", None)

        x = self.linear_1(x)
        x = self.non_linearity_1(x)
        if self.num_heads > 1 and heads is not None and mask_head_stage1 is not None:
            x = mask_head_stage1(x, heads, self.num_heads)
        x = self.linear_mid(x)
        x = self.non_linearity_2(x)
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


class LinearDipoleReadoutBlock:
    """
    Shared base class for linear dipole readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        dipole_only: bool = False,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "LinearDipoleReadoutBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")

        self.irreps_in = make_irreps(irreps_in)
        self.dipole_only = bool(dipole_only)
        self.cueq_config = cueq_config
        if self.dipole_only:
            self.irreps_out = make_irreps("1x1o")
        else:
            self.irreps_out = make_irreps("1x0e + 1x1o")
        self.linear = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any) -> Any:
        return self.linear(x)


class NonLinearDipoleReadoutBlock:
    """
    Shared base class for non-linear dipole readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        mlp_irreps: Any,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "NonLinearDipoleReadoutBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_gate = backend.require("make_gate")

        self.irreps_in = make_irreps(irreps_in)
        self.MLP_irreps = make_irreps(mlp_irreps)
        self.hidden_irreps = self.MLP_irreps
        self.gate = gate
        self.dipole_only = bool(dipole_only)
        self.cueq_config = cueq_config

        if self.dipole_only:
            self.irreps_out = make_irreps("1x1o")
        else:
            self.irreps_out = make_irreps("1x0e + 1x1o")

        irreps_scalars, irreps_gates, irreps_gated = _build_gated_irreps(
            make_irreps=make_irreps,
            hidden_irreps=self.hidden_irreps,
            irreps_out=self.irreps_out,
        )

        self.equivariant_nonlin = make_gate(
            irreps_scalars=irreps_scalars,
            irreps_gates=irreps_gates,
            irreps_gated=irreps_gated,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_1 = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_2 = make_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any) -> Any:
        x = self.linear_1(x)
        x = self.equivariant_nonlin(x)
        return self.linear_2(x)


class LinearDipolePolarReadoutBlock:
    """
    Shared base class for linear dipole-polarizability readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        use_polarizability: bool = True,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "LinearDipolePolarReadoutBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")

        self.irreps_in = make_irreps(irreps_in)
        self.use_polarizability = bool(use_polarizability)
        self.cueq_config = cueq_config
        if not self.use_polarizability:
            raise ValueError(
                "LinearDipolePolarReadoutBlock requires use_polarizability=True."
            )
        self.irreps_out = make_irreps("2x0e + 1x1o + 1x2e")
        self.linear = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any) -> Any:
        return self.linear(x)


class NonLinearDipolePolarReadoutBlock:
    """
    Shared base class for non-linear dipole-polarizability readout wrappers.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        irreps_in: Any,
        mlp_irreps: Any,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "NonLinearDipolePolarReadoutBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_gate = backend.require("make_gate")

        self.irreps_in = make_irreps(irreps_in)
        self.MLP_irreps = make_irreps(mlp_irreps)
        self.hidden_irreps = self.MLP_irreps
        self.gate = gate
        self.use_polarizability = bool(use_polarizability)
        self.cueq_config = cueq_config
        if not self.use_polarizability:
            raise ValueError(
                "NonLinearDipolePolarReadoutBlock requires use_polarizability=True."
            )
        self.irreps_out = make_irreps("2x0e + 1x1o + 1x2e")

        irreps_scalars, irreps_gates, irreps_gated = _build_gated_irreps(
            make_irreps=make_irreps,
            hidden_irreps=self.hidden_irreps,
            irreps_out=self.irreps_out,
        )

        self.equivariant_nonlin = make_gate(
            irreps_scalars=irreps_scalars,
            irreps_gates=irreps_gates,
            irreps_gated=irreps_gated,
            gate=self.gate,
            cueq_config=self.cueq_config,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_1 = make_linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_2 = make_linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(self, x: Any) -> Any:
        x = self.linear_1(x)
        x = self.equivariant_nonlin(x)
        return self.linear_2(x)


class RadialEmbeddingBlock:
    """
    Build radial basis features and optional distance transforms for edges.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "RadialEmbeddingBlock")
        make_bessel_basis = backend.require("make_bessel_basis")
        make_gaussian_basis = backend.require("make_gaussian_basis")
        make_chebychev_basis = backend.require("make_chebychev_basis")
        make_polynomial_cutoff = backend.require("make_polynomial_cutoff")
        make_agnesi_transform = backend.require("make_agnesi_transform")
        make_soft_transform = backend.require("make_soft_transform")

        self.r_max = float(r_max)
        self.num_bessel = int(num_bessel)
        self.num_polynomial_cutoff = int(num_polynomial_cutoff)
        self.radial_type = str(radial_type)
        self.distance_transform = str(distance_transform)
        self.apply_cutoff = bool(apply_cutoff)

        if self.radial_type == "bessel":
            self.basis_fn = make_bessel_basis(
                r_max=self.r_max,
                num_basis=self.num_bessel,
                rngs=rngs,
            )
        elif self.radial_type == "gaussian":
            self.basis_fn = make_gaussian_basis(
                r_max=self.r_max,
                num_basis=self.num_bessel,
                rngs=rngs,
            )
        elif self.radial_type == "chebyshev":
            self.basis_fn = make_chebychev_basis(
                r_max=self.r_max,
                num_basis=self.num_bessel,
            )
        else:
            raise ValueError(f"Unknown radial_type: {self.radial_type}")

        if self.distance_transform == "Agnesi":
            self.distance_transform_module = make_agnesi_transform(rngs=rngs)
        elif self.distance_transform == "Soft":
            self.distance_transform_module = make_soft_transform(rngs=rngs)
        else:
            self.distance_transform_module = None

        self.cutoff_fn = make_polynomial_cutoff(
            r_max=self.r_max,
            p=self.num_polynomial_cutoff,
        )
        self.out_dim = self.num_bessel

    def forward(
        self,
        edge_lengths: Any,
        node_attrs: Any,
        edge_index: Any,
        atomic_numbers: Any,
        node_attrs_index: Any = None,
    ) -> tuple[Any, Any]:
        cutoff = self.cutoff_fn(edge_lengths)
        transformed_lengths = edge_lengths
        if self.distance_transform_module is not None:
            if node_attrs_index is None:
                transformed_lengths = self.distance_transform_module(
                    edge_lengths,
                    node_attrs,
                    edge_index,
                    atomic_numbers,
                )
            else:
                transformed_lengths = self.distance_transform_module(
                    edge_lengths,
                    node_attrs,
                    edge_index,
                    atomic_numbers,
                    node_attrs_index=node_attrs_index,
                )
        radial = self.basis_fn(transformed_lengths)
        if self.apply_cutoff:
            return radial * cutoff, None
        return radial, cutoff


class EquivariantProductBasisBlock:
    """
    Apply symmetric contraction and an optional skip connection to node features.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        node_feats_irreps: Any,
        target_irreps: Any,
        correlation: int,
        use_sc: bool = True,
        num_elements: int | None = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: bool | None = None,
        cueq_config: Any = None,
        oeq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "EquivariantProductBasisBlock")
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        make_symmetric_contraction = backend.require("make_symmetric_contraction")

        self.node_feats_irreps = make_irreps(node_feats_irreps)
        self.target_irreps = make_irreps(target_irreps)
        self.correlation = int(correlation)
        self.use_sc = bool(use_sc)
        self.num_elements = num_elements
        self.use_agnostic_product = bool(use_agnostic_product)
        self.use_reduced_cg = use_reduced_cg
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config

        num_elements_local = self.num_elements
        if self.use_agnostic_product:
            num_elements_local = 1

        self.symmetric_contractions = make_symmetric_contraction(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            num_elements=num_elements_local,
            use_reduced_cg=self.use_reduced_cg,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
            rngs=rngs,
        )

        self.linear = make_linear(
            irreps_in=self.target_irreps,
            irreps_out=self.target_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(
        self,
        node_feats: Any,
        sc: Any,
        node_attrs: Any,
        node_attrs_index: Any = None,
    ) -> Any:
        backend = self.BACKEND
        make_ones = backend.require("make_ones")
        make_index_attrs = backend.require("make_index_attrs")
        transpose_mul_ir = backend.require("transpose_mul_ir")

        if self.use_agnostic_product:
            node_attrs = make_ones(node_feats=node_feats, width=1)
            node_attrs_index = None

        use_cueq = False
        if self.cueq_config is not None:
            if self.cueq_config.enabled and (
                self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
            ):
                use_cueq = True

        if use_cueq:
            layout_str = getattr(self.cueq_config, "layout_str", "mul_ir")
            features = node_feats
            if layout_str == "mul_ir":
                features = transpose_mul_ir(features)
            features = features.reshape(features.shape[0], -1)
            index_attrs = make_index_attrs(
                node_attrs=node_attrs,
                node_attrs_index=node_attrs_index,
            )
            node_feats = self.symmetric_contractions(features, index_attrs)
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class ScaleShiftBlock:
    """
    Shared base class for scale-shift wrappers.
    """

    BACKEND: ModelBackend

    def init(self, *, scale: Any, shift: Any) -> None:
        backend = _require_backend(self, "ScaleShiftBlock")
        make_scale_shift = backend.require("make_scale_shift")
        self.scale = make_scale_shift(self, name="scale", value=scale)
        self.shift = make_scale_shift(self, name="shift", value=shift)

    def forward(self, x: Any, head: Any) -> Any:
        backend = self.BACKEND
        get_scale_shift = backend.require("get_scale_shift")
        atleast_1d = backend.require("atleast_1d")

        scale = atleast_1d(get_scale_shift(self.scale))[head]
        shift = atleast_1d(get_scale_shift(self.shift))[head]
        return scale * x + shift

    def __repr__(self) -> str:
        backend = self.BACKEND
        get_scale_shift = backend.require("get_scale_shift")
        atleast_1d = backend.require("atleast_1d")
        to_numpy = backend.require("to_numpy")

        scale_vals = to_numpy(atleast_1d(get_scale_shift(self.scale)))
        shift_vals = to_numpy(atleast_1d(get_scale_shift(self.shift)))
        formatted_scale = ", ".join(f"{float(value):.4f}" for value in scale_vals)
        formatted_shift = ", ".join(f"{float(value):.4f}" for value in shift_vals)
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"


class AtomicEnergiesBlock:
    """
    Shared base class for element one-hot to atomic-energy projections.
    """

    BACKEND: ModelBackend

    def init(self, *, atomic_energies: Any) -> None:
        backend = _require_backend(self, "AtomicEnergiesBlock")
        make_atomic_energies = backend.require("make_atomic_energies")
        self.atomic_energies = make_atomic_energies(self, atomic_energies)

    def forward(self, x: Any) -> Any:
        backend = self.BACKEND
        get_atomic_energies = backend.require("get_atomic_energies")
        atleast_2d = backend.require("atleast_2d")
        matmul = backend.require("matmul")
        transpose = backend.require("transpose")

        atomic_energies = get_atomic_energies(self.atomic_energies)
        return matmul(x, transpose(atleast_2d(atomic_energies)))

    def __repr__(self) -> str:
        backend = self.BACKEND
        get_atomic_energies = backend.require("get_atomic_energies")
        atleast_2d = backend.require("atleast_2d")
        to_numpy = backend.require("to_numpy")

        atomic_energies = get_atomic_energies(self.atomic_energies)
        energies_np = to_numpy(atleast_2d(atomic_energies))
        formatted_energies = ", ".join(
            "[" + ", ".join(f"{value:.4f}" for value in row) + "]"
            for row in energies_np
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class InteractionBlock:
    """
    Shared base class for message-passing interaction blocks.

    Concrete subclasses define `_setup()` and `forward()` for different message
    update rules, while this base class owns backend-agnostic configuration,
    radial MLP defaults, and LAMMPS ghost-atom handling.
    """

    BACKEND: ModelBackend

    def init(
        self,
        *,
        node_attrs_irreps: Any,
        node_feats_irreps: Any,
        edge_attrs_irreps: Any,
        edge_feats_irreps: Any,
        target_irreps: Any,
        hidden_irreps: Any,
        avg_num_neighbors: float,
        edge_irreps: Any = None,
        radial_MLP: list[int] | None = None,
        cueq_config: Any = None,
        oeq_config: Any = None,
        rngs: Any = None,
    ) -> None:
        backend = _require_backend(self, "InteractionBlock")
        make_irreps = backend.require("make_irreps")

        self.node_attrs_irreps = make_irreps(node_attrs_irreps)
        self.node_feats_irreps = make_irreps(node_feats_irreps)
        self.edge_attrs_irreps = make_irreps(edge_attrs_irreps)
        self.edge_feats_irreps = make_irreps(edge_feats_irreps)
        self.target_irreps = make_irreps(target_irreps)
        self.hidden_irreps = make_irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = list(radial_MLP)

        if edge_irreps is None:
            self.edge_irreps = make_irreps(self.node_feats_irreps)
        else:
            self.edge_irreps = make_irreps(edge_irreps)

        if self.oeq_config and getattr(self.oeq_config, "conv_fusion", False):
            self.conv_fusion = self.oeq_config.conv_fusion
        if self.cueq_config and getattr(self.cueq_config, "conv_fusion", False):
            self.conv_fusion = self.cueq_config.conv_fusion

        self._setup(rngs=rngs)

    def _setup(self, *, rngs: Any = None) -> None:
        raise NotImplementedError

    def handle_lammps(
        self,
        node_feats: Any,
        lammps_class: Any,
        lammps_natoms: tuple[int, int],
        first_layer: bool,
    ) -> Any:
        """Pad and exchange node features when running under LAMMPS."""
        backend = self.BACKEND
        lammps_mp_apply = getattr(backend, "lammps_mp_apply", None)
        cat = getattr(backend, "cat", None)
        make_zeros = getattr(backend, "make_zeros", None)
        if (
            lammps_class is None
            or first_layer
            or lammps_mp_apply is None
            or cat is None
            or make_zeros is None
        ):
            return node_feats
        _, n_total = lammps_natoms
        pad = make_zeros(
            (n_total, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = cat([node_feats, pad], dim=0)
        return lammps_mp_apply(node_feats, lammps_class)

    def truncate_ghosts(self, tensor: Any, n_real: int | None = None) -> Any:
        """Drop ghost atoms from a tensor after a LAMMPS exchange step."""
        return tensor[:n_real] if n_real is not None else tensor

    @staticmethod
    def resolve_real_atom_count(
        lammps_class: Any, lammps_natoms: tuple[int, int]
    ) -> int | None:
        """Return the number of real atoms when running with LAMMPS ghosts."""
        return int(lammps_natoms[0]) if lammps_class is not None else None

    @staticmethod
    def apply_cutoff(value: Any, cutoff: Any = None) -> Any:
        """Apply an optional cutoff factor to edge-aligned values."""
        return value if cutoff is None else value * cutoff

    def truncate_tensors(self, n_real: int | None, *tensors: Any) -> tuple[Any, ...]:
        """Apply LAMMPS ghost truncation to several tensors at once."""
        return tuple(self.truncate_ghosts(tensor, n_real) for tensor in tensors)

    def scatter_edges_to_nodes(
        self,
        edge_values: Any,
        edge_index: Any,
        *,
        dim_size: int,
    ) -> Any:
        """Sum edge-aligned quantities onto receiver nodes."""
        scatter_sum = self.BACKEND.require("scatter_sum")
        return scatter_sum(
            src=edge_values,
            index=edge_index[1],
            dim=0,
            dim_size=dim_size,
        )

    def prepare_message_inputs(
        self,
        *,
        node_feats: Any,
        edge_feats: Any,
        cutoff: Any,
        lammps_class: Any,
        lammps_natoms: tuple[int, int],
        first_layer: bool,
    ) -> tuple[Any, Any, int | None]:
        """Lift node features to the edge space and prepare tensor-product weights."""
        n_real = self.resolve_real_atom_count(lammps_class, lammps_natoms)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.apply_cutoff(self.conv_tp_weights(edge_feats), cutoff)
        return node_feats, tp_weights, n_real

    def convolve_messages(
        self,
        *,
        node_feats: Any,
        edge_attrs: Any,
        tp_weights: Any,
        edge_index: Any,
    ) -> Any:
        """Apply the equivariant convolution and aggregate edge messages."""
        if hasattr(self, "conv_fusion"):
            return self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        edge_messages = self.conv_tp(
            node_feats[edge_index[0]],
            edge_attrs,
            tp_weights,
        )
        return self.scatter_edges_to_nodes(
            edge_messages,
            edge_index,
            dim_size=node_feats.shape[0],
        )

    def compute_density(
        self,
        *,
        edge_feats: Any,
        edge_index: Any,
        num_nodes: int,
        cutoff: Any,
    ) -> Any:
        """Compute receiver-node densities from radial edge features."""
        tanh = self.BACKEND.require("tanh")
        edge_density = tanh(self.density_fn(edge_feats) ** 2)
        edge_density = self.apply_cutoff(edge_density, cutoff)
        return self.scatter_edges_to_nodes(
            edge_density,
            edge_index,
            dim_size=num_nodes,
        )

    def setup_message_path(
        self,
        *,
        skip_irreps_in: Any,
        skip_irreps_out: Any,
        use_density: bool = False,
        rngs: Any = None,
    ) -> None:
        """Build the common tensor-product message path shared by interaction blocks."""
        backend = self.BACKEND
        make_linear = backend.require("make_linear")
        tp_out = backend.require("tp_out_irreps_with_instructions")
        make_tensor_product = backend.require("make_tensor_product")
        make_fully_connected_net = backend.require("make_fully_connected_net")
        make_fully_connected_tensor_product = backend.require(
            "make_fully_connected_tensor_product"
        )
        make_reshape_irreps = backend.require("reshape_irreps")
        silu = backend.require("silu")

        self.linear_up = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.edge_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        irreps_mid, instructions = tp_out(
            irreps1=self.edge_irreps,
            irreps2=self.edge_attrs_irreps,
            target_irreps=self.target_irreps,
        )
        self.conv_tp = make_tensor_product(
            irreps_in1=self.edge_irreps,
            irreps_in2=self.edge_attrs_irreps,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
            rngs=rngs,
        )
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = make_fully_connected_net(
            hs=[input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            act=silu,
            rngs=rngs,
        )
        self.irreps_out = self.target_irreps
        self.linear = make_linear(
            irreps_in=irreps_mid,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.skip_tp = make_fully_connected_tensor_product(
            irreps_in1=skip_irreps_in,
            irreps_in2=self.node_attrs_irreps,
            irreps_out=skip_irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        if use_density:
            self.density_fn = make_fully_connected_net(
                hs=[input_dim, 1],
                act=silu,
                rngs=rngs,
            )
        self.reshape = make_reshape_irreps(
            irreps=self.irreps_out,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
    ) -> Any:
        raise NotImplementedError


class RealAgnosticInteractionBlock(InteractionBlock):
    """Non-residual interaction block with element-agnostic tensor products."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        self.setup_message_path(
            skip_irreps_in=self.target_irreps,
            skip_irreps_out=self.target_irreps,
            rngs=rngs,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        lammps_class: Any = None,
        first_layer: bool = False,
    ) -> tuple[Any, None]:
        node_feats, tp_weights, n_real = self.prepare_message_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            cutoff=cutoff,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        message = self.convolve_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message, node_attrs = self.truncate_tensors(n_real, message, node_attrs)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return self.reshape(message), None


class RealAgnosticResidualInteractionBlock(InteractionBlock):
    """Residual interaction block with an equivariant skip connection."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        self.setup_message_path(
            skip_irreps_in=self.node_feats_irreps,
            skip_irreps_out=self.hidden_irreps,
            rngs=rngs,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_class: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> tuple[Any, Any]:
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats, tp_weights, n_real = self.prepare_message_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            cutoff=cutoff,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        message = self.convolve_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message, node_attrs, sc = self.truncate_tensors(n_real, message, node_attrs, sc)
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc


class RealAgnosticDensityInteractionBlock(InteractionBlock):
    """Density-normalized interaction block without an explicit residual path."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        self.setup_message_path(
            skip_irreps_in=self.target_irreps,
            skip_irreps_out=self.target_irreps,
            use_density=True,
            rngs=rngs,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_class: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> tuple[Any, None]:
        num_nodes = node_feats.shape[0]
        node_feats, tp_weights, n_real = self.prepare_message_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            cutoff=cutoff,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        density = self.compute_density(
            edge_feats=edge_feats,
            edge_index=edge_index,
            num_nodes=num_nodes,
            cutoff=cutoff,
        )
        message = self.convolve_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message, node_attrs, density = self.truncate_tensors(
            n_real,
            message,
            node_attrs,
            density,
        )
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return self.reshape(message), None


class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    """Density-normalized interaction block with a residual skip path."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        self.setup_message_path(
            skip_irreps_in=self.node_feats_irreps,
            skip_irreps_out=self.hidden_irreps,
            use_density=True,
            rngs=rngs,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_class: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> tuple[Any, Any]:
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats, tp_weights, n_real = self.prepare_message_inputs(
            node_feats=node_feats,
            edge_feats=edge_feats,
            cutoff=cutoff,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        density = self.compute_density(
            edge_feats=edge_feats,
            edge_index=edge_index,
            num_nodes=num_nodes,
            cutoff=cutoff,
        )
        message = self.convolve_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message, node_attrs, density, sc = self.truncate_tensors(
            n_real,
            message,
            node_attrs,
            density,
            sc,
        )
        message = self.linear(message) / (density + 1)
        return self.reshape(message), sc


class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    """Residual interaction block whose radial weights depend on node context."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        backend = self.BACKEND
        make_irreps = backend.require("make_irreps")
        make_linear = backend.require("make_linear")
        tp_out = backend.require("tp_out_irreps_with_instructions")
        make_tensor_product = backend.require("make_tensor_product")
        make_fully_connected_net = backend.require("make_fully_connected_net")
        make_reshape_irreps = backend.require("reshape_irreps")
        silu = backend.require("silu")

        self.node_feats_down_irreps = make_irreps("64x0e")
        self.linear_up = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.edge_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        irreps_mid, instructions = tp_out(
            irreps1=self.edge_irreps,
            irreps2=self.edge_attrs_irreps,
            target_irreps=self.target_irreps,
        )
        self.conv_tp = make_tensor_product(
            irreps_in1=self.edge_irreps,
            irreps_in2=self.edge_attrs_irreps,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
            rngs=rngs,
        )
        self.linear_down = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.node_feats_down_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = make_fully_connected_net(
            hs=[input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            act=silu,
            rngs=rngs,
        )
        self.irreps_out = self.target_irreps
        self.linear = make_linear(
            irreps_in=irreps_mid,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.reshape = make_reshape_irreps(
            irreps=self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.skip_linear = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_class: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> tuple[Any, Any]:
        del node_attrs, lammps_class, lammps_natoms, first_layer
        cat = self.BACKEND.require("cat")

        sender = edge_index[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = cat(
            [edge_feats, node_feats_down[sender], node_feats_down[edge_index[1]]],
            dim=-1,
        )
        tp_weights = self.apply_cutoff(
            self.conv_tp_weights(augmented_edge_feats),
            cutoff,
        )
        message = self.convolve_messages(
            node_feats=node_feats_up,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc


class RealAgnosticResidualNonLinearInteractionBlock(InteractionBlock):
    """Residual interaction block with a non-linear equivariant update head."""

    BACKEND: ModelBackend

    def _setup(self, *, rngs: Any = None) -> None:
        backend = self.BACKEND
        make_irreps = backend.require("make_irreps")
        make_irrep = backend.require("make_irrep")
        make_linear = backend.require("make_linear")
        tp_out = backend.require("tp_out_irreps_with_instructions")
        make_tensor_product = backend.require("make_tensor_product")
        make_radial_mlp = backend.require("make_radial_mlp")
        make_custom_gate = backend.require("make_custom_gate")
        make_reshape_irreps = backend.require("reshape_irreps")
        make_transpose_irreps_layout = backend.require("make_transpose_irreps_layout")
        make_parameter = backend.require("make_parameter")
        init_uniform_ = backend.require("init_uniform_")
        silu = backend.require("silu")
        sigmoid = backend.require("sigmoid")

        node_scalar_irreps = make_irreps(
            [(self.node_feats_irreps.count(make_irrep(0, 1)), (0, 1))]
        )
        self.source_embedding = make_linear(
            irreps_in=self.node_attrs_irreps,
            irreps_out=node_scalar_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.target_embedding = make_linear(
            irreps_in=self.node_attrs_irreps,
            irreps_out=node_scalar_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_up = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.edge_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        init_uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        init_uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        irreps_mid, instructions = tp_out(
            irreps1=self.edge_irreps,
            irreps2=self.edge_attrs_irreps,
            target_irreps=self.target_irreps,
        )
        self.conv_tp = make_tensor_product(
            irreps_in1=self.edge_irreps,
            irreps_in2=self.edge_attrs_irreps,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
            rngs=rngs,
        )

        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = make_radial_mlp(
            hs=[input_dim + 2 * node_scalar_irreps.dim]
            + self.radial_MLP
            + [self.conv_tp.weight_numel],
            rngs=rngs,
        )
        self.irreps_out = self.target_irreps
        self.skip_tp = make_linear(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.reshape = make_reshape_irreps(
            irreps=self.irreps_out,
            cueq_config=self.cueq_config,
        )

        irreps_scalars = make_irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_gated = make_irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        )
        irreps_gates = make_irreps([(mul, make_irrep(0, 1)) for mul, _ in irreps_gated])
        self.equivariant_nonlin = make_custom_gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[silu for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
            cueq_config=self.cueq_config,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_res = make_linear(
            irreps_in=self.edge_irreps,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_1 = make_linear(
            irreps_in=irreps_mid,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_2 = make_linear(
            irreps_in=self.irreps_out,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.density_fn = make_radial_mlp(
            hs=[input_dim + 2 * node_scalar_irreps.dim, 64, 1],
            rngs=rngs,
        )
        self.alpha = make_parameter(self, name="alpha", value=20.0, requires_grad=True)
        self.beta = make_parameter(self, name="beta", value=0.0, requires_grad=True)
        self.transpose_mul_ir = make_transpose_irreps_layout(
            irreps=self.irreps_nonlin,
            source="ir_mul",
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self.transpose_ir_mul = make_transpose_irreps_layout(
            irreps=self.irreps_out,
            source="mul_ir",
            target="ir_mul",
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: Any,
        node_feats: Any,
        edge_attrs: Any,
        edge_feats: Any,
        edge_index: Any,
        cutoff: Any = None,
        lammps_class: Any = None,
        lammps_natoms: tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> tuple[Any, Any]:
        cat = self.BACKEND.require("cat")

        num_nodes = node_feats.shape[0]
        n_real = self.resolve_real_atom_count(lammps_class, lammps_natoms)
        sc = self.skip_tp(node_feats)
        node_feats = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )

        source_embedding = self.source_embedding(node_attrs)
        target_embedding = self.target_embedding(node_attrs)
        edge_feats = cat(
            [
                edge_feats,
                source_embedding[edge_index[0]],
                target_embedding[edge_index[1]],
            ],
            dim=-1,
        )
        tp_weights = self.apply_cutoff(self.conv_tp_weights(edge_feats), cutoff)
        density = self.compute_density(
            edge_feats=edge_feats,
            edge_index=edge_index,
            num_nodes=num_nodes,
            cutoff=cutoff,
        )
        message = self.convolve_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            tp_weights=tp_weights,
            edge_index=edge_index,
        )
        message, density, sc, node_feats_res = self.truncate_tensors(
            n_real,
            message,
            density,
            sc,
            node_feats_res,
        )
        message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
        if self.transpose_mul_ir is not None:
            message = self.transpose_mul_ir(message)
        message = self.equivariant_nonlin(message)
        if self.transpose_ir_mul is not None:
            message = self.transpose_ir_mul(message)
        message = self.linear_2(message)
        return self.reshape(message), sc
