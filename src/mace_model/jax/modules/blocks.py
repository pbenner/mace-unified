from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from mace_model.core.modules.backends import use_backend
from mace_model.core.modules.blocks import (
    AtomicEnergiesBlock as CoreAtomicEnergiesBlock,
)
from mace_model.core.modules.blocks import (
    EquivariantProductBasisBlock as CoreEquivariantProductBasisBlock,
)
from mace_model.core.modules.blocks import InteractionBlock as CoreInteractionBlock
from mace_model.core.modules.blocks import (
    LinearDipolePolarReadoutBlock as CoreLinearDipolePolarReadoutBlock,
)
from mace_model.core.modules.blocks import (
    LinearDipoleReadoutBlock as CoreLinearDipoleReadoutBlock,
)
from mace_model.core.modules.blocks import (
    LinearNodeEmbeddingBlock as CoreLinearNodeEmbeddingBlock,
)
from mace_model.core.modules.blocks import LinearReadoutBlock as CoreLinearReadoutBlock
from mace_model.core.modules.blocks import (
    NonLinearBiasReadoutBlock as CoreNonLinearBiasReadoutBlock,
)
from mace_model.core.modules.blocks import (
    NonLinearDipolePolarReadoutBlock as CoreNonLinearDipolePolarReadoutBlock,
)
from mace_model.core.modules.blocks import (
    NonLinearDipoleReadoutBlock as CoreNonLinearDipoleReadoutBlock,
)
from mace_model.core.modules.blocks import (
    NonLinearReadoutBlock as CoreNonLinearReadoutBlock,
)
from mace_model.core.modules.blocks import (
    RadialEmbeddingBlock as CoreRadialEmbeddingBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticAttResidualInteractionBlock as CoreRealAgnosticAttResidualInteractionBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticDensityInteractionBlock as CoreRealAgnosticDensityInteractionBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock as CoreRealAgnosticDensityResidualInteractionBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticInteractionBlock as CoreRealAgnosticInteractionBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticResidualInteractionBlock as CoreRealAgnosticResidualInteractionBlock,
)
from mace_model.core.modules.blocks import (
    RealAgnosticResidualNonLinearInteractionBlock as CoreRealAgnosticResidualNonLinearInteractionBlock,
)
from mace_model.core.modules.blocks import ScaleShiftBlock as CoreScaleShiftBlock

from mace_model.jax.adapters.e3nn import Irreps
from ..adapters.nnx.torch import nxx_auto_import_from_torch
from .backends import JAX_BACKEND

if TYPE_CHECKING:
    from mace_model.jax.adapters.cuequivariance import CuEquivarianceConfig
else:
    CuEquivarianceConfig = Any


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearNodeEmbeddingBlock(CoreLinearNodeEmbeddingBlock, nnx.Module):
    """
    Flax/JAX unified node-embedding block.
    """

    irreps_in: Irreps
    irreps_out: Irreps
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreLinearNodeEmbeddingBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearReadoutBlock(CoreLinearReadoutBlock, nnx.Module):
    """
    Flax/JAX unified linear readout block.
    """

    irreps_in: Irreps
    irrep_out: Irreps = Irreps("0e")
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        irrep_out: Irreps = Irreps("0e"),
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            irrep_out=irrep_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreLinearReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearDipoleReadoutBlock(CoreLinearDipoleReadoutBlock, nnx.Module):
    """
    Flax/JAX unified linear dipole readout block.
    """

    irreps_in: Irreps
    dipole_only: bool = False
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            dipole_only=dipole_only,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreLinearDipoleReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearDipoleReadoutBlock(CoreNonLinearDipoleReadoutBlock, nnx.Module):
    """
    Flax/JAX unified non-linear dipole readout block.
    """

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    dipole_only: bool = False
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            dipole_only=dipole_only,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreNonLinearDipoleReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearReadoutBlock(CoreNonLinearReadoutBlock, nnx.Module):
    """
    Flax/JAX unified readout block.

    Signature mirrors mace_model.jax.modules.blocks.NonLinearReadoutBlock.
    """

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable | None
    irrep_out: Irreps = Irreps("0e")
    num_heads: int = 1
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable | None,
        irrep_out: Irreps = Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreNonLinearReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearBiasReadoutBlock(CoreNonLinearBiasReadoutBlock, nnx.Module):
    """
    Flax/JAX unified non-linear bias readout block.
    """

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable | None
    irrep_out: Irreps = Irreps("0e")
    num_heads: int = 1
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable | None,
        irrep_out: Irreps = Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreNonLinearBiasReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearDipolePolarReadoutBlock(CoreLinearDipolePolarReadoutBlock, nnx.Module):
    """
    Flax/JAX unified linear dipole-polar readout block.
    """

    irreps_in: Irreps
    use_polarizability: bool = True
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            use_polarizability=use_polarizability,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreLinearDipolePolarReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearDipolePolarReadoutBlock(
    CoreNonLinearDipolePolarReadoutBlock, nnx.Module
):
    """
    Flax/JAX unified non-linear dipole-polar readout block.
    """

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    use_polarizability: bool = True
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            use_polarizability=use_polarizability,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    __call__ = CoreNonLinearDipolePolarReadoutBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RadialEmbeddingBlock(CoreRadialEmbeddingBlock, nnx.Module):
    """
    Flax/JAX unified radial embedding block.
    """

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    radial_type: str = "bessel"
    distance_transform: str = "None"
    apply_cutoff: bool = True

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.init(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
            rngs=rngs,
        )

    def __call__(
        self,
        edge_lengths: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ):
        return CoreRadialEmbeddingBlock.forward(
            self,
            edge_lengths=edge_lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            node_attrs_index=node_attrs_index,
        )


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class ScaleShiftBlock(CoreScaleShiftBlock, nnx.Module):
    """
    Flax/JAX unified scale-shift block.
    """

    def __init__(self, scale: float | jnp.ndarray, shift: float | jnp.ndarray) -> None:
        self.init(scale=scale, shift=shift)

    __call__ = CoreScaleShiftBlock.forward
    __repr__ = CoreScaleShiftBlock.__repr__


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AtomicEnergiesBlock(CoreAtomicEnergiesBlock, nnx.Module):
    """
    Flax/JAX unified atomic-energies block.
    """

    atomic_energies_init: np.ndarray | jnp.ndarray

    def __init__(
        self,
        atomic_energies_init: np.ndarray | jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        self.atomic_energies_init = atomic_energies_init
        self.init(atomic_energies=atomic_energies_init)

    __call__ = CoreAtomicEnergiesBlock.forward
    __repr__ = CoreAtomicEnergiesBlock.__repr__


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class InteractionBlock(CoreInteractionBlock, nnx.Module):
    """
    Flax/JAX unified interaction-block base wrapper.
    """

    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        edge_irreps: Irreps | None = None,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: Any = None,
        attn_num_heads: int | None = None,
        attn_head_dim: int | None = None,
        attn_gate_mode: str | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        del attn_num_heads
        del attn_head_dim
        del attn_gate_mode
        self.init(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=target_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            edge_irreps=edge_irreps,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            rngs=rngs,
        )

    __call__ = CoreInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticInteractionBlock(CoreRealAgnosticInteractionBlock, InteractionBlock):
    __call__ = CoreRealAgnosticInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticResidualInteractionBlock(
    CoreRealAgnosticResidualInteractionBlock, InteractionBlock
):
    __call__ = CoreRealAgnosticResidualInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticDensityInteractionBlock(
    CoreRealAgnosticDensityInteractionBlock, InteractionBlock
):
    __call__ = CoreRealAgnosticDensityInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticDensityResidualInteractionBlock(
    CoreRealAgnosticDensityResidualInteractionBlock, InteractionBlock
):
    __call__ = CoreRealAgnosticDensityResidualInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticAttResidualInteractionBlock(
    CoreRealAgnosticAttResidualInteractionBlock, InteractionBlock
):
    __call__ = CoreRealAgnosticAttResidualInteractionBlock.forward


@use_backend(JAX_BACKEND)
class RealAgnosticResidualNonLinearInteractionBlock(
    CoreRealAgnosticResidualNonLinearInteractionBlock, InteractionBlock
):
    __call__ = CoreRealAgnosticResidualNonLinearInteractionBlock.forward


@use_backend(JAX_BACKEND)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EquivariantProductBasisBlock(CoreEquivariantProductBasisBlock, nnx.Module):
    """
    Flax/JAX unified equivariant-product basis block.
    """

    node_feats_irreps: Irreps
    target_irreps: Irreps
    correlation: int
    use_sc: bool = True
    num_elements: int | None = None
    use_agnostic_product: bool = False
    use_reduced_cg: bool | None = None
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        node_feats_irreps: Irreps,
        target_irreps: Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: int | None = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: bool | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        replace_symmetric_contraction: bool = False,
        replacement_hidden_irreps: Irreps | None = None,
        replacement_depth: int = 2,
        replacement_use_species_conditioning: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        del replacement_hidden_irreps
        del replacement_depth
        del replacement_use_species_conditioning
        if replace_symmetric_contraction:
            raise NotImplementedError(
                "Unified JAX EquivariantProductBasisBlock does not support "
                "replace_symmetric_contraction yet."
            )
        self.init(
            node_feats_irreps=node_feats_irreps,
            target_irreps=target_irreps,
            correlation=correlation,
            use_sc=use_sc,
            num_elements=num_elements,
            use_agnostic_product=use_agnostic_product,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=None,
            rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jnp.ndarray,
        sc: jnp.ndarray | None,
        node_attrs: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ):
        return CoreEquivariantProductBasisBlock.forward(
            self,
            node_feats=node_feats,
            sc=sc,
            node_attrs=node_attrs,
            node_attrs_index=node_attrs_index,
        )


__all__ = [
    "AtomicEnergiesBlock",
    "EquivariantProductBasisBlock",
    "InteractionBlock",
    "LinearDipolePolarReadoutBlock",
    "LinearDipoleReadoutBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearBiasReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearReadoutBlock",
    "RadialEmbeddingBlock",
    "RealAgnosticAttResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
    "ScaleShiftBlock",
]
