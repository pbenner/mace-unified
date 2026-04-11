from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
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

from mace_model.torch.adapters.e3nn import o3

from .backends import TORCH_BACKEND

if TYPE_CHECKING:
    from mace_model.torch.adapters.cuequivariance import CuEquivarianceConfig, OEQConfig
else:
    CuEquivarianceConfig = Any
    OEQConfig = Any


@use_backend(TORCH_BACKEND)
class LinearNodeEmbeddingBlock(CoreLinearNodeEmbeddingBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.init(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreLinearNodeEmbeddingBlock.forward


@use_backend(TORCH_BACKEND)
class LinearReadoutBlock(CoreLinearReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            irrep_out=irrep_out,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreLinearReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class LinearDipoleReadoutBlock(CoreLinearDipoleReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            dipole_only=dipole_only,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreLinearDipoleReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class NonLinearDipoleReadoutBlock(CoreNonLinearDipoleReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            dipole_only=dipole_only,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreNonLinearDipoleReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class NonLinearReadoutBlock(CoreNonLinearReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable | None,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreNonLinearReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class NonLinearBiasReadoutBlock(CoreNonLinearBiasReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable | None,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            irrep_out=irrep_out,
            num_heads=num_heads,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreNonLinearBiasReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class LinearDipolePolarReadoutBlock(CoreLinearDipolePolarReadoutBlock, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            use_polarizability=use_polarizability,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreLinearDipolePolarReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class NonLinearDipolePolarReadoutBlock(
    CoreNonLinearDipolePolarReadoutBlock, torch.nn.Module
):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        del oeq_config
        self.init(
            irreps_in=irreps_in,
            mlp_irreps=MLP_irreps,
            gate=gate,
            use_polarizability=use_polarizability,
            cueq_config=cueq_config,
            rngs=None,
        )

    forward = CoreNonLinearDipolePolarReadoutBlock.forward


@use_backend(TORCH_BACKEND)
class RadialEmbeddingBlock(CoreRadialEmbeddingBlock, torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.init(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
            rngs=None,
        )

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return CoreRadialEmbeddingBlock.forward(
            self,
            edge_lengths=edge_lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            node_attrs_index=None,
        )


@use_backend(TORCH_BACKEND)
class ScaleShiftBlock(CoreScaleShiftBlock, torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        torch.nn.Module.__init__(self)
        self.init(scale=scale, shift=shift)

    forward = CoreScaleShiftBlock.forward
    __repr__ = CoreScaleShiftBlock.__repr__


@use_backend(TORCH_BACKEND)
class AtomicEnergiesBlock(CoreAtomicEnergiesBlock, torch.nn.Module):
    def __init__(self, atomic_energies: np.ndarray | torch.Tensor):
        torch.nn.Module.__init__(self)
        self.init(atomic_energies=atomic_energies)

    forward = CoreAtomicEnergiesBlock.forward
    __repr__ = CoreAtomicEnergiesBlock.__repr__


@use_backend(TORCH_BACKEND)
class EquivariantProductBasisBlock(CoreEquivariantProductBasisBlock, torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: int | None = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: bool | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.init(
            node_feats_irreps=node_feats_irreps,
            target_irreps=target_irreps,
            correlation=correlation,
            use_sc=use_sc,
            num_elements=num_elements,
            use_agnostic_product=use_agnostic_product,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            rngs=None,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        return CoreEquivariantProductBasisBlock.forward(
            self,
            node_feats=node_feats,
            sc=sc,
            node_attrs=node_attrs,
            node_attrs_index=None,
        )


@use_backend(TORCH_BACKEND)
class InteractionBlock(CoreInteractionBlock, torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        edge_irreps: o3.Irreps | None = None,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: OEQConfig | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
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
            rngs=None,
        )

    forward = CoreInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticInteractionBlock(CoreRealAgnosticInteractionBlock, InteractionBlock):
    forward = CoreRealAgnosticInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticResidualInteractionBlock(
    CoreRealAgnosticResidualInteractionBlock, InteractionBlock
):
    forward = CoreRealAgnosticResidualInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticDensityInteractionBlock(
    CoreRealAgnosticDensityInteractionBlock, InteractionBlock
):
    forward = CoreRealAgnosticDensityInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticDensityResidualInteractionBlock(
    CoreRealAgnosticDensityResidualInteractionBlock, InteractionBlock
):
    forward = CoreRealAgnosticDensityResidualInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticAttResidualInteractionBlock(
    CoreRealAgnosticAttResidualInteractionBlock, InteractionBlock
):
    forward = CoreRealAgnosticAttResidualInteractionBlock.forward


@use_backend(TORCH_BACKEND)
class RealAgnosticResidualNonLinearInteractionBlock(
    CoreRealAgnosticResidualNonLinearInteractionBlock, InteractionBlock
):
    forward = CoreRealAgnosticResidualNonLinearInteractionBlock.forward


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
