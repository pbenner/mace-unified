from __future__ import annotations

import pytest

try:
    import cuequivariance_jax  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"cuequivariance_jax is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

from mace_model.jax.modules import blocks as jax_blocks
from mace_model.jax.modules import embeddings as jax_embeddings
from mace_model.jax.modules import models as jax_models
from mace_model.jax.modules import radial as jax_radial
from mace_model.torch.modules import blocks as torch_blocks
from mace_model.torch.modules import embeddings as torch_embeddings
from mace_model.torch.modules import models as torch_models
from mace_model.torch.modules import radial as torch_radial

TORCH_BLOCK_NAMES = [
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

JAX_BLOCK_NAMES = TORCH_BLOCK_NAMES
RADIAL_NAMES = [
    "AgnesiTransform",
    "BesselBasis",
    "ChebychevBasis",
    "GaussianBasis",
    "PolynomialCutoff",
    "RadialMLP",
    "SoftTransform",
    "ZBLBasis",
]
TORCH_MODEL_NAMES = [
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
]
JAX_MODEL_NAMES = [
    "MACE",
    "ScaleShiftMACE",
]


def test_torch_block_surface_exports_expected_symbols():
    for name in TORCH_BLOCK_NAMES:
        assert hasattr(torch_blocks, name)


def test_jax_block_surface_exports_expected_symbols():
    for name in JAX_BLOCK_NAMES:
        assert hasattr(jax_blocks, name)


def test_embedding_surface_exports_expected_symbols():
    assert hasattr(torch_embeddings, "GenericJointEmbedding")
    assert hasattr(jax_embeddings, "GenericJointEmbedding")


def test_model_surface_exports_expected_symbols():
    for name in TORCH_MODEL_NAMES:
        assert hasattr(torch_models, name)
    for name in JAX_MODEL_NAMES:
        assert hasattr(jax_models, name)


def test_radial_surface_exports_expected_symbols():
    for name in RADIAL_NAMES:
        assert hasattr(torch_radial, name)
        assert hasattr(jax_radial, name)
