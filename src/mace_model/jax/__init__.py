from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgnesiTransform",
    "AtomicEnergiesBlock",
    "BesselBasis",
    "ChebychevBasis",
    "EquivariantProductBasisBlock",
    "GaussianBasis",
    "GenericJointEmbedding",
    "InteractionBlock",
    "LinearDipolePolarReadoutBlock",
    "LinearDipoleReadoutBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "MACE",
    "NonLinearBiasReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "RadialEmbeddingBlock",
    "RadialMLP",
    "RealAgnosticAttResidualInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticDensityResidualInteractionBlock",
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
    "ScaleShiftBlock",
    "ScaleShiftMACE",
    "SoftTransform",
    "ZBLBasis",
]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module(".modules", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
