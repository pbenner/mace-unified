from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgnesiTransform",
    "AtomicEnergiesBlock",
    "AtomicDielectricMACE",
    "AtomicDipolesMACE",
    "BesselBasis",
    "ChebychevBasis",
    "EquivariantProductBasisBlock",
    "EnergyDipolesMACE",
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
    "TorchInferenceWrapper",
    "ZBLBasis",
    "compile_model",
    "export_model",
    "graph_to_inference_args",
    "make_inference_wrapper",
]


def __getattr__(name: str):
    if name in __all__:
        if name in {
            "TorchInferenceWrapper",
            "compile_model",
            "export_model",
            "graph_to_inference_args",
            "make_inference_wrapper",
        }:
            return getattr(import_module(".tools", __name__), name)
        return getattr(import_module(".modules", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
