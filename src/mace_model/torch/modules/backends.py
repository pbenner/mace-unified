from __future__ import annotations

import numpy as np
import torch
from mace_model.torch.adapters.cuequivariance import (
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
    TransposeIrrepsLayoutWrapper,
)
from mace_model.core.modules.backends import define_backend
from mace_model.core.modules.irreps_utils import (
    CachedIrrepsReshaper,
    tp_out_irreps_with_instructions,
)
from mace_model.torch.adapters.e3nn import nn, o3

from ..tools.compile import simplify_if_compile
from ..tools.scatter import scatter_sum
from ..tools.utils import LAMMPS_MP
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
)


def _tp_out_irreps_with_instructions_torch(
    irreps1: o3.Irreps,
    irreps2: o3.Irreps,
    target_irreps: o3.Irreps,
) -> tuple[o3.Irreps, list[tuple[int, int, int, str, bool]]]:
    return tp_out_irreps_with_instructions(
        make_irreps=o3.Irreps,
        irreps1=irreps1,
        irreps2=irreps2,
        target_irreps=target_irreps,
    )


class _ReshapeIrreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, cueq_config: object | None = None) -> None:
        super().__init__()
        self._reshaper = CachedIrrepsReshaper(
            make_irreps=o3.Irreps,
            irreps=irreps,
            cueq_config=cueq_config,
        )
        self.irreps = self._reshaper.irreps
        self.dims = self._reshaper.dims
        self.muls = self._reshaper.muls
        self.cueq_config = self._reshaper.cueq_config

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._reshaper.reshape(
            tensor,
            concat_fields=lambda fields, axis: torch.cat(fields, dim=axis),
        )


def _mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(
        x.shape[0],
        x.shape[1] // num_heads,
        num_heads,
        device=x.device,
        dtype=x.dtype,
    )
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask


@define_backend(name="torch")
class _TorchBackendSpec:
    @staticmethod
    def make_irreps(value):
        return o3.Irreps(value)

    @staticmethod
    def make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        del rngs
        return Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
        )

    @staticmethod
    def make_bias_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        del cueq_config, rngs
        return o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            biases=True,
        )

    @staticmethod
    def make_activation(*, hidden_irreps, gate, cueq_config):
        del cueq_config
        return simplify_if_compile(nn.Activation)(
            irreps_in=hidden_irreps,
            acts=[gate],
        )

    @staticmethod
    def make_gate(*, irreps_scalars, irreps_gates, irreps_gated, gate, cueq_config):
        del cueq_config
        return nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

    @staticmethod
    def make_bessel_basis(*, r_max, num_basis, rngs):
        del rngs
        return BesselBasis(r_max=r_max, num_basis=num_basis)

    @staticmethod
    def make_gaussian_basis(*, r_max, num_basis, rngs):
        del rngs
        return GaussianBasis(r_max=r_max, num_basis=num_basis)

    @staticmethod
    def make_chebychev_basis(*, r_max, num_basis):
        return ChebychevBasis(r_max=r_max, num_basis=num_basis)

    @staticmethod
    def make_polynomial_cutoff(*, r_max, p):
        return PolynomialCutoff(r_max=r_max, p=p)

    @staticmethod
    def make_agnesi_transform(*, rngs):
        del rngs
        return AgnesiTransform()

    @staticmethod
    def make_soft_transform(*, rngs):
        del rngs
        return SoftTransform()

    @staticmethod
    def make_joint_embedders(*, specs, feature_names, rngs):
        del rngs
        embedders = torch.nn.ModuleDict()
        for name in feature_names:
            spec = specs[name]
            emb_dim = int(spec["emb_dim"])
            use_bias = bool(spec.get("use_bias", True))

            if spec["type"] == "categorical":
                embedders[name] = torch.nn.Embedding(
                    int(spec["num_classes"]),
                    emb_dim,
                )
            elif spec["type"] == "continuous":
                in_dim = int(spec.get("in_dim", 1))
                embedders[name] = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, emb_dim, bias=use_bias),
                    torch.nn.SiLU(),
                    torch.nn.Linear(emb_dim, emb_dim, bias=use_bias),
                )
            else:
                raise ValueError(f"Unknown type {spec['type']} for feature {name}")
        return embedders

    @staticmethod
    def make_joint_projection(*, total_dim, out_dim, rngs):
        del rngs
        return torch.nn.Sequential(
            torch.nn.Linear(int(total_dim), int(out_dim), bias=False),
            torch.nn.SiLU(),
        )

    @staticmethod
    def make_joint_categorical_indices(*, feat, offset):
        return (feat + offset).long().squeeze(-1)

    @staticmethod
    def make_symmetric_contraction(
        *,
        irreps_in,
        irreps_out,
        correlation,
        num_elements,
        use_reduced_cg,
        cueq_config,
        oeq_config,
        rngs,
    ):
        del rngs
        return SymmetricContractionWrapper(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

    @staticmethod
    def make_tensor_product(
        *,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights,
        internal_weights,
        cueq_config,
        oeq_config,
        rngs=None,
    ):
        del rngs
        return TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

    @staticmethod
    def make_fully_connected_tensor_product(
        *,
        irreps_in1,
        irreps_in2,
        irreps_out,
        cueq_config,
        rngs=None,
    ):
        del rngs
        return FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            cueq_config=cueq_config,
        )

    @staticmethod
    def make_fully_connected_net(*, hs, act, rngs=None):
        del rngs
        return nn.FullyConnectedNet(hs, act)

    @staticmethod
    def make_radial_mlp(*, hs, rngs=None):
        del rngs
        return RadialMLP(hs)

    @staticmethod
    def make_transpose_irreps_layout(*, irreps, source, target, cueq_config):
        return TransposeIrrepsLayoutWrapper(
            irreps=irreps,
            source=source,
            target=target,
            cueq_config=cueq_config,
        )

    @staticmethod
    def make_custom_gate(
        *,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
        cueq_config,
    ):
        del cueq_config
        return nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_gated,
        )

    @staticmethod
    def tp_out_irreps_with_instructions(*, irreps1, irreps2, target_irreps):
        return _tp_out_irreps_with_instructions_torch(
            irreps1,
            irreps2,
            target_irreps,
        )

    @staticmethod
    def reshape_irreps(*, irreps, cueq_config):
        return _ReshapeIrreps(irreps, cueq_config=cueq_config)

    @staticmethod
    def scatter_sum(src, index, dim=-1, dim_size=None, indices_are_sorted=False):
        del indices_are_sorted
        return scatter_sum(src=src, index=index, dim=dim, dim_size=dim_size)

    stack = staticmethod(torch.stack)
    sum = staticmethod(torch.sum)

    @staticmethod
    def make_irrep(l, p):
        return o3.Irrep(l, p)

    init_uniform_ = staticmethod(torch.nn.init.uniform_)
    tanh = staticmethod(torch.tanh)
    silu = staticmethod(torch.nn.functional.silu)
    sigmoid = staticmethod(torch.sigmoid)
    cat = staticmethod(lambda values, dim=-1: torch.cat(values, dim=dim))
    make_zeros = staticmethod(torch.zeros)

    @staticmethod
    def make_parameter(module, *, name, value, requires_grad=True):
        module.register_parameter(
            name,
            torch.nn.Parameter(
                torch.tensor(value, dtype=torch.get_default_dtype()),
                requires_grad=requires_grad,
            ),
        )
        return getattr(module, name)

    lammps_mp_apply = staticmethod(LAMMPS_MP.apply)

    @staticmethod
    def make_scale_shift(module, *, name, value):
        module.register_buffer(
            name,
            torch.tensor(value, dtype=torch.get_default_dtype()),
        )
        return getattr(module, name)

    @staticmethod
    def get_scale_shift(value):
        return value

    @staticmethod
    def make_atomic_energies(module, atomic_energies):
        module.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )
        return module.atomic_energies

    @staticmethod
    def get_atomic_energies(atomic_energies):
        return atomic_energies

    @staticmethod
    def make_ones(*, node_feats, width):
        return torch.ones(
            (node_feats.shape[0], int(width)),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )

    @staticmethod
    def make_index_attrs(*, node_attrs, node_attrs_index):
        del node_attrs_index
        return torch.nonzero(node_attrs)[:, 1].int()

    transpose_mul_ir = staticmethod(lambda x: torch.transpose(x, 1, 2))
    atleast_1d = staticmethod(torch.atleast_1d)
    atleast_2d = staticmethod(torch.atleast_2d)
    matmul = staticmethod(torch.matmul)
    transpose = staticmethod(lambda x: x.T)
    mask_head = staticmethod(_mask_head)
    mask_head_stage1 = staticmethod(_mask_head)

    @staticmethod
    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)


TORCH_BACKEND = _TorchBackendSpec
