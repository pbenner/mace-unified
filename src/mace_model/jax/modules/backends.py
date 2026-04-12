from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from flax import nnx
from mace_model.core.modules.backends import define_backend
from mace_model.core.modules.irreps_utils import (
    CachedIrrepsReshaper,
    tp_out_irreps_with_instructions,
)

from mace_model.jax.adapters.e3nn import nn
from mace_model.jax.adapters.e3nn import Irrep, Irreps, IrrepsArray
from mace_model.jax.adapters.cuequivariance import (
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
)
from mace_model.jax.adapters.cuequivariance.utility import (
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
)
from ..tools.dtype import default_dtype
from ..tools.scatter import scatter_sum as jax_scatter_sum

from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
)


class _ContinuousEmbed(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.lin1 = nnx.Linear(in_dim, out_dim, use_bias=use_bias, rngs=rngs)
        self.lin2 = nnx.Linear(out_dim, out_dim, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.lin1(x)
        x = jnn.silu(x)
        return self.lin2(x)


class _JointProjection(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs) -> None:
        self.lin = nnx.Linear(in_dim, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnn.silu(self.lin(x))


class _NoOpTransposeIrrepsLayout:
    def __init__(self, irreps: Irreps, source: str, target: str):
        self.irreps = Irreps(irreps)
        self.source = source
        self.target = target

    def __call__(self, x):
        return_irreps = isinstance(x, IrrepsArray)
        array = x.array if return_irreps else x

        if self.source == self.target:
            transposed = array
        elif self.source == "ir_mul" and self.target == "mul_ir":
            transposed = ir_mul_to_mul_ir(array, self.irreps)
        elif self.source == "mul_ir" and self.target == "ir_mul":
            transposed = mul_ir_to_ir_mul(array, self.irreps)
        else:
            raise ValueError(
                f"Unsupported irreps layout transpose {self.source!r} -> {self.target!r}."
            )

        if return_irreps:
            return IrrepsArray(x.irreps, transposed, layout_str=self.target)
        return transposed


def _tp_out_irreps_with_instructions_jax(
    irreps1: Irreps,
    irreps2: Irreps,
    target_irreps: Irreps,
) -> tuple[Irreps, list]:
    return tp_out_irreps_with_instructions(
        make_irreps=Irreps,
        irreps1=irreps1,
        irreps2=irreps2,
        target_irreps=target_irreps,
    )


class _ReshapeIrreps:
    def __init__(self, irreps: Irreps, cueq_config: object | None = None):
        self._reshaper = CachedIrrepsReshaper(
            make_irreps=Irreps,
            irreps=irreps,
            cueq_config=cueq_config,
        )
        self.irreps = self._reshaper.irreps
        self._dims = self._reshaper.dims
        self._muls = self._reshaper.muls
        self._total_dim = self._reshaper.total_dim
        self.cueq_config = self._reshaper.cueq_config

    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        array = getattr(tensor, "array", tensor)
        return self._reshaper.reshape(
            array,
            concat_fields=lambda fields, axis: jnp.concatenate(fields, axis=axis),
            validate_input=True,
        )


def _mask_head(x: jnp.ndarray | IrrepsArray, head: int, num_heads: int):
    return_irreps = isinstance(x, IrrepsArray)
    array = x.array if return_irreps else x
    batch, features = array.shape
    head_dim = features // num_heads
    mask = jnp.zeros((batch, head_dim, num_heads), dtype=array.dtype)
    idx = jnp.arange(batch)
    mask = mask.at[idx, :, head].set(1)
    mask = jnp.transpose(mask, (0, 2, 1)).reshape(array.shape)
    masked = array * mask
    if return_irreps:
        return IrrepsArray(x.irreps, masked, layout_str=x.layout_str)
    return masked


@define_backend(name="jax")
class _JaxBackendSpec:
    @staticmethod
    def make_irreps(value):
        return Irreps(value)

    @staticmethod
    def make_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        return Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    @staticmethod
    def make_bias_linear(*, irreps_in, irreps_out, cueq_config, rngs):
        return Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    @staticmethod
    def make_activation(*, hidden_irreps, gate, cueq_config):
        return nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate],
            layout_str=getattr(cueq_config, "layout_str", "mul_ir"),
        )

    @staticmethod
    def make_gate(*, irreps_scalars, irreps_gates, irreps_gated, gate, cueq_config):
        return nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
            layout_str=getattr(cueq_config, "layout_str", "mul_ir"),
        )

    @staticmethod
    def make_bessel_basis(*, r_max, num_basis, rngs):
        return BesselBasis(r_max=r_max, num_basis=num_basis, rngs=rngs)

    @staticmethod
    def make_gaussian_basis(*, r_max, num_basis, rngs):
        return GaussianBasis(r_max=r_max, num_basis=num_basis, rngs=rngs)

    @staticmethod
    def make_chebychev_basis(*, r_max, num_basis):
        return ChebychevBasis(r_max=r_max, num_basis=num_basis)

    @staticmethod
    def make_polynomial_cutoff(*, r_max, p):
        return PolynomialCutoff(r_max=r_max, p=p)

    @staticmethod
    def make_agnesi_transform(*, rngs):
        return AgnesiTransform(rngs=rngs)

    @staticmethod
    def make_soft_transform(*, rngs):
        return SoftTransform(rngs=rngs)

    @staticmethod
    def make_radial_mlp(*, hs, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        return RadialMLP(hs, rngs=rngs)

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
        del oeq_config
        return TensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            cueq_config=cueq_config,
            rngs=rngs,
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
        return FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    @staticmethod
    def make_fully_connected_net(*, hs, act, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        return nn.FullyConnectedNet(hs, act, rngs=rngs)

    @staticmethod
    def make_transpose_irreps_layout(*, irreps, source, target, cueq_config):
        del cueq_config
        return _NoOpTransposeIrrepsLayout(irreps=irreps, source=source, target=target)

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
        return nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_gated,
            layout_str=getattr(cueq_config, "layout_str", "mul_ir"),
        )

    @staticmethod
    def tp_out_irreps_with_instructions(*, irreps1, irreps2, target_irreps):
        return _tp_out_irreps_with_instructions_jax(irreps1, irreps2, target_irreps)

    @staticmethod
    def reshape_irreps(*, irreps, cueq_config):
        return _ReshapeIrreps(irreps, cueq_config=cueq_config)

    @staticmethod
    def make_joint_embedders(*, specs, feature_names, rngs):
        embedders = nnx.Dict()
        for name in feature_names:
            spec = specs[name]
            emb_dim = int(spec["emb_dim"])

            if spec["type"] == "categorical":
                embedders[name] = nnx.Embed(
                    num_embeddings=int(spec["num_classes"]),
                    features=emb_dim,
                    rngs=rngs,
                )
            elif spec["type"] == "continuous":
                embedders[name] = _ContinuousEmbed(
                    in_dim=int(spec.get("in_dim", 1)),
                    out_dim=emb_dim,
                    use_bias=bool(spec.get("use_bias", True)),
                    rngs=rngs,
                )
            else:
                raise ValueError(f"Unknown feature type {spec['type']!r}")
        return embedders

    @staticmethod
    def make_joint_projection(*, total_dim, out_dim, rngs):
        return _JointProjection(int(total_dim), int(out_dim), rngs=rngs)

    @staticmethod
    def make_joint_categorical_indices(*, feat, offset):
        return (feat + offset).astype(jnp.int32).squeeze(-1)

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
        del oeq_config
        return SymmetricContractionWrapper(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    @staticmethod
    def make_scale_shift(module, *, name, value):
        del module, name
        return nnx.Param(jnp.asarray(value, dtype=default_dtype()))

    @staticmethod
    def get_scale_shift(value):
        return jax.lax.stop_gradient(value)

    @staticmethod
    def make_atomic_energies(module, atomic_energies):
        del module
        init_values = jnp.asarray(atomic_energies, dtype=default_dtype())
        return nnx.Param(init_values)

    @staticmethod
    def get_atomic_energies(atomic_energies):
        return jax.lax.stop_gradient(atomic_energies)

    @staticmethod
    def make_ones(*, node_feats, width):
        return jnp.ones((node_feats.shape[0], int(width)), dtype=node_feats.dtype)

    @staticmethod
    def make_index_attrs(*, node_attrs, node_attrs_index):
        if node_attrs_index is None:
            return jnp.argmax(node_attrs, axis=1).astype(jnp.int32)
        index_attrs = jnp.asarray(node_attrs_index, dtype=jnp.int32)
        if index_attrs.ndim != 1:
            return jnp.argmax(node_attrs, axis=1).astype(jnp.int32)
        return index_attrs.reshape(-1)

    transpose_mul_ir = staticmethod(lambda x: jnp.transpose(x, (0, 2, 1)))
    atleast_1d = staticmethod(jnp.atleast_1d)
    atleast_2d = staticmethod(jnp.atleast_2d)
    matmul = staticmethod(jnp.matmul)
    transpose = staticmethod(lambda x: x.T)
    mask_head = staticmethod(_mask_head)
    cat = staticmethod(lambda values, dim=-1: jnp.concatenate(values, axis=dim))
    stack = staticmethod(lambda values, dim=0: jnp.stack(values, axis=dim))
    sum = staticmethod(lambda value, dim: jnp.sum(value, axis=dim))
    to_numpy = staticmethod(np.asarray)
    scatter_sum = staticmethod(
        lambda src,
        index,
        dim=-1,
        dim_size=None,
        indices_are_sorted=False: jax_scatter_sum(
            src=src,
            index=index,
            dim=dim,
            dim_size=dim_size,
            indices_are_sorted=indices_are_sorted,
        )
    )

    @staticmethod
    def make_parameter(module, *, name, value, requires_grad=True):
        del requires_grad
        param = nnx.Param(jnp.asarray(value, dtype=default_dtype()))
        setattr(module, name, param)
        return param

    @staticmethod
    def make_irrep(l, p):
        return Irrep(l, p)

    @staticmethod
    def init_uniform_(value, a=-0.05, b=0.05):
        current = jnp.asarray(value.get_value())
        updated = jnp.linspace(a, b, current.size, dtype=current.dtype).reshape(
            current.shape
        )
        value[...] = updated
        return value

    tanh = staticmethod(jnp.tanh)
    silu = staticmethod(jnn.silu)
    sigmoid = staticmethod(jnn.sigmoid)
    make_zeros = staticmethod(jnp.zeros)


JAX_BACKEND = _JaxBackendSpec
