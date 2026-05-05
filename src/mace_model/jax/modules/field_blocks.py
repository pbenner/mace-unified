from __future__ import annotations

import math
from typing import Any

import cuequivariance as cue
import jax
import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

from mace_model.core.modules.irreps_utils import (
    tp_out_irreps_with_instructions as _core_tp_out_irreps_with_instructions,
)
from mace_model.core.modules.polar import layout_from_cueq_config
from mace_model.jax.adapters.cuequivariance import CuEquivarianceConfig, Linear
from mace_model.jax.adapters.e3nn import Irrep, Irreps, nn
from mace_model.jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_model.jax.tools.dtype import default_dtype

from .backends import JAX_BACKEND
from .radial import RadialMLP


def _tp_out_irreps_with_instructions(
    irreps1: Irreps,
    irreps2: Irreps,
    target_irreps: Irreps,
):
    return _core_tp_out_irreps_with_instructions(
        make_irreps=Irreps,
        irreps1=irreps1,
        irreps2=irreps2,
        target_irreps=target_irreps,
    )


def _scalar_bias_slices(irreps: Irreps) -> tuple[list[tuple[int, int, int, int]], int]:
    slices = []
    offset = 0
    bias_offset = 0
    for mul, ir in Irreps(irreps):
        size = int(mul) * int(ir.dim)
        if int(ir.l) == 0 and int(ir.p) == 1:
            slices.append((offset, offset + size, bias_offset, bias_offset + int(mul)))
            bias_offset += int(mul)
        offset += size
    return slices, bias_offset


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class _BiasedLinear(nnx.Module):
    """Cue-backed linear layer with e3nn-style scalar biases."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.linear = Linear(
            self.irreps_in,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
            rngs=rngs,
        )
        self._bias_slices, bias_dim = _scalar_bias_slices(self.irreps_out)
        self.bias = nnx.Param(jnp.zeros((bias_dim,), dtype=default_dtype()))

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        out = self.linear(value)
        if not self._bias_slices:
            return out

        bias = jnp.asarray(self.bias, dtype=out.dtype)
        for out_start, out_stop, bias_start, bias_stop in self._bias_slices:
            out = out.at[..., out_start:out_stop].add(bias[bias_start:bias_stop])
        return out


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MultiLayerFeatureMixer(nnx.Module):
    def __init__(
        self,
        node_feats_irreps: Irreps,
        num_interactions: int,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.linears = nnx.List(
            [
                Linear(
                    node_feats_irreps,
                    node_feats_irreps,
                    cueq_config=cueq_config,
                    rngs=rngs,
                )
                for _ in range(int(num_interactions))
            ]
        )

    def __call__(self, all_node_feats: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(all_node_feats[0])
        for layer_index, linear in enumerate(self.linears):
            out = out + linear(all_node_feats[layer_index])
        return out


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EnvironmentDependentSpinSourceBlock(nnx.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        max_l: int,
        zero_charges: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.zero_charges = bool(zero_charges)
        self.irreps_out = 2 * Irreps.spherical_harmonics(int(max_l))
        self.linear = Linear(
            irreps_in,
            self.irreps_out,
            cueq_config=cueq_config,
            rngs=rngs,
        )

    def __call__(self, node_feats: jnp.ndarray) -> jnp.ndarray:
        multipoles = self.linear(node_feats)
        if self.zero_charges:
            multipoles = multipoles.at[:, 0].set(0.0)
        return jnp.expand_dims(multipoles, axis=-2)


class PotentialEmbeddingBlock(nnx.Module):
    def __init__(
        self,
        potential_irreps: Irreps,
        node_feats_irreps: Irreps,
        node_attrs_irreps: Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        self.potential_irreps = Irreps(potential_irreps)
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.node_attrs_irreps = Irreps(node_attrs_irreps)
        self.cueq_config = cueq_config
        self._setup(rngs=rngs, **kwargs)

    def _setup(self, *, rngs: nnx.Rngs, **kwargs) -> None:
        del rngs, kwargs
        raise NotImplementedError

    def __call__(
        self,
        potential_feats: jnp.ndarray,
        node_feats: jnp.ndarray,
        node_attrs: jnp.ndarray,
        *args,
    ) -> jnp.ndarray:
        raise NotImplementedError


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticChargeBiasedLinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self, charges_irreps: Irreps, *, rngs: nnx.Rngs) -> None:
        self.potential_linear = Linear(
            self.potential_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            rngs=rngs,
        )
        self.node_feats_linear = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            rngs=rngs,
        )
        self.charges_irreps = Irreps(charges_irreps)
        self.charge_embedding = Linear(
            self.charges_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            rngs=rngs,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._potential_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.potential_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._node_feats_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._charges_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._node_feats_from_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.node_feats_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def __call__(
        self,
        potential_feats: jnp.ndarray,
        node_feats: jnp.ndarray,
        node_attrs: jnp.ndarray,
        local_charges: jnp.ndarray,
    ) -> jnp.ndarray:
        del node_attrs
        potential_in = self._potential_to_mul_ir(potential_feats)
        node_feats_in = self._node_feats_to_mul_ir(node_feats)
        charges_in = self._charges_to_mul_ir(local_charges)

        potential_emb = self.potential_linear(potential_in)
        node_feats_emb = self.node_feats_linear(node_feats_in)
        charges_emb = self.charge_embedding(charges_in)
        return self._node_feats_from_mul_ir(
            potential_emb + node_feats_emb + charges_emb
        )


class NoNonLinearity(nnx.Module):
    def __init__(self, invar_irreps: Irreps) -> None:
        self.irreps = Irreps(invar_irreps)

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        return value


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MLPNonLinearity(nnx.Module):
    def __init__(self, invar_irreps: Irreps, *, rngs: nnx.Rngs) -> None:
        channels = Irreps(invar_irreps).count(Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels],
            jnn.silu,
            rngs=rngs,
        )

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(value)


class FieldUpdateBlock(nnx.Module):
    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        potential_irreps: Irreps,
        charges_irreps: Irreps,
        field_norm_factor: float,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: Any | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        del oeq_config
        self.node_attrs_irreps = Irreps(node_attrs_irreps)
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.edge_attrs_irreps = Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = Irreps(edge_feats_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.hidden_irreps = Irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.potential_irreps = Irreps(potential_irreps)
        self.charges_irreps = Irreps(charges_irreps)
        self.radial_MLP = radial_MLP
        self.field_norm_factor = jnp.asarray(
            float(field_norm_factor),
            dtype=default_dtype(),
        )
        self.cueq_config = cueq_config
        self._setup(rngs=rngs, **kwargs)

    def _setup(self, *, rngs: nnx.Rngs, **kwargs) -> None:
        del rngs, kwargs
        raise NotImplementedError

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        potential_features: jnp.ndarray,
        local_charges: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError


def instructions_for_sparse_tp(
    feat_in1: Irreps,
    feat_in2: Irreps,
    feat_out: Irreps,
):
    channels1 = Irreps(feat_in1).count(Irrep(0, 1))
    channels2 = Irreps(feat_in2).count(Irrep(0, 1))
    channels3 = Irreps(feat_out).count(Irrep(0, 1))
    if channels1 != channels2 or channels1 != channels3:
        raise ValueError('Sparse tensor product scalar channels must match.')
    _, instructions = _tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    return [(i, j, 0, mode, trainable) for i, j, _k, mode, trainable in instructions]


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SparseUvuTensorProduct(nnx.Module):
    """JAX sparse `uvu` tensor product backed by cue Clebsch-Gordan data."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: list[tuple[int, int, int, str, bool]],
        layout: str = 'mul_ir',
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        if layout not in {'mul_ir', 'ir_mul'}:
            raise ValueError(f'Unsupported sparse TP layout {layout!r}.')
        self.layout = layout

        normalized_instructions = [
            tuple(instruction) if len(instruction) == 6 else (*tuple(instruction), 1.0)
            for instruction in instructions
        ]
        self.instructions = normalized_instructions
        self.shared_weights = True
        self.internal_weights = True

        in1_slices = self.irreps_in1.slices()
        in2_slices = self.irreps_in2.slices()
        out_slices = self.irreps_out.slices()
        path_meta = []
        cgs = []

        def num_elements(instruction) -> int:
            _i_in1, i_in2, _i_out, mode, _has_weight, _path_weight = instruction
            if mode != 'uvu':
                raise NotImplementedError(
                    "SparseUvuTensorProduct only supports 'uvu' connection mode."
                )
            mul2, _ir2 = self.irreps_in2[int(i_in2)]
            return int(mul2)

        normalization_coefficients = []
        for instruction in normalized_instructions:
            i_in1, i_in2, i_out, mode, _has_weight, path_weight = instruction
            if mode != 'uvu':
                raise NotImplementedError(
                    "SparseUvuTensorProduct only supports 'uvu' connection mode."
                )
            _mul1, ir1 = self.irreps_in1[int(i_in1)]
            _mul2, ir2 = self.irreps_in2[int(i_in2)]
            _mul_out, ir_out = self.irreps_out[int(i_out)]
            if int(ir1.p) * int(ir2.p) != int(ir_out.p):
                raise ValueError('Sparse tensor-product path has incompatible parity.')
            x = sum(
                num_elements(other)
                for other in normalized_instructions
                if int(other[2]) == int(i_out)
            )
            alpha = float(ir_out.dim)
            if x > 0:
                alpha /= float(x)
            alpha *= float(path_weight)
            normalization_coefficients.append(math.sqrt(alpha))

        output_mask_fields = []
        for out_index, (mul, ir) in enumerate(self.irreps_out):
            active = any(
                int(instruction[2]) == out_index
                and normalization_coefficients[instruction_index] != 0
                for instruction_index, instruction in enumerate(normalized_instructions)
            )
            value = 1.0 if active else 0.0
            output_mask_fields.append(
                jnp.full(
                    (int(mul) * int(ir.dim),),
                    value,
                    dtype=default_dtype(),
                )
            )
        self.output_mask = (
            jnp.concatenate(output_mask_fields)
            if output_mask_fields
            else jnp.ones((0,), dtype=default_dtype())
        )

        weight_offset = 0
        for path_index, (instruction, path_weight) in enumerate(
            zip(normalized_instructions, normalization_coefficients)
        ):
            i1, i2, io, _mode, _has_weight, _raw_path_weight = instruction
            i1 = int(i1)
            i2 = int(i2)
            io = int(io)
            mul1, ir1 = self.irreps_in1[i1]
            mul2, ir2 = self.irreps_in2[i2]
            mul_out, ir_out = self.irreps_out[io]
            dim1 = int(ir1.dim)
            dim2 = int(ir2.dim)
            dim_out = int(ir_out.dim)
            if int(mul_out) != int(mul1):
                raise NotImplementedError(
                    'SparseUvuTensorProduct requires output multiplicity to match '
                    'the first input multiplicity.'
                )
            cgs.append(
                tuple(
                    jnp.asarray(
                        cue.O3.clebsch_gordan(ir1, ir2, ir_out)[0],
                        dtype=default_dtype(),
                    ).tolist()
                )
            )

            weight_size = int(mul1) * int(mul2)
            path_meta.append(
                (
                    in1_slices[i1].start,
                    in1_slices[i1].stop,
                    in2_slices[i2].start,
                    in2_slices[i2].stop,
                    out_slices[io].start,
                    out_slices[io].stop,
                    weight_offset,
                    weight_offset + weight_size,
                    float(path_weight),
                    int(mul1),
                    int(mul2),
                    dim1,
                    dim2,
                    dim_out,
                    path_index,
                )
            )
            weight_offset += weight_size
        self._path_meta = tuple(path_meta)
        self._cgs = tuple(cgs)
        self.weight_numel = weight_offset
        self.weight = nnx.Param(
            jax.random.normal(rngs(), (self.weight_numel,), dtype=default_dtype())
        )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                'SparseUvuTensorProduct expects flattened [batch, irreps.dim] tensors.'
            )

        batch = x1.shape[0]
        out = jnp.zeros((batch, self.irreps_out.dim), dtype=x1.dtype)
        weight_values = jnp.asarray(self.weight, dtype=x1.dtype)

        def to_mul_ir(block: jnp.ndarray, mul: int, dim: int) -> jnp.ndarray:
            if self.layout == 'mul_ir':
                return block.reshape(batch, mul, dim)
            return jnp.swapaxes(block.reshape(batch, dim, mul), 1, 2)

        def from_mul_ir(value: jnp.ndarray) -> jnp.ndarray:
            if self.layout == 'mul_ir':
                return value.reshape(batch, -1)
            return jnp.swapaxes(value, 1, 2).reshape(batch, -1)

        for (
            in1_start,
            in1_stop,
            in2_start,
            in2_stop,
            out_start,
            out_stop,
            weight_start,
            weight_stop,
            path_weight,
            mul1,
            mul2,
            dim1,
            dim2,
            dim_out,
            cg_index,
        ) in self._path_meta:
            in1_block = x1[:, in1_start:in1_stop]
            in2_block = x2[:, in2_start:in2_stop]
            out_block = out[:, out_start:out_stop]
            weight = weight_values[weight_start:weight_stop].reshape(mul1, mul2)
            cg = jnp.asarray(self._cgs[cg_index], dtype=x1.dtype)
            x1_view = to_mul_ir(in1_block, mul1, dim1)
            x2_view = to_mul_ir(in2_block, mul2, dim2)
            coupled = jnp.einsum('bua,bvd,adc->buvc', x1_view, x2_view, cg)
            contribution = path_weight * jnp.einsum('buvc,uv->buc', coupled, weight)
            out_block_view = to_mul_ir(out_block, mul1, dim_out)
            out = out.at[:, out_start:out_stop].set(
                from_mul_ir(out_block_view + contribution)
            )

        return out * jnp.asarray(self.output_mask, dtype=out.dtype)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class GeneralNonLinearBiasReadoutBlock(nnx.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Any,
        irrep_out: Irreps = Irreps('0e'),
        irreps_out: Irreps | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: Any | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        del oeq_config
        self.hidden_irreps = Irreps(MLP_irreps)
        self.irreps_out = Irreps(irrep_out if irreps_out is None else irreps_out)
        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if int(ir.l) == 0 and ir in self.irreps_out
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if int(ir.l) > 0 and ir in self.irreps_out
            ]
        )
        irreps_gates = Irreps([(mul, (0, 1)) for mul, _ in irreps_gated])
        activation_fn = gate if gate is not None else jnn.silu
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[activation_fn for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[jnn.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=cueq_config,
            rngs=rngs,
        )
        self.linear_mid = _BiasedLinear(
            self.hidden_irreps,
            self.irreps_nonlin,
            rngs=rngs,
        )
        self.linear_2 = _BiasedLinear(
            self.hidden_irreps,
            self.irreps_out,
            rngs=rngs,
        )
        layout_str = layout_from_cueq_config(cueq_config)
        self._tp_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.irreps_nonlin,
            source=layout_str,
            target='mul_ir',
            cueq_config=cueq_config,
        )
        self._tp_from_mul_ir_out = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.irreps_out,
            source='mul_ir',
            target=layout_str,
            cueq_config=cueq_config,
        )

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        value = self.linear_1(value)
        value = self._tp_to_mul_ir(value)
        value = self.equivariant_nonlin(value)
        value = self.linear_mid(value)
        value = self.equivariant_nonlin(value)
        value = self.linear_2(value)
        return self._tp_from_mul_ir_out(value)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: type[
            PotentialEmbeddingBlock
        ] = AgnosticChargeBiasedLinearPotentialEmbedding,
        nonlinearity_cls: type[nnx.Module] = NoNonLinearity,
        num_elements: int | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        del kwargs, nonlinearity_cls, num_elements
        invar_irreps = Irreps(f'{self.node_feats_irreps.count(Irrep(0, 1))}x0e')
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.source_embedding = Linear(
            self.node_attrs_irreps,
            invar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        dot_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
            rngs=rngs,
        )
        self.nonlinearity = RadialMLP(
            [2 * invar_irreps.dim] + [64, 64, 64] + [invar_irreps.dim],
            rngs=rngs,
        )
        _, tp_instructions = _tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=tp_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
            rngs=rngs,
        )
        mlp_irreps = (
            (32 * Irreps.spherical_harmonics(self.charges_irreps.lmax))
            .sort()[0]
            .simplify()
        )
        self.readout = GeneralNonLinearBiasReadoutBlock(
            irreps_in=self.node_feats_irreps,
            MLP_irreps=mlp_irreps,
            gate=jnn.silu,
            irreps_out=self.charges_irreps + Irreps('2x0e'),
            cueq_config=None,
            rngs=rngs,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._readout_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._readout_from_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.charges_irreps + Irreps('2x0e'),
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        potential_features: jnp.ndarray,
        local_charges: jnp.ndarray,
    ) -> jnp.ndarray:
        del edge_attrs, edge_feats, edge_index
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs,
            local_charges,
        )
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        source_embedding = self.source_embedding(node_attrs)
        invariant_descriptors_embedded = jnp.concatenate(
            [invariant_descriptors, source_embedding],
            axis=-1,
        )
        nonlin_feats = self.nonlinearity(invariant_descriptors_embedded)
        new_feats = self.tp_out(node_feats, nonlin_feats)
        multipoles = self.readout(self._readout_to_mul_ir(new_feats))
        return self._readout_from_mul_ir(multipoles)


class PostScfReadout(nnx.Module):
    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        potential_irreps: Irreps,
        charges_irreps: Irreps,
        radial_MLP: list[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        oeq_config: Any | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        del oeq_config
        self.node_attrs_irreps = Irreps(node_attrs_irreps)
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.edge_attrs_irreps = Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = Irreps(edge_feats_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.hidden_irreps = Irreps(hidden_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.potential_irreps = Irreps(potential_irreps)
        self.charges_irreps = Irreps(charges_irreps)
        self.cueq_config = cueq_config
        self._setup(rngs=rngs, **kwargs)

    def _setup(self, *, rngs: nnx.Rngs, **kwargs) -> None:
        del rngs, kwargs
        raise NotImplementedError

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        field_feats: jnp.ndarray,
        charges_0: jnp.ndarray,
        charges_induced: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class OneBodyMLPFieldReadout(PostScfReadout):
    def _setup(self, *, rngs: nnx.Rngs, **kwargs) -> None:
        del kwargs
        invar_irreps = Irreps(f'{self.node_feats_irreps.count(Irrep(0, 1))}x0e')
        self.linear_up_q = _BiasedLinear(
            self.charges_irreps,
            self.node_feats_irreps,
            rngs=rngs,
        )
        self.linear_up_v = _BiasedLinear(
            self.potential_irreps,
            self.node_feats_irreps,
            rngs=rngs,
        )
        layout_str = layout_from_cueq_config(self.cueq_config)
        self._q_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.charges_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._v_to_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.potential_irreps,
            source=layout_str,
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self._up_from_mul_ir = JAX_BACKEND.make_transpose_irreps_layout(
            irreps=self.node_feats_irreps,
            source='mul_ir',
            target=layout_str,
            cueq_config=self.cueq_config,
        )
        dot_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_q = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
            rngs=rngs,
        )
        self.dot_products_v = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=dot_instructions,
            layout=layout_from_cueq_config(self.cueq_config),
            rngs=rngs,
        )
        self.mlp = RadialMLP([2 * invar_irreps.dim] + [128, 128, 128] + [1], rngs=rngs)

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        field_feats: jnp.ndarray,
        charges_0: jnp.ndarray,
        charges_induced: jnp.ndarray,
    ) -> jnp.ndarray:
        del node_attrs, edge_attrs, edge_feats, edge_index
        q_in = self._q_to_mul_ir(charges_induced + charges_0)
        q_up = self._up_from_mul_ir(self.linear_up_q(q_in))
        v_in = self._v_to_mul_ir(field_feats)
        v_up = self._up_from_mul_ir(self.linear_up_v(v_in))
        invar_feats = jnp.concatenate(
            [
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ],
            axis=-1,
        )
        return jnp.squeeze(self.mlp(invar_feats), axis=-1)


field_update_blocks = {
    'AgnosticEmbeddedOneBodyVariableUpdate': AgnosticEmbeddedOneBodyVariableUpdate,
}

field_readout_blocks = {
    'OneBodyMLPFieldReadout': OneBodyMLPFieldReadout,
}


__all__ = [
    'AgnosticChargeBiasedLinearPotentialEmbedding',
    'AgnosticEmbeddedOneBodyVariableUpdate',
    'EnvironmentDependentSpinSourceBlock',
    'FieldUpdateBlock',
    'GeneralNonLinearBiasReadoutBlock',
    'MLPNonLinearity',
    'MultiLayerFeatureMixer',
    'NoNonLinearity',
    'OneBodyMLPFieldReadout',
    'PostScfReadout',
    'PotentialEmbeddingBlock',
    'SparseUvuTensorProduct',
    'field_readout_blocks',
    'field_update_blocks',
    'instructions_for_sparse_tp',
]
