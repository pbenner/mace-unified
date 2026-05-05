"""Shared PolarMACE setup helpers for backend model wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

DEFAULT_FIELD_READOUT_TYPE = 'OneBodyMLPFieldReadout'
DEFAULT_FIXEDPOINT_UPDATE_TYPE = 'AgnosticEmbeddedOneBodyVariableUpdate'


class PolarMACEConstructorArgs(NamedTuple):
    """Required constructor values normalized before backend init."""

    hidden_irreps: Any
    mlp_irreps: Any
    gate: Any
    avg_num_neighbors: float
    num_interactions: int
    num_elements: int


class PolarIrrepsLayout(NamedTuple):
    """Irreps used by PolarMACE long-range field modules."""

    charges_irreps: Any
    field_irreps: Any
    potential_irreps: Any
    node_attr_irreps: Any
    edge_feats_irreps: Any
    field_update_sh_irreps: Any
    from_ell_max_field_update: int
    field_interaction_irreps: Any


def layout_from_cueq_config(cueq_config: Any) -> str:
    """Return the data layout requested by cuequivariance configuration."""
    if cueq_config is not None and getattr(cueq_config, 'enabled', False):
        return str(getattr(cueq_config, 'layout_str', 'mul_ir'))
    return 'mul_ir'


def coerce_mlp_irreps(value: Any, make_irreps: Any) -> Any:
    """Coerce the MLP irreps option, preserving legacy integer shorthand."""
    if isinstance(value, int):
        return make_irreps(f'{int(value)}x0e')
    return make_irreps(str(value))


def resolve_named_type(
    value: Any,
    registry: Mapping[str, Any],
    option_name: str,
) -> Any:
    """Resolve string config values against a backend-specific class registry."""
    if not isinstance(value, str):
        return value
    try:
        return registry[value]
    except KeyError as exc:
        known = ', '.join(sorted(registry))
        raise ValueError(
            f'Unknown {option_name} {value!r}; expected one of {known}.'
        ) from exc


def normalize_field_feature_widths(
    field_feature_widths: Sequence[float],
) -> list[float]:
    widths = [float(width) for width in field_feature_widths]
    if not widths:
        raise ValueError('field_feature_widths must contain at least one width.')
    return widths


def normalize_field_feature_norms(
    field_feature_norms: Sequence[float] | None,
) -> list[float] | None:
    if field_feature_norms is None:
        return None
    return [float(value) for value in field_feature_norms]


def expand_field_feature_norms(
    *,
    field_feature_norms: Sequence[float] | None,
    field_feature_widths: Sequence[float],
    field_feature_max_l: int,
) -> list[float]:
    """Expand per-width/per-ell norms to the flattened irreps layout."""
    widths = normalize_field_feature_widths(field_feature_widths)
    norms = normalize_field_feature_norms(field_feature_norms)
    max_l = int(field_feature_max_l)
    expected_norms = len(widths) * (max_l + 1)
    if norms is None:
        norms = [1.0] * expected_norms
    elif len(norms) != expected_norms:
        raise ValueError(
            'field_feature_norms must contain '
            f'{expected_norms} values, got {len(norms)}.'
        )

    expanded_norms: list[float] = []
    for ell in range(max_l + 1):
        for width_index in range(len(widths)):
            norm = norms[ell * len(widths) + width_index]
            expanded_norms.extend([norm] * (2 * ell + 1))
    return expanded_norms


def estimate_gto_basis_kspace_cutoff(
    sigmas: Sequence[float],
    max_l: int,
) -> float:
    """Heuristic k-space cutoff used by the GTO electrostatic descriptors."""
    widths = normalize_field_feature_widths(sigmas)
    return 0.75 * (1.0 / min(widths)) * (int(max_l) + 1) ** 0.3 * 3.0


class PolarMACEModel:
    """Backend-independent PolarMACE constructor and layout helpers."""

    @staticmethod
    def prepare_polar_base_kwargs(
        kwargs: Mapping[str, Any],
        *,
        readout_cls: type[Any],
    ) -> dict[str, Any]:
        """Normalize kwargs passed through to the regular MACE backbone."""
        prepared = dict(kwargs)
        heads = prepared.get('heads', ['Default']) or ['Default']
        prepared['heads'] = heads
        prepared.setdefault('readout_cls', readout_cls)
        prepared.setdefault('use_agnostic_product', True)
        prepared.setdefault('apply_cutoff', True)
        prepared.setdefault('atomic_inter_scale', [1.0] * len(heads))
        prepared.setdefault('atomic_inter_shift', [0.0] * len(heads))
        prepared.pop('field_dependence_type', None)
        prepared.pop('final_field_readout_type', None)
        prepared['keep_last_layer_irreps'] = True
        return prepared

    @staticmethod
    def require_polar_mace_kwargs(
        kwargs: Mapping[str, Any],
        *,
        make_irreps: Any,
    ) -> PolarMACEConstructorArgs:
        """Fetch constructor values needed before the base MACE init runs."""
        try:
            return PolarMACEConstructorArgs(
                hidden_irreps=make_irreps(kwargs['hidden_irreps']),
                mlp_irreps=coerce_mlp_irreps(kwargs['MLP_irreps'], make_irreps),
                gate=kwargs['gate'],
                avg_num_neighbors=float(kwargs['avg_num_neighbors']),
                num_interactions=int(kwargs['num_interactions']),
                num_elements=int(kwargs['num_elements']),
            )
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise KeyError(
                f"Missing required argument '{missing}' in kwargs for PolarMACE. "
                'Pass all ScaleShiftMACE/MACE constructor args as keyword arguments.'
            ) from exc

    def initialize_polar_common_attributes(
        self,
        *,
        kspace_cutoff_factor: float,
        atomic_multipoles_max_l: int,
        atomic_multipoles_smearing_width: float,
        field_feature_max_l: int,
        field_feature_widths: Sequence[float],
        num_recursion_steps: int,
        field_si: bool,
        include_electrostatic_self_interaction: bool,
        add_local_electron_energy: bool,
        quadrupole_feature_corrections: bool,
        return_electrostatic_potentials: bool,
        field_feature_norms: Sequence[float] | None,
        field_norm_factor: float | None,
        fixedpoint_update_config: Mapping[str, Any] | None,
        field_readout_config: Mapping[str, Any] | None,
    ) -> None:
        """Store backend-independent PolarMACE constructor attributes."""
        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.atomic_multipoles_max_l = int(atomic_multipoles_max_l)
        self.atomic_multipoles_smearing_width = float(atomic_multipoles_smearing_width)
        self.field_feature_max_l = int(field_feature_max_l)
        self.field_feature_widths = normalize_field_feature_widths(field_feature_widths)
        self.num_recursion_steps = int(num_recursion_steps)
        self.field_si = bool(field_si)
        self.include_electrostatic_self_interaction = bool(
            include_electrostatic_self_interaction
        )
        self.add_local_electron_energy = bool(add_local_electron_energy)
        self.quadrupole_feature_corrections = bool(quadrupole_feature_corrections)
        self.return_electrostatic_potentials = bool(return_electrostatic_potentials)
        self._field_feature_norms = normalize_field_feature_norms(field_feature_norms)
        self.field_norm_factor = (
            1.0 if field_norm_factor is None else float(field_norm_factor)
        )
        self._fixedpoint_update_config = dict(fixedpoint_update_config or {})
        self._field_readout_config = dict(field_readout_config or {})
        self.keep_last_layer_irreps = True

    @staticmethod
    def make_polar_irreps_layout(
        *,
        make_irreps: Any,
        make_irrep: Any,
        hidden_irreps: Any,
        atomic_multipoles_max_l: int,
        field_feature_max_l: int,
        field_feature_widths: Sequence[float],
        num_elements: int,
        radial_embedding_out_dim: int,
        max_ell_field_update: int = 2,
    ) -> PolarIrrepsLayout:
        """Build the long-range field irreps shared by backend modules."""
        feature_widths = normalize_field_feature_widths(field_feature_widths)
        charges_irreps = 2 * make_irreps.spherical_harmonics(
            int(atomic_multipoles_max_l)
        )
        lr_sh_irreps = make_irreps.spherical_harmonics(int(field_feature_max_l))
        field_irreps = (lr_sh_irreps * len(feature_widths)).sort()[0].simplify()
        potential_irreps = field_irreps * 2
        node_attr_irreps = make_irreps([(int(num_elements), (0, 1))])
        edge_feats_irreps = make_irreps(f'{int(radial_embedding_out_dim)}x0e')
        field_update_sh_irreps = make_irreps.spherical_harmonics(
            int(max_ell_field_update)
        )
        from_ell_max_field_update = (int(max_ell_field_update) + 1) ** 2
        num_features = hidden_irreps.count(make_irrep(0, 1))
        field_interaction_irreps = (
            (field_update_sh_irreps * num_features).sort()[0].simplify()
        )
        return PolarIrrepsLayout(
            charges_irreps=charges_irreps,
            field_irreps=field_irreps,
            potential_irreps=potential_irreps,
            node_attr_irreps=node_attr_irreps,
            edge_feats_irreps=edge_feats_irreps,
            field_update_sh_irreps=field_update_sh_irreps,
            from_ell_max_field_update=from_ell_max_field_update,
            field_interaction_irreps=field_interaction_irreps,
        )

    @staticmethod
    def resolve_field_update_config(
        config: Mapping[str, Any] | None,
        *,
        field_update_registry: Mapping[str, Any],
        potential_embedding_registry: Mapping[str, Any],
        nonlinearity_registry: Mapping[str, Any],
        default_potential_embedding_cls: type[Any],
    ) -> tuple[type[Any], dict[str, Any]]:
        """Resolve backend field-update classes from a serialized config."""
        update_config = dict(config or {})
        field_update_cls = resolve_named_type(
            update_config.pop('type', DEFAULT_FIXEDPOINT_UPDATE_TYPE),
            field_update_registry,
            'fixedpoint_update_config.type',
        )
        update_config['potential_embedding_cls'] = resolve_named_type(
            update_config.get(
                'potential_embedding_cls',
                default_potential_embedding_cls,
            ),
            potential_embedding_registry,
            'fixedpoint_update_config.potential_embedding_cls',
        )
        if 'nonlinearity_cls' in update_config:
            update_config['nonlinearity_cls'] = resolve_named_type(
                update_config['nonlinearity_cls'],
                nonlinearity_registry,
                'fixedpoint_update_config.nonlinearity_cls',
            )
        return field_update_cls, update_config

    @staticmethod
    def resolve_field_readout_config(
        config: Mapping[str, Any] | None,
        *,
        field_readout_registry: Mapping[str, Any],
    ) -> tuple[type[Any], dict[str, Any]]:
        """Resolve backend field-readout classes from a serialized config."""
        readout_config = dict(config or {})
        field_readout_cls = resolve_named_type(
            readout_config.pop('type', DEFAULT_FIELD_READOUT_TYPE),
            field_readout_registry,
            'field_readout_config.type',
        )
        return field_readout_cls, readout_config


__all__ = [
    'DEFAULT_FIELD_READOUT_TYPE',
    'DEFAULT_FIXEDPOINT_UPDATE_TYPE',
    'PolarIrrepsLayout',
    'PolarMACEConstructorArgs',
    'PolarMACEModel',
    'coerce_mlp_irreps',
    'estimate_gto_basis_kspace_cutoff',
    'expand_field_feature_norms',
    'layout_from_cueq_config',
    'normalize_field_feature_norms',
    'normalize_field_feature_widths',
    'resolve_named_type',
]
