from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.special
from scipy.constants import pi

FIELD_CONSTANT = 1 / (5.526349406 * 1e-3)
CUBIC_MADELUNG = 2.837297


def normalization_denominator_np(
    sigmas: Sequence[float],
    max_l: int,
    normalize: str,
) -> np.ndarray:
    if normalize not in {'multipoles', 'none', 'receiver'}:
        raise ValueError("normalize must be one of 'multipoles', 'none', or 'receiver'")

    sigmas_np = np.asarray(sigmas, dtype=np.float64)
    ls = np.arange(int(max_l) + 1, dtype=np.float64)
    if normalize == 'multipoles':
        l_dep_part = (
            np.sqrt(4 * pi / (2 * ls + 1))
            * 2 ** ((2 * ls + 1) / 2)
            * scipy.special.gamma((2 * ls + 3) / 2)
        )
        return l_dep_part[None, :] * sigmas_np[:, None] ** (2 * ls[None, :] + 3)
    if normalize == 'receiver':
        l_dep_part = 2 ** ((ls + 1) / 2) * scipy.special.gamma((ls + 3) / 2)
        return l_dep_part[None, :] * sigmas_np[:, None] ** (ls[None, :] + 3)
    return np.ones((len(sigmas_np), len(ls)), dtype=np.float64)


def cl_sigma_np(l_value: int, sigma: float, normalize: str) -> float:
    return float(
        1.0
        / normalization_denominator_np([float(sigma)], int(l_value), normalize)[
            0, int(l_value)
        ]
    )


def expanded_l_indices_np(max_l: int) -> np.ndarray:
    ls = np.arange(int(max_l) + 1, dtype=np.int64)
    return np.repeat(ls, 2 * ls + 1)


def phase_factors_np(max_l: int) -> tuple[np.ndarray, np.ndarray]:
    expanded_l = expanded_l_indices_np(max_l).astype(np.float64)
    ones = np.ones_like(expanded_l)
    even_mask = expanded_l.astype(np.int64) % 2 == 0
    real_phase_factors = np.zeros_like(ones)
    imag_phase_factors = np.zeros_like(ones)
    real_phase_factors[even_mask] = np.power(
        -ones[even_mask], expanded_l[even_mask] / 2
    )
    imag_phase_factors[~even_mask] = -np.power(
        -ones[~even_mask],
        (expanded_l[~even_mask] - 1.0) / 2,
    )
    return real_phase_factors, imag_phase_factors


def output_permutation_np(max_l: int, n_radial: int) -> np.ndarray:
    indices: list[int] = []
    block = (int(max_l) + 1) ** 2
    for l_value in range(int(max_l) + 1):
        for radial_index in range(int(n_radial)):
            offset = radial_index * block
            indices.extend(range(l_value**2 + offset, (l_value + 1) ** 2 + offset))
    return np.asarray(indices, dtype=np.int64)


def external_field_matrix_np(
    l_receive: int,
    sigmas_receive: Sequence[float],
    normalize_receive: str,
) -> np.ndarray:
    projections_dim = (int(l_receive) + 1) ** 2 * len(sigmas_receive)
    matrix = np.zeros((projections_dim, 4), dtype=np.float64)

    for sigma_index, sigma in enumerate(sigmas_receive):
        matrix[sigma_index, 0] = (
            cl_sigma_np(0, float(sigma), normalize_receive)
            * np.sqrt(8 * pi)
            * scipy.special.gamma(1.5)
            * float(sigma) ** 3
        )

    if int(l_receive) >= 1:
        for sigma_index, sigma in enumerate(sigmas_receive):
            magnitude = (
                cl_sigma_np(1, float(sigma), normalize_receive)
                * np.sqrt(1.5)
                * float(sigma) ** 5
                * 2
                * pi
            )
            for m_value in range(3):
                matrix[len(sigmas_receive) + sigma_index * 3 + m_value, 1 + m_value] = (
                    magnitude
                )

    permutation_matrix = np.asarray(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=np.float64,
    )
    return matrix @ permutation_matrix


def self_interaction_constants_np(
    *,
    l_source: int,
    sigma_source: float,
    l_receive: int,
    sigmas_receive: Sequence[float],
    normalize_source: str,
    normalize_receive: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    overlap_constants = np.zeros(
        (len(sigmas_receive) * (min(int(l_receive), int(l_source)) + 1) ** 2),
        dtype=np.float64,
    )

    for l_value in range(min(int(l_source), int(l_receive)) + 1):
        for sigma_index, sigma_receive in enumerate(sigmas_receive):
            grid = np.linspace(
                0.0001, 10 * max(float(sigma_receive), sigma_source), 10000
            )
            f_total = _integral_f1(grid, l_value, float(sigma_receive)) + _integral_f2(
                grid,
                l_value,
                float(sigma_receive),
            )
            integrand = (
                np.power(grid, l_value + 2)
                * np.exp(-0.5 * np.power(grid, 2) / sigma_source**2)
                * f_total
            )
            value = np.trapezoid(integrand, x=grid)
            prefactor = FIELD_CONSTANT / (2 * l_value + 1)
            cl_source = cl_sigma_np(l_value, sigma_source, normalize_source)
            cl_receive = cl_sigma_np(l_value, float(sigma_receive), normalize_receive)
            for m_value in range(2 * l_value + 1):
                overlap_constants[
                    len(sigmas_receive) * l_value**2
                    + m_value
                    + sigma_index * (2 * l_value + 1)
                ] = cl_source * cl_receive * prefactor * value

    indices: list[int] = []
    for l_value in range(min(int(l_source), int(l_receive)) + 1):
        for _sigma_index in range(len(sigmas_receive)):
            for m_value in range(2 * l_value + 1):
                indices.append(l_value**2 + m_value)

    non_zero_terms = len(sigmas_receive) * (min(int(l_source), int(l_receive)) + 1) ** 2
    return overlap_constants, np.asarray(indices, dtype=np.int64), non_zero_terms


def _integral_f1(r_value: np.ndarray, l_value: int, sigma: float) -> np.ndarray:
    r_part = np.power(r_value, -(l_value + 1))
    gammas = scipy.special.gammainc(
        (2 * l_value + 3) / 2,
        0.5 * np.power(r_value, 2) / sigma**2,
    ) * scipy.special.gamma((2 * l_value + 3) / 2) - scipy.special.gammainc(
        l_value + 1.5,
        0,
    ) * scipy.special.gamma(l_value + 0.5)
    return 2 ** (l_value + 0.5) * sigma ** (2 * l_value + 3) * gammas * r_part


def _integral_f2(r_value: np.ndarray, l_value: int, sigma: float) -> np.ndarray:
    return (
        sigma**2
        * np.power(r_value, l_value)
        * np.exp(-0.5 * np.power(r_value, 2) / sigma**2)
    )


__all__ = [
    'CUBIC_MADELUNG',
    'FIELD_CONSTANT',
    'cl_sigma_np',
    'expanded_l_indices_np',
    'external_field_matrix_np',
    'normalization_denominator_np',
    'output_permutation_np',
    'phase_factors_np',
    'self_interaction_constants_np',
]
