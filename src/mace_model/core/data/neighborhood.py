from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from matscipy.neighbours import neighbour_list as _neighbour_list
except ImportError:  # pragma: no cover - fallback path
    _neighbour_list = None
    from ase.neighborlist import primitive_neighbor_list as _primitive_neighbor_list


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,
    true_self_interaction: bool = False,
):
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or np.allclose(cell, 0.0):
        cell = np.identity(3, dtype=float)
    else:
        cell = np.asarray(cell, dtype=float).copy()

    assert len(pbc) == 3 and all(isinstance(value, (bool, np.bool_)) for value in pbc)
    assert cell.shape == (3, 3)

    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.abs(positions)) + 1
    for axis, periodic in enumerate(pbc):
        if not periodic:
            cell[axis, :] = max_positions * 5 * cutoff * identity[axis, :]

    if _neighbour_list is not None:
        sender, receiver, unit_shifts = _neighbour_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
        )
    else:  # pragma: no cover - fallback path
        sender, receiver, unit_shifts = _primitive_neighbor_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            self_interaction=True,
            use_scaled_positions=False,
        )

    if not true_self_interaction:
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    edge_index = np.stack((sender, receiver))
    shifts = np.dot(unit_shifts, cell)
    return edge_index, shifts, unit_shifts, cell
