from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch

from mace_model.torch.tools.scatter import scatter_sum


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions],
        grad_outputs=[torch.ones_like(energy)],
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(positions)
    return -gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    forces, virials = torch.autograd.grad(
        outputs=[energy],
        inputs=[positions, displacement],
        grad_outputs=[torch.ones_like(energy)],
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3), dtype=positions.dtype, device=positions.device)
    return -forces, -virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3, 3, dtype=positions.dtype, device=positions.device
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum("be,bec->bc", unit_shifts, cell[batch[sender]])
    return positions, shifts, displacement


def compute_hessians_vmap(
    forces: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    eye = torch.eye(num_elements, device=forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            eye
        )[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros(
            (positions.shape[0], forces.shape[0], 3, 3),
            device=positions.device,
            dtype=positions.dtype,
        )
    return gradient


def compute_hessians_loop(
    forces: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hessian.append(
            torch.zeros_like(positions) if hess_row is None else hess_row.detach()
        )
    return torch.stack(hessian)


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
):
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=(training or compute_hessian or compute_edge_forces),
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=(training or compute_hessian or compute_edge_forces),
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    hessian = (
        compute_hessians_vmap(forces, positions)
        if compute_hessian and forces is not None
        else None
    )
    edge_forces = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy, positions=vectors, training=(training or compute_hessian)
        )
        if edge_forces is not None:
            edge_forces = -edge_forces
    return forces, virials, stress, hessian, edge_forces


def get_atomic_virials_stresses(
    edge_forces: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
    num_atoms: int,
    batch: torch.Tensor,
    cell: torch.Tensor,
):
    edge_virial = torch.einsum("zi,zj->zij", edge_forces, vectors)
    atom_virial_sender = scatter_sum(
        src=edge_virial, index=edge_index[0], dim=0, dim_size=num_atoms
    )
    atom_virial_receiver = scatter_sum(
        src=edge_virial, index=edge_index[1], dim=0, dim_size=num_atoms
    )
    atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
    atom_virial = (atom_virial + atom_virial.transpose(-1, -2)) / 2
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    atom_volume = volume[batch].view(-1, 1, 1)
    atom_stress = atom_virial / atom_volume
    atom_stress = torch.where(
        torch.abs(atom_stress) < 1e10, atom_stress, torch.zeros_like(atom_stress)
    )
    return -atom_virial, atom_stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-9,
):
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    if normalize:
        return vectors / (lengths + eps), lengths
    return vectors, lengths


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[torch.Tensor]
    lammps_natoms: Tuple[int, int] = (0, 0)


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    vectors: torch.Tensor
    lengths: torch.Tensor
    cell: torch.Tensor
    node_heads: torch.Tensor
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
) -> GraphContext:
    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    )
    if lammps_mliap:
        n_real, n_total = data["natoms"][0], data["natoms"][1]
        num_graphs = 2
        num_atoms_arange = torch.arange(n_real, device=data["node_attrs"].device)
        displacement = None
        positions = torch.zeros(
            (int(n_real), 3), dtype=data["vectors"].dtype, device=data["vectors"].device
        )
        cell = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["vectors"].dtype,
            device=data["vectors"].device,
        )
        vectors = data["vectors"].requires_grad_(True)
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        ikw = InteractionKwargs(data["lammps_class"], (n_real, n_total))
    else:
        if not torch.compiler.is_compiling():
            data["positions"].requires_grad_(True)
        positions = data["positions"]
        cell = data["cell"]
        num_atoms_arange = torch.arange(positions.shape[0], device=positions.device)
        num_graphs = int(data["ptr"].numel() - 1)
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        if compute_virials or compute_stress or compute_displacement:
            p, s, displacement = get_symmetric_displacement(
                positions=positions,
                unit_shifts=data["unit_shifts"],
                cell=cell,
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )
            data["positions"], data["shifts"] = p, s
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        ikw = InteractionKwargs(None, (0, 0))
    return GraphContext(
        is_lammps=lammps_mliap,
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )


__all__ = [
    "GraphContext",
    "InteractionKwargs",
    "compute_forces",
    "compute_forces_virials",
    "compute_hessians_loop",
    "compute_hessians_vmap",
    "get_atomic_virials_stresses",
    "get_edge_vectors_and_lengths",
    "get_outputs",
    "get_symmetric_displacement",
    "prepare_graph",
]
