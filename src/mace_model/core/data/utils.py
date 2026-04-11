from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import ase.data
import numpy as np

Positions = np.ndarray
Cell = np.ndarray
Pbc = tuple[bool, bool, bool]

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    POLARIZABILITY = "polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    ELEC_TEMP = "elec_temp"

    @staticmethod
    def keydict() -> dict[str, str]:
        return {f"{member.name.lower()}_key": member.value for member in DefaultKeys}


class AtomicNumberTable:
    def __init__(self, zs: Iterable[int]):
        self.zs = list(zs)

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self) -> str:
        return f"AtomicNumberTable: {tuple(self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(int(atomic_number))


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    return AtomicNumberTable(sorted({int(z) for z in zs}))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray,
    z_table: AtomicNumberTable,
) -> np.ndarray:
    to_index = np.vectorize(z_table.z_to_index)
    return to_index(atomic_numbers)


@dataclass
class KeySpecification:
    info_keys: dict[str, str] = field(default_factory=dict)
    arrays_keys: dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[dict[str, str]] = None,
        arrays_keys: Optional[dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        info_keys = {}
        arrays_keys = {}
        for key, value in DefaultKeys.keydict().items():
            name = key[:-4]
            if name in {"forces", "charges"}:
                arrays_keys[name] = value
            else:
                info_keys[name] = value
        return instance.update(info_keys=info_keys, arrays_keys=arrays_keys)


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions
    properties: dict[str, Any]
    property_weights: dict[str, float]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    weight: float = 1.0
    config_type: str = DEFAULT_CONFIG_TYPE
    head: str = "Default"


def config_from_atoms(
    atoms,
    key_specification: KeySpecification = KeySpecification(),
    config_type_weights: Optional[dict[str, float]] = None,
    head_name: str = "Default",
) -> Configuration:
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc().tolist())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", DEFAULT_CONFIG_TYPE)
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type,
        1.0,
    )

    properties = {}
    property_weights = {}
    for name in list(key_specification.arrays_keys) + list(key_specification.info_keys):
        property_weights[name] = atoms.info.get(f"config_{name}_weight", 1.0)

    for name, atoms_key in key_specification.info_keys.items():
        properties[name] = atoms.info.get(atoms_key, None)
        if atoms_key not in atoms.info:
            property_weights[name] = 0.0

    for name, atoms_key in key_specification.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if atoms_key not in atoms.arrays:
            property_weights[name] = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        properties=properties,
        weight=weight,
        property_weights=property_weights,
        head=head_name,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )
