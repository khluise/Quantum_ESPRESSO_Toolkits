"""Structure handling and QE input generation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import spglib
from ase import Atoms
from ase.data import atomic_masses
from ase.io import read


@dataclass
class AdvancedSettings:
    calculation: str = "scf"
    conv_thr: Optional[float] = None
    mixing_beta: Optional[float] = None
    electron_maxstep: Optional[int] = None
    diagonalization: Optional[str] = None
    mixing_mode: Optional[str] = None
    occupations: Optional[str] = None
    smearing: Optional[str] = None
    degauss: Optional[float] = None
    nspin_override: Optional[int] = None
    k_mode: str = "automatic"
    k_points: Tuple[int, int, int] = (6, 6, 6)
    k_shift: Tuple[int, int, int] = (0, 0, 0)
    starting_magnetization: Dict[str, float] = field(default_factory=dict)
    nosym: bool = False
    noinv: bool = False
    tstress: bool = False
    tprnfor: bool = False
    wf_collect: bool = False


@dataclass
class MagnetismPreset:
    key: str
    label: str
    occupations: str
    smearing: Optional[str]
    degauss: Optional[float]
    nspin: int
    default_magnetization: float = 0.0


MAGNETISM_PRESETS: List[MagnetismPreset] = [
    MagnetismPreset("non_magnetic", "Non-magnetic insulator", "fixed", None, None, 1, 0.0),
    MagnetismPreset("metal", "Metal (cold smearing 0.02)", "smearing", "cold", 0.02, 1, 0.0),
    MagnetismPreset("ferromagnetic", "Ferromagnetic (nspin=2)", "smearing", "cold", 0.02, 2, 0.5),
    MagnetismPreset("antiferromagnetic", "Antiferromagnetic template", "smearing", "cold", 0.02, 2, 0.5),
]


PROTOCOL_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sssp_precision": {"ecutwfc": 50.0, "ecutrho_factor": 8.0},
    "sssp_efficiency": {"ecutwfc": 40.0, "ecutrho_factor": 8.0},
    "pseudo_dojo_pbe": {"ecutwfc": 55.0, "ecutrho_factor": 10.0},
    "pseudo_dojo_lda": {"ecutwfc": 50.0, "ecutrho_factor": 10.0},
}


XC_FUNCTIONALS: Dict[str, str] = {
    "PBE": "pbe",
    "PBEsol": "pbesol",
    "LDA (PZ)": "lda",
    "SCAN": "scan",
}


class PseudoLibrary:
    """Minimal pseudopotential mapper with protocol/XC awareness."""

    def __init__(self) -> None:
        self.cache_dir = Path.home() / ".qe_input_gui" / "pseudos"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve(
        self,
        protocol: str,
        xc_tag: str,
        elements: Sequence[str],
    ) -> Tuple[Dict[str, str], float, float]:
        defaults = PROTOCOL_DEFAULTS.get(protocol, {"ecutwfc": 50.0, "ecutrho_factor": 8.0})
        ecutwfc = defaults["ecutwfc"]
        ecutrho = defaults["ecutwfc"] * defaults.get("ecutrho_factor", 8.0)
        suffix = f"{protocol}-{xc_tag}".replace(" ", "_")
        mapping: Dict[str, str] = {}
        for element in dict.fromkeys(elements):
            mapping[element] = f"{element}_{suffix}.UPF"
        return mapping, float(ecutwfc), float(ecutrho)


def refine_structure_with_spglib(structure: Atoms, symprec: float = 0.01) -> Atoms:
    cell = (structure.get_cell(), structure.get_scaled_positions(), structure.get_atomic_numbers())
    primitive = spglib.find_primitive(cell, symprec=symprec)
    if primitive is None:
        return structure
    lattice, scaled_positions, atomic_numbers = primitive
    refined = Atoms(numbers=atomic_numbers, cell=lattice, pbc=True)
    refined.set_scaled_positions(scaled_positions)
    return refined


def kpoints_from_kdist(cell: np.ndarray, k_dist: float = 0.15) -> Tuple[int, int, int]:
    lengths = np.linalg.norm(cell, axis=1)
    grid = []
    for length in lengths:
        if length <= 0:
            grid.append(1)
            continue
        nk = max(1, int(math.ceil(1.0 / (k_dist * length))))
        grid.append(nk)
    return tuple(grid)


def build_qe_input(
    structure: Atoms,
    protocol: str,
    xc_label: str,
    magnetism: MagnetismPreset,
    pseudo_map: Dict[str, str],
    ecutwfc: float,
    ecutrho: float,
    k_points: Tuple[int, int, int],
    k_shift: Tuple[int, int, int],
    settings: AdvancedSettings,
) -> str:
    natoms = len(structure)
    species = structure.get_chemical_symbols()
    unique_species = list(dict.fromkeys(species))

    control_lines = ["&CONTROL"]
    control_lines.append(f"    calculation='{settings.calculation}'")
    control_lines.append("    prefix='calc'")
    control_lines.append("    pseudo_dir='./'")
    control_lines.append("    outdir='./out'")
    if settings.wf_collect:
        control_lines.append("    wf_collect=.true.")
    control_lines.append("/")

    system_lines = ["&SYSTEM", "    ibrav=0"]
    system_lines.append(f"    nat={natoms}")
    system_lines.append(f"    ntyp={len(unique_species)}")
    system_lines.append(f"    ecutwfc={ecutwfc}")
    system_lines.append(f"    ecutrho={ecutrho}")

    occ_choice = settings.occupations or magnetism.occupations
    if occ_choice == "smearing":
        system_lines.append("    occupations='smearing'")
        smear = settings.smearing or magnetism.smearing or "cold"
        system_lines.append(f"    smearing='{smear}'")
        degauss_val = settings.degauss if settings.degauss is not None else (magnetism.degauss or 0.02)
        system_lines.append(f"    degauss={degauss_val}")
    else:
        system_lines.append("    occupations='fixed'")

    nspin = settings.nspin_override or (magnetism.nspin if magnetism.nspin > 1 else None)
    if nspin and nspin > 1:
        system_lines.append(f"    nspin={nspin}")
        for spec in unique_species:
            mag = settings.starting_magnetization.get(spec)
            if mag is None and magnetism.default_magnetization:
                mag = magnetism.default_magnetization
            if mag is not None:
                system_lines.append(f"    starting_magnetization('{spec}') = {mag}")

    if settings.nosym:
        system_lines.append("    nosym=.true.")
    if settings.noinv:
        system_lines.append("    noinv=.true.")
    if settings.tstress:
        system_lines.append("    tstress=.true.")
    if settings.tprnfor:
        system_lines.append("    tprnfor=.true.")
    system_lines.append("/")

    electrons_lines = ["&ELECTRONS"]
    conv_thr = settings.conv_thr or (2e-10 * natoms)
    electrons_lines.append(f"    conv_thr={conv_thr}")
    if settings.mixing_beta is not None:
        electrons_lines.append(f"    mixing_beta={settings.mixing_beta}")
    if settings.electron_maxstep is not None:
        electrons_lines.append(f"    electron_maxstep={settings.electron_maxstep}")
    if settings.diagonalization:
        electrons_lines.append(f"    diagonalization='{settings.diagonalization}'")
    if settings.mixing_mode:
        electrons_lines.append(f"    mixing_mode='{settings.mixing_mode}'")
    electrons_lines.append("/")

    atomic_species_lines = ["ATOMIC_SPECIES"]
    numbers = structure.get_atomic_numbers()
    for spec in unique_species:
        idx = species.index(spec)
        mass = atomic_masses[numbers[idx]]
        atomic_species_lines.append(f"{spec} {mass:.6f} {pseudo_map[spec]}")

    atomic_positions_lines = ["ATOMIC_POSITIONS crystal"]
    scaled_positions = structure.get_scaled_positions()
    for sym, pos in zip(species, scaled_positions):
        coords = " ".join(f"{coord:.10f}" for coord in pos)
        atomic_positions_lines.append(f"{sym} {coords}")

    cell_lines = ["CELL_PARAMETERS angstrom"]
    for vec in structure.get_cell():
        cell_lines.append(" ".join(f"{x:.10f}" for x in vec))

    kpoints_lines = ["K_POINTS automatic"]
    kpoints_lines.append(
        "{} {} {} {} {} {}".format(
            int(k_points[0]),
            int(k_points[1]),
            int(k_points[2]),
            int(k_shift[0]),
            int(k_shift[1]),
            int(k_shift[2]),
        )
    )

    blocks = [
        "\n".join(control_lines),
        "\n".join(system_lines),
        "\n".join(electrons_lines),
        "\n".join(atomic_species_lines),
        "\n".join(atomic_positions_lines),
        "\n".join(cell_lines),
        "\n".join(kpoints_lines),
    ]
    header = [f"! protocol: {protocol}", f"! xc_functional: {xc_label}"]
    return "\n\n".join(header + blocks) + "\n"


def read_structure(path: str, fmt: str | None = None) -> Atoms:
    kwargs = {"format": fmt} if fmt else {}
    return read(path, **kwargs)
