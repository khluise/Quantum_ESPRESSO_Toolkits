#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read


# ============================================================
# Default settings
# ============================================================

DEFAULT_SYMPREC = 1.0e-5

KPOINT_OFFSET = (0, 0, 0)

SUPPORTED_STRUCTURE_FORMATS = {
    ".cif": "cif",
    ".xsf": "xsf",
    ".cell": "castep-cell",
    ".vasp": "vasp",
    ".poscar": "vasp",
}

VASP_BASENAMES = {
    "POSCAR",
    "CONTCAR",
}

DEFAULT_PROTOCOL = "balanced"
DEFAULT_MAGNETISM = "non-magnetic-fractional"


@dataclass(frozen=True)
class ProtocolSettings:
    kpoint_spacing: float
    degauss: float
    ecutwfc: float
    ecutrho: float
    etot_per_atom: float
    conv_per_atom: float
    forc_conv_thr: float


PROTOCOL_SETTINGS = {
    "fast": ProtocolSettings(
        kpoint_spacing=0.30,
        degauss=0.0275,
        ecutwfc=35.0,
        ecutrho=280.0,
        etot_per_atom=1.0e-4,
        conv_per_atom=4.0e-10,
        forc_conv_thr=1.0e-3,
    ),
    "balanced": ProtocolSettings(
        kpoint_spacing=0.15,
        degauss=0.0200,
        ecutwfc=35.0,
        ecutrho=280.0,
        etot_per_atom=1.0e-5,
        conv_per_atom=2.0e-10,
        forc_conv_thr=1.0e-4,
    ),
    "stringent": ProtocolSettings(
        kpoint_spacing=0.10,
        degauss=0.0125,
        ecutwfc=60.0,
        ecutrho=480.0,
        etot_per_atom=5.0e-6,
        conv_per_atom=1.0e-10,
        forc_conv_thr=5.0e-5,
    ),
}

DEFAULT_PROTOCOL_SETTINGS = PROTOCOL_SETTINGS[DEFAULT_PROTOCOL]

DEFAULT_KPOINT_SPACING = DEFAULT_PROTOCOL_SETTINGS.kpoint_spacing
DEGAUSS = DEFAULT_PROTOCOL_SETTINGS.degauss
ECUTWFC = DEFAULT_PROTOCOL_SETTINGS.ecutwfc
ECUTRHO = DEFAULT_PROTOCOL_SETTINGS.ecutrho
ETOT_PER_ATOM = DEFAULT_PROTOCOL_SETTINGS.etot_per_atom
CONV_PER_ATOM = DEFAULT_PROTOCOL_SETTINGS.conv_per_atom
FORC_CONV_THR = DEFAULT_PROTOCOL_SETTINGS.forc_conv_thr

MAGNETISM_CHOICES = (
    "non-magnetic-fractional",
    "non-magnetic-fixed",
)

ODD_ELECTRON_WARNING = (
    "Warning: The current system has an odd number of electrons, so the choice "
    "to treat it as a non-magnetic insulator has been overruled, and the input "
    "generated is for the non-magnetic metal case."
)

QE_OUTDIR = "./out/"
PSEUDO_DIR = "./pseudo/"


# ============================================================
# Formatting helpers
# ============================================================

def qe_float(value: float) -> str:
    return f"{value:18.10e}".replace("e", "d")


def qe_bool(value: bool) -> str:
    return ".true." if value else ".false."


def clean_fraction(x: float, tol: float = 5e-11) -> float:
    if abs(x) < tol:
        return 0.0

    if abs(x - 1.0) < tol:
        return 0.0

    return float(x)


# ============================================================
# Structure handling
# ============================================================

def structure_format_for_path(structure_path: Path) -> str:
    if structure_path.name.upper() in VASP_BASENAMES:
        return "vasp"

    suffix = structure_path.suffix.lower()

    if suffix not in SUPPORTED_STRUCTURE_FORMATS:
        supported = ", ".join(
            [
                ".cif",
                ".xsf",
                ".cell",
                ".vasp",
                ".poscar",
                "POSCAR",
                "CONTCAR",
            ]
        )
        raise ValueError(
            f"Unsupported structure format for {structure_path}. "
            f"Supported inputs: {supported}."
        )

    return SUPPORTED_STRUCTURE_FORMATS[suffix]


def read_structure(structure_path: Path) -> Atoms:
    ase_format = structure_format_for_path(structure_path)

    atoms = read(str(structure_path), format=ase_format)

    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"No structure found in {structure_path}")

        atoms = atoms[0]

    atoms.pbc = True

    return atoms


def read_cif(cif_path: Path) -> Atoms:
    return read_structure(cif_path)


def refine_primitive(
    atoms: Atoms,
    symprec: float,
) -> Atoms:

    lattice = np.array(atoms.cell, dtype=float)

    scaled_positions = atoms.get_scaled_positions(wrap=True)

    atomic_numbers_array = atoms.get_atomic_numbers()

    primitive = spglib.find_primitive(
        (lattice, scaled_positions, atomic_numbers_array),
        symprec=symprec,
    )

    if primitive is None:
        return atoms

    prim_lattice, prim_scaled_positions, prim_numbers = primitive

    return Atoms(
        numbers=list(prim_numbers),
        scaled_positions=np.array(prim_scaled_positions, dtype=float),
        cell=np.array(prim_lattice, dtype=float),
        pbc=True,
    )


def species_in_structure_order(atoms: Atoms) -> list[str]:
    species = []

    for symbol in atoms.get_chemical_symbols():
        if symbol not in species:
            species.append(symbol)

    return species


def formula_prefix(atoms: Atoms) -> str:
    formula = atoms.get_chemical_formula(mode="metal")

    formula = formula.replace(" ", "")
    formula = formula.replace("(", "")
    formula = formula.replace(")", "")

    return formula


# ============================================================
# K-point generation
# ============================================================

def reciprocal_lengths_2pi(
    atoms: Atoms,
) -> tuple[float, float, float]:

    cell = np.array(atoms.cell, dtype=float)

    a1, a2, a3 = cell

    volume = float(np.dot(a1, np.cross(a2, a3)))

    if abs(volume) < 1e-12:
        raise ValueError("Invalid cell volume")

    b1 = 2.0 * math.pi * np.cross(a2, a3) / volume

    b2 = 2.0 * math.pi * np.cross(a3, a1) / volume

    b3 = 2.0 * math.pi * np.cross(a1, a2) / volume

    return (
        float(np.linalg.norm(b1)),
        float(np.linalg.norm(b2)),
        float(np.linalg.norm(b3)),
    )


def kmesh_from_spacing(
    atoms: Atoms,
    kpoint_spacing: float,
) -> tuple[int, int, int]:

    lengths = reciprocal_lengths_2pi(atoms)

    return tuple(
        max(1, math.ceil(length / kpoint_spacing))
        for length in lengths
    )


# ============================================================
# Protocol and magnetism handling
# ============================================================

def settings_for_protocol(
    protocol: str,
    kpoint_spacing_override: float | None,
) -> ProtocolSettings:

    settings = PROTOCOL_SETTINGS[protocol]

    if kpoint_spacing_override is not None:
        settings = replace(
            settings,
            kpoint_spacing=kpoint_spacing_override,
        )

    return settings


def total_electron_count(atoms: Atoms) -> int:
    return int(sum(atoms.get_atomic_numbers()))


def resolve_magnetism(
    atoms: Atoms,
    requested_magnetism: str,
) -> tuple[str, str | None]:

    if requested_magnetism != "non-magnetic-fixed":
        return requested_magnetism, None

    if total_electron_count(atoms) % 2 == 0:
        return requested_magnetism, None

    return "non-magnetic-fractional", ODD_ELECTRON_WARNING


# ============================================================
# QE input generation
# ============================================================

def make_qe_input(
    atoms: Atoms,
    settings: ProtocolSettings,
    magnetism: str,
) -> str:

    nat = len(atoms)

    species = species_in_structure_order(atoms)

    ntyp = len(species)

    prefix = formula_prefix(atoms)

    etot_conv_thr = nat * settings.etot_per_atom

    conv_thr = nat * settings.conv_per_atom

    k1, k2, k3 = kmesh_from_spacing(
        atoms,
        settings.kpoint_spacing,
    )

    lines = []

    # ---------------- CONTROL ----------------

    lines.append("&CONTROL")

    lines.append("  calculation = 'scf'")

    lines.append(f"  etot_conv_thr = {qe_float(etot_conv_thr)}")

    lines.append(f"  forc_conv_thr = {qe_float(settings.forc_conv_thr)}")

    lines.append(f"  outdir = '{QE_OUTDIR}'")

    lines.append(f"  prefix = '{prefix}'")

    lines.append(f"  pseudo_dir = '{PSEUDO_DIR}'")

    lines.append(f"  tprnfor = {qe_bool(True)}")

    lines.append(f"  tstress = {qe_bool(True)}")

    lines.append("  verbosity = 'high'")

    lines.append("/")

    # ---------------- SYSTEM ----------------

    lines.append("&SYSTEM")

    if magnetism == "non-magnetic-fractional":
        lines.append(f"  degauss = {qe_float(settings.degauss)}")

    lines.append(f"  ecutrho = {qe_float(settings.ecutrho)}")

    lines.append(f"  ecutwfc = {qe_float(settings.ecutwfc)}")

    lines.append("  ibrav = 0")

    lines.append(f"  nat = {nat}")

    lines.append("  nosym = .false.")

    lines.append(f"  ntyp = {ntyp}")

    if magnetism == "non-magnetic-fixed":
        lines.append("  occupations = 'fixed'")
    else:
        lines.append("  occupations = 'smearing'")

        lines.append("  smearing = 'cold'")

    lines.append("/")

    # ---------------- ELECTRONS ----------------

    lines.append("&ELECTRONS")

    lines.append(f"  conv_thr = {qe_float(conv_thr)}")

    lines.append("  electron_maxstep = 80")

    lines.append(f"  mixing_beta = {qe_float(0.4)}")

    lines.append("/")

    # ---------------- ATOMIC_SPECIES ----------------

    lines.append("ATOMIC_SPECIES")

    for symbol in species:

        mass = atomic_masses[atomic_numbers[symbol]]

        pseudo_name = f"{symbol}.pseudo_file.UPF"

        lines.append(
            f"{symbol:<6} {mass:.4f} {pseudo_name}"
        )

    # ---------------- ATOMIC_POSITIONS ----------------

    lines.append("ATOMIC_POSITIONS crystal")

    symbols = atoms.get_chemical_symbols()

    scaled_positions = atoms.get_scaled_positions(wrap=True)

    for symbol, pos in zip(symbols, scaled_positions):

        x, y, z = [clean_fraction(v) for v in pos]

        lines.append(
            f"{symbol:<2} "
            f"{x:20.10f}"
            f"{y:19.10f}"
            f"{z:19.10f}"
        )

    # ---------------- K_POINTS ----------------

    lines.append("K_POINTS automatic")

    lines.append(
        f"{k1} {k2} {k3} "
        f"{KPOINT_OFFSET[0]} "
        f"{KPOINT_OFFSET[1]} "
        f"{KPOINT_OFFSET[2]}"
    )

    # ---------------- CELL_PARAMETERS ----------------

    lines.append("CELL_PARAMETERS angstrom")

    for vec in np.array(atoms.cell, dtype=float):

        lines.append(
            f"{vec[0]:18.10f}"
            f"{vec[1]:19.10f}"
            f"{vec[2]:19.10f}"
        )

    return "\n".join(lines) + "\n"


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "structure",
        help="Input structure file (.cif, .xsf, .cell, POSCAR, CONTCAR, .vasp, .poscar)",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="scf.in",
        help="Output QE input filename",
    )

    # --------------------------------------------------------
    # User-exposed options
    # --------------------------------------------------------

    parser.add_argument(
        "--kpoint-spacing",
        type=float,
        default=None,
        help=(
            "Override protocol k-point spacing in 1/angstrom "
            f"(default from --protocol {DEFAULT_PROTOCOL}: {DEFAULT_KPOINT_SPACING})"
        ),
    )

    parser.add_argument(
        "--protocol",
        choices=list(PROTOCOL_SETTINGS),
        default=DEFAULT_PROTOCOL,
        help=f"Generation protocol (default: {DEFAULT_PROTOCOL})",
    )

    parser.add_argument(
        "--magnetism",
        "--magnetization",
        dest="magnetism",
        choices=MAGNETISM_CHOICES,
        default=DEFAULT_MAGNETISM,
        help=(
            "Occupation mode: non-magnetic-fractional for metals, "
            f"non-magnetic-fixed for insulators (default: {DEFAULT_MAGNETISM})"
        ),
    )

    parser.add_argument(
        "--refine",
        choices=["yes", "no"],
        default="yes",
        help="Refine primitive structure using spglib (default: yes)",
    )

    parser.add_argument(
        "--symprec",
        type=float,
        default=DEFAULT_SYMPREC,
        help=f"spglib symmetry tolerance (default: {DEFAULT_SYMPREC})",
    )

    args = parser.parse_args()

    if args.kpoint_spacing is not None and args.kpoint_spacing <= 0:
        parser.error("--kpoint-spacing must be greater than zero")

    structure_path = Path(args.structure).resolve()

    output_path = Path(args.output).resolve()

    if not structure_path.exists():
        raise FileNotFoundError(structure_path)

    atoms = read_structure(structure_path)

    if args.refine == "yes":
        atoms = refine_primitive(
            atoms,
            symprec=args.symprec,
        )

    settings = settings_for_protocol(
        protocol=args.protocol,
        kpoint_spacing_override=args.kpoint_spacing,
    )

    magnetism, warning = resolve_magnetism(
        atoms,
        requested_magnetism=args.magnetism,
    )

    if warning is not None:
        print(warning, file=sys.stderr)

    qe_input = make_qe_input(
        atoms,
        settings=settings,
        magnetism=magnetism,
    )

    output_path.write_text(qe_input)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
