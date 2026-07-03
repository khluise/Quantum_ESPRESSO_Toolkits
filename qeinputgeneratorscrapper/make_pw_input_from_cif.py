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
from ase.lattice import identify_lattice


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

IBRAV_AUTO = "auto"
SUPPORTED_NONZERO_IBRAVS = {
    -13,
    -12,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    14,
}
SUPPORTED_IBRAVS = {0, *SUPPORTED_NONZERO_IBRAVS}
QE_IBRAV_BY_ASE_BRAVAIS = {
    "CUB": 1,
    "FCC": 2,
    "BCC": 3,
    "HEX": 4,
    "RHL": 5,
    "TET": 6,
    "BCT": 7,
    "ORC": 8,
    "ORCC": 9,
    "ORCF": 10,
    "ORCI": 11,
    "MCL": -12,
    "MCLC": -13,
    "TRI": 14,
}
QE_CELL_TRANSFORM_TOL = 1.0e-5


@dataclass(frozen=True)
class ProtocolSettings:
    kpoint_spacing: float
    degauss: float
    ecutwfc: float
    ecutrho: float
    etot_per_atom: float
    conv_per_atom: float
    forc_conv_thr: float


@dataclass(frozen=True)
class QELattice:
    ibrav: int
    parameters: tuple[tuple[str, float], ...]
    cell: np.ndarray


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
# QE lattice / ibrav handling
# ============================================================

def parse_ibrav_argument(value: str) -> int | str:
    normalized = value.strip().lower()

    if normalized == IBRAV_AUTO:
        return IBRAV_AUTO

    try:
        ibrav = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ibrav must be 0, auto, or a supported QE ibrav integer"
        ) from exc

    if ibrav not in SUPPORTED_IBRAVS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_IBRAVS))
        raise argparse.ArgumentTypeError(
            f"Unsupported --ibrav {ibrav}. Supported values: auto, {supported}"
        )

    return ibrav


def qe_lattice_float(value: float) -> str:
    return f"{value:18.10f}"


def bravais_parameter(
    bravais_lattice: object,
    name: str,
) -> float:

    value = getattr(bravais_lattice, name, None)

    if value is None:
        parameters = getattr(bravais_lattice, "_parameters", {})
        value = parameters.get(name)

    if value is None:
        raise ValueError(
            f"ASE Bravais lattice {bravais_lattice} does not provide {name!r}"
        )

    return float(value)


def qe_cell_from_ibrav(
    ibrav: int,
    bravais_lattice: object,
) -> QELattice:

    a = bravais_parameter(bravais_lattice, "a")
    parameters: list[tuple[str, float]] = [("A", a)]

    if ibrav == 1:
        cell = np.array(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a],
            ],
            dtype=float,
        )

    elif ibrav == 2:
        half_a = 0.5 * a
        cell = np.array(
            [
                [-half_a, 0.0, half_a],
                [0.0, half_a, half_a],
                [-half_a, half_a, 0.0],
            ],
            dtype=float,
        )

    elif ibrav == 3:
        half_a = 0.5 * a
        cell = np.array(
            [
                [half_a, half_a, half_a],
                [-half_a, half_a, half_a],
                [-half_a, -half_a, half_a],
            ],
            dtype=float,
        )

    elif ibrav == 4:
        c = bravais_parameter(bravais_lattice, "c")
        parameters.append(("C", c))
        cell = np.array(
            [
                [a, 0.0, 0.0],
                [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
                [0.0, 0.0, c],
            ],
            dtype=float,
        )

    elif ibrav == 5:
        alpha = math.radians(bravais_parameter(bravais_lattice, "alpha"))
        cos_alpha = math.cos(alpha)
        parameters.append(("cosBC", cos_alpha))
        tx = math.sqrt((1.0 - cos_alpha) / 2.0)
        ty = math.sqrt((1.0 - cos_alpha) / 6.0)
        tz = math.sqrt((1.0 + 2.0 * cos_alpha) / 3.0)
        cell = a * np.array(
            [
                [tx, -ty, tz],
                [0.0, 2.0 * ty, tz],
                [-tx, -ty, tz],
            ],
            dtype=float,
        )

    elif ibrav in {6, 7}:
        c = bravais_parameter(bravais_lattice, "c")
        parameters.append(("C", c))

        if ibrav == 6:
            cell = np.array(
                [
                    [a, 0.0, 0.0],
                    [0.0, a, 0.0],
                    [0.0, 0.0, c],
                ],
                dtype=float,
            )
        else:
            half_a = 0.5 * a
            half_c = 0.5 * c
            cell = np.array(
                [
                    [half_a, -half_a, half_c],
                    [half_a, half_a, half_c],
                    [-half_a, -half_a, half_c],
                ],
                dtype=float,
            )

    elif ibrav in {8, 9, 10, 11}:
        b = bravais_parameter(bravais_lattice, "b")
        c = bravais_parameter(bravais_lattice, "c")
        parameters.extend([("B", b), ("C", c)])

        if ibrav == 8:
            cell = np.array(
                [
                    [a, 0.0, 0.0],
                    [0.0, b, 0.0],
                    [0.0, 0.0, c],
                ],
                dtype=float,
            )
        elif ibrav == 9:
            cell = np.array(
                [
                    [0.5 * a, 0.5 * b, 0.0],
                    [-0.5 * a, 0.5 * b, 0.0],
                    [0.0, 0.0, c],
                ],
                dtype=float,
            )
        elif ibrav == 10:
            cell = np.array(
                [
                    [0.5 * a, 0.0, 0.5 * c],
                    [0.5 * a, 0.5 * b, 0.0],
                    [0.0, 0.5 * b, 0.5 * c],
                ],
                dtype=float,
            )
        else:
            cell = np.array(
                [
                    [0.5 * a, 0.5 * b, 0.5 * c],
                    [-0.5 * a, 0.5 * b, 0.5 * c],
                    [-0.5 * a, -0.5 * b, 0.5 * c],
                ],
                dtype=float,
            )

    elif ibrav in {-12, -13}:
        b = bravais_parameter(bravais_lattice, "b")
        c = bravais_parameter(bravais_lattice, "c")
        beta_degrees = getattr(bravais_lattice, "beta", None)

        if beta_degrees is None:
            beta_degrees = getattr(bravais_lattice, "alpha", None)

        if beta_degrees is None:
            parameters_dict = getattr(bravais_lattice, "_parameters", {})
            beta_degrees = parameters_dict.get("beta", parameters_dict.get("alpha"))

        if beta_degrees is None:
            raise ValueError(
                f"ASE Bravais lattice {bravais_lattice} does not provide a monoclinic angle"
            )

        beta = math.radians(float(beta_degrees))
        cos_beta = math.cos(beta)
        sin_beta = math.sin(beta)
        parameters.extend([("B", b), ("C", c), ("cosAC", cos_beta)])

        if ibrav == -12:
            cell = np.array(
                [
                    [a, 0.0, 0.0],
                    [0.0, b, 0.0],
                    [c * cos_beta, 0.0, c * sin_beta],
                ],
                dtype=float,
            )
        else:
            cell = np.array(
                [
                    [0.5 * a, 0.5 * b, 0.0],
                    [-0.5 * a, 0.5 * b, 0.0],
                    [c * cos_beta, 0.0, c * sin_beta],
                ],
                dtype=float,
            )

    elif ibrav == 14:
        b = bravais_parameter(bravais_lattice, "b")
        c = bravais_parameter(bravais_lattice, "c")
        alpha = math.radians(bravais_parameter(bravais_lattice, "alpha"))
        beta = math.radians(bravais_parameter(bravais_lattice, "beta"))
        gamma = math.radians(bravais_parameter(bravais_lattice, "gamma"))
        cos_alpha = math.cos(alpha)
        cos_beta = math.cos(beta)
        cos_gamma = math.cos(gamma)
        sin_gamma = math.sin(gamma)
        z_term = (
            1.0
            + 2.0 * cos_alpha * cos_beta * cos_gamma
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
        )

        if sin_gamma <= 0.0 or z_term < 0.0:
            raise ValueError("Invalid triclinic cell parameters")

        parameters.extend(
            [
                ("B", b),
                ("C", c),
                ("cosAB", cos_gamma),
                ("cosAC", cos_beta),
                ("cosBC", cos_alpha),
            ]
        )
        cell = np.array(
            [
                [a, 0.0, 0.0],
                [b * cos_gamma, b * sin_gamma, 0.0],
                [
                    c * cos_beta,
                    c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                    c * math.sqrt(max(0.0, z_term)) / sin_gamma,
                ],
            ],
            dtype=float,
        )

    else:
        raise ValueError(f"Unsupported nonzero ibrav: {ibrav}")

    return QELattice(
        ibrav=ibrav,
        parameters=tuple(parameters),
        cell=cell,
    )


def lattice_transform_to_qe_cell(
    atoms: Atoms,
    qe_lattice: QELattice,
    tolerance: float = QE_CELL_TRANSFORM_TOL,
) -> np.ndarray:

    input_cell = np.array(atoms.cell, dtype=float)

    transform = input_cell @ np.linalg.inv(qe_lattice.cell)

    rounded_transform = np.rint(transform)

    if not np.allclose(transform, rounded_transform, atol=tolerance, rtol=0.0):
        max_delta = float(np.max(np.abs(transform - rounded_transform)))
        raise ValueError(
            "QE ibrav cell is not an integer-basis match for the ASE cell "
            f"(max transform delta {max_delta:.3e})"
        )

    determinant = round(float(np.linalg.det(rounded_transform)))

    if abs(determinant) != 1:
        raise ValueError(
            "QE ibrav cell changes the primitive-cell volume "
            f"(integer transform determinant {determinant})"
        )

    rebuilt_cell = rounded_transform @ qe_lattice.cell

    if not np.allclose(input_cell, rebuilt_cell, atol=tolerance, rtol=0.0):
        max_delta = float(np.max(np.abs(input_cell - rebuilt_cell)))
        raise ValueError(
            "QE ibrav cell does not reconstruct the ASE cell "
            f"(max cell delta {max_delta:.3e} angstrom)"
        )

    return rounded_transform


def fractional_positions_for_qe_lattice(
    atoms: Atoms,
    qe_lattice: QELattice | None,
) -> np.ndarray:

    scaled_positions = atoms.get_scaled_positions(wrap=True)

    if qe_lattice is None:
        return scaled_positions

    transform = lattice_transform_to_qe_cell(atoms, qe_lattice)

    return scaled_positions @ transform


def resolve_qe_lattice(
    atoms: Atoms,
    ibrav_request: int | str,
    symprec: float,
) -> tuple[QELattice | None, str | None]:

    if ibrav_request == 0:
        return None, None

    try:
        bravais_lattice, _operation = identify_lattice(
            atoms.cell,
            eps=symprec,
            pbc=True,
        )
    except Exception as exc:
        if ibrav_request == IBRAV_AUTO:
            return (
                None,
                "Warning: Could not detect a Bravais lattice for --ibrav auto; "
                "falling back to ibrav = 0.",
            )

        raise ValueError(
            f"Could not detect Bravais lattice needed to validate --ibrav {ibrav_request}"
        ) from exc

    detected_name = bravais_lattice.name

    detected_ibrav = QE_IBRAV_BY_ASE_BRAVAIS.get(detected_name)

    if detected_ibrav is None:
        if ibrav_request == IBRAV_AUTO:
            return (
                None,
                f"Warning: ASE Bravais lattice {detected_name} is not mapped to QE ibrav; "
                "falling back to ibrav = 0.",
            )

        raise ValueError(
            f"ASE Bravais lattice {detected_name} cannot be written as --ibrav {ibrav_request}"
        )

    if ibrav_request == IBRAV_AUTO:
        requested_ibrav = detected_ibrav
    else:
        requested_ibrav = int(ibrav_request)

        if requested_ibrav != detected_ibrav:
            raise ValueError(
                f"Requested --ibrav {requested_ibrav}, but ASE detected "
                f"{detected_name} / QE ibrav {detected_ibrav}"
            )

    try:
        qe_lattice = qe_cell_from_ibrav(
            requested_ibrav,
            bravais_lattice,
        )

        lattice_transform_to_qe_cell(
            atoms,
            qe_lattice,
        )
    except ValueError as exc:
        if ibrav_request == IBRAV_AUTO:
            return (
                None,
                "Warning: Detected QE ibrav "
                f"{requested_ibrav} from ASE Bravais lattice {detected_name}, "
                f"but validation failed ({exc}); falling back to ibrav = 0.",
            )

        raise

    return qe_lattice, None


# ============================================================
# QE input generation
# ============================================================

def make_qe_input(
    atoms: Atoms,
    settings: ProtocolSettings,
    magnetism: str,
    qe_lattice: QELattice | None = None,
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

    if qe_lattice is None:
        lines.append("  ibrav = 0")
    else:
        lines.append(f"  ibrav = {qe_lattice.ibrav}")

        for name, value in qe_lattice.parameters:
            lines.append(f"  {name} = {qe_lattice_float(value)}")

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

    scaled_positions = fractional_positions_for_qe_lattice(
        atoms,
        qe_lattice,
    )

    for symbol, pos in zip(symbols, scaled_positions):

        wrapped_pos = np.mod(pos, 1.0)

        x, y, z = [clean_fraction(v) for v in wrapped_pos]

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

    if qe_lattice is not None:
        return "\n".join(lines) + "\n"

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

    parser.add_argument(
        "--ibrav",
        type=parse_ibrav_argument,
        default=0,
        metavar="{0,auto,N}",
        help=(
            "QE Bravais lattice mode: 0 keeps CELL_PARAMETERS, auto detects a "
            "nonzero ibrav when safe, or pass a supported QE ibrav integer "
            "(default: 0)"
        ),
    )

    args = parser.parse_args()

    if args.kpoint_spacing is not None and args.kpoint_spacing <= 0:
        parser.error("--kpoint-spacing must be greater than zero")

    if args.symprec <= 0:
        parser.error("--symprec must be greater than zero")

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

    try:
        qe_lattice, warning = resolve_qe_lattice(
            atoms,
            ibrav_request=args.ibrav,
            symprec=args.symprec,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if warning is not None:
        print(warning, file=sys.stderr)

    qe_input = make_qe_input(
        atoms,
        settings=settings,
        magnetism=magnetism,
        qe_lattice=qe_lattice,
    )

    output_path.write_text(qe_input)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
