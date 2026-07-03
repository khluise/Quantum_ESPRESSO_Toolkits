"""Build workflow workspaces, manifests, and initial state files.

This module is intentionally GUI-independent.  The Workflow Creator tab can
collect user choices, pass them into these helpers, and leave execution to a
future workflow runner.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from xml.etree import ElementTree


WORKFLOW_VERSION = 1
RY_TO_EV = 13.605693

SOURCE_DIR = "source"
CONVERGENCE_DIR = "convergence"
CUTOFF_CONVERGENCE_DIR = "cutoff_convergence"
KPOINT_CONVERGENCE_DIR = "kpoint_convergence"
RELAXATION_DIR = "relaxation"
BANDS_DIR = "bands"
WANNIER90_DIR = "wannier90"
# DOS/PDOS workflow directory (SCF -> NSCF -> projwfc.x)
DOS_DIR = "dos"
# Phonon workflow directory (SCF -> ph.x)
PHONON_DIR = "phonon"
ELASTIC_DIR = "elastic"
FERMI_SURFACE_DIR = "fermi_surface"
CHARGE_DENSITY_DIR = "charge_density"
EPC_MODE = "epc_interpolated"
TETRA_EPC_MODE = "epc_tetrahedra"
DEFAULT_PHONON_Q_GRID = [3, 3, 3]
DEFAULT_PHDOS_NK = [20, 20, 20]
DEFAULT_EPC_FIT_KMESH = [24, 24, 24]
DEFAULT_EPC_SCF_KMESH = [12, 12, 12]
DEFAULT_EPC_Q_GRID = [3, 3, 3]
DEFAULT_PHONON_SCF_KMESH = [16, 16, 16]
DEFAULT_EPC_LAMBDA_Q_POINTS = [
    [0.0000000, 0.0000000, 0.0000000, 1.00],
    [-0.2500000, -0.2500000, 0.2500000, 8.00],
    [-0.5000000, -0.5000000, 0.5000000, 4.00],
    [0.0000000, 0.0000000, 0.5000000, 6.00],
    [-0.2500000, -0.2500000, 0.7500000, 24.00],
    [-0.5000000, -0.5000000, 1.0000000, 12.00],
    [0.0000000, 0.0000000, 1.0000000, 3.00],
    [-0.5000000, 0.0000000, 1.0000000, 6.00],
]
# Fat-band is an optional post-processing stage inside the same band workflow directory.
FAT_BANDS_DIR = BANDS_DIR
INPUTS_OUTPUTS_DIR = "inputs_and_outputs"
LOGS_DIR = "logs"
DATA_PLOTS_DIR = "data_and_plots"
PREPARED_QE_TEMPLATE_NAME = "QE_input.template"
PSEUDO_DIR_NAME = "pseudo"

QE_INPUT_SUFFIXES = {".in", ".pwi", ".pw", ".qe", ".txt"}
LOCAL_STRUCTURE_SUFFIXES = {".cif", ".xsf", ".cell", ".vasp", ".poscar"}
LOCAL_STRUCTURE_BASENAMES = {"POSCAR", "CONTCAR"}

PROJECT_ROOT = Path(__file__).resolve().parent
QE_INPUT_GENERATOR_DIR = PROJECT_ROOT / "qeinputgeneratorscrapper"
LOCAL_QE_INPUT_GENERATOR_SCRIPT = QE_INPUT_GENERATOR_DIR / "make_pw_input_from_cif.py"
MC_QE_INPUT_GENERATOR_SCRIPT = QE_INPUT_GENERATOR_DIR / "mc_qeinputgenerator_web_automation_client.py"
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0

PROTOCOL_NUMERICAL_SETTINGS = {
    "fast": {
        "kpoint_spacing": 0.30,
        "degauss": 0.0275,
        "ecutwfc": 35.0,
        "ecutrho": 280.0,
        "etot_per_atom": 1.0e-4,
        "conv_per_atom": 4.0e-10,
        "forc_conv_thr": 1.0e-3,
    },
    "balanced": {
        "kpoint_spacing": 0.15,
        "degauss": 0.0200,
        "ecutwfc": 35.0,
        "ecutrho": 280.0,
        "etot_per_atom": 1.0e-5,
        "conv_per_atom": 2.0e-10,
        "forc_conv_thr": 1.0e-4,
    },
    "stringent": {
        "kpoint_spacing": 0.10,
        "degauss": 0.0125,
        "ecutwfc": 60.0,
        "ecutrho": 480.0,
        "etot_per_atom": 5.0e-6,
        "conv_per_atom": 1.0e-10,
        "forc_conv_thr": 5.0e-5,
    },
}

RELAXATION_IONS_NAMELIST = [
    "&IONS",
    "   ion_dynamics = 'bfgs'",
    "/",
]
RELAXATION_CELL_NAMELIST = [
    "&CELL",
    "   cell_dofree    = 'ibrav'",
    "   cell_dynamics  = 'bfgs'",
    "   press          = 0.D0",
    "   press_conv_thr = 0.5D0",
    "/",
]

DEFAULT_RELAX_PRESS = "0.D0"
DEFAULT_RELAX_PRESS_CONV_THR = "0.5D0"
DEFAULT_RELAX_ELECTRON_MAXSTEP = "80"
DEFAULT_RELAX_DIAGONALIZATION = "david"
DEFAULT_RELAX_ION_DYNAMICS = "bfgs"
DEFAULT_RELAX_CELL_DYNAMICS = "bfgs"
RELAX_DIAGONALIZATION_VALUES = {"david", "cg", "par0", "rmm-davidson", "rmm-paro"}
RELAX_ION_DYNAMICS_VALUES = {"bfgs", "damp"}
RELAX_CELL_DYNAMICS_VALUES = {"bfgs", "damp-pr", "damp-w"}
RELAX_SYSTEM_LATTICE_KEYS = {
    "ibrav",
    "celldm(1)",
    "celldm(2)",
    "celldm(3)",
    "celldm(4)",
    "celldm(5)",
    "celldm(6)",
    "a",
    "b",
    "c",
    "cosab",
    "cosac",
    "cosbc",
}


def sanitize_job_name(name: str) -> str:
    """Return a filesystem-friendly workflow/job name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip())
    return cleaned or "qe_workflow"


def _run_generator_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log: Callable[[str], None] | None = None,
) -> None:
    """Run a generator command and optionally stream captured output to a logger."""

    if log:
        log("Running: " + " ".join(command))
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout and log:
        log(stdout)
    if stderr and log:
        log(stderr)
    if result.returncode != 0:
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(detail)


def _is_local_generator_supported(path: Path) -> bool:
    if path.name.upper() in LOCAL_STRUCTURE_BASENAMES:
        return True
    return path.suffix.lower() in LOCAL_STRUCTURE_SUFFIXES


def _truthy_option(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _spin_orbit_enabled(options: Mapping[str, Any]) -> bool:
    spin_orbit = options.get("spin_orbit")
    if isinstance(spin_orbit, str):
        return spin_orbit.strip().lower() in {"1", "true", "yes", "on"}
    return bool(spin_orbit)


def _resolve_workflow_magnetism(options: Mapping[str, Any]) -> str:
    magnetism = str(options.get("magnetism") or "").strip()
    if magnetism:
        return magnetism
    electronic = str(options.get("electronic_type") or "").strip().lower()
    if electronic == "insulator":
        return "non-magnetic-fixed"
    return "non-magnetic-fractional"


def _generate_workflow_qe_input(
    source_path: str | Path,
    output_path: str | Path,
    options: Mapping[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Generate the prepared QE input used as the template for workflow tasks."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"The source file '{source}' does not exist.")

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    opts = dict(options or {})
    use_materials_cloud = _truthy_option(opts.get("use_materials_cloud"))
    if not use_materials_cloud and not _is_local_generator_supported(source):
        use_materials_cloud = True

    if use_materials_cloud:
        _generate_workflow_materials_cloud(source, target, opts, log=log)
    else:
        _generate_workflow_local(source, target, opts, log=log)

    copy_selected_workflow_pseudos(opts, target.parent / PSEUDO_DIR_NAME)
    _finalize_prepared_qe_template(target, opts)

    if not target.is_file():
        raise RuntimeError(f"Workflow conversion finished but did not create {target}")
    return target


def copy_selected_workflow_pseudos(options: Mapping[str, Any], pseudo_dir: Path) -> None:
    """Copy GUI-selected workflow pseudopotentials into a source pseudo directory."""

    selected_paths: list[Path] = []
    for key in ("local_pseudo_files", "full_relativistic_pseudo_files"):
        mapping = options.get(key)
        if not isinstance(mapping, Mapping):
            continue
        for raw_path in mapping.values():
            path = Path(str(raw_path or "")).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Selected workflow pseudopotential is missing: {path}")
            selected_paths.append(path)

    if not selected_paths:
        return

    pseudo_dir.mkdir(parents=True, exist_ok=True)
    for source in selected_paths:
        target = pseudo_dir / source.name
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def _finalize_prepared_qe_template(path: Path, options: Mapping[str, Any]) -> None:
    """Normalize the prepared workflow template after conversion/copying.

    The source template is the neutral starting point for initial relaxation and
    convergence.  Final-accuracy choices such as spin-orbit/full-relativistic
    pseudos should be rendered by the later final-relaxation task.
    """

    text = path.read_text(encoding="utf-8", errors="ignore")
    updated = remove_namelist_keys(text, "system", {"noncolin", "lspinorb"})
    updated = apply_manual_ecutwfc_to_qe_text(updated, options)
    updated = apply_2d_structure_options_to_qe_text(updated, options)

    pseudo_dir = path.parent / PSEUDO_DIR_NAME
    if not pseudo_dir.is_dir():
        if updated != text:
            path.write_text(updated, encoding="utf-8")
        return

    pseudo_map = discover_pseudo_files(pseudo_dir, prefer_non_full_relativistic=True)
    updated = rewrite_atomic_species_pseudos(updated, pseudo_map, pseudo_dir=pseudo_dir)
    updated = set_namelist_values(updated, "control", {"pseudo_dir": qe_quote("./pseudo/")})
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def apply_manual_ecutwfc_to_qe_text(text: str, options: Mapping[str, Any] | None) -> str:
    manual_ecutwfc = manual_ecutwfc_from_options(options)
    if manual_ecutwfc is None:
        return text
    return set_namelist_values(
        text,
        "system",
        {
            "ecutwfc": format_qe_number(manual_ecutwfc),
            "ecutrho": format_qe_number(manual_ecutwfc * 8.0),
        },
    )


def apply_2d_structure_options_to_qe_text(text: str, options: Mapping[str, Any] | None) -> str:
    if not isinstance(options, Mapping) or not _truthy_option(options.get("convert_2d_structure")):
        return text
    return set_namelist_values(
        text,
        "system",
        {
            "assume_isolated": qe_quote("2D"),
            "vdw_corr": qe_quote("Grimme-D3"),
        },
    )


def _generate_workflow_local(
    source: Path,
    target: Path,
    options: Mapping[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    if not LOCAL_QE_INPUT_GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"Local generator script not found: {LOCAL_QE_INPUT_GENERATOR_SCRIPT}")

    protocol = str(options.get("protocol") or "balanced")
    magnetism = _resolve_workflow_magnetism(options)

    command = [
        sys.executable,
        str(LOCAL_QE_INPUT_GENERATOR_SCRIPT),
        str(source),
        str(target),
        "--protocol",
        protocol,
        "--magnetism",
        magnetism,
    ]

    kpoint_distance = options.get("kpoint_distance")
    if kpoint_distance is not None and str(kpoint_distance).strip() != "":
        command.extend(["--kpoint-spacing", str(kpoint_distance)])

    refine_cell = str(options.get("refine_cell") or "").strip().lower()
    if refine_cell in {"yes", "no"}:
        command.extend(["--refine", refine_cell])

    symprec = options.get("symprec")
    if symprec is not None and str(symprec).strip() != "":
        command.extend(["--symprec", str(symprec)])

    _run_generator_command(command, cwd=QE_INPUT_GENERATOR_DIR, log=log)


def _generate_workflow_materials_cloud(
    source: Path,
    target: Path,
    options: Mapping[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    if not MC_QE_INPUT_GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"Materials Cloud generator script not found: {MC_QE_INPUT_GENERATOR_SCRIPT}")

    file_format = str(options.get("file_format") or "auto")
    protocol = str(options.get("protocol") or "balanced")
    xc = str(options.get("xc") or "PBEsol")
    magnetism = _resolve_workflow_magnetism(options)
    refine_cell = str(options.get("refine_cell") or "no").strip().lower()
    if refine_cell not in {"yes", "no"}:
        refine_cell = "no"

    with tempfile.TemporaryDirectory(prefix="workflow_convert_", dir=str(target.parent)) as tmp_dir:
        staging_dir = Path(tmp_dir) / "materials_cloud"
        command = [
            sys.executable,
            str(MC_QE_INPUT_GENERATOR_SCRIPT),
            str(source),
            "--out",
            str(staging_dir),
            "--download-zip",
            "--file-format",
            file_format,
            "--protocol",
            protocol,
            "--xc",
            xc,
            "--magnetism",
            magnetism,
            "--refine-cell",
            refine_cell,
        ]

        tolerance = str(options.get("tolerance") or "").strip()
        if tolerance:
            command.extend(["--tolerance", tolerance])

        pseudo_override = str(options.get("pseudo_override") or "").strip()
        if pseudo_override:
            command.extend(["--pseudo-override", pseudo_override])
        kmesh_override = str(options.get("kmesh_override") or "").strip()
        if kmesh_override:
            command.extend(["--kmesh-override", kmesh_override])
        threshold_override = str(options.get("threshold_override") or "").strip()
        if threshold_override:
            command.extend(["--threshold-override", threshold_override])

        _run_generator_command(command, cwd=QE_INPUT_GENERATOR_DIR, log=log)

        zip_path = staging_dir / "qe_input_and_pseudos.zip"
        if not zip_path.is_file():
            fallback = staging_dir / "scf.in"
            if fallback.is_file():
                shutil.copy2(fallback, target)
                return
            raise RuntimeError(f"Materials Cloud did not create the expected ZIP file: {zip_path}")

        extracted_dir = Path(tmp_dir) / "extracted"
        _safe_extract_zip(zip_path, extracted_dir)

        pwscf_input = _find_pwscf_input(extracted_dir)
        if pwscf_input is None:
            raise RuntimeError("The Materials Cloud ZIP did not contain PWscf/pwscf.in.")

        shutil.copy2(pwscf_input, target)

        pseudo_dir = pwscf_input.parent / "pseudo"
        if pseudo_dir.is_dir():
            shutil.copytree(pseudo_dir, target.parent / "pseudo", dirs_exist_ok=True)


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Refusing to extract unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def _find_pwscf_input(root: Path) -> Path | None:
    preferred = root / "PWscf" / "pwscf.in"
    if preferred.is_file():
        return preferred
    for candidate in root.rglob("pwscf.in"):
        if candidate.is_file():
            return candidate
    return None


def _apply_spin_orbit_flags(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    updated = _ensure_system_flags(
        text,
        {
            "noncolin": ".true.",
            "lspinorb": ".true.",
        },
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _ensure_system_flags(text: str, flags: Mapping[str, str]) -> str:
    lines = text.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("&system"):
            start_idx = idx
            break
    if start_idx is None:
        return text

    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("/"):
            end_idx = idx
            break
    if end_idx is None:
        return text

    indent = _namelist_assignment_indent(lines, start_idx + 1, end_idx)
    found: set[str] = set()
    for idx in range(start_idx + 1, end_idx):
        raw = lines[idx]
        cleaned = raw.split("!", 1)[0]
        if "=" not in cleaned:
            continue
        key = cleaned.split("=", 1)[0].strip().lower()
        if key in flags:
            line_indent = raw[: len(raw) - len(raw.lstrip())] or indent
            lines[idx] = f"{line_indent}{key} = {flags[key]}"
            found.add(key)

    insert_lines = [f"{indent}{key} = {flags[key]}" for key in flags if key not in found]
    if insert_lines:
        lines[end_idx:end_idx] = insert_lines

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def _namelist_assignment_indent(lines: Sequence[str], start_idx: int, end_idx: int) -> str:
    for idx in range(start_idx, end_idx):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped or stripped.startswith("!"):
            continue
        body = raw.split("!", 1)[0]
        if "=" not in body:
            continue
        return raw[: len(raw) - len(raw.lstrip())]
    return "  "


def render_cutoff_convergence_input(
    template_text: str,
    *,
    ecutwfc: int | float,
    workflow_root: str | Path,
    child_workdir: str | Path,
    child_id: str,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render one cutoff-convergence SCF input from QE_input.template."""

    root = Path(workflow_root).expanduser().resolve()
    child_dir = root / Path(child_workdir)
    pseudo_dir = root / SOURCE_DIR / PSEUDO_DIR_NAME
    relative_pseudo_dir = relative_path_for_qe(child_dir, pseudo_dir)
    spin_orbit = bool(spin_orbit_required)

    rendered = set_namelist_values(
        template_text,
        "control",
        {
            "calculation": qe_quote("scf"),
            "outdir": qe_quote("./out/"),
            "prefix": qe_quote(child_id),
            "pseudo_dir": qe_quote(relative_pseudo_dir),
            "tprnfor": ".true.",
            "tstress": ".true.",
        },
    )
    rendered = remove_namelists(rendered, {"ions", "cell"})
    rendered = set_namelist_values(
        rendered,
        "system",
        {
            "ecutwfc": format_qe_number(float(ecutwfc)),
            "ecutrho": format_qe_number(float(ecutwfc) * 8.0),
        },
    )
    if spin_orbit:
        rendered = _ensure_system_flags(
            rendered,
            {
                "noncolin": ".true.",
                "lspinorb": ".true.",
            },
        )
    else:
        rendered = remove_namelist_keys(rendered, "system", {"noncolin", "lspinorb"})

    if pseudo_dir.is_dir():
        pseudo_map = discover_pseudo_files(
            pseudo_dir,
            prefer_full_relativistic=spin_orbit,
            prefer_non_full_relativistic=not spin_orbit,
        )
        rendered = rewrite_atomic_species_pseudos(rendered, pseudo_map, pseudo_dir=pseudo_dir)
        if spin_orbit:
            species = atomic_species_entries(rendered)
            validate_full_relativistic_pseudos(species, pseudo_dir)
    elif spin_orbit:
        raise FileNotFoundError(
            f"Spin-orbit coupling requires full-relativistic pseudopotentials in {pseudo_dir}"
        )

    return rendered


def render_kpoint_convergence_input(
    template_text: str,
    *,
    kmesh: Sequence[int],
    workflow_root: str | Path,
    child_workdir: str | Path,
    child_id: str,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render one k-point-convergence SCF input from QE_input.template."""

    root = Path(workflow_root).expanduser().resolve()
    child_dir = root / Path(child_workdir)
    pseudo_dir = root / SOURCE_DIR / PSEUDO_DIR_NAME
    relative_pseudo_dir = relative_path_for_qe(child_dir, pseudo_dir)
    spin_orbit = bool(spin_orbit_required)

    rendered = set_namelist_values(
        template_text,
        "control",
        {
            "calculation": qe_quote("scf"),
            "outdir": qe_quote("./out/"),
            "prefix": qe_quote(child_id),
            "pseudo_dir": qe_quote(relative_pseudo_dir),
            "tprnfor": ".true.",
            "tstress": ".true.",
        },
    )
    rendered = remove_namelists(rendered, {"ions", "cell"})
    if spin_orbit:
        rendered = _ensure_system_flags(
            rendered,
            {
                "noncolin": ".true.",
                "lspinorb": ".true.",
            },
        )
    else:
        rendered = remove_namelist_keys(rendered, "system", {"noncolin", "lspinorb"})
    if ecutwfc is not None:
        rendered = set_namelist_values(
            rendered,
            "system",
            {
                "ecutwfc": format_qe_number(float(ecutwfc)),
                "ecutrho": format_qe_number(float(ecutwfc) * 8.0),
            },
        )
    rendered = set_k_points_automatic(rendered, [int(value) for value in kmesh[:3]])

    if pseudo_dir.is_dir():
        pseudo_map = discover_pseudo_files(
            pseudo_dir,
            prefer_full_relativistic=spin_orbit,
            prefer_non_full_relativistic=not spin_orbit,
        )
        rendered = rewrite_atomic_species_pseudos(rendered, pseudo_map, pseudo_dir=pseudo_dir)
        if spin_orbit:
            species = atomic_species_entries(rendered)
            validate_full_relativistic_pseudos(species, pseudo_dir)
    elif spin_orbit:
        raise FileNotFoundError(
            f"Spin-orbit coupling requires full-relativistic pseudopotentials in {pseudo_dir}"
        )

    return rendered


def remove_namelists(text: str, namelists: set[str]) -> str:
    lines = text.splitlines()
    targets = {f"&{name}".lower() for name in namelists}
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip().lower()
        if any(stripped.startswith(target) for target in targets):
            idx += 1
            while idx < len(lines):
                if lines[idx].strip().startswith("/"):
                    idx += 1
                    break
                idx += 1
            continue
        kept.append(lines[idx])
        idx += 1
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(kept) + suffix


def ensure_relaxation_namelists(text: str) -> str:
    """Insert the standard QE relax namelists before ATOMIC_SPECIES."""

    cleaned = remove_namelists(text, {"ions", "cell"})
    lines = cleaned.splitlines()
    insert_idx = relaxation_namelist_insert_index(lines)
    block = [
        "",
        *RELAXATION_IONS_NAMELIST,
        "",
        *RELAXATION_CELL_NAMELIST,
        "",
    ]

    while insert_idx < len(lines) and not lines[insert_idx].strip():
        del lines[insert_idx]
    lines[insert_idx:insert_idx] = block

    suffix = "\n" if cleaned.endswith("\n") else ""
    return "\n".join(lines) + suffix


def normalize_qe_fortran_number(value: Any, *, default: str | None = None, label: str = "value") -> str:
    """Normalize a user-entered number to compact QE Fortran ``D`` notation."""

    text = str(value if value is not None else "").strip()
    if not text:
        if default is not None:
            return default
        raise ValueError(f"{label} must be a number.")

    exponent_match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eEdD]([+-]?\d+)", text)
    if exponent_match:
        mantissa = exponent_match.group(1)
        if mantissa.startswith("."):
            mantissa = "0" + mantissa
        elif mantissa.startswith("-."):
            mantissa = "-0" + mantissa[1:]
        exponent = int(exponent_match.group(2))
        try:
            float(f"{mantissa}e{exponent}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a number.") from exc
        return f"{mantissa}D{exponent}"

    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = float(numerator.strip()) / float(denominator.strip())
        else:
            number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    except ZeroDivisionError as exc:
        raise ValueError(f"{label} must be a number.") from exc

    compact = f"{number:.12g}"
    if "e" in compact or "E" in compact:
        mantissa, exponent = re.split("[eE]", compact, maxsplit=1)
        exponent_value = int(exponent)
        return f"{mantissa}D{exponent_value}"
    return f"{compact}D0"


def apply_relaxation_pressure_options(text: str, options: Mapping[str, Any]) -> str:
    """Apply workflow pressure settings to the relaxation ``&CELL`` namelist."""

    pressure = normalize_qe_fortran_number(
        options.get("relax_pressure"),
        default=DEFAULT_RELAX_PRESS,
        label="pressure",
    )
    pressure_conv_thr = normalize_qe_fortran_number(
        options.get("relax_pressure_conv_thr"),
        default=DEFAULT_RELAX_PRESS_CONV_THR,
        label="pressure convergence threshold",
    )
    return set_namelist_values(
        text,
        "cell",
        {
            "press": pressure,
            "press_conv_thr": pressure_conv_thr,
        },
    )


def normalize_relaxation_parameters(
    options: Mapping[str, Any] | None,
    *,
    legacy_conversion_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize workflow relaxation parameters from GUI/CLI/manifest sources."""

    raw = dict(options or {})
    legacy = dict(legacy_conversion_options or {})

    nstep = normalize_optional_positive_int_text(raw.get("nstep"), "nstep")
    electron_maxstep = normalize_optional_positive_int_text(
        raw.get("electron_maxstep", DEFAULT_RELAX_ELECTRON_MAXSTEP),
        "electron_maxstep",
    )
    if not electron_maxstep:
        electron_maxstep = DEFAULT_RELAX_ELECTRON_MAXSTEP

    diagonalization = str(raw.get("diagonalization") or DEFAULT_RELAX_DIAGONALIZATION).strip().lower()
    if diagonalization not in RELAX_DIAGONALIZATION_VALUES:
        diagonalization = DEFAULT_RELAX_DIAGONALIZATION

    ion_dynamics = str(raw.get("ion_dynamics") or DEFAULT_RELAX_ION_DYNAMICS).strip().lower()
    if ion_dynamics not in RELAX_ION_DYNAMICS_VALUES:
        ion_dynamics = DEFAULT_RELAX_ION_DYNAMICS

    requested_cell = str(raw.get("cell_dynamics") or "").strip().lower()
    if ion_dynamics == "bfgs":
        cell_dynamics = "bfgs"
    else:
        cell_dynamics = requested_cell if requested_cell in {"damp-pr", "damp-w"} else "damp-pr"

    pressure_source = raw.get("relax_pressure", legacy.get("relax_pressure"))
    pressure_conv_source = raw.get("relax_pressure_conv_thr", legacy.get("relax_pressure_conv_thr"))
    pressure = normalize_qe_fortran_number(
        pressure_source,
        default=DEFAULT_RELAX_PRESS,
        label="pressure",
    )
    pressure_conv_thr = normalize_qe_fortran_number(
        pressure_conv_source,
        default=DEFAULT_RELAX_PRESS_CONV_THR,
        label="pressure convergence threshold",
    )

    return {
        "nstep": nstep,
        "electron_maxstep": electron_maxstep,
        "diagonalization": diagonalization,
        "ion_dynamics": ion_dynamics,
        "cell_dynamics": cell_dynamics,
        "convert_2d_structure": _truthy_option(raw.get("convert_2d_structure")),
        "use_ibrav_auto": _truthy_option(raw.get("use_ibrav_auto") or raw.get("ibrav_auto")),
        "relax_pressure": pressure,
        "relax_pressure_conv_thr": pressure_conv_thr,
    }


def normalize_optional_positive_int_text(value: Any, label: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return str(number)


def apply_relaxation_parameters(
    text: str,
    parameters: Mapping[str, Any] | None,
) -> str:
    """Apply workflow relaxation parameters to ``pw.x`` relax/vc-relax input."""

    options = normalize_relaxation_parameters(parameters)
    rendered = text
    control_values: dict[str, str] = {}
    if str(options.get("nstep") or "").strip():
        control_values["nstep"] = str(options["nstep"])
    if control_values:
        rendered = set_namelist_values(rendered, "control", control_values)
    rendered = set_namelist_values(
        rendered,
        "electrons",
        {
            "electron_maxstep": str(options["electron_maxstep"]),
            "diagonalization": qe_quote(str(options["diagonalization"])),
        },
    )
    rendered = set_namelist_values(
        rendered,
        "ions",
        {"ion_dynamics": qe_quote(str(options["ion_dynamics"]))},
    )
    rendered = set_namelist_values(
        rendered,
        "cell",
        {
            "cell_dofree": qe_quote("2Dxy") if bool(options.get("convert_2d_structure")) else qe_quote("ibrav"),
            "cell_dynamics": qe_quote(str(options["cell_dynamics"])),
            "press": str(options["relax_pressure"]),
            "press_conv_thr": str(options["relax_pressure_conv_thr"]),
        },
    )
    if bool(options.get("use_ibrav_auto")):
        rendered, _warning = apply_ibrav_auto_to_qe_input(rendered)
    else:
        rendered = force_qe_input_ibrav_zero(rendered)
    return rendered


def force_qe_input_ibrav_zero(text: str) -> str:
    """Keep the workflow default explicit-cell representation."""

    if extract_cell_vectors(text) is None:
        return text
    return set_namelist_values(text, "system", {"ibrav": "0"})


def apply_ibrav_auto_to_qe_input(text: str) -> tuple[str, str | None]:
    """Try to convert an explicit-cell QE input to nonzero ``ibrav``.

    Returns ``(updated_text, warning)``.  On any unsafe or unsupported case, the
    input is returned as ``ibrav = 0`` with ``CELL_PARAMETERS`` preserved.
    """

    try:
        atoms = atoms_from_qe_explicit_cell_input(text)
        if atoms is None:
            return force_qe_input_ibrav_zero(text), "Could not parse explicit-cell QE structure for ibrav auto."

        from qeinputgeneratorscrapper.make_pw_input_from_cif import (  # type: ignore
            DEFAULT_SYMPREC,
            IBRAV_AUTO,
            fractional_positions_for_qe_lattice,
            qe_lattice_float,
            resolve_qe_lattice,
        )

        qe_lattice, warning = resolve_qe_lattice(atoms, ibrav_request=IBRAV_AUTO, symprec=DEFAULT_SYMPREC)
        if qe_lattice is None:
            return force_qe_input_ibrav_zero(text), warning or "Could not detect a safe nonzero ibrav."

        scaled_positions = fractional_positions_for_qe_lattice(atoms, qe_lattice)
        symbols = atoms.get_chemical_symbols()
        atomic_lines = []
        for symbol, position in zip(symbols, scaled_positions):
            x, y, z = [float(value) % 1.0 for value in position[:3]]
            atomic_lines.append(f"{symbol:<2} {x:20.10f}{y:19.10f}{z:19.10f}")

        rendered = replace_qe_card(text, "ATOMIC_POSITIONS", "ATOMIC_POSITIONS crystal", atomic_lines)
        rendered = remove_qe_card(rendered, "CELL_PARAMETERS")
        rendered = remove_namelist_keys(rendered, "system", RELAX_SYSTEM_LATTICE_KEYS)
        system_values = {"ibrav": str(qe_lattice.ibrav), "nosym": ".true."}
        for name, value in qe_lattice.parameters:
            system_values[str(name)] = qe_lattice_float(float(value))
        rendered = set_namelist_values(rendered, "system", system_values)
        return rendered, warning
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for workflow robustness
        return force_qe_input_ibrav_zero(text), f"Could not apply ibrav auto: {exc}"


def atoms_from_qe_explicit_cell_input(text: str) -> Any | None:
    cell = extract_cell_vectors(text)
    if cell is None:
        return None
    atomic = extract_qe_atomic_positions(text)
    if atomic is None:
        return None

    header, rows = atomic
    mode = qe_card_unit(header, default="alat")
    symbols: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            return None
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            return None
        symbols.append(parts[0])

    try:
        from ase import Atoms  # type: ignore
    except Exception:
        return None

    if mode.startswith("crystal"):
        return Atoms(symbols=symbols, scaled_positions=coords, cell=cell, pbc=True)

    if mode in {"angstrom", "ang"}:
        return Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)

    scale = qe_position_scale_to_angstrom(text, mode)
    if scale is None:
        return None
    positions = [[component * scale for component in coord] for coord in coords]
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def qe_position_scale_to_angstrom(text: str, mode: str) -> float | None:
    unit = str(mode or "").strip().lower()
    if unit in {"bohr", "a.u.", "au"}:
        return BOHR_TO_ANGSTROM
    if unit in {"alat", ""}:
        system = parse_namelist_values(text, "system")
        alat_bohr = parse_qe_float(system.get("celldm(1)") or system.get("celldm1"))
        if alat_bohr is not None:
            return alat_bohr * BOHR_TO_ANGSTROM
        alat_angstrom = parse_qe_float(system.get("a"))
        if alat_angstrom is not None:
            return alat_angstrom
    return None


def extract_qe_atomic_positions(text: str) -> tuple[str, list[str]] | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith("ATOMIC_POSITIONS"):
            continue
        rows: list[str] = []
        end_idx = idx + 1
        while end_idx < len(lines):
            candidate = lines[end_idx].strip()
            if not candidate:
                break
            upper = candidate.upper()
            if upper.startswith(("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS")):
                break
            if candidate.startswith("&") or candidate.startswith("/"):
                break
            rows.append(lines[end_idx])
            end_idx += 1
        return line, rows
    return None


def qe_card_unit(header: str, *, default: str = "") -> str:
    stripped = str(header or "").strip()
    match = re.search(r"\(([^)]+)\)", stripped)
    if match:
        return match.group(1).strip().lower()
    parts = stripped.split(None, 1)
    if len(parts) > 1:
        return parts[1].strip().lower().strip("{}")
    return default


def remove_qe_card(text: str, card_name: str) -> str:
    lines = text.splitlines()
    target = card_name.strip().upper()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith(target):
            start_idx = idx
            break
    if start_idx is None:
        return text
    end_idx = start_idx + 1
    while end_idx < len(lines):
        stripped = lines[end_idx].strip()
        if not stripped:
            break
        upper = stripped.upper()
        if upper.startswith(("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS")):
            break
        if stripped.startswith("&") or stripped.startswith("/"):
            break
        end_idx += 1
    suffix = "\n" if text.endswith("\n") else ""
    del lines[start_idx:end_idx]
    return "\n".join(lines) + suffix


def relaxation_namelist_insert_index(lines: Sequence[str]) -> int:
    """Return the place where &IONS/&CELL should sit in a relax input."""

    _, electrons_end = find_namelist_bounds(lines, "electrons")
    if electrons_end is not None:
        return electrons_end + 1

    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_SPECIES"):
            return idx
    return len(lines)


def render_initial_relaxation_input(
    template_text: str,
    *,
    calculation: str,
    workflow_root: str | Path,
    child_workdir: str | Path,
    child_id: str,
    conversion_options: Mapping[str, Any] | None = None,
    relaxation_options: Mapping[str, Any] | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the rough initial relax/vc-relax input from QE_input.template."""

    calc = normalize_relaxation_calculation(calculation)
    root = Path(workflow_root).expanduser().resolve()
    child_dir = root / Path(child_workdir)
    pseudo_dir = root / SOURCE_DIR / PSEUDO_DIR_NAME
    relative_pseudo_dir = relative_path_for_qe(child_dir, pseudo_dir)
    options = dict(conversion_options or {})
    spin_orbit = _spin_orbit_enabled(options) if spin_orbit_required is None else bool(spin_orbit_required)
    settings = protocol_numerical_settings("fast")
    nat = max(1, parse_qe_int(parse_namelist_values(template_text, "system").get("nat"), default=1))
    magnetism = _resolve_workflow_magnetism(options)

    rendered = set_namelist_values(
        template_text,
        "control",
        {
            "calculation": qe_quote(calc),
            "etot_conv_thr": format_qe_fortran_float(nat * settings["etot_per_atom"]),
            "forc_conv_thr": format_qe_fortran_float(settings["forc_conv_thr"]),
            "outdir": qe_quote("./out/"),
            "prefix": qe_quote(child_id),
            "pseudo_dir": qe_quote(relative_pseudo_dir),
            "tprnfor": ".true.",
            "tstress": ".true.",
        },
    )
    if spin_orbit:
        rendered = _ensure_system_flags(
            rendered,
            {
                "noncolin": ".true.",
                "lspinorb": ".true.",
            },
        )
    else:
        rendered = remove_namelist_keys(rendered, "system", {"noncolin", "lspinorb"})

    manual_ecutwfc = manual_ecutwfc_from_options(options)
    ecutwfc = manual_ecutwfc if manual_ecutwfc is not None else settings["ecutwfc"]
    ecutrho = ecutwfc * 8.0 if manual_ecutwfc is not None else settings["ecutrho"]
    system_values = {
        "ecutwfc": format_qe_number(ecutwfc),
        "ecutrho": format_qe_number(ecutrho),
    }
    if magnetism == "non-magnetic-fixed":
        rendered = remove_namelist_keys(rendered, "system", {"degauss", "smearing"})
        system_values["occupations"] = qe_quote("fixed")
    else:
        system_values.update(
            {
                "degauss": format_qe_fortran_float(settings["degauss"]),
                "occupations": qe_quote("smearing"),
                "smearing": qe_quote("cold"),
            }
        )
    rendered = set_namelist_values(rendered, "system", system_values)
    rendered = set_namelist_values(
        rendered,
        "electrons",
        {
            "conv_thr": format_qe_fortran_float(nat * settings["conv_per_atom"]),
            "electron_maxstep": "80",
            "mixing_beta": format_qe_fortran_float(0.4),
        },
    )

    kmesh = kmesh_from_qe_input(rendered, settings["kpoint_spacing"])
    if kmesh:
        rendered = set_k_points_automatic(rendered, kmesh)

    if pseudo_dir.is_dir():
        pseudo_map = discover_pseudo_files(
            pseudo_dir,
            prefer_full_relativistic=spin_orbit,
            prefer_non_full_relativistic=not spin_orbit,
        )
        rendered = rewrite_atomic_species_pseudos(rendered, pseudo_map, pseudo_dir=pseudo_dir)
        if spin_orbit:
            species = atomic_species_entries(rendered)
            validate_full_relativistic_pseudos(species, pseudo_dir)
    elif spin_orbit:
        raise FileNotFoundError(
            f"Spin-orbit coupling requires full-relativistic pseudopotentials in {pseudo_dir}"
        )

    rendered = ensure_relaxation_namelists(rendered)
    parameters = normalize_relaxation_parameters(relaxation_options, legacy_conversion_options=options)
    parameters["convert_2d_structure"] = _truthy_option(options.get("convert_2d_structure"))
    return apply_relaxation_parameters(rendered, parameters)


def render_final_relaxation_input(
    template_text: str,
    *,
    calculation: str,
    workflow_root: str | Path,
    child_workdir: str | Path,
    child_id: str,
    ecutwfc: int | float | None = None,
    kmesh: Sequence[int] | None = None,
    conversion_options: Mapping[str, Any] | None = None,
    relaxation_options: Mapping[str, Any] | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the final relax/vc-relax input using converged workflow values."""

    calc = normalize_relaxation_calculation(calculation)
    root = Path(workflow_root).expanduser().resolve()
    child_dir = root / Path(child_workdir)
    pseudo_dir = root / SOURCE_DIR / PSEUDO_DIR_NAME
    relative_pseudo_dir = relative_path_for_qe(child_dir, pseudo_dir)
    options = dict(conversion_options or {})
    spin_orbit = _spin_orbit_enabled(options) if spin_orbit_required is None else bool(spin_orbit_required)

    rendered = set_namelist_values(
        template_text,
        "control",
        {
            "calculation": qe_quote(calc),
            "outdir": qe_quote("./out/"),
            "prefix": qe_quote(child_id),
            "pseudo_dir": qe_quote(relative_pseudo_dir),
            "tprnfor": ".true.",
            "tstress": ".true.",
        },
    )

    if ecutwfc is not None:
        rendered = set_namelist_values(
            rendered,
            "system",
            {
                "ecutwfc": format_qe_number(float(ecutwfc)),
                "ecutrho": format_qe_number(float(ecutwfc) * 8.0),
            },
        )

    if kmesh is not None:
        rendered = set_k_points_automatic(rendered, [int(value) for value in kmesh[:3]])

    if spin_orbit:
        rendered = _ensure_system_flags(
            rendered,
            {
                "noncolin": ".true.",
                "lspinorb": ".true.",
            },
        )
    else:
        rendered = remove_namelist_keys(rendered, "system", {"noncolin", "lspinorb"})

    if pseudo_dir.is_dir():
        pseudo_map = discover_pseudo_files(
            pseudo_dir,
            prefer_full_relativistic=spin_orbit,
            prefer_non_full_relativistic=not spin_orbit,
        )
        rendered = rewrite_atomic_species_pseudos(rendered, pseudo_map, pseudo_dir=pseudo_dir)
        if spin_orbit:
            species = atomic_species_entries(rendered)
            validate_full_relativistic_pseudos(species, pseudo_dir)
    elif spin_orbit:
        raise FileNotFoundError(
            f"Spin-orbit coupling requires full-relativistic pseudopotentials in {pseudo_dir}"
        )

    rendered = ensure_relaxation_namelists(rendered)
    parameters = normalize_relaxation_parameters(relaxation_options, legacy_conversion_options=options)
    parameters["convert_2d_structure"] = _truthy_option(options.get("convert_2d_structure"))
    return apply_relaxation_parameters(rendered, parameters)


def render_band_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float = 0.10,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
    outdir: str = "./out/",
) -> str:
    """Render the SCF stage used by band/fat-band workflows."""

    root = Path(workflow_root).expanduser().resolve()
    child_dir = root / Path(child_workdir)
    pseudo_dir = root / SOURCE_DIR / PSEUDO_DIR_NAME
    relative_pseudo_dir = relative_path_for_qe(child_dir, pseudo_dir)
    spin_orbit = bool(spin_orbit_required)

    rendered = set_namelist_values(
        template_text,
        "control",
        {
            "calculation": qe_quote("scf"),
            "outdir": qe_quote(outdir or "./out/"),
            "prefix": qe_quote(prefix),
            "pseudo_dir": qe_quote(relative_pseudo_dir),
            "tprnfor": ".true.",
            "tstress": ".true.",
        },
    )
    rendered = remove_namelists(rendered, {"ions", "cell"})

    system_values: dict[str, str] = {}
    if ecutwfc is not None:
        system_values["ecutwfc"] = format_qe_number(float(ecutwfc))
        system_values["ecutrho"] = format_qe_number(float(ecutwfc) * 8.0)
    if system_values:
        rendered = set_namelist_values(rendered, "system", system_values)

    if spin_orbit:
        rendered = _ensure_system_flags(
            rendered,
            {
                "noncolin": ".true.",
                "lspinorb": ".true.",
            },
        )
    else:
        rendered = remove_namelist_keys(rendered, "system", {"noncolin", "lspinorb"})

    if kmesh is not None and len(kmesh) >= 3:
        rendered = set_k_points_automatic(rendered, [int(value) for value in kmesh[:3]])
    elif kpoint_spacing > 0:
        kmesh = kmesh_from_qe_input(rendered, float(kpoint_spacing))
        if kmesh:
            rendered = set_k_points_automatic(rendered, kmesh)

    if pseudo_dir.is_dir():
        pseudo_map = discover_pseudo_files(
            pseudo_dir,
            prefer_full_relativistic=spin_orbit,
            prefer_non_full_relativistic=not spin_orbit,
        )
        rendered = rewrite_atomic_species_pseudos(rendered, pseudo_map, pseudo_dir=pseudo_dir)
        if spin_orbit:
            species = atomic_species_entries(rendered)
            validate_full_relativistic_pseudos(species, pseudo_dir)
    elif spin_orbit:
        raise FileNotFoundError(
            f"Spin-orbit coupling requires full-relativistic pseudopotentials in {pseudo_dir}"
        )

    return rendered


def render_fermi_surface_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float = 0.10,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
    outdir: str = "./out/",
) -> str:
    """Render the SCF stage used by the Fermi-surface workflow."""

    rendered = render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=kpoint_spacing,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
        outdir=outdir,
    )
    rendered = set_namelist_values(
        rendered,
        "system",
        {
            "occupations": qe_quote("tetrahedra_opt"),
        },
    )
    return remove_namelist_keys(rendered, "system", {"smearing", "degauss"})


def render_charge_density_pp_input(
    *,
    prefix: str,
    outdir: str = "./out/",
    fileout: str = "{prefix}_rho.cube",
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
) -> str:
    """Render pp.x input for charge-density cube export."""

    safe_prefix = str(prefix or "charge_density").strip() or "charge_density"
    resolved_fileout = str(fileout or "{prefix}_rho.cube").strip() or "{prefix}_rho.cube"
    resolved_fileout = resolved_fileout.replace("{prefix}", safe_prefix)
    nx_value = max(1, int(nx or 64))
    ny_value = max(1, int(ny or 64))
    nz_value = max(1, int(nz or 64))
    lines = [
        "&INPUTPP",
        f"     prefix={qe_quote(safe_prefix)},",
        f"     outdir={qe_quote(outdir or './out/')},",
        "     plot_num=8,",
        "/",
        "&PLOT",
        "iflag=3,",
        "output_format=6,",
        f"fileout={qe_quote(resolved_fileout)},",
        f"nx={nx_value},ny={ny_value},nz={nz_value},",
        "/",
        "",
    ]
    return "\n".join(lines)


def render_dos_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float = 0.30,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the SCF stage used by DOS/PDOS workflows."""

    rendered = render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=kpoint_spacing,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )
    rendered = remove_namelist_keys(rendered, "system", {"smearing", "degauss"})
    rendered = set_namelist_values(
        rendered,
        "system",
        {
            "occupations": qe_quote("tetrahedra_opt"),
        },
    )
    return rendered


def render_dos_nscf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float = 0.10,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the NSCF stage used by DOS/PDOS workflows."""

    rendered = render_dos_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=kpoint_spacing,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )
    return set_namelist_values(rendered, "control", {"calculation": qe_quote("nscf")})


def render_tetra_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render a tetrahedra-occupation SCF stage for phonon EPC helpers."""

    rendered = render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=0.0,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )
    rendered = remove_namelist_keys(rendered, "system", {"smearing", "degauss"})
    return set_namelist_values(rendered, "system", {"occupations": qe_quote("tetrahedra_opt")})


def render_phonon_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float = 0.0,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the SCF stage used by phonon workflows."""

    return render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=kpoint_spacing,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )


def render_elastic_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str = "elastic",
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the SCF stage used by thermo_pw elastic-property workflows."""

    return render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=0.0,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
        outdir="./",
    )


def render_thermo_control(options: Mapping[str, Any] | None = None) -> str:
    """Render thermo_pw elastic-control input."""

    normalized = normalize_elastic_options(options or {})
    lines = [
        "&INPUT_THERMO",
        f"  what = {qe_quote(str(normalized['what']))},",
        f"  elastic_algorithm = {qe_quote(str(normalized['elastic_algorithm']))},",
        f"  ngeo_strain = {int(normalized['ngeo_strain'])},",
        f"  frozen_ions = {'.true.' if bool(normalized['frozen_ions']) else '.false.'},",
        "  find_ibrav = .true.,",
        "/",
        "",
    ]
    return "\n".join(lines)


def render_epc_scf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kmesh: Sequence[int],
    la2f: bool = False,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render an example03-style metallic SCF input for EPC workflows."""

    rendered = render_phonon_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kmesh=kmesh,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )
    rendered = set_namelist_values(
        rendered,
        "system",
        {
            "occupations": qe_quote("smearing"),
            "smearing": qe_quote("marzari-vanderbilt"),
            "degauss": "0.05",
        },
    )
    if la2f:
        rendered = set_namelist_values(rendered, "system", {"la2F": ".true."})
    else:
        rendered = remove_namelist_keys(rendered, "system", {"la2f"})
    return rendered


def render_phonon_input(
    template_text: str,
    *,
    prefix: str,
    outdir: str = "./out/",
    tr2_ph: str = "1.0d-12",
    alpha_mix: str | float = "0.7",
    fildyn: str = "phonon.dyn",
    q_point: Sequence[float] | None = None,
    ldisp: bool = False,
    nq_grid: Sequence[int] | None = None,
    trans: bool = True,
    epsil: bool = False,
    nmix_ph: int = 4,
    verbosity: str = "default",
    fildvscf: str = "",
    fildrho: str = "",
    lshift_q: bool = False,
    electron_phonon: str = "",
    el_ph_sigma: str = "",
    el_ph_nsigma: int | None = None,
    elph_nk_grid: Sequence[int] | None = None,
    inputa2f_nfreq: int | None = None,
) -> str:
    """Render a ph.x input for either a single q-point or an ldisp q-grid."""

    entries: list[tuple[str, str]] = [
        ("tr2_ph", str(tr2_ph or "1.0d-12")),
        ("alpha_mix(1)", str(alpha_mix if str(alpha_mix).strip() else "0.7")),
        ("prefix", qe_quote(prefix)),
        ("outdir", qe_quote(outdir or "./out/")),
        ("fildyn", qe_quote(fildyn or "phonon.dyn")),
        ("trans", ".true." if trans else ".false."),
        ("nmix_ph", str(max(1, int(nmix_ph or 4)))),
        ("verbosity", qe_quote(verbosity or "default")),
    ]
    if fildvscf:
        entries.append(("fildvscf", qe_quote(fildvscf)))
    if fildrho:
        entries.append(("fildrho", qe_quote(fildrho)))
    if lshift_q:
        entries.append(("lshift_q", ".true."))
    if electron_phonon:
        entries.append(("electron_phonon", qe_quote(electron_phonon)))
    if el_ph_sigma:
        entries.append(("el_ph_sigma", str(el_ph_sigma)))
    if el_ph_nsigma is not None:
        entries.append(("el_ph_nsigma", str(max(1, int(el_ph_nsigma)))))
    if elph_nk_grid is not None:
        nk_values = normalize_phonon_nq_grid(elph_nk_grid)
        entries.extend(
            [
                ("nk1", str(nk_values[0])),
                ("nk2", str(nk_values[1])),
                ("nk3", str(nk_values[2])),
            ]
        )
    if epsil:
        entries.append(("epsil", ".true."))
    if ldisp:
        nq_values = normalize_phonon_nq_grid(nq_grid)
        entries.extend(
            [
                ("ldisp", ".true."),
                ("nq1", str(nq_values[0])),
                ("nq2", str(nq_values[1])),
                ("nq3", str(nq_values[2])),
            ]
        )

    width = max(len(key) for key, _value in entries)
    lines = ["&inputph"]
    for key, value in entries:
        lines.append(f"  {key.ljust(width)} = {value}")
    lines.append("/")

    if not ldisp:
        q_values = list(q_point or [0.0, 0.0, 0.0])
        while len(q_values) < 3:
            q_values.append(0.0)
        lines.append(" ".join(format_phonon_q_component(value) for value in q_values[:3]))
    if inputa2f_nfreq is not None:
        lines.extend(["&INPUTa2F", f"  nfreq = {max(1, int(inputa2f_nfreq))}", "/"])
    lines.append("")
    return "\n".join(lines)


def render_q2r_input(
    *,
    fildyn: str = "phonon.dyn",
    flfrc: str = "phonon.fc",
    zasr: str = "crystal",
    la2f: bool = False,
    lshift_q: bool = False,
) -> str:
    """Render the q2r.x force-constants input for a phonon dispersion workflow."""

    entries = [
        ("fildyn", qe_quote(fildyn or "phonon.dyn")),
        ("zasr", qe_quote(zasr or "crystal")),
        ("flfrc", qe_quote(flfrc or "phonon.fc")),
    ]
    if la2f:
        entries.append(("la2F", ".true."))
    if lshift_q:
        entries.append(("lshift_q", ".true."))
    width = max(len(key) for key, _value in entries)
    lines = ["&input"]
    for key, value in entries:
        lines.append(f"  {key.ljust(width)} = {value}")
    lines.extend(["/", ""])
    return "\n".join(lines)


def render_matdyn_input(
    template_text: str,
    *,
    flfrc: str = "phonon.fc",
    flfrq: str = "phonon.freq.band",
    flvec: str = "",
    q_path_lines: Sequence[str] | None = None,
    asr: str = "crystal",
    q_in_band_form: bool = True,
    la2f: bool = False,
) -> str:
    """Render the matdyn.x phonon-dispersion input."""

    entries: list[tuple[str, str]] = [
        ("asr", qe_quote(asr or "crystal")),
        ("flfrc", qe_quote(flfrc or "phonon.fc")),
        ("flfrq", qe_quote(flfrq or "phonon.freq.band")),
        ("q_in_band_form", ".true." if q_in_band_form else ".false."),
    ]
    if flvec:
        entries.append(("flvec", qe_quote(flvec)))
    if la2f:
        entries.append(("la2F", ".true."))
    width = max(len(key) for key, _value in entries)
    lines = ["&input"]
    for key, value in entries:
        lines.append(f"  {key.ljust(width)} = {value}")
    lines.append("/")
    lines.extend(normalize_matdyn_q_path(q_path_lines))
    lines.append("")
    return "\n".join(lines)


def render_phonon_dos_input(
    template_text: str,
    *,
    flfrc: str = "phonon.fc",
    fldos: str = "phonon.phdos",
    flvec: str = "",
    nk_grid: Sequence[int] | None = None,
    asr: str = "crystal",
    flfrq: str = "phonon.freq.dos",
    ndos: int | None = None,
    la2f: bool = False,
) -> str:
    """Render the matdyn.x phonon DOS input."""

    nk_values = normalize_phonon_nq_grid(nk_grid)
    entries: list[tuple[str, str]] = [
        ("asr", qe_quote(asr or "crystal")),
        ("dos", ".true."),
        ("flfrc", qe_quote(flfrc or "phonon.fc")),
        ("fldos", qe_quote(fldos or "phonon.phdos")),
        ("nk1", str(nk_values[0])),
        ("nk2", str(nk_values[1])),
        ("nk3", str(nk_values[2])),
    ]
    if flvec:
        entries.append(("flvec", qe_quote(flvec)))
    if flfrq:
        entries.append(("flfrq", qe_quote(flfrq)))
    if ndos is not None:
        entries.append(("ndos", str(max(1, int(ndos)))))
    if la2f:
        entries.append(("la2F", ".true."))
    width = max(len(key) for key, _value in entries)
    lines = ["&input"]
    for key, value in entries:
        lines.append(f"  {key.ljust(width)} = {value}")
    lines.extend(["/", ""])
    return "\n".join(lines)


def render_lambda_input(
    *,
    emax: str = "10",
    degauss: str = "0.1",
    smearing_method: int = 1,
    mu: str = "0.10",
    q_points: Sequence[Sequence[Any]] | None = None,
) -> str:
    """Render example03-style lambda.x input."""

    points = normalize_epc_lambda_q_points(q_points)
    lines = [
        f"{emax}  {degauss}  {int(smearing_method)}    ! emax, degauss, smearing method",
        f"    {len(points)}          ! Number of q-points for which EPC is calculated,",
    ]
    for point in points:
        qx, qy, qz, weight = point
        lines.append(f"{qx: .7f} {qy: .7f} {qz: .7f}  {weight: .2f}")
    for index in range(1, len(points) + 1):
        lines.append(f"elph_dir/elph.inp_lambda.{index}")
    lines.append(f"{mu}                     ! mu for modified Allen-Dynes formula")
    lines.append("")
    return "\n".join(lines)


def render_band_calculation_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoints_crystal_b: Sequence[str],
    nbnd: int | None = None,
    ecutwfc: int | float | None = None,
    spin_orbit_required: bool | None = None,
) -> str:
    """Render the PW bands stage for band/fat-band workflows."""

    rendered = render_band_scf_input(
        template_text,
        workflow_root=workflow_root,
        child_workdir=child_workdir,
        prefix=prefix,
        kpoint_spacing=0.0,
        ecutwfc=ecutwfc,
        spin_orbit_required=spin_orbit_required,
    )
    rendered = set_namelist_values(rendered, "control", {"calculation": qe_quote("bands")})

    if nbnd is not None and int(nbnd) > 0:
        rendered = set_namelist_values(rendered, "system", {"nbnd": str(int(nbnd))})

    card_lines = [str(line).rstrip() for line in kpoints_crystal_b if str(line).strip()]
    if len(card_lines) < 2:
        raise ValueError("K_POINTS crystal_b requires at least a header and one data line.")
    header = card_lines[0]
    if not header.strip().upper().startswith("K_POINTS"):
        header = "K_POINTS crystal_b"
        data_lines = card_lines
    else:
        data_lines = card_lines[1:]
    rendered = replace_qe_card(rendered, "K_POINTS", header, data_lines)
    return rendered


def render_bandsx_input(
    *,
    prefix: str,
    outdir: str = "./out/",
    filband: str = "bands.out",
) -> str:
    """Render a bands.x input file."""

    lines = [
        "&BANDS",
        f"  prefix = {qe_quote(prefix)}",
        f"  outdir = {qe_quote(outdir)}",
        f"  filband = {qe_quote(filband)}",
        "/",
        "",
    ]
    return "\n".join(lines)


def render_projwfc_input(
    *,
    prefix: str,
    outdir: str = "./out/",
    lsym: bool | None = None,
    filproj: str = "",
) -> str:
    """Render a projwfc.x input file."""

    lines = [
        "&PROJWFC",
        f"    prefix = {qe_quote(prefix)}",
        f"    outdir = {qe_quote(outdir)}",
    ]
    if lsym is not None:
        lines.append(f"    lsym = {'.true.' if lsym else '.false.'}")
    if str(filproj or "").strip():
        lines.append(f"    filproj = {qe_quote(str(filproj).strip())}")
    lines.extend(["/", ""])
    return "\n".join(lines)


def render_qe_input_with_structure(template_text: str, structure_data: Mapping[str, Any]) -> str:
    """Return QE input text with ATOMIC_POSITIONS/CELL_PARAMETERS replaced."""

    rendered = str(template_text)
    atomic = structure_data.get("atomic_positions") if isinstance(structure_data, Mapping) else None
    cell = structure_data.get("cell_parameters") if isinstance(structure_data, Mapping) else None
    species_card = structure_data.get("atomic_species") if isinstance(structure_data, Mapping) else None

    if isinstance(species_card, Mapping):
        header = str(species_card.get("header") or "ATOMIC_SPECIES")
        lines = [str(line) for line in species_card.get("lines") or []]
        rendered = replace_qe_card(rendered, "ATOMIC_SPECIES", header, lines)
        if lines:
            rendered = set_namelist_values(rendered, "system", {"ntyp": str(len(lines))})

    if isinstance(atomic, Mapping):
        header = str(atomic.get("header") or "ATOMIC_POSITIONS angstrom")
        lines = [str(line) for line in atomic.get("lines") or []]
        rendered = replace_qe_card(rendered, "ATOMIC_POSITIONS", header, lines)
        rendered = set_namelist_values(rendered, "system", {"nat": str(len(lines))})

        species = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in species:
                species.append(parts[0])
        if species:
            rendered = set_namelist_values(rendered, "system", {"ntyp": str(len(species))})

    if isinstance(cell, Mapping):
        header = str(cell.get("header") or "CELL_PARAMETERS angstrom")
        lines = [str(line) for line in cell.get("lines") or []]
        rendered = replace_qe_card(rendered, "CELL_PARAMETERS", header, lines)

    return rendered


def replace_qe_card(text: str, card_name: str, header: str, data_lines: Sequence[str]) -> str:
    lines = text.splitlines()
    target = card_name.strip().upper()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith(target):
            start_idx = idx
            break
    if start_idx is None:
        suffix = "\n" if text.endswith("\n") else ""
        block = ["", str(header), *[str(line) for line in data_lines]]
        return "\n".join([*lines, *block]) + suffix

    end_idx = start_idx + 1
    while end_idx < len(lines):
        stripped = lines[end_idx].strip()
        if not stripped:
            break
        upper = stripped.upper()
        if upper.startswith(
            (
                "ATOMIC_SPECIES",
                "ATOMIC_POSITIONS",
                "CELL_PARAMETERS",
                "K_POINTS",
                "CONSTRAINTS",
                "OCCUPATIONS",
            )
        ):
            break
        if stripped.startswith("&") or stripped.startswith("/"):
            break
        end_idx += 1

    replacement = [str(header), *[str(line) for line in data_lines]]
    lines[start_idx:end_idx] = replacement
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def normalize_relaxation_calculation(calculation: str) -> str:
    calc = str(calculation or "").strip().lower().replace("_", "-")
    if calc in {"vc-relax", "qe.vc-relax"}:
        return "vc-relax"
    return "relax"


def protocol_numerical_settings(protocol: str) -> dict[str, float]:
    key = str(protocol or "balanced").strip().lower()
    settings = PROTOCOL_NUMERICAL_SETTINGS.get(key) or PROTOCOL_NUMERICAL_SETTINGS["balanced"]
    return {name: float(value) for name, value in settings.items()}


def manual_ecutwfc_from_options(options: Mapping[str, Any] | None) -> float | None:
    if not isinstance(options, Mapping):
        return None
    value = options.get("manual_ecutwfc")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def manual_cutoff_skip_notice(ecutwfc: int | float) -> str:
    return (
        f"Cutoff convergence skipped because manual ecutwfc override "
        f"{float(ecutwfc):g} Ry was set; ecutrho will use 8x this value."
    )


def remove_namelist_keys(text: str, namelist: str, keys: set[str]) -> str:
    lines = text.splitlines()
    start_idx, end_idx = find_namelist_bounds(lines, namelist)
    if start_idx is None or end_idx is None:
        return text

    lowered = {str(key).strip().lower() for key in keys}
    kept: list[str] = []
    for idx, line in enumerate(lines):
        if start_idx < idx < end_idx:
            body = line.split("!", 1)[0]
            if "=" in body:
                key = body.split("=", 1)[0].strip().lower()
                if key in lowered:
                    continue
        kept.append(line)

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(kept) + suffix


def relative_path_for_qe(from_dir: Path, target_dir: Path) -> str:
    try:
        relative = Path(os_path_relpath(target_dir, from_dir))
    except ValueError:
        return str(target_dir)
    text = relative.as_posix()
    return text if text.endswith("/") else f"{text}/"


def os_path_relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(str(target), str(start))


def qe_input_has_spin_orbit(text: str) -> bool:
    system = parse_namelist_values(text, "system")
    return is_qe_true(system.get("lspinorb", "")) or is_qe_true(system.get("noncolin", ""))


def is_qe_true(value: str) -> bool:
    return str(value or "").strip().strip(",").strip().lower() in {".true.", "true", "t", "1"}


def set_namelist_values(text: str, namelist: str, values: Mapping[str, str]) -> str:
    lines = text.splitlines()
    start_idx, end_idx = find_namelist_bounds(lines, namelist)
    if start_idx is None or end_idx is None:
        return text

    indent = _namelist_assignment_indent(lines, start_idx + 1, end_idx)
    remaining = {str(key).lower(): str(value) for key, value in values.items()}
    for idx in range(start_idx + 1, end_idx):
        body = lines[idx].split("!", 1)[0]
        if "=" not in body:
            continue
        key = body.split("=", 1)[0].strip().lower()
        if key not in remaining:
            continue
        line_indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())] or indent
        lines[idx] = f"{line_indent}{key} = {remaining.pop(key)}"

    if remaining:
        insert_lines = [f"{indent}{key} = {value}" for key, value in remaining.items()]
        lines[end_idx:end_idx] = insert_lines

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def parse_namelist_values(text: str, namelist: str) -> dict[str, str]:
    lines = text.splitlines()
    start_idx, end_idx = find_namelist_bounds(lines, namelist)
    if start_idx is None or end_idx is None:
        return {}

    parsed: dict[str, str] = {}
    for idx in range(start_idx + 1, end_idx):
        body = lines[idx].split("!", 1)[0]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        parsed[key.strip().lower()] = value.strip().rstrip(",")
    return parsed


def find_namelist_bounds(lines: Sequence[str], namelist: str) -> tuple[int | None, int | None]:
    target = f"&{namelist}".lower()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith(target):
            start_idx = idx
            break
    if start_idx is None:
        return None, None

    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("/"):
            return start_idx, idx
    return start_idx, None


def format_qe_number(value: float) -> str:
    return f"{float(value):18.10e}".replace("e", "d")


def format_qe_fortran_float(value: float) -> str:
    return f"{float(value):18.10e}".replace("e", "d").strip()


def format_phonon_q_component(value: Any) -> str:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) < 1.0e-12:
        return f"{float(rounded):.1f}"
    formatted = f"{number:.10f}".rstrip("0").rstrip(".")
    return formatted or "0.0"


def qe_quote(value: str) -> str:
    cleaned = str(value).strip().strip("'\"")
    return f"'{cleaned}'"


def parse_qe_int(value: Any, *, default: int = 0) -> int:
    text = str(value or "").strip().strip(",")
    if not text:
        return default
    try:
        return int(float(text.replace("d", "e").replace("D", "e")))
    except (TypeError, ValueError):
        return default


def parse_qe_float(value: Any) -> float | None:
    text = str(value or "").strip().strip(",")
    if not text:
        return None
    try:
        return float(text.replace("d", "e").replace("D", "e"))
    except (TypeError, ValueError):
        return None


def kmesh_from_qe_input(text: str, spacing: float) -> tuple[int, int, int] | None:
    cell = extract_cell_vectors(text)
    if cell is None:
        return None
    reciprocal_lengths = reciprocal_lengths_2pi(cell)
    if reciprocal_lengths is None:
        return None
    mesh = []
    for length in reciprocal_lengths:
        mesh.append(max(1, int(math.ceil(length / max(float(spacing), 1.0e-12)))))
    return tuple(mesh[:3])  # type: ignore[return-value]


def extract_cell_vectors(text: str) -> list[list[float]] | None:
    lines = text.splitlines()
    system = parse_namelist_values(text, "system")
    alat_bohr = parse_qe_float(system.get("celldm(1)") or system.get("celldm1"))
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith("CELL_PARAMETERS"):
            continue
        unit = "alat"
        match = re.search(r"\(([^)]+)\)", stripped)
        if match:
            unit = match.group(1).strip().lower()
        else:
            parts = stripped.split()
            if len(parts) > 1:
                unit = parts[1].strip().lower()
        vectors: list[list[float]] = []
        for raw in lines[idx + 1 : idx + 4]:
            parts = raw.split()
            if len(parts) < 3:
                return None
            try:
                vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                return None
        if unit in {"angstrom", "ang"}:
            return vectors
        if unit in {"bohr", "a.u.", "au"}:
            return [[component * BOHR_TO_ANGSTROM for component in vector] for vector in vectors]
        if unit == "alat" and alat_bohr is not None:
            scale = alat_bohr * BOHR_TO_ANGSTROM
            return [[component * scale for component in vector] for vector in vectors]
        return None
    return None


def reciprocal_lengths_2pi(cell: Sequence[Sequence[float]]) -> tuple[float, float, float] | None:
    try:
        a1 = [float(value) for value in cell[0][:3]]
        a2 = [float(value) for value in cell[1][:3]]
        a3 = [float(value) for value in cell[2][:3]]
    except (IndexError, TypeError, ValueError):
        return None
    volume = dot(a1, cross(a2, a3))
    if abs(volume) < 1.0e-12:
        return None
    b1 = [2.0 * math.pi * value / volume for value in cross(a2, a3)]
    b2 = [2.0 * math.pi * value / volume for value in cross(a3, a1)]
    b3 = [2.0 * math.pi * value / volume for value in cross(a1, a2)]
    return vector_norm(b1), vector_norm(b2), vector_norm(b3)


def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def set_k_points_automatic(text: str, mesh: Sequence[int]) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith("K_POINTS"):
            continue
        nk1, nk2, nk3 = [max(1, int(value)) for value in mesh[:3]]
        new_header = "K_POINTS automatic"
        new_mesh = f"{nk1} {nk2} {nk3} 0 0 0"
        end_idx = idx + 1
        while end_idx + 1 < len(lines):
            candidate = lines[end_idx + 1].strip()
            if not candidate:
                break
            upper = candidate.upper()
            if upper.startswith(("CELL_PARAMETERS", "ATOMIC_POSITIONS", "ATOMIC_SPECIES", "K_POINTS")):
                break
            if candidate.startswith("&") or candidate.startswith("/"):
                break
            end_idx += 1
        lines[idx : end_idx + 1] = [new_header, new_mesh]
        suffix = "\n" if text.endswith("\n") else ""
        return "\n".join(lines) + suffix
    return text


def discover_pseudo_files(
    pseudo_dir: Path,
    *,
    prefer_full_relativistic: bool = False,
    prefer_non_full_relativistic: bool = False,
) -> dict[str, Path]:
    if prefer_full_relativistic and prefer_non_full_relativistic:
        raise ValueError("Only one pseudopotential relativistic preference can be requested.")

    mapping: dict[str, Path] = {}
    for path in sorted(pseudo_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".upf", ".upf1"}:
            continue
        element = pseudo_element(path)
        if not element:
            element = element_from_pseudo_filename(path.name)
        if not element:
            continue
        current = mapping.get(element)
        if current is None or should_replace_pseudo(
            current,
            path,
            prefer_full_relativistic=prefer_full_relativistic,
            prefer_non_full_relativistic=prefer_non_full_relativistic,
        ):
            mapping[element] = path
    return mapping


def should_replace_pseudo(
    current: Path,
    candidate: Path,
    *,
    prefer_full_relativistic: bool,
    prefer_non_full_relativistic: bool,
) -> bool:
    current_is_full = pseudo_relativistic_type(current).lower() == "full"
    candidate_is_full = pseudo_relativistic_type(candidate).lower() == "full"
    if prefer_full_relativistic:
        if candidate_is_full != current_is_full:
            return candidate_is_full
    if prefer_non_full_relativistic:
        if candidate_is_full != current_is_full:
            return not candidate_is_full
    return False


def pseudo_element(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    match = re.search(r"<PP_HEADER\b[^>]*\belement\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_element_symbol(match.group(1))

    old_match = re.search(r"<PP_HEADER>\s*[\r\n]+\s*\S+\s+Version Number\s*[\r\n]+\s*([A-Za-z]{1,2})\s+Element", text)
    if old_match:
        return normalize_element_symbol(old_match.group(1))
    return ""


def element_from_pseudo_filename(name: str) -> str:
    match = re.match(r"([A-Za-z][a-z]?)", name)
    return normalize_element_symbol(match.group(1) if match else "")


def normalize_element_symbol(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", str(value or ""))
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:].lower()


def rewrite_atomic_species_pseudos(
    text: str,
    pseudo_map: Mapping[str, Path],
    *,
    pseudo_dir: Path,
) -> str:
    entries = atomic_species_entries(text)
    if not entries:
        return text

    lines = text.splitlines()
    for entry in entries:
        element = normalize_element_symbol(entry["element"])
        pseudo_path = pseudo_map.get(element)
        if pseudo_path is None:
            raise FileNotFoundError(f"No pseudopotential found for {element} in {pseudo_dir}")
        idx = int(entry["line_index"])
        tokens = lines[idx].split()
        if len(tokens) < 3:
            continue
        tokens[2] = pseudo_path.name
        indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
        lines[idx] = indent + " ".join(tokens)

    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def atomic_species_entries(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    entries: list[dict[str, Any]] = []
    in_block = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if not in_block:
            if upper.startswith("ATOMIC_SPECIES"):
                in_block = True
            continue

        if not stripped:
            break
        if upper.startswith(("ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS")) or stripped.startswith("&"):
            break
        if stripped.startswith("!"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 3:
            entries.append(
                {
                    "line_index": idx,
                    "element": tokens[0],
                    "mass": tokens[1],
                    "pseudo": tokens[2],
                }
            )
    return entries


def extract_relaxed_structure_from_pw_output(text: str) -> dict[str, Any]:
    """Extract the final atomic positions and cell from a pw.x relax/vc-relax output."""

    final_text = final_coordinates_region(text) or text
    atomic = extract_last_qe_card(final_text, "ATOMIC_POSITIONS")
    cell = extract_last_qe_card(final_text, "CELL_PARAMETERS", max_data_lines=3)
    result: dict[str, Any] = {}
    if atomic:
        result["atomic_positions"] = atomic
    if cell:
        result["cell_parameters"] = cell
    return result


def extract_relaxed_structure_from_qe_xml(xml_path: str | Path) -> dict[str, Any]:
    """Extract the final structure and useful metadata from QE XML output."""

    path = Path(xml_path).expanduser()
    root = ElementTree.parse(path).getroot()
    output = first_xml_child(root, "output")
    structure = first_xml_child(output, "atomic_structure") if output is not None else None
    if structure is None:
        structures = list(iter_xml(root, "atomic_structure"))
        structure = structures[-1] if structures else None
    if structure is None:
        raise ValueError(f"No atomic_structure block found in {path}")

    cell_node = first_xml_child(structure, "cell")
    positions_node = first_xml_child(structure, "atomic_positions")
    if cell_node is None or positions_node is None:
        raise ValueError(f"Incomplete atomic_structure block in {path}")

    species_node = first_xml_child(output, "atomic_species") if output is not None else None
    if species_node is None:
        species_node = first_xml_child(first_xml_child(root, "input"), "atomic_species")

    cell_vectors_bohr = []
    for name in ("a1", "a2", "a3"):
        vector_node = first_xml_child(cell_node, name)
        if vector_node is None:
            raise ValueError(f"Missing {name} vector in {path}")
        cell_vectors_bohr.append(parse_xml_vector(vector_node.text))

    atom_lines = []
    for atom in xml_children(positions_node, "atom"):
        symbol = str(atom.attrib.get("name") or "").strip()
        if not symbol:
            continue
        position = [value * BOHR_TO_ANGSTROM for value in parse_xml_vector(atom.text)]
        atom_lines.append(
            f"{symbol:<2} "
            f"{position[0]:20.10f}"
            f"{position[1]:19.10f}"
            f"{position[2]:19.10f}"
        )

    cell_lines = []
    for vector in cell_vectors_bohr:
        angstrom = [value * BOHR_TO_ANGSTROM for value in vector]
        cell_lines.append(
            f"{angstrom[0]:18.10f}"
            f"{angstrom[1]:19.10f}"
            f"{angstrom[2]:19.10f}"
        )

    result: dict[str, Any] = {
        "source": "qe_xml",
        "xml_file": str(path),
        "atomic_positions": {
            "header": "ATOMIC_POSITIONS angstrom",
            "lines": atom_lines,
        },
        "cell_parameters": {
            "header": "CELL_PARAMETERS angstrom",
            "lines": cell_lines,
        },
    }
    if species_node is not None:
        species_lines = []
        for species in xml_children(species_node, "species"):
            symbol = str(species.attrib.get("name") or "").strip()
            mass = xml_float(first_xml_child(species, "mass"))
            pseudo = text_or_empty(first_xml_child(species, "pseudo_file"))
            if symbol and mass is not None and pseudo:
                species_lines.append(f"{symbol:<2} {mass:.10g} {pseudo}")
        if species_lines:
            result["atomic_species"] = {
                "header": "ATOMIC_SPECIES",
                "lines": species_lines,
            }

    energy = xml_float(first_xml_descendant(output, "etot") if output is not None else None)
    if energy is not None:
        result["total_energy_ry"] = energy * HARTREE_TO_RY

    convergence = extract_xml_convergence(output)
    if convergence:
        result["convergence"] = convergence

    exit_status = xml_int(first_xml_child(root, "exit_status"))
    if exit_status is not None:
        result["exit_status"] = exit_status
    return result


def extract_xml_convergence(output: ElementTree.Element | None) -> dict[str, Any]:
    if output is None:
        return {}
    convergence_info = first_xml_child(output, "convergence_info")
    if convergence_info is None:
        return {}
    result: dict[str, Any] = {}
    scf = first_xml_child(convergence_info, "scf_conv")
    scf_achieved = None
    if scf is not None:
        scf_achieved = xml_bool(first_xml_child(scf, "convergence_achieved"))
        result["scf"] = {
            "convergence_achieved": scf_achieved,
            "n_scf_steps": xml_int(first_xml_child(scf, "n_scf_steps")),
            "scf_error": xml_float(first_xml_child(scf, "scf_error")),
        }
    opt = first_xml_child(convergence_info, "opt_conv")
    opt_achieved = None
    if opt is not None:
        opt_achieved = xml_bool(first_xml_child(opt, "convergence_achieved"))
        result["optimization"] = {
            "convergence_achieved": opt_achieved,
            "n_opt_steps": xml_int(first_xml_child(opt, "n_opt_steps")),
            "grad_norm": xml_float(first_xml_child(opt, "grad_norm")),
        }
    if scf_achieved is not None or opt_achieved is not None:
        result["scf_converged"] = scf_achieved
        result["optimization_converged"] = opt_achieved
    if scf_achieved is not None and opt_achieved is not None:
        result["relaxation_converged"] = bool(scf_achieved and opt_achieved)
    return result


def parse_xml_vector(text: str | None) -> list[float]:
    parts = str(text or "").split()
    if len(parts) < 3:
        raise ValueError(f"Expected a 3-vector, got: {text!r}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def first_xml_child(parent: ElementTree.Element | None, name: str) -> ElementTree.Element | None:
    if parent is None:
        return None
    for child in list(parent):
        if xml_name(child) == name:
            return child
    return None


def first_xml_descendant(parent: ElementTree.Element | None, name: str) -> ElementTree.Element | None:
    if parent is None:
        return None
    for child in parent.iter():
        if xml_name(child) == name:
            return child
    return None


def xml_children(parent: ElementTree.Element | None, name: str) -> list[ElementTree.Element]:
    if parent is None:
        return []
    return [child for child in list(parent) if xml_name(child) == name]


def iter_xml(parent: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in parent.iter() if xml_name(child) == name]


def xml_name(node: ElementTree.Element) -> str:
    return str(node.tag).rsplit("}", 1)[-1]


def xml_float(node: ElementTree.Element | None) -> float | None:
    if node is None or node.text is None:
        return None
    try:
        return float(node.text.strip())
    except ValueError:
        return None


def xml_int(node: ElementTree.Element | None) -> int | None:
    value = xml_float(node)
    return int(value) if value is not None else None


def xml_bool(node: ElementTree.Element | None) -> bool | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip().lower()
    if text in {"true", "t", "1", ".true."}:
        return True
    if text in {"false", "f", "0", ".false."}:
        return False
    return None


def text_or_empty(node: ElementTree.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def final_coordinates_region(text: str) -> str:
    lower = text.lower()
    begin = lower.rfind("begin final coordinates")
    if begin < 0:
        return ""
    end = lower.find("end final coordinates", begin)
    if end < 0:
        return text[begin:]
    return text[begin:end]


def extract_last_qe_card(text: str, card_name: str, *, max_data_lines: int | None = None) -> dict[str, Any]:
    lines = text.splitlines()
    matches: list[dict[str, Any]] = []
    target = card_name.upper()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.upper().startswith(target):
            continue
        data: list[str] = []
        for raw in lines[idx + 1 :]:
            candidate = raw.strip()
            if not candidate:
                if data:
                    break
                continue
            upper = candidate.upper()
            if upper.startswith(
                (
                    "ATOMIC_POSITIONS",
                    "CELL_PARAMETERS",
                    "ATOMIC_SPECIES",
                    "K_POINTS",
                    "BEGIN",
                    "END",
                )
            ):
                break
            if candidate.startswith("&") or candidate.startswith("/"):
                break
            if re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:$", candidate):
                break
            data.append(raw.rstrip())
            if max_data_lines is not None and len(data) >= max_data_lines:
                break
        if data:
            matches.append({"header": stripped, "lines": data})
    return matches[-1] if matches else {}


def validate_full_relativistic_pseudos(
    species_entries: Sequence[Mapping[str, Any]],
    pseudo_dir: Path,
) -> None:
    for entry in species_entries:
        pseudo_name = str(entry.get("pseudo") or "").strip()
        if not pseudo_name:
            continue
        pseudo_path = pseudo_dir / pseudo_name
        if not pseudo_path.is_file():
            raise FileNotFoundError(f"Spin-orbit pseudo file not found: {pseudo_path}")
        if pseudo_relativistic_type(pseudo_path).lower() != "full":
            element = entry.get("element") or pseudo_name
            raise ValueError(
                f"Spin-orbit coupling requires a full-relativistic pseudopotential for {element}: {pseudo_path}"
            )


def pseudo_relativistic_type(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    match = re.search(
        r"<PP_HEADER\b[^>]*\brelativistic\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return str(match.group(1)).strip()
    lowered = text.lower()
    if "fully-relativistic" in lowered or "full relativistic" in lowered:
        return "full"
    if "scalar-relativistic" in lowered or "scalar relativistic" in lowered:
        return "scalar"
    if "non-relativistic" in lowered or "non relativistic" in lowered:
        return "none"
    return ""


@dataclass
class WorkflowBuildRequest:
    """User selections needed to create a workflow manifest."""

    name: str
    output_directory: str | Path
    source_structure: str | Path
    selected_workflows: Sequence[str] = field(default_factory=tuple)
    relaxation_mode: str = "as_is"
    description: str = ""
    queue: str = "default"
    conversion_options: Mapping[str, Any] = field(default_factory=dict)
    relaxation_options: Mapping[str, Any] = field(default_factory=dict)
    convergence_options: Mapping[str, Any] = field(default_factory=dict)
    band_options: Mapping[str, Any] = field(default_factory=dict)
    wannier_options: Mapping[str, Any] = field(default_factory=dict)
    dos_options: Mapping[str, Any] = field(default_factory=dict)
    fermi_surface_options: Mapping[str, Any] = field(default_factory=dict)
    charge_density_options: Mapping[str, Any] = field(default_factory=dict)
    phonon_options: Mapping[str, Any] = field(default_factory=dict)
    elastic_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowFiles:
    """Paths produced by workflow file generation."""

    workflow_root: Path
    manifest_path: Path
    state_path: Path
    manifest: dict[str, Any]
    state: dict[str, Any]


def create_workflow_files(
    request: WorkflowBuildRequest,
    *,
    create_directories: bool = True,
    copy_source: bool = True,
    overwrite: bool = False,
) -> WorkflowFiles:
    """Create a workflow workspace and write manifest/state JSON files."""

    manifest = build_workflow_manifest(request)
    state = build_initial_workflow_state(manifest)

    workflow_root = Path(manifest["workflow_root"]).expanduser()
    manifest_path = workflow_root / "workflow_manifest.json"
    state_path = workflow_root / "workflow_state.json"

    if not overwrite and (manifest_path.exists() or state_path.exists()):
        raise FileExistsError(
            f"Workflow files already exist in {workflow_root}. "
            "Use overwrite=True only when intentionally replacing them."
        )

    if create_directories:
        create_workflow_directories(manifest, state)
        if copy_source:
            copy_source_file(manifest)
        copy_manifest_pseudo_files(manifest)

    write_json_file(manifest_path, manifest)
    write_json_file(state_path, state)

    return WorkflowFiles(
        workflow_root=workflow_root,
        manifest_path=manifest_path,
        state_path=state_path,
        manifest=manifest,
        state=state,
    )


def build_workflow_manifest(request: WorkflowBuildRequest) -> dict[str, Any]:
    """Build the immutable workflow recipe from GUI/user selections."""

    name = (request.name or "workflow").strip()
    safe_name = sanitize_job_name(name)
    workflow_root = Path(request.output_directory).expanduser() / safe_name
    source_path = Path(request.source_structure).expanduser()
    selected = normalize_selected_workflows(request.selected_workflows)
    relaxation_mode = normalize_relaxation_mode(request.relaxation_mode)
    now = datetime.now()
    created_at = now.isoformat(timespec="microseconds")
    workflow_id = f"{safe_name}_{now.strftime('%Y%m%d_%H%M%S_%f')}"

    conversion_options = dict(request.conversion_options or {})
    manual_ecutwfc = manual_ecutwfc_from_options(conversion_options)
    if manual_ecutwfc is not None:
        conversion_options["manual_ecutwfc"] = manual_ecutwfc
    else:
        conversion_options.pop("manual_ecutwfc", None)
    force_conversion = should_force_source_conversion(source_path, conversion_options)
    source_info = build_source_info(source_path, force_conversion=force_conversion)
    convergence_options = normalize_convergence_options(request.convergence_options)
    relaxation_options = normalize_relaxation_parameters(
        request.relaxation_options,
        legacy_conversion_options=conversion_options,
    )
    band_options = normalize_band_options(request.band_options)
    wannier_options = normalize_wannier_options(request.wannier_options)
    dos_options = normalize_dos_options(request.dos_options)
    fermi_surface_options = normalize_fermi_surface_options(request.fermi_surface_options)
    charge_density_options = normalize_charge_density_options(request.charge_density_options)
    elastic_options = normalize_elastic_options(request.elastic_options)
    phonon_options = normalize_phonon_options(request.phonon_options)

    tasks = build_manifest_tasks(
        selected_workflows=selected,
        relaxation_mode=relaxation_mode,
        relaxation_options=relaxation_options,
        convergence_options=convergence_options,
        band_options=band_options,
        wannier_options=wannier_options,
        dos_options=dos_options,
        fermi_surface_options=fermi_surface_options,
        charge_density_options=charge_density_options,
        elastic_options=elastic_options,
        phonon_options=phonon_options,
        manual_ecutwfc=manual_ecutwfc,
    )
    notices: list[str] = []
    cutoff_options = convergence_options.get("cutoff") if isinstance(convergence_options, Mapping) else {}
    if manual_ecutwfc is not None and isinstance(cutoff_options, Mapping) and cutoff_options.get("enabled", False):
        notices.append(manual_cutoff_skip_notice(manual_ecutwfc))

    return {
        "version": WORKFLOW_VERSION,
        "workflow_id": workflow_id,
        "name": name,
        "description": request.description or "",
        "created_at": created_at,
        "workflow_root": str(workflow_root.resolve()),
        "directories": workflow_directories(),
        "notices": notices,
        "source": source_info,
        "global": {
            "queue": request.queue or "default",
        },
        "conversion": conversion_options,
        "relaxation": {
            "mode": relaxation_mode,
            "parameters": relaxation_options,
        },
        "selected_workflows": selected,
        "convergence": convergence_options,
        "bands": band_options,
        "wannier90": wannier_options,
        "dos": dos_options,
        "fermi_surface": fermi_surface_options,
        "charge_density": charge_density_options,
        "elastic": elastic_options,
        "phonon": phonon_options,
        "tasks": tasks,
    }


def should_force_source_conversion(source_path: Path, conversion_options: Mapping[str, Any]) -> bool:
    """Return whether an already-QE source should be regenerated during preparation."""

    if not conversion_options:
        return False
    if _truthy_option(conversion_options.get("force_conversion")):
        return True
    return source_path.suffix.lower() not in QE_INPUT_SUFFIXES


def build_initial_workflow_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the mutable state tree used by the workflow runner and GUI."""

    tasks: dict[str, Any] = {}
    for task in manifest.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "").strip()
        task_type = str(task.get("type") or "").strip()
        if not task_id or not task_type:
            continue
        tasks[task_id] = build_initial_task_state(task, manifest)

    notices = manifest.get("notices") if isinstance(manifest.get("notices"), list) else []
    return {
        "version": manifest.get("version", WORKFLOW_VERSION),
        "workflow_id": manifest.get("workflow_id", ""),
        "manifest_file": "workflow_manifest.json",
        "created_at": manifest.get("created_at", ""),
        "updated_at": datetime.now().isoformat(timespec="microseconds"),
        "status": "queued",
        "current_task": None,
        "notices": list(notices),
        "tasks": tasks,
        "results": {
            "ecutwfc": None,
            "kmesh": None,
            "initial_relaxation_output": None,
            "relaxation_output": None,
            "final_relaxation_output": None,
            "band_scf_output": None,
            "band_pw_output": None,
            "bandx_output": None,
            "projwfc_output": None,
            "fatband": None,
            "fatband_projection_csv_dir": None,
            "fatband_projection_csv": None,
            "fatband_csv_metadata": None,
            "fatband_projection_error": None,
            "fatband_plots": None,
            "fatband_color_mode": None,
            "fatband_error": None,
            "dos_scf_output": None,
            "dos_nscf_output": None,
            "dos_projwfc_output": None,
            "pdos_sumpdos_output": None,
            "pdos_sumpdos_data": None,
            "pdos_sumpdos_fermi_ev": None,
            "pdos_postprocess_output": None,
            "pdos_postprocess_metadata": None,
            "pdos_csv_files": None,
            "pdos_plot_files": None,
            "dos_summary_csv": None,
            "dos_summary_plot": None,
            "pdos_combined_plot": None,
            "pdos_combined_plot_mode": None,
            "dos": None,
            "fermi_surface_scf_output": None,
            "fermi_surface_velocity_output": None,
            "fermi_surface_frmsf": None,
            "fermi_surface": None,
            "charge_density_scf_output": None,
            "charge_density_pp_output": None,
            "charge_density_cube": None,
            "charge_density": None,
            "elastic_scf_input": None,
            "elastic_thermo_control": None,
            "elastic_output": None,
            "elastic": None,
            "phonon_scf_output": None,
            "phonon_output": None,
            "phonon_dyn_file": None,
            "phonon_q2r_output": None,
            "phonon_matdyn_output": None,
            "phonon_force_constants_file": None,
            "phonon_frequency_file": None,
            "phonon_phdos_output": None,
            "phonon_dos_file": None,
            "phonon_dos_data": None,
            "phonon_dos_plot": None,
            "phonon_band_modes_file": None,
            "phonon_dos_modes_file": None,
            "phonon_projected_band_data": None,
            "phonon_projected_dos_data": None,
            "phonon_projected_combined_plot": None,
            "phonon_projected_replot_script": None,
            "phonon_epc_fit_scf_output": None,
            "phonon_epc_scf_output": None,
            "phonon_epc_ph_output": None,
            "phonon_epc_q2r_output": None,
            "phonon_epc_matdyn_output": None,
            "phonon_epc_phdos_output": None,
            "phonon_lambda_output": None,
            "phonon_epc_frequency_file": None,
            "phonon_epc_dos_file": None,
            "phonon_epc_lambda": None,
            "phonon": None,
            "wannier90_scf_output": None,
            "wannier90_pw_output": None,
            "wannier90_bandx_output": None,
            "wannier90_nscf_output": None,
            "wannier90_projwfc_output": None,
            "wannier90_filproj": None,
            "wannier90_projectability_output": None,
            "wannier90_projectability_fit_output": None,
            "wannier90_projectability_fit": None,
            "wannier90_projectability_fit_model": None,
            "wannier90_projectability_mu": None,
            "wannier90_projectability_sigma": None,
            "wannier90_scdm_fitting_output": None,
            "wannier90_pw2wan_input": None,
            "wannier90_win_input": None,
            "wannier90_preprocess_output": None,
            "wannier90_nnkp": None,
            "wannier90_pw2wan_output": None,
            "wannier90_pw2wannier90_output": None,
            "wannier90_run_output": None,
            "wannier90_wout": None,
            "wannier90": None,
        },
        "events": [],
    }


def build_initial_task_state(task: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create the initial state node for a manifest task."""

    task_id = str(task.get("id") or "")
    task_type = str(task.get("type") or "")
    label = str(task.get("label") or task_id)
    status = "queued" if not task.get("depends_on") else "waiting"
    children: list[dict[str, Any]]

    if task_type == "structure.prepare":
        children = [build_prepare_source_child(manifest)]
    elif task_type == "convergence.cutoff":
        children = build_cutoff_children(task)
    elif task_type == "convergence.kpoints":
        children = build_kpoint_children(task)
    elif task_type in {"qe.relax", "qe.vc_relax"}:
        children = [build_relaxation_child(task)]
    elif task_type in {"qe.bands_workflow", "qe.fat_bands_workflow"}:
        children = build_bands_children(task)
    elif task_type == "qe.wannier90_workflow":
        children = build_wannier90_children(task)
    elif task_type == "qe.dos_workflow":
        children = build_dos_children(task)
    elif task_type == "qe.fermi_surface_workflow":
        children = build_fermi_surface_children(task)
    elif task_type == "qe.charge_density_workflow":
        children = build_charge_density_children(task)
    elif task_type == "qe.elastic_workflow":
        children = build_elastic_children(task)
    elif task_type == "qe.phonon_workflow":
        children = build_phonon_children(task)
    elif task_type == "postprocess.fatband_csv":
        children = [build_fatband_csv_child(task)]
    elif task_type == "postprocess.fatband_plot":
        children = [build_fatband_plot_child(task)]
    else:
        children = [build_generic_child(task)]

    planned_total = len(children)
    may_expand = bool(task.get("may_expand", False))
    task_state = {
        "id": task_id,
        "type": task_type,
        "label": label,
        "status": status,
        "depends_on": list(task.get("depends_on") or []),
        "completed": 0,
        "planned_total": planned_total,
        "may_expand": may_expand,
        "params": dict(task.get("params") or {}),
        "running_child": None,
        "children": children,
        "result": {},
    }
    expansion = build_initial_expansion_state(task, children)
    if expansion:
        task_state["expansion"] = expansion
    return task_state


def build_initial_expansion_state(
    task: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Initial dynamic-growth metadata for expandable tasks."""

    task_type = str(task.get("type") or "")
    params = task.get("params") or {}
    if task_type == "convergence.cutoff":
        maximum = params.get("maximum")
        step = int(params.get("step") or 10)
        expand_by = int(params.get("expand_by") or 2)
        next_value = None
        if children:
            last_params = children[-1].get("params") or {}
            last_value = last_params.get("ecutwfc")
            if last_value is not None:
                candidate = int(last_value) + step
                if maximum is None or candidate <= int(maximum):
                    next_value = candidate
        elif params.get("start") is not None:
            next_value = int(params["start"])
        return {
            "expand_by": max(1, expand_by),
            "next_value": next_value,
            "maximum": maximum,
        }

    if task_type == "convergence.kpoints":
        return {
            "expand_by": max(1, int(params.get("expand_by") or 2)),
            "next_spacing": params.get("start_spacing"),
            "max_mesh_size": params.get("max_mesh_size"),
        }

    return {}


def build_manifest_tasks(
    *,
    selected_workflows: Sequence[str],
    relaxation_mode: str,
    relaxation_options: Mapping[str, Any],
    convergence_options: Mapping[str, Any],
    band_options: Mapping[str, Any],
    wannier_options: Mapping[str, Any],
    dos_options: Mapping[str, Any],
    fermi_surface_options: Mapping[str, Any],
    charge_density_options: Mapping[str, Any],
    elastic_options: Mapping[str, Any],
    phonon_options: Mapping[str, Any],
    manual_ecutwfc: int | float | None = None,
) -> list[dict[str, Any]]:
    """Build a serial task graph from selected workflow options."""

    tasks: list[dict[str, Any]] = [
        {
            "id": "prepare_source",
            "type": "structure.prepare",
            "label": "Structure preparation",
            "depends_on": [],
            "params": {},
        }
    ]
    dependency = "prepare_source"

    selected = set(selected_workflows)
    convergence_task_specs: list[dict[str, Any]] = []
    if "convergence" in selected:
        cutoff = dict((convergence_options.get("cutoff") or {}) if isinstance(convergence_options, Mapping) else {})
        if cutoff.get("enabled", False) and manual_ecutwfc is None:
            convergence_task_specs.append(
                {
                    "id": "cutoff_convergence",
                    "type": "convergence.cutoff",
                    "label": "Cutoff energy convergence",
                    "params": cutoff,
                    "may_expand": True,
                }
            )

        kpoints = dict((convergence_options.get("kpoints") or {}) if isinstance(convergence_options, Mapping) else {})
        if kpoints.get("enabled", False):
            convergence_task_specs.append(
                {
                    "id": "kpoint_convergence",
                    "type": "convergence.kpoints",
                    "label": "K-points convergence",
                    "params": kpoints,
                    "may_expand": True,
                }
            )

    has_relaxation = relaxation_mode in {"atomic_positions", "full_geometry"}
    has_convergence = bool(convergence_task_specs)

    if has_relaxation and has_convergence:
        # When both are selected: do a rough initial relaxation first so convergence
        # runs on a better structure, then do final relaxation after convergence.
        initial_task = build_relaxation_task(relaxation_mode, "initial", dependency, relaxation_options)
        tasks.append(initial_task)
        dependency = "initial_relaxation"

    for spec in convergence_task_specs:
        task = dict(spec)
        params = dict(task.get("params") or {})
        params.setdefault("template_source", str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME))
        params.setdefault("structure_source", "prepared_input")
        if has_relaxation and has_convergence:
            params["structure_source"] = "initial_relaxation"
        params.setdefault("pseudo_policy", "non_full_relativistic")
        task["params"] = params
        task["depends_on"] = [dependency]
        tasks.append(task)
        dependency = str(task["id"])

    if has_relaxation:
        final_task = build_relaxation_task(relaxation_mode, "final", dependency, relaxation_options)
        final_params = dict(final_task.get("params") or {})
        if has_convergence:
            # Final relaxation should start from the initial-relaxed structure.
            final_params["structure_source"] = "initial_relaxation"
            final_params["uses_converged_values"] = True
        else:
            # No convergence selected: final relaxation runs directly from prepared source.
            final_params["structure_source"] = "prepared_input"
            final_params["uses_converged_values"] = False
        final_task["params"] = final_params
        tasks.append(final_task)
        dependency = "final_relaxation"

    selected_set = set(selected_workflows)
    if (
        "bands" in selected_set
        or "band" in selected_set
        or "fat_bands" in selected_set
        or "fat_band" in selected_set
        or "fatbands" in selected_set
    ):
        band_variant = "fat_bands" if _truthy_option((band_options or {}).get("calculate_fat_band")) else "bands"
        if band_variant == "bands" and any(
            key in selected_set for key in {"fat_bands", "fat_band", "fatbands"}
        ):
            band_variant = "fat_bands"
        tasks.append(
            {
                "id": "bands_workflow",
                "type": "qe.bands_workflow",
                "label": "Band structure workflow",
                "depends_on": [dependency],
                "params": {
                    "variant": band_variant,
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "band_options": dict(band_options or {}),
                },
            }
        )
        dependency = "bands_workflow"
        if band_variant == "fat_bands":
            tasks.append(
                {
                    "id": "fatband_csv_extraction",
                    "type": "postprocess.fatband_csv",
                    "label": "Fat-band CSV extraction",
                    "depends_on": ["bands_workflow"],
                    "params": {
                        "source_task": "bands_workflow",
                        "filband": str((band_options or {}).get("filband") or "bands.out"),
                    },
                }
            )
            tasks.append(
                {
                    "id": "fatband_plotting",
                    "type": "postprocess.fatband_plot",
                    "label": "Fat-band plotting",
                    "depends_on": ["fatband_csv_extraction"],
                    "params": {
                        "source_task": "bands_workflow",
                        "filband": str((band_options or {}).get("filband") or "bands.out"),
                        "kpoints_crystal_b": list((band_options or {}).get("kpoints_crystal_b") or []),
                        "fatband_color_mode": str((band_options or {}).get("fatband_color_mode") or "dominant"),
                    },
                }
            )
            dependency = "fatband_plotting"

    if "pdos" in selected_set or "dos" in selected_set:
        # DOS/PDOS runs after the latest completed upstream stage. When fat-band
        # post-processing is enabled, this naturally becomes fatband_plotting.
        dos_dependency = dependency
        tasks.append(
            {
                "id": "dos_workflow",
                "type": "qe.dos_workflow",
                "label": "Projected density of states workflow",
                "depends_on": [dos_dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    # PDOS requires tetrahedra occupations in both SCF and NSCF,
                    # so it always runs a dedicated DOS SCF stage.
                    "reuse_band_scf": False,
                    "dos_options": dict(dos_options or {}),
                },
            }
        )
        dependency = "dos_workflow"

    if "fermi_surface" in selected_set or "fermisurface" in selected_set:
        tasks.append(
            {
                "id": "fermi_surface_workflow",
                "type": "qe.fermi_surface_workflow",
                "label": "Fermi surface workflow",
                "depends_on": [dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "fermi_surface_options": dict(fermi_surface_options or {}),
                },
            }
        )
        dependency = "fermi_surface_workflow"

    if "charge_density" in selected_set or "chargedensity" in selected_set:
        tasks.append(
            {
                "id": "charge_density_workflow",
                "type": "qe.charge_density_workflow",
                "label": "Charge density workflow",
                "depends_on": [dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "charge_density_options": dict(charge_density_options or {}),
                },
            }
        )
        dependency = "charge_density_workflow"

    if "elastic" in selected_set:
        tasks.append(
            {
                "id": "elastic_workflow",
                "type": "qe.elastic_workflow",
                "label": "Elastic properties workflow",
                "depends_on": [dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "elastic_options": dict(elastic_options or {}),
                },
            }
        )
        dependency = "elastic_workflow"

    if "phonon" in selected_set:
        tasks.append(
            {
                "id": "phonon_workflow",
                "type": "qe.phonon_workflow",
                "label": "Phonon workflow",
                "depends_on": [dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "phonon_options": dict(phonon_options or {}),
                },
            }
        )
        dependency = "phonon_workflow"

    if "wannier" in selected_set or "wannier90" in selected_set or "winner90" in selected_set:
        tasks.append(
            {
                "id": "wannier90_workflow",
                "type": "qe.wannier90_workflow",
                "label": "Wannier90 + SCDM workflow",
                "depends_on": [dependency],
                "params": {
                    "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
                    "structure_source": "final_relaxation" if has_relaxation else "prepared_input",
                    "uses_converged_values": True,
                    "band_options": dict(band_options or {}),
                    "wannier_options": dict(wannier_options or {}),
                },
            }
        )
        dependency = "wannier90_workflow"

    for key in selected_workflows:
        if key == "convergence":
            continue
        if key in {
            "bands",
            "band",
            "fat_bands",
            "fat_band",
            "fatbands",
            "pdos",
            "dos",
            "fermi_surface",
            "fermisurface",
            "charge_density",
            "chargedensity",
            "elastic",
            "phonon",
            "wannier",
            "wannier90",
            "winner90",
        }:
            continue
        meta = workflow_task_metadata(key)
        tasks.append(
            {
                "id": meta["id"],
                "type": meta["type"],
                "label": meta["label"],
                "depends_on": [dependency],
                "params": {},
            }
        )
        dependency = meta["id"]

    return tasks


def build_relaxation_task(
    relaxation_mode: str,
    stage: str,
    dependency: str,
    relaxation_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calc_type = "qe.vc_relax" if relaxation_mode == "full_geometry" else "qe.relax"
    base_label = "Full geometry relaxation" if relaxation_mode == "full_geometry" else "Atomic positions relaxation"
    stage_key = stage if stage in {"initial", "final", "single"} else "single"
    if stage_key == "initial":
        task_id = "initial_relaxation"
        label = f"Initial rough {base_label.lower()}"
        params = {
            "mode": relaxation_mode,
            "stage": "initial",
            "protocol": "fast",
            "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
            "structure_source": "prepared_input",
            "pseudo_policy": "non_full_relativistic",
            "uses_converged_values": False,
        }
    elif stage_key == "final":
        task_id = "final_relaxation"
        label = f"Final accuracy {base_label.lower()}"
        params = {
            "mode": relaxation_mode,
            "stage": "final",
            "protocol": "selected",
            "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
            "structure_source": "prepared_input",
            "pseudo_policy": "final_accuracy",
            "uses_converged_values": True,
        }
    else:
        task_id = "relaxation"
        label = base_label
        params = {
            "mode": relaxation_mode,
            "stage": "single",
            "protocol": "selected",
            "template_source": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
            "structure_source": "prepared_input",
            "pseudo_policy": "prepared_template",
            "uses_converged_values": False,
        }

    params["relaxation_parameters"] = dict(relaxation_options or {})
    return {
        "id": task_id,
        "type": calc_type,
        "label": label,
        "depends_on": [dependency],
        "params": params,
    }


def build_source_info(source_path: Path, *, force_conversion: bool = False) -> dict[str, Any]:
    """Return source-file metadata and source-directory relative paths."""

    suffix = source_path.suffix.lower()
    requires_conversion = force_conversion or suffix not in QE_INPUT_SUFFIXES
    source_name = source_path.name or "structure"

    return {
        "original": str(source_path.resolve()),
        "stored_original": str(Path(SOURCE_DIR) / source_name),
        "prepared_input": str(Path(SOURCE_DIR) / PREPARED_QE_TEMPLATE_NAME),
        "requires_conversion": requires_conversion,
    }


def workflow_directories() -> dict[str, str]:
    """Relative directory layout used by all workflows."""

    return {
        "source": SOURCE_DIR,
        "convergence": CONVERGENCE_DIR,
        "cutoff_convergence": str(Path(CONVERGENCE_DIR) / CUTOFF_CONVERGENCE_DIR),
        "kpoint_convergence": str(Path(CONVERGENCE_DIR) / KPOINT_CONVERGENCE_DIR),
        "relaxation": RELAXATION_DIR,
        "bands": BANDS_DIR,
        "dos": DOS_DIR,
        "charge_density": CHARGE_DENSITY_DIR,
        "elastic": ELASTIC_DIR,
        "phonon": PHONON_DIR,
        "wannier90": WANNIER90_DIR,
        "fat_bands": FAT_BANDS_DIR,
        "inputs_and_outputs": INPUTS_OUTPUTS_DIR,
        "logs": LOGS_DIR,
        "data_and_plots": DATA_PLOTS_DIR,
    }


def normalize_selected_workflows(selected_workflows: Sequence[str]) -> list[str]:
    """Normalize Step 3 workflow keys while preserving user order."""

    selected: list[str] = []
    seen: set[str] = set()
    aliases = {
        "band": "bands",
        "fatband": "fat_bands",
        "fatbands": "fat_bands",
        "fat_band": "fat_bands",
        "fermi": "fermi_surface",
        "fermisurface": "fermi_surface",
        "fermi_surfaces": "fermi_surface",
        "charge": "charge_density",
        "chargedensity": "charge_density",
        "charge_densities": "charge_density",
        "elastic_properties": "elastic",
        "elastic_property": "elastic",
        "elastic_constants": "elastic",
        "winner90": "wannier90",
        "wannier": "wannier90",
    }
    for raw in selected_workflows or []:
        for part in str(raw).replace(";", ",").split(","):
            key = part.strip().lower().replace("-", "_").replace(" ", "_")
            key = aliases.get(key, key)
            if not key or key in seen:
                continue
            selected.append(key)
            seen.add(key)
    return selected


def normalize_relaxation_mode(mode: str) -> str:
    """Normalize Step 2 relaxation keys."""

    key = (mode or "as_is").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "as_is",
        "asis": "as_is",
        "structure_as_is": "as_is",
        "atomic": "atomic_positions",
        "positions": "atomic_positions",
        "relax": "atomic_positions",
        "full": "full_geometry",
        "vc_relax": "full_geometry",
        "vc-relax": "full_geometry",
    }
    return aliases.get(key, key if key in {"as_is", "atomic_positions", "full_geometry"} else "as_is")


def normalize_convergence_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize convergence options from GUI or CLI sources."""

    raw = dict(options or {})
    cutoff_raw = raw.get("cutoff") if isinstance(raw.get("cutoff"), Mapping) else {}
    kpoints_raw = raw.get("kpoints") if isinstance(raw.get("kpoints"), Mapping) else {}

    cutoff = normalize_cutoff_options(cutoff_raw)
    kpoints = normalize_kpoint_options(kpoints_raw)

    return {
        "cutoff": cutoff,
        "kpoints": kpoints,
    }


def normalize_band_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize band/fat-band settings."""

    raw = dict(options or {})
    kpoint_spacing = positive_float(raw, ["kpoint_spacing", "scf_kpoint_spacing"], 0.10)
    nbnd = optional_int(raw, ["nbnd"])
    if nbnd is not None and nbnd <= 0:
        nbnd = None
    calculate_fat_band = _truthy_option(
        raw.get("calculate_fat_band", raw.get("fat_band", raw.get("fatbands", False)))
    )
    fatband_color_raw = str(
        raw.get("fatband_color_mode")
        or raw.get("fatband_coloring")
        or raw.get("color_mode")
        or "dominant"
    ).strip().lower().replace("_", "-")
    fatband_color_mode = (
        "blended"
        if fatband_color_raw in {"v2-basis", "v2basis", "blend", "blended", "v2-element-mix", "element-mix"}
        else "dominant"
    )

    weight = optional_int(raw, ["kpoint_weight", "path_weight", "weight"])
    if weight is None or weight <= 0:
        weight = 50

    crystal_b = raw.get("kpoints_crystal_b")
    if isinstance(crystal_b, Sequence) and not isinstance(crystal_b, (str, bytes)):
        kpoints_crystal_b = [str(line).rstrip() for line in crystal_b if str(line).strip()]
    else:
        kpoints_crystal_b = []

    selected_paths_raw = raw.get("selected_paths")
    selected_paths: list[list[str]] = []
    if isinstance(selected_paths_raw, Sequence) and not isinstance(selected_paths_raw, (str, bytes)):
        for path in selected_paths_raw:
            if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
                continue
            labels = [str(label).strip() for label in path if str(label).strip()]
            if labels:
                selected_paths.append(labels)

    return {
        "calculate_fat_band": calculate_fat_band,
        "kpoint_spacing": kpoint_spacing,
        "nbnd": nbnd,
        "kpoint_weight": int(weight),
        "selected_paths": selected_paths,
        "kpoints_crystal_b": kpoints_crystal_b,
        "filband": str(raw.get("filband") or "bands.out"),
        "fatband_color_mode": fatband_color_mode,
    }


def normalize_wannier_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Wannier90 + SCDM workflow settings."""

    raw = dict(options or {})
    nbnd = optional_int(raw, ["wannier_nbnd", "nbnd", "nband"])
    if nbnd is None or nbnd <= 0:
        nbnd = 64

    scf_kmesh = normalize_epc_kmesh(
        raw.get("scf_kmesh", raw.get("kmesh", raw.get("mp_grid"))),
        [8, 8, 6],
    )
    weight = optional_int(raw, ["kpoint_weight", "path_weight", "weight"])
    if weight is None or weight <= 0:
        weight = 50

    crystal_b = raw.get("kpoints_crystal_b")
    if isinstance(crystal_b, Sequence) and not isinstance(crystal_b, (str, bytes)):
        kpoints_crystal_b = [str(line).rstrip() for line in crystal_b if str(line).strip()]
    else:
        kpoints_crystal_b = []

    selected_paths_raw = raw.get("selected_paths")
    selected_paths: list[list[str]] = []
    if isinstance(selected_paths_raw, Sequence) and not isinstance(selected_paths_raw, (str, bytes)):
        for path in selected_paths_raw:
            if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
                continue
            labels = [str(label).strip() for label in path if str(label).strip()]
            if labels:
                selected_paths.append(labels)

    return {
        "kpoint_spacing": positive_float(raw, ["kpoint_spacing", "scf_kpoint_spacing"], 0.10),
        "scf_kmesh": scf_kmesh,
        "kmesh": scf_kmesh,
        "wannier_nbnd": int(nbnd),
        "nbnd": int(nbnd),
        "kpoint_weight": int(weight),
        "selected_paths": selected_paths,
        "kpoints_crystal_b": kpoints_crystal_b,
        "write_unk": _truthy_option(raw.get("write_unk", False)),
        "num_iter": positive_int(raw, ["num_iter", "num_inter"], 500),
        "dis_num_iter": positive_int(raw, ["dis_num_iter"], 500),
        "filband": str(raw.get("filband") or "bands.out"),
        "prefix": str(raw.get("prefix") or "").strip(),
    }


def normalize_dos_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize DOS/PDOS workflow settings."""

    raw = dict(options or {})
    scf_kpoint_spacing = positive_float(raw, ["scf_kpoint_spacing", "kpoint_spacing", "kspacing_scf"], 0.30)
    nscf_kpoint_spacing = positive_float(raw, ["nscf_kpoint_spacing", "kpoint_spacing_nscf", "kspacing_nscf"], 0.10)
    use_converged_kmesh = _truthy_option(raw.get("use_converged_kmesh", raw.get("use_converged_kpoints", True)))
    prefix = str(raw.get("prefix") or "").strip()

    return {
        "scf_kpoint_spacing": scf_kpoint_spacing,
        "nscf_kpoint_spacing": nscf_kpoint_spacing,
        "use_converged_kmesh": use_converged_kmesh,
        "reuse_band_scf": False,
        "prefix": prefix,
    }


def normalize_fermi_surface_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Fermi surface workflow settings."""

    raw = dict(options or {})
    kpoint_spacing = positive_float(raw, ["kpoint_spacing", "scf_kpoint_spacing", "kspacing"], 0.10)
    prefix = str(raw.get("prefix") or "").strip()
    return {
        "kpoint_spacing": kpoint_spacing,
        "prefix": prefix,
    }


def normalize_charge_density_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize charge-density workflow settings."""

    raw = dict(options or {})
    kpoint_spacing = positive_float(raw, ["kpoint_spacing", "scf_kpoint_spacing", "kspacing"], 0.10)
    prefix = str(raw.get("prefix") or "").strip()
    fileout = str(raw.get("fileout") or raw.get("file_out") or "{prefix}_rho.cube").strip()
    if not fileout:
        fileout = "{prefix}_rho.cube"
    nx = optional_int(raw, ["nx"])
    ny = optional_int(raw, ["ny"])
    nz = optional_int(raw, ["nz"])
    return {
        "kpoint_spacing": kpoint_spacing,
        "prefix": prefix,
        "fileout": fileout,
        "nx": int(nx) if nx and nx > 0 else 64,
        "ny": int(ny) if ny and ny > 0 else 64,
        "nz": int(nz) if nz and nz > 0 else 64,
    }


def normalize_elastic_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize thermo_pw elastic workflow settings."""

    raw = dict(options or {})
    what = str(raw.get("what") or "scf_elastic_constants").strip().strip("'\"").lower()
    allowed_what = {"scf_elastic_constants", "mur_lc_elastic_constants"}
    if what not in allowed_what:
        what = "scf_elastic_constants"

    algorithm = (
        str(raw.get("elastic_algorithm") or raw.get("algorithm") or "standard")
        .strip()
        .strip("'\"")
        .lower()
    )
    allowed_algorithms = {"standard", "advanced", "energy_std", "energy"}
    if algorithm not in allowed_algorithms:
        algorithm = "standard"

    default_ngeo = 6 if algorithm == "energy" else 4
    ngeo_strain = optional_int(raw, ["ngeo_strain", "ngeo"])
    if ngeo_strain is None or ngeo_strain <= 0:
        ngeo_strain = default_ngeo

    return {
        "what": what,
        "elastic_algorithm": algorithm,
        "ngeo_strain": int(ngeo_strain),
        "frozen_ions": _truthy_option(raw.get("frozen_ions", False)),
        "find_ibrav": True,
    }


def normalize_phonon_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize phonon workflow settings."""

    raw = dict(options or {})
    q2r_raw = raw.get("q2r") if isinstance(raw.get("q2r"), Mapping) else {}
    matdyn_raw = raw.get("matdyn") if isinstance(raw.get("matdyn"), Mapping) else {}
    scf_kpoint_spacing = positive_float(raw, ["scf_kpoint_spacing", "kpoint_spacing", "kspacing_scf"], 0.0)
    use_converged_kmesh = _truthy_option(raw.get("use_converged_kmesh", raw.get("use_converged_kpoints", False)))
    q_point_raw = raw.get("q_point", raw.get("qpoint", raw.get("q")))
    q_point = normalize_q_point(q_point_raw)
    mode = str(raw.get("mode") or raw.get("calculation_mode") or raw.get("workflow_mode") or "").strip().lower()
    mode = mode.replace("-", "_").replace(" ", "_")
    if mode in {"gamma", "single", "singleq", "single_q", "q_point"}:
        mode = "single_q"
    elif mode in {"dispersion", "bands", "phonon_bands", "ldisp", "grid"}:
        mode = "dispersion"
    elif mode in {
        "epc",
        "electron_phonon",
        "electron_phonon_coupling",
        "lambda",
        "a2f",
        "epc_interpolated",
        "tetra",
        "tetrahedra",
        "lambda_tetra",
        "epc_tetra",
        "epc_tetrahedra",
        "alpha2f",
    }:
        mode = EPC_MODE
    else:
        mode = "dispersion" if _truthy_option(raw.get("ldisp", True)) else "single_q"
    ldisp = mode in {"dispersion", EPC_MODE}
    nq_grid = normalize_phonon_nq_grid(raw.get("nq", raw.get("q_grid", raw.get("qgrid"))), DEFAULT_PHONON_Q_GRID)
    prefix = str(raw.get("prefix") or "").strip()
    fildyn = str(raw.get("fildyn") or "phonon.dyn").strip() or "phonon.dyn"
    flfrc = str(raw.get("flfrc") or q2r_raw.get("flfrc") or matdyn_raw.get("flfrc") or "phonon.fc").strip() or "phonon.fc"
    flfrq = str(raw.get("flfrq") or matdyn_raw.get("flfrq") or "phonon.freq.band").strip() or "phonon.freq.band"
    phdos_flfrq = (
        str(raw.get("phdos_flfrq") or raw.get("dos_flfrq") or matdyn_raw.get("phdos_flfrq") or "phonon.freq.dos")
        .strip()
        or "phonon.freq.dos"
    )
    fldos = str(raw.get("fldos") or matdyn_raw.get("fldos") or "phonon.phdos").strip() or "phonon.phdos"
    matdyn_flvec = str(raw.get("matdyn_flvec") or matdyn_raw.get("flvec") or "phonon_band.modes").strip() or "phonon_band.modes"
    phdos_flvec = str(raw.get("phdos_flvec") or raw.get("dos_flvec") or "phonon_dos.modes").strip() or "phonon_dos.modes"
    epc_fldos = str(raw.get("epc_fldos") or raw.get("a2f_fldos") or "phonon.dos").strip() or "phonon.dos"
    tr2_ph = str(raw.get("tr2_ph") or "1.0d-12").strip() or "1.0d-12"
    epc_tr2_ph_raw = str(raw.get("epc_tr2_ph") or "").strip()
    epc_tr2_ph = epc_tr2_ph_raw or (tr2_ph if str(raw.get("tr2_ph") or "").strip() else "1.0d-10")
    alpha_mix = str(raw.get("alpha_mix") or raw.get("alpha_mix(1)") or "0.7").strip() or "0.7"
    nmix_ph = positive_int(raw, ["nmix_ph"], 4)
    verbosity = str(raw.get("verbosity") or "default").strip() or "default"
    band_enabled = _truthy_option(
        raw.get("band_enabled", raw.get("phonon_band_enabled", raw.get("matdyn_enabled", True)))
    )
    dos_enabled = _truthy_option(raw.get("dos_enabled", raw.get("phdos_enabled", True)))
    if mode == "dispersion" and not band_enabled and not dos_enabled:
        band_enabled = True
    projection_enabled = _truthy_option(raw.get("phonon_projection_plot_enabled", raw.get("projection_plot_enabled", True)))
    phdos_nk = normalize_phonon_nq_grid(raw.get("phdos_nk", raw.get("dos_nk", raw.get("nk"))), DEFAULT_PHDOS_NK)
    q_path = normalize_matdyn_q_path(
        raw.get("matdyn_q_path", raw.get("q_path", raw.get("matdyn_path", matdyn_raw.get("q_path"))))
    )

    return {
        "mode": mode,
        "calculation_mode": mode,
        "scf_kpoint_spacing": scf_kpoint_spacing,
        "use_converged_kmesh": use_converged_kmesh,
        "q_point": q_point,
        "ldisp": ldisp,
        "nq": nq_grid,
        "q_grid": nq_grid,
        "prefix": prefix,
        "fildyn": fildyn,
        "flfrc": flfrc,
        "flfrq": flfrq,
        "phdos_flfrq": phdos_flfrq,
        "fldos": fldos,
        "matdyn_flvec": matdyn_flvec,
        "phdos_flvec": phdos_flvec,
        "epc_fldos": epc_fldos,
        "band_enabled": bool(band_enabled),
        "dos_enabled": bool(dos_enabled),
        "phonon_projection_plot_enabled": bool(projection_enabled and mode == "dispersion" and band_enabled and dos_enabled),
        "phdos_nk": phdos_nk,
        "scf_kmesh": normalize_epc_kmesh(
            raw.get("scf_kmesh", raw.get("phonon_scf_kmesh", raw.get("phonon_kmesh"))),
            DEFAULT_PHONON_SCF_KMESH,
        ),
        "epc_fit_kmesh": normalize_epc_kmesh(
            raw.get("epc_fit_kmesh", raw.get("fit_kmesh", raw.get("dense_fit_kmesh"))),
            DEFAULT_EPC_FIT_KMESH,
        ),
        "epc_scf_kmesh": normalize_epc_kmesh(
            raw.get("epc_scf_kmesh", raw.get("epc_phonon_scf_kmesh")),
            DEFAULT_EPC_SCF_KMESH,
        ),
        "epc_q_grid": normalize_phonon_nq_grid(raw.get("epc_q_grid", raw.get("epc_nq", raw.get("a2f_q_grid"))), DEFAULT_EPC_Q_GRID),
        "fildvscf": str(raw.get("fildvscf") or "dvscf").strip() or "dvscf",
        "electron_phonon": str(raw.get("electron_phonon") or "interpolated").strip() or "interpolated",
        "epc_tr2_ph": epc_tr2_ph,
        "epc_ndos": positive_int(raw, ["epc_ndos", "ndos"], 50),
        "lambda_emax": str(raw.get("lambda_emax") or raw.get("emax") or "auto").strip() or "auto",
        "lambda_degauss": str(raw.get("lambda_degauss") or "0.1").strip() or "0.1",
        "lambda_smearing_method": positive_int(raw, ["lambda_smearing_method", "smearing_method"], 1),
        "lambda_mu": str(raw.get("lambda_mu") or raw.get("mu") or "0.10").strip() or "0.10",
        "lambda_expected_tc": optional_float(raw, ["lambda_expected_tc", "expected_tc", "experimental_tc"]),
        "lambda_tc_tolerance": positive_float(raw, ["lambda_tc_tolerance", "tc_tolerance"], 1.0),
        "lambda_degauss_scan_max": min(0.8, positive_float(raw, ["lambda_degauss_scan_max", "max_degauss"], 0.8)),
        "lambda_q_points": normalize_epc_lambda_q_points(raw.get("lambda_q_points", raw.get("epc_lambda_q_points"))),
        "q2r_zasr": str(raw.get("q2r_zasr") or q2r_raw.get("zasr") or "crystal").strip() or "crystal",
        "matdyn_asr": str(raw.get("matdyn_asr") or matdyn_raw.get("asr") or "crystal").strip() or "crystal",
        "matdyn_q_in_band_form": _truthy_option(
            raw.get("matdyn_q_in_band_form", matdyn_raw.get("q_in_band_form", True))
        ),
        "matdyn_q_path": q_path,
        "tr2_ph": tr2_ph,
        "alpha_mix": alpha_mix,
        "nmix_ph": nmix_ph,
        "verbosity": verbosity,
        "trans": _truthy_option(raw.get("trans", True)),
        "epsil": _truthy_option(raw.get("epsil", False)),
    }


def normalize_cutoff_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize cutoff convergence settings."""

    raw = dict(options or {})
    enabled = _truthy_option(raw.get("enabled", False))
    start = optional_int(raw, ["start", "min", "minimum", "min_ecutwfc", "ecut_start"])
    step = positive_int(raw, ["step", "ecut_step", "step_size"], 10)
    maximum = positive_int(raw, ["maximum", "max", "max_ecutwfc", "ecut_max"], 120)
    window_size = positive_int(raw, ["window_size", "initial_window", "planned_window"], 5)
    expand_by = positive_int(raw, ["expand_by", "growth_size", "extension_size"], 2)
    energy_delta_ry = optional_float(raw, ["energy_delta_ry", "threshold_ry"])
    if energy_delta_ry is None:
        energy_delta_ev = optional_float(raw, ["energy_delta_ev", "threshold_ev"])
        energy_delta_ry = (energy_delta_ev / RY_TO_EV) if energy_delta_ev is not None else 0.001 / RY_TO_EV

    if start is not None and start <= 0:
        start = None
    if start is not None:
        maximum = max(start, maximum)

    return {
        "enabled": enabled,
        "start": start,
        "step": step,
        "maximum": maximum,
        "window_size": window_size,
        "expand_by": expand_by,
        "energy_delta_ry": max(0.0, float(energy_delta_ry)),
    }


def normalize_kpoint_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize k-point convergence settings."""

    raw = dict(options or {})
    enabled = _truthy_option(raw.get("enabled", False))
    start_spacing = optional_float(raw, ["start_spacing", "spacing", "max_spacing"])
    spacing_step = positive_float(raw, ["spacing_step", "step"], 0.025)
    max_mesh_size = positive_int(raw, ["max_mesh_size", "maximum_mesh_size"], 14000)
    window_size = positive_int(raw, ["window_size", "initial_window", "planned_window"], 5)
    expand_by = positive_int(raw, ["expand_by", "growth_size", "extension_size"], 2)
    planned_meshes = raw.get("planned_meshes") or []
    energy_delta_ry = optional_float(raw, ["energy_delta_ry", "threshold_ry"])
    if energy_delta_ry is None:
        energy_delta_ev = optional_float(raw, ["energy_delta_ev", "threshold_ev"])
        energy_delta_ry = (energy_delta_ev / RY_TO_EV) if energy_delta_ev is not None else 0.001 / RY_TO_EV

    return {
        "enabled": enabled,
        "start_spacing": start_spacing,
        "spacing_step": spacing_step,
        "max_mesh_size": max_mesh_size,
        "window_size": window_size,
        "expand_by": expand_by,
        "planned_meshes": normalize_meshes(planned_meshes),
        "energy_delta_ry": max(0.0, float(energy_delta_ry)),
    }


def build_prepare_source_child(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Initial state child for copying or converting the source structure."""

    source = manifest.get("source") or {}
    requires_conversion = bool(source.get("requires_conversion", True))
    label = "convert_structure.workflow" if requires_conversion else "prepare_source.workflow"
    log_name = "convert_structure.log" if requires_conversion else "prepare_source.log"
    return {
        "id": "convert_structure" if requires_conversion else "prepare_source",
        "label": label,
        "status": "queued",
        "workdir": SOURCE_DIR,
        "input": source.get("stored_original", ""),
        "output": source.get("prepared_input", ""),
        "log": str(Path(LOGS_DIR) / log_name),
        "archive_inputs_outputs": [],
    }


def build_cutoff_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for the current cutoff convergence window."""

    params = task.get("params") or {}
    start = params.get("start")
    step = int(params.get("step") or 10)
    maximum = int(params.get("maximum") or 120)
    window_size = int(params.get("window_size") or 5)
    if start is None:
        return []

    values: list[int] = []
    current = int(start)
    while current <= maximum and len(values) < window_size:
        values.append(current)
        current += step

    return [build_cutoff_child(value) for value in values]


def build_cutoff_child(ecutwfc: int) -> dict[str, Any]:
    """State child for one cutoff SCF calculation."""

    child_id = f"ecut{int(ecutwfc)}"
    workdir = Path(CONVERGENCE_DIR) / CUTOFF_CONVERGENCE_DIR / child_id
    input_name = f"{child_id}.scf.in"
    output_name = f"{child_id}.scf.out"
    return {
        "id": child_id,
        "label": f"{child_id}.scf.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / input_name),
            str(Path(INPUTS_OUTPUTS_DIR) / output_name),
        ],
        "params": {"ecutwfc": int(ecutwfc)},
        "energy_ry": None,
    }


def build_kpoint_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for k-point convergence, if meshes are already known."""

    params = task.get("params") or {}
    meshes = params.get("planned_meshes") or []
    return [build_kpoint_child(mesh) for mesh in normalize_meshes(meshes)]


def build_kpoint_child(mesh: Sequence[int]) -> dict[str, Any]:
    """State child for one k-point SCF calculation."""

    nk1, nk2, nk3 = [int(value) for value in mesh[:3]]
    child_id = f"kmesh_{nk1}x{nk2}x{nk3}"
    workdir = Path(CONVERGENCE_DIR) / KPOINT_CONVERGENCE_DIR / child_id
    input_name = f"{child_id}.scf.in"
    output_name = f"{child_id}.scf.out"
    return {
        "id": child_id,
        "label": f"{child_id}.scf.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / input_name),
            str(Path(INPUTS_OUTPUTS_DIR) / output_name),
        ],
        "params": {"kmesh": [nk1, nk2, nk3]},
        "energy_ry": None,
    }


def build_relaxation_child(task: Mapping[str, Any]) -> dict[str, Any]:
    """Initial child job for either relax or vc-relax."""

    task_type = str(task.get("type") or "qe.relax")
    task_id = str(task.get("id") or "relaxation")
    params = dict(task.get("params") or {})
    stage = str(params.get("stage") or "single").strip().lower()
    if stage not in {"initial", "final", "single"}:
        stage = "single"

    calc = "vc-relax" if task_type == "qe.vc_relax" else "relax"
    child_id = task_id if task_id else "relaxation"
    file_stem = calc if stage == "single" else f"{stage}_{calc}"
    input_name = f"{file_stem}.in"
    output_name = f"{file_stem}.out"
    workdir = Path(RELAXATION_DIR) if stage == "single" else Path(RELAXATION_DIR) / stage
    child_params = dict(params)
    child_params["calculation"] = calc
    return {
        "id": child_id,
        "label": f"{file_stem}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{file_stem}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / input_name),
            str(Path(INPUTS_OUTPUTS_DIR) / output_name),
        ],
        "params": child_params,
    }


def build_bands_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for band/fat-band workflows."""

    params = dict(task.get("params") or {})
    variant = str(params.get("variant") or "bands").strip().lower()
    base_dir = FAT_BANDS_DIR if variant == "fat_bands" else BANDS_DIR
    include_projwfc = variant == "fat_bands"
    children = [
        build_bands_child(base_dir, child_id="scf", input_name="scf.in", output_name="scf.out", engine="pw.x"),
        build_bands_child(
            base_dir,
            child_id="pwbands",
            input_name="pwbands.in",
            output_name="pwbands.out",
            engine="pw.x",
        ),
        build_bands_child(
            base_dir,
            child_id="bandx",
            input_name="bandx.in",
            output_name="bandx.out",
            engine="bands.x",
        ),
    ]
    if include_projwfc:
        children.append(
            build_bands_child(
                base_dir,
                child_id="projwfc",
                input_name="projwfc.in",
                output_name="projwfc.out",
                engine="projwfc.x",
            )
        )
    return children


def build_wannier90_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for the Wannier90 + SCDM workflow."""

    _ = task
    return [
        build_bands_child(WANNIER90_DIR, child_id="scf", input_name="scf.in", output_name="scf.out", engine="pw.x"),
        build_bands_child(
            WANNIER90_DIR,
            child_id="pwbands",
            input_name="pwbands.in",
            output_name="pwbands.out",
            engine="pw.x",
        ),
        build_bands_child(
            WANNIER90_DIR,
            child_id="bandx",
            input_name="bandx.in",
            output_name="bandx.out",
            engine="bands.x",
        ),
        build_bands_child(WANNIER90_DIR, child_id="nscf", input_name="nscf.in", output_name="nscf.out", engine="pw.x"),
        build_bands_child(
            WANNIER90_DIR,
            child_id="projwfc",
            input_name="projwfc.in",
            output_name="projwfc.out",
            engine="projwfc.x",
        ),
        build_wannier90_scdm_fitting_child(),
        build_wannier90_pw2wan_input_child(),
        build_wannier90_win_input_child(),
        build_wannier90_preprocess_child(),
        build_wannier90_pw2wan_run_child(),
        build_wannier90_run_child(),
    ]


def build_wannier90_scdm_fitting_child() -> dict[str, Any]:
    """Generate projectability weights and fit the SCDM window parameters."""

    return {
        "id": "scdm_fitting",
        "label": "SCDM fitting",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "projwfc.out"),
        "output": str(Path(WANNIER90_DIR) / "projectability_fit.json"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_scdm_fitting.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_projwfc.out"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_projectability_fit.json"),
        ],
        "params": {
            "engine": "python",
            "stage": "scdm_fitting",
        },
    }

def build_wannier90_pw2wan_input_child() -> dict[str, Any]:
    """Render pw2wannier90.x input from the selected SCDM fit."""

    return {
        "id": "pw2wan_input",
        "label": "pw2wan_input.workflow",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "projectability_fit.json"),
        "output": str(Path(WANNIER90_DIR) / "pw2wan.in"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_pw2wan_input.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_projectability_fit.json"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_pw2wan.in"),
        ],
        "params": {
            "engine": "python",
            "stage": "pw2wan_input",
        },
    }


def build_wannier90_win_input_child() -> dict[str, Any]:
    """Render the Wannier90 control file from workflow band/kmesh data."""

    return {
        "id": "wannier_control",
        "label": "wannier_control.workflow",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "pw2wan.in"),
        "output": str(Path(WANNIER90_DIR) / "wannier_control.win"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_wannier_control.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_pw2wan.in"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_wannier_control.win"),
        ],
        "params": {
            "engine": "python",
            "stage": "wannier_control",
        },
    }


def build_wannier90_preprocess_child() -> dict[str, Any]:
    """Run wannier90.x -pp to generate the nnkp file."""

    return {
        "id": "wannier90_preprocess",
        "label": "wannier90_preprocess.workflow",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "wannier_control.win"),
        "output": str(Path(WANNIER90_DIR) / "wannier_control.nnkp"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_wannier90_preprocess.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_wannier_control.win"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_wannier_control.nnkp"),
        ],
        "params": {
            "engine": "wannier90.x",
            "stage": "wannier90_preprocess",
        },
    }


def build_wannier90_pw2wan_run_child() -> dict[str, Any]:
    """Run pw2wannier90.x after Wannier90 preprocessing."""

    return {
        "id": "pw2wan_run",
        "label": "pw2wan_run.workflow",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "pw2wan.in"),
        "output": str(Path(WANNIER90_DIR) / "pw2wan.out"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_pw2wan_run.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_pw2wan.in"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_pw2wan.out"),
        ],
        "params": {
            "engine": "pw2wannier90.x",
            "stage": "pw2wan_run",
        },
    }


def build_wannier90_run_child() -> dict[str, Any]:
    """Run the final wannier90.x calculation."""

    return {
        "id": "wannier90_run",
        "label": "wannier90_run.workflow",
        "status": "queued",
        "workdir": WANNIER90_DIR,
        "input": str(Path(WANNIER90_DIR) / "wannier_control.win"),
        "output": str(Path(WANNIER90_DIR) / "wannier_control.wout"),
        "log": str(Path(LOGS_DIR) / f"{WANNIER90_DIR}_wannier90_run.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_wannier_control.win"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{WANNIER90_DIR}_wannier_control.wout"),
        ],
        "params": {
            "engine": "wannier90.x",
            "stage": "wannier90_run",
        },
    }


def build_dos_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for DOS/PDOS workflows."""

    _ = task
    return [
        build_dos_child(child_id="scf", input_name="scf.in", output_name="scf.out", engine="pw.x"),
        build_dos_child(child_id="nscf", input_name="nscf.in", output_name="nscf.out", engine="pw.x"),
        build_dos_child(child_id="projwfc", input_name="projwfc.in", output_name="projwfc.out", engine="projwfc.x"),
        {
            "id": "pdos_sumpdos",
            "label": "pdos_sumpdos.workflow",
            "status": "queued",
            "workdir": DOS_DIR,
            "input": str(Path(DOS_DIR) / "projwfc.out"),
            "output": str(Path(DATA_PLOTS_DIR) / "pdos_sumpdos_data.json"),
            "log": str(Path(LOGS_DIR) / f"{DOS_DIR}_pdos_sumpdos.log"),
            "archive_inputs_outputs": [
                str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_projwfc.out"),
                str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_pdos_sumpdos_data.json"),
            ],
            "params": {
                "engine": "python",
                "stage": "pdos_sumpdos",
            },
        },
        {
            "id": "pdos_plotting",
            "label": "pdos_plotting.workflow",
            "status": "queued",
            "workdir": DOS_DIR,
            "input": str(Path(DATA_PLOTS_DIR) / "pdos_sumpdos_data.json"),
            "output": str(Path(DATA_PLOTS_DIR) / "pdos_postprocess.json"),
            "log": str(Path(LOGS_DIR) / f"{DOS_DIR}_pdos_plotting.log"),
            "archive_inputs_outputs": [
                str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_pdos_sumpdos_data.json"),
                str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_pdos_postprocess.json"),
            ],
            "params": {
                "engine": "python",
                "stage": "pdos_plotting",
            },
        },
    ]


def build_fermi_surface_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for Fermi-surface velocity workflows."""

    _ = task
    return [
        build_fermi_surface_child(
            child_id="scf",
            input_name="scf.in",
            output_name="scf.out",
            engine="pw.x",
        ),
        build_fermi_surface_child(
            child_id="fermi_velocity",
            input_name="scf.in",
            output_name="FS_vf.out",
            engine="fermi_velocity.x",
        ),
    ]


def build_fermi_surface_child(
    *,
    child_id: str,
    input_name: str,
    output_name: str,
    engine: str,
) -> dict[str, Any]:
    """State child for one stage of the Fermi-surface workflow."""

    workdir = Path(FERMI_SURFACE_DIR)
    archive_inputs_outputs = [
        str(Path(INPUTS_OUTPUTS_DIR) / f"{FERMI_SURFACE_DIR}_{child_id}_{input_name}"),
        str(Path(INPUTS_OUTPUTS_DIR) / f"{FERMI_SURFACE_DIR}_{child_id}_{output_name}"),
    ]
    return {
        "id": child_id,
        "label": f"{child_id}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{FERMI_SURFACE_DIR}_{child_id}.log"),
        "archive_inputs_outputs": archive_inputs_outputs,
        "params": {
            "engine": engine,
            "stage": child_id,
        },
    }


def build_charge_density_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for charge-density cube workflows."""

    _ = task
    return [
        build_charge_density_child(
            child_id="scf",
            input_name="scf.in",
            output_name="scf.out",
            engine="pw.x",
        ),
        build_charge_density_child(
            child_id="pp",
            input_name="pp.in",
            output_name="pp.out",
            engine="pp.x",
        ),
    ]


def build_charge_density_child(
    *,
    child_id: str,
    input_name: str,
    output_name: str,
    engine: str,
) -> dict[str, Any]:
    """State child for one stage of the charge-density workflow."""

    workdir = Path(CHARGE_DENSITY_DIR)
    return {
        "id": child_id,
        "label": f"{child_id}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{CHARGE_DENSITY_DIR}_{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{CHARGE_DENSITY_DIR}_{child_id}_{input_name}"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{CHARGE_DENSITY_DIR}_{child_id}_{output_name}"),
        ],
        "params": {
            "engine": engine,
            "stage": child_id,
        },
    }


def build_elastic_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for elastic-property workflows."""

    _ = task
    return [build_elastic_child()]


def build_phonon_children(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Initial child jobs for the phonon workflow."""

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    phonon_options = params.get("phonon_options") if isinstance(params.get("phonon_options"), Mapping) else {}
    mode = str(phonon_options.get("mode") or phonon_options.get("calculation_mode") or "dispersion").lower()
    if mode == EPC_MODE:
        return [
            build_phonon_child(child_id="epc_fit_scf", input_name="epc_fit_scf.in", output_name="epc_fit_scf.out", engine="pw.x"),
            build_phonon_child(child_id="epc_scf", input_name="epc_scf.in", output_name="epc_scf.out", engine="pw.x"),
            build_phonon_child(child_id="epc_ph", input_name="epc_ph.in", output_name="epc_ph.out", engine="ph.x"),
            build_phonon_child(child_id="epc_q2r", input_name="epc_q2r.in", output_name="epc_q2r.out", engine="q2r.x"),
            build_phonon_child(
                child_id="epc_matdyn",
                input_name="epc_matdyn.in",
                output_name="epc_matdyn.out",
                engine="matdyn.x",
            ),
            build_phonon_child(
                child_id="epc_phdos",
                input_name="epc_phdos.in",
                output_name="epc_phdos.out",
                engine="matdyn.x",
            ),
            build_phonon_ploting_child(phonon_options),
            build_phonon_child(child_id="lambda", input_name="lambda.in", output_name="lambda.out", engine="lambda.x"),
            build_phonon_child(
                child_id="lambda_ploting",
                input_name="lambda.out",
                output_name="lambda_ploting.json",
                engine="python",
            ),
        ]
    children = [
        build_phonon_child(child_id="scf", input_name="scf.in", output_name="scf.out", engine="pw.x"),
        build_phonon_child(child_id="ph", input_name="ph.in", output_name="ph.out", engine="ph.x"),
    ]
    if mode != "single_q":
        children.append(
            build_phonon_child(child_id="q2r", input_name="q2r.in", output_name="q2r.out", engine="q2r.x")
        )
        if bool(phonon_options.get("band_enabled", True)):
            children.append(
                build_phonon_child(
                    child_id="matdyn",
                    input_name="matdyn.in",
                    output_name="matdyn.out",
                    engine="matdyn.x",
                ),
            )
        if bool(phonon_options.get("dos_enabled", True)):
            children.append(
                build_phonon_child(
                    child_id="phdos",
                    input_name="phdos.in",
                    output_name="phdos.out",
                    engine="matdyn.x",
                )
            )
        if (
            bool(phonon_options.get("phonon_projection_plot_enabled", False))
            and bool(phonon_options.get("band_enabled", True))
            and bool(phonon_options.get("dos_enabled", True))
        ):
            children.append(build_phonon_ploting_child(phonon_options))
    return children


def build_phonon_ploting_child(phonon_options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": "ploting",
        "label": "ploting_phonon_postprocess.workflow",
        "status": "queued",
        "workdir": DATA_PLOTS_DIR,
        "input": str(Path(PHONON_DIR) / str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")),
        "output": str(Path(DATA_PLOTS_DIR) / "phonon_projected_plot_data.json"),
        "log": str(Path(LOGS_DIR) / f"{PHONON_DIR}_projected.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / Path(str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")).name),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{PHONON_DIR}_projected_plot_data.json"),
        ],
        "params": {
            "engine": "python",
            "stage": "ploting",
        },
    }


def build_phonon_child(
    *,
    child_id: str,
    input_name: str,
    output_name: str,
    engine: str,
) -> dict[str, Any]:
    """State child for one stage of the phonon workflow."""

    workdir = Path(PHONON_DIR)
    input_base = Path(input_name).name
    output_base = Path(output_name).name
    return {
        "id": child_id,
        "label": f"{child_id}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{PHONON_DIR}_{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{PHONON_DIR}_{input_base}"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{PHONON_DIR}_{output_base}"),
        ],
        "params": {
            "engine": engine,
            "stage": child_id,
        },
    }


def build_dos_child(
    *,
    child_id: str,
    input_name: str,
    output_name: str,
    engine: str,
) -> dict[str, Any]:
    """State child for one stage of the DOS/PDOS workflow."""

    workdir = Path(DOS_DIR)
    input_base = Path(input_name).name
    output_base = Path(output_name).name
    return {
        "id": child_id,
        "label": f"{child_id}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{DOS_DIR}_{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_{input_base}"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{DOS_DIR}_{output_base}"),
        ],
        "params": {
            "engine": engine,
            "stage": child_id,
        },
    }


def build_elastic_child() -> dict[str, Any]:
    """State child for the thermo_pw elastic workflow."""

    workdir = Path(ELASTIC_DIR)
    return {
        "id": "elastic",
        "label": "elastic.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / "scf.in"),
        "control": str(workdir / "thermo_control"),
        "output": str(workdir / "thermo_pw.out"),
        "log": str(Path(LOGS_DIR) / "elastic.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / "elastic_scf.in"),
            str(Path(INPUTS_OUTPUTS_DIR) / "elastic_thermo_control"),
            str(Path(INPUTS_OUTPUTS_DIR) / "elastic_thermo_pw.out"),
        ],
        "params": {
            "engine": "thermo_pw.x",
            "stage": "elastic",
        },
    }


def build_fatband_csv_child(task: Mapping[str, Any]) -> dict[str, Any]:
    """Initial child job for extracting fat-band projection CSV tables."""

    _ = task
    return {
        "id": "fatband_csv",
        "label": "fatband_csv.workflow",
        "status": "queued",
        "workdir": BANDS_DIR,
        "input": str(Path(BANDS_DIR) / "projwfc.out"),
        "output": str(Path(DATA_PLOTS_DIR) / "fatband_projection_csv" / "all_states.csv"),
        "log": str(Path(LOGS_DIR) / "fatband_csv.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / "bands_projwfc.out"),
            str(Path(INPUTS_OUTPUTS_DIR) / "fatband_all_states.csv"),
        ],
        "params": {
            "stage": "fatband_csv",
        },
    }


def build_fatband_plot_child(task: Mapping[str, Any]) -> dict[str, Any]:
    """Initial child job for plotting fat-band figures from extracted CSV data."""

    params = task.get("params") or {}
    color_mode = str(params.get("fatband_color_mode") or "dominant")
    return {
        "id": "fatband_plot",
        "label": "fatband_plot.workflow",
        "status": "queued",
        "workdir": BANDS_DIR,
        "input": str(Path(DATA_PLOTS_DIR) / "fatband_projection_csv" / "all_states.csv"),
        "output": str(Path(DATA_PLOTS_DIR) / "fatband_plot_data.json"),
        "log": str(Path(LOGS_DIR) / "fatband_plot.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / "fatband_all_states.csv"),
            str(Path(INPUTS_OUTPUTS_DIR) / "fatband_plot_data.json"),
        ],
        "params": {
            "stage": "fatband_plot",
            "fatband_color_mode": color_mode,
        },
    }


def build_bands_child(
    base_dir: str,
    *,
    child_id: str,
    input_name: str,
    output_name: str,
    engine: str,
) -> dict[str, Any]:
    """State child for one stage of the band/fat-band workflow."""

    workdir = Path(base_dir)
    return {
        "id": child_id,
        "label": f"{child_id}.workflow",
        "status": "queued",
        "workdir": str(workdir),
        "input": str(workdir / input_name),
        "output": str(workdir / output_name),
        "log": str(Path(LOGS_DIR) / f"{base_dir}_{child_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / f"{base_dir}_{input_name}"),
            str(Path(INPUTS_OUTPUTS_DIR) / f"{base_dir}_{output_name}"),
        ],
        "params": {
            "engine": engine,
            "stage": child_id,
        },
    }


def build_generic_child(task: Mapping[str, Any]) -> dict[str, Any]:
    """Initial child job for future workflow task types."""

    task_id = str(task.get("id") or "workflow_task")
    workdir = task_id
    input_name = f"{task_id}.in"
    output_name = f"{task_id}.out"
    return {
        "id": task_id,
        "label": f"{task_id}.workflow",
        "status": "queued",
        "workdir": workdir,
        "input": str(Path(workdir) / input_name),
        "output": str(Path(workdir) / output_name),
        "log": str(Path(LOGS_DIR) / f"{task_id}.log"),
        "archive_inputs_outputs": [
            str(Path(INPUTS_OUTPUTS_DIR) / input_name),
            str(Path(INPUTS_OUTPUTS_DIR) / output_name),
        ],
        "params": {},
    }


def workflow_task_metadata(key: str) -> dict[str, str]:
    """Map Step 3 workflow keys to generic future task metadata."""

    normalized = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
    known = {
        "bands": {"id": "bands", "type": "qe.bands", "label": "Band structure"},
        "band": {"id": "bands", "type": "qe.bands", "label": "Band structure"},
        "fat_bands": {"id": "fat_bands", "type": "qe.fat_bands", "label": "Fat-band"},
        "fat_band": {"id": "fat_bands", "type": "qe.fat_bands", "label": "Fat-band"},
        "fatbands": {"id": "fat_bands", "type": "qe.fat_bands", "label": "Fat-band"},
        "pdos": {"id": "dos_workflow", "type": "qe.dos_workflow", "label": "Projected density of states workflow"},
        "dos": {"id": "dos_workflow", "type": "qe.dos_workflow", "label": "Projected density of states workflow"},
        "fermi_surface": {"id": "fermi_surface_workflow", "type": "qe.fermi_surface_workflow", "label": "Fermi surface workflow"},
        "fermisurface": {"id": "fermi_surface_workflow", "type": "qe.fermi_surface_workflow", "label": "Fermi surface workflow"},
        "charge_density": {"id": "charge_density_workflow", "type": "qe.charge_density_workflow", "label": "Charge density workflow"},
        "chargedensity": {"id": "charge_density_workflow", "type": "qe.charge_density_workflow", "label": "Charge density workflow"},
        "elastic": {"id": "elastic_workflow", "type": "qe.elastic_workflow", "label": "Elastic properties workflow"},
        "elastic_properties": {"id": "elastic_workflow", "type": "qe.elastic_workflow", "label": "Elastic properties workflow"},
        "elastic_property": {"id": "elastic_workflow", "type": "qe.elastic_workflow", "label": "Elastic properties workflow"},
        "elastic_constants": {"id": "elastic_workflow", "type": "qe.elastic_workflow", "label": "Elastic properties workflow"},
        "phonon": {"id": "phonon_workflow", "type": "qe.phonon_workflow", "label": "Phonon workflow"},
        "wannier": {"id": "wannier90_workflow", "type": "qe.wannier90_workflow", "label": "Wannier90 + SCDM workflow"},
        "wannier90": {"id": "wannier90_workflow", "type": "qe.wannier90_workflow", "label": "Wannier90 + SCDM workflow"},
        "winner90": {"id": "wannier90_workflow", "type": "qe.wannier90_workflow", "label": "Wannier90 + SCDM workflow"},
    }
    if normalized in known:
        return known[normalized]
    safe = sanitize_job_name(normalized or "workflow_task")
    return {"id": safe, "type": f"custom.{safe}", "label": safe.replace("_", " ").title()}


def create_workflow_directories(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Create the workflow directory tree, including currently planned child dirs."""

    root = Path(str(manifest.get("workflow_root") or "")).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    for relative in [SOURCE_DIR, INPUTS_OUTPUTS_DIR, LOGS_DIR, DATA_PLOTS_DIR]:
        (root / relative).mkdir(parents=True, exist_ok=True)

    for task in manifest.get("tasks", []) or []:
        if not isinstance(task, Mapping):
            continue
        task_type = str(task.get("type") or "")
        if task_type == "convergence.cutoff":
            (root / CONVERGENCE_DIR / CUTOFF_CONVERGENCE_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "convergence.kpoints":
            (root / CONVERGENCE_DIR / KPOINT_CONVERGENCE_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type in {"qe.relax", "qe.vc_relax"}:
            (root / RELAXATION_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.bands_workflow":
            (root / BANDS_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.fat_bands_workflow":
            (root / FAT_BANDS_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.dos_workflow":
            (root / DOS_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.fermi_surface_workflow":
            (root / FERMI_SURFACE_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.charge_density_workflow":
            (root / CHARGE_DENSITY_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.elastic_workflow":
            (root / ELASTIC_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.phonon_workflow":
            (root / PHONON_DIR).mkdir(parents=True, exist_ok=True)
        elif task_type == "qe.wannier90_workflow":
            (root / WANNIER90_DIR).mkdir(parents=True, exist_ok=True)

    tasks = state.get("tasks") or {}
    if isinstance(tasks, Mapping):
        for task_state in tasks.values():
            if not isinstance(task_state, Mapping):
                continue
            for child in task_state.get("children", []) or []:
                if not isinstance(child, Mapping):
                    continue
                workdir = str(child.get("workdir") or "").strip()
                if workdir:
                    (root / workdir).mkdir(parents=True, exist_ok=True)


def copy_source_file(manifest: Mapping[str, Any]) -> None:
    """Copy the user-selected source file into source/ when available."""

    root = Path(str(manifest.get("workflow_root") or "")).expanduser()
    source = manifest.get("source") or {}
    original = Path(str(source.get("original") or "")).expanduser()
    stored_relative = str(source.get("stored_original") or "").strip()
    if not original.is_file() or not stored_relative:
        return
    stored = root / stored_relative
    stored.parent.mkdir(parents=True, exist_ok=True)
    if original.resolve() == stored.resolve():
        return
    shutil.copy2(original, stored)


def copy_manifest_pseudo_files(manifest: Mapping[str, Any]) -> None:
    """Copy manifest-selected pseudopotentials into source/pseudo."""

    root = Path(str(manifest.get("workflow_root") or "")).expanduser()
    conversion_options = manifest.get("conversion")
    if not isinstance(conversion_options, Mapping):
        return
    copy_selected_workflow_pseudos(conversion_options, root / SOURCE_DIR / PSEUDO_DIR_NAME)


def write_json_file(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def optional_int(raw: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def positive_int(raw: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    value = optional_int(raw, keys)
    if value is None:
        return max(1, int(default))
    return max(1, int(value))


def optional_float(raw: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def positive_float(raw: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    value = optional_float(raw, keys)
    if value is None:
        return max(0.0, float(default))
    return max(0.0, float(value))


def normalize_meshes(meshes: Any) -> list[list[int]]:
    """Normalize a sequence of k-point meshes into [[nk1, nk2, nk3], ...]."""

    normalized: list[list[int]] = []
    if not isinstance(meshes, Sequence) or isinstance(meshes, (str, bytes)):
        return normalized
    for mesh in meshes:
        if isinstance(mesh, str):
            parts = mesh.lower().replace("x", " ").replace(",", " ").split()
        elif isinstance(mesh, Sequence):
            parts = list(mesh)
        else:
            continue
        if len(parts) < 3:
            continue
        try:
            nk1, nk2, nk3 = [max(1, int(value)) for value in parts[:3]]
        except (TypeError, ValueError):
            continue
        normalized.append([nk1, nk2, nk3])
    return normalized


def normalize_q_point(value: Any) -> list[float]:
    """Normalize a ph.x q-vector, defaulting to Gamma."""

    if isinstance(value, str):
        parts: list[Any] = value.replace(",", " ").split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        return [0.0, 0.0, 0.0]
    if len(parts) < 3:
        return [0.0, 0.0, 0.0]
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]


def normalize_phonon_nq_grid(value: Any, default: Sequence[int] | None = None) -> list[int]:
    """Normalize a ph.x/matdyn grid using the caller's workflow default."""

    default_grid = [max(1, int(float(part))) for part in (default or DEFAULT_PHONON_Q_GRID)[:3]]
    if len(default_grid) < 3:
        default_grid = list(DEFAULT_PHONON_Q_GRID)
    if isinstance(value, str):
        parts: list[Any] = value.lower().replace("x", " ").replace(",", " ").split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        return default_grid
    if len(parts) < 3:
        return default_grid
    try:
        return [max(1, int(float(parts[0]))), max(1, int(float(parts[1]))), max(1, int(float(parts[2])))]
    except (TypeError, ValueError):
        return default_grid


def normalize_epc_kmesh(value: Any, default: Sequence[int]) -> list[int]:
    meshes = normalize_meshes([value])
    if meshes:
        return meshes[0]
    meshes = normalize_meshes(value)
    if meshes:
        return meshes[0]
    return [max(1, int(item)) for item in list(default)[:3]]


def normalize_epc_lambda_q_points(value: Any) -> list[list[float]]:
    """Normalize lambda.x q-points as qx qy qz weight rows."""

    rows: list[Any]
    if value is None or value == "":
        rows = list(DEFAULT_EPC_LAMBDA_Q_POINTS)
    elif isinstance(value, str):
        rows = [line.strip() for line in value.replace(";", "\n").splitlines() if line.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        rows = list(value)
    else:
        rows = list(DEFAULT_EPC_LAMBDA_Q_POINTS)

    normalized: list[list[float]] = []
    for row in rows:
        if isinstance(row, str):
            parts = row.split("!", 1)[0].replace(",", " ").split()
        elif isinstance(row, Mapping):
            parts = [row.get("qx"), row.get("qy"), row.get("qz"), row.get("weight", row.get("w", 1.0))]
        elif isinstance(row, Sequence) and not isinstance(row, (bytes, bytearray)):
            parts = list(row)
        else:
            continue
        if len(parts) < 4:
            continue
        try:
            normalized.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except (TypeError, ValueError):
            continue
    return normalized or [list(point) for point in DEFAULT_EPC_LAMBDA_Q_POINTS]


def default_matdyn_q_path() -> list[str]:
    """Default crystal-coordinate path for a first phonon dispersion input."""

    return [
        "5",
        "0.0 0.0 0.0 40",
        "0.5 0.0 0.0 40",
        "0.3333333333 0.3333333333 0.0 40",
        "0.0 0.0 0.0 40",
        "0.0 0.0 0.5 1",
    ]


def normalize_matdyn_q_path(value: Any) -> list[str]:
    """Normalize matdyn.x band-form q-path lines."""

    if isinstance(value, str):
        lines = [line.strip() for line in value.replace(";", "\n").splitlines() if line.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        lines = [str(line).strip() for line in value if str(line).strip()]
    else:
        lines = []
    if not lines:
        return default_matdyn_q_path()

    first_parts = lines[0].split()
    if len(first_parts) == 1:
        try:
            int(float(first_parts[0]))
            return lines
        except (TypeError, ValueError):
            pass
    return [str(len(lines)), *lines]
