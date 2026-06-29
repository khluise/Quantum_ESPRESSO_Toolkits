"""Job configuration utilities for Quantum ESPRESSO Helper."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import SystemConfig
from .io import parse_scf_seed, render_scf_template


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMAND_PREFS_PATH = PROJECT_ROOT / "command_preferences.json"
NO_MPI_DEFAULT_COMMANDS = {"alpha2f.x", "projwfc.x"}


@dataclass
class JobConfig:
    name: str
    calculation: str
    mpi_procs: int
    queue: str
    working_directory: Optional[str] = None
    pseudo_directory: Optional[str] = None
    input_files: Sequence[str] = ()
    pseudo_files: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        files = self.input_files or ()
        self.input_files = tuple(str(path) for path in files)
        mapping: Dict[str, str] = {}
        for element, path in (self.pseudo_files or {}).items():
            if not element or not path:
                continue
            mapping[str(element)] = str(path)
        self.pseudo_files = mapping

    def sanitized_name(self) -> str:
        return sanitize_job_name(self.name or "qe_job")

    def script_filename(self) -> str:
        return f"{self.sanitized_name()}.sh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calculation": self.calculation,
            "mpi_procs": int(self.mpi_procs),
            "queue": self.queue,
            "working_directory": self.working_directory,
            "pseudo_directory": self.pseudo_directory,
            "input_files": list(self.input_files),
            "pseudo_files": dict(self.pseudo_files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobConfig":
        input_files = data.get("input_files") or []
        if isinstance(input_files, (str, bytes)):
            input_files = [input_files]
        try:
            mpi_procs = int(data.get("mpi_procs", 1))
        except Exception:  # noqa: BLE001
            mpi_procs = 1

        pseudo_files_raw = data.get("pseudo_files") or {}
        pseudo_files: Dict[str, str] = {}
        if isinstance(pseudo_files_raw, dict):
            for element, path in pseudo_files_raw.items():
                if path:
                    pseudo_files[str(element)] = str(path)

        return cls(
            name=str(data.get("name", "")),
            calculation=str(data.get("calculation", "")),
            mpi_procs=max(1, mpi_procs),
            queue=str(data.get("queue", "default")),
            working_directory=data.get("working_directory") or None,
            pseudo_directory=data.get("pseudo_directory") or None,
            input_files=input_files,
            pseudo_files=pseudo_files,
        )

def sanitize_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or "qe_job"

def create_run_command(
    calculation_type: str,
    input_file: str,
    output_file: str,
    number_of_cores: int,
    *,
    validate_input: bool = True,
) -> str:
    if not calculation_type or not isinstance(calculation_type, str):
        raise ValueError("calculation_type must be a non-empty string")
    if not input_file or not output_file:
        raise ValueError("input_file and output_file must be provided")
    if not isinstance(number_of_cores, int) or number_of_cores < 1:
        raise ValueError("number_of_cores must be an integer >= 1")

    calc = calculation_type.strip().lower()

    binaries = {
        "scf": "pw.x",
        "nscf": "pw.x",
        "relax": "pw.x",
        "bands": "pw.x",
        "bandx": "bands.x",
        "dos": "dos.x",
        "pdos": "projwfc.x",
        "test": "test.x",
        "phonon": "ph.x",
        "q2r": "q2r.x",
        "matdyn": "matdyn.x",
        "epsilon": "epsilon.x",
        "xspectra": "xspectra.x",
        "pw2wannier90": "pw2wannier90.x",
        "wannier90": "wannier90.x",
    }
    aliases = {
        "self_consistent": "scf",
        "non_self_consistent": "nscf",
        "vc-relax": "relax",
        "band": "bands",
        "band_structure": "bands",
        "density_of_states": "dos",
        "partial_dos": "pdos",
        "phonon_calculation": "phonon",
        "force_constants": "q2r",
        "phonon_dispersion": "matdyn",
        "matdyn.x": "matdyn",
        "optics": "epsilon",
        "eps": "epsilon",
        "wan": "wannier90",
        "wannier": "wannier90",
        "pw2wan": "pw2wannier90",
    }
    if calc in aliases:
        calc = aliases[calc]
    binary = binaries.get(calc)
    if binary is None:
        raise ValueError(f"Unknown calculation_type '{calculation_type}'. Contact support to add this type.")

    command_base = _resolve_command_base(binary)
    template = f"{command_base} -in {{infile}} | tee {{outfile}}"

    if validate_input and not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
    return template.format(n=number_of_cores, infile=input_file, outfile=output_file)


def _load_command_preferences() -> dict[str, Any]:
    if not COMMAND_PREFS_PATH.exists():
        return {}
    try:
        with COMMAND_PREFS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _resolve_command_base(binary: str) -> str:
    """Resolve command base from command_preferences.json with sensible fallbacks."""
    prefs = _load_command_preferences()
    commands = prefs.get("commands") or {}
    command_base = ""

    if isinstance(commands, dict):
        candidate_keys = [binary]
        if binary == "pw2wannier90.x":
            candidate_keys.append("wannier2pw.x")
        for key in candidate_keys:
            value = str(commands.get(key, "")).strip()
            if value:
                command_base = value
                break

    if command_base:
        return command_base

    use_mpirun = bool(prefs.get("use_mpirun", True))
    mpirun_suffix = str(prefs.get("mpirun_suffix", "mpirun")).strip()
    if use_mpirun and binary not in NO_MPI_DEFAULT_COMMANDS:
        prefix = mpirun_suffix or "mpirun"
        return f"{prefix} {binary}"
    return binary

def generate_wannier_win(
    seedname: str,
    *,
    num_wann: int = 10,
    num_bands: int = 20,
    mp_grid: tuple[int, int, int] = (4, 4, 4),
    projections: Optional[Sequence[str]] = None,
) -> str:
    nk1, nk2, nk3 = mp_grid
    lines: list[str] = []
    lines.append(f"# Wannier90 input (.win) for seed '{seedname}'")
    lines.append("# Generated by Quantum ESPRESSO Helper GUI; edit as needed")
    lines.append("")
    lines.append(f"num_wann = {num_wann}")
    lines.append(f"num_bands = {num_bands}")
    lines.append(f"mp_grid = {nk1} {nk2} {nk3}")
    lines.append("!")
    lines.append("! Optional disentanglement window (edit/remove if not used)")
    lines.append("dis_win_min = -10.0")
    lines.append("dis_win_max = 10.0")
    lines.append("")

    if projections:
        lines.append("begin projections")
        for proj in projections:
            lines.append(str(proj))
        lines.append("end projections")
        lines.append("")
    else:
        lines.append("! projections = sp3; d; random")
        lines.append("")

    lines.append("! bands_plot = true")
    lines.append("! begin kpoint_path")
    lines.append("! \tG 0.0 0.0 0.0  X 0.5 0.0 0.5")
    lines.append("! end kpoint_path")
    lines.append("")
    return "\n".join(lines)


def apply_pseudo_settings_to_text(
    text: str,
    mapping: Dict[str, str] | None = None,
    *,
    pseudo_dir: str | None = None,
) -> str:
    """Rewrite QE input text with pseudo assignments and control overrides."""
    if not text:
        return text

    updated = text
    if mapping:
        updated = _rewrite_atomic_species(updated, mapping)
    if pseudo_dir is not None:
        updated = _rewrite_pseudo_dir(updated, pseudo_dir)
    return updated


def ensure_control_verbosity(text: str, level: str = "'high'") -> str:
    """Ensure the &CONTROL namelist contains a verbosity setting."""
    if not text:
        return text

    lines = text.splitlines()
    result: list[str] = []
    in_control = False
    inserted = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if not in_control and upper.startswith("&CONTROL"):
            in_control = True
            inserted = False
            result.append(line)
            continue

        if in_control:
            if stripped == "/":
                if not inserted:
                    result.append(f"   verbosity = {level}")
                    inserted = True
                result.append(line)
                in_control = False
                continue

            body, comment = _split_comment(line)
            if "=" in body:
                key = body.split("=", 1)[0].strip().lower()
                if key == "verbosity":
                    indent = body[: len(body) - len(body.lstrip())]
                    stripped_body = body.strip()
                    trailing_comma = "," if stripped_body.endswith(",") else ""
                    trailing_ws = body[len(body.rstrip()):]
                    new_body = f"{indent}verbosity = {level}{trailing_comma}{trailing_ws}"
                    result.append(new_body + comment)
                    inserted = True
                    continue

            result.append(line)
            continue

        result.append(line)

    if in_control and not inserted:
        result.append(f"   verbosity = {level}")

    ensured = "\n".join(result)
    if text.endswith("\n") and not ensured.endswith("\n"):
        ensured += "\n"
    return ensured


def _rewrite_atomic_species(text: str, mapping: Dict[str, str]) -> str:
    if not text or not mapping:
        return text

    lookup = {}
    for key, value in mapping.items():
        if not key or not value:
            continue
        element_key = str(key).lower()
        try:
            name = Path(value).name or str(Path(value))
        except Exception:
            name = str(value)
        lookup[element_key] = name
    if not lookup:
        return text

    lines = text.splitlines()
    result: list[str] = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        if not in_block:
            result.append(line)
            if stripped.upper().startswith("ATOMIC_SPECIES"):
                in_block = True
            continue

        if not stripped or stripped.startswith("&") or stripped.upper().startswith(("ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS")):
            in_block = False
            result.append(line)
            continue

        body, comment = (line.split("!", 1) + [""])[:2]
        comment = f"!{comment}" if comment else ""
        indent_len = len(body) - len(body.lstrip())
        indent = body[:indent_len]
        tokens = body.strip().split()
        if len(tokens) >= 3:
            element = tokens[0]
            replacement = lookup.get(element.lower())
            if replacement:
                base_tokens = tokens[:]
                base_tokens[2] = replacement
                new_line = indent + " ".join(base_tokens)
                if comment:
                    new_line = f"{new_line} {comment.strip()}"
                result.append(new_line)
                continue

        result.append(line)

    return "\n".join(result)


def _rewrite_pseudo_dir(text: str, pseudo_dir: str) -> str:
    cleaned = (pseudo_dir or "").strip()
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    literal = f"'{cleaned}'" if cleaned else "''"

    lines = text.splitlines()
    rewritten: list[str] = []

    for line in lines:
        body, comment = _split_comment(line)
        stripped = body.strip()
        if stripped.lower().startswith("pseudo_dir"):
            lhs, sep, rhs = body.partition("=")
            if sep:
                leading = rhs[: len(rhs) - len(rhs.lstrip())]
                rhs_stripped = rhs.strip()
                suffix = "," if rhs_stripped.endswith(",") else ""
                trail_count = len(rhs) - len(rhs.rstrip(" "))
                tail_spaces = " " * trail_count
                new_body = f"{lhs}{sep}{leading}{literal}{suffix}{tail_spaces}"
                rewritten.append(new_body + comment)
                continue
        rewritten.append(line)

    return "\n".join(rewritten)


def _split_comment(line: str) -> tuple[str, str]:
    if "!" not in line:
        return line, ""
    idx = line.find("!")
    return line[:idx], line[idx:]


def generate_dummy_inputs(
    config: JobConfig,
    *,
    base_text: str | None = None,
    system_config: SystemConfig | None = None,
) -> dict[str, str]:
    if not config.input_files:
        return {}

    primary_source = Path(config.input_files[0])
    primary_name = primary_source.name
    calc_label = (config.calculation or "scf").lower()

    resolved_config = system_config or SystemConfig.default()

    base_text = (base_text or "").strip()
    if base_text:
        main_content = ensure_control_verbosity(base_text)
    elif calc_label == "scf" and primary_source.exists():
        try:
            seed = parse_scf_seed(str(primary_source))
            main_content = ensure_control_verbosity(
                render_scf_template(seed, config=resolved_config)
            )
        except Exception:  # noqa: BLE001
            main_content = (
                f"! Placeholder input for '{calc_label}' generated by the GUI\n"
                f"! Source file: {primary_name}\n\n"
                f"&CONTROL\n    calculation = '{calc_label}'\n    verbosity = 'high'\n/\n"
            )
    else:
        main_content = (
            f"! Placeholder input for '{calc_label}' generated by the GUI\n"
            f"! Source file: {primary_name}\n\n"
            f"&CONTROL\n    calculation = '{calc_label}'\n    verbosity = 'high'\n/\n"
        )

    generated: dict[str, str] = {primary_name: main_content}

    stem = primary_source.stem
    if calc_label in {"dos", "pdos"}:
        extra_name = f"{stem}.{calc_label}.in"
        generated[extra_name] = (
            f"! Secondary placeholder for {calc_label}\n"
            f"! Derived from {primary_name}\n\n"
            "BEGIN_DATA\n    ! add k-point sampling here\nEND_DATA\n"
        )
    elif calc_label in {"bands", "phonon"}:
        extra_name = f"{stem}.{calc_label}.setup"
        generated[extra_name] = (
            f"# Setup instructions for {calc_label}\n"
            "# TODO: replace with domain-specific generator.\n"
        )
    elif calc_label in {"optics", "epsilon"}:
        extra_name = f"{stem}.epsilon.in"
        generated[extra_name] = (
            f"! Placeholder input for optical properties (epsilon.x)\n"
            f"! Derived from {primary_name}\n\n"
            "&INPUTPP\n    ! set calculation flags for epsilon.x here\n/\n"
        )
    elif calc_label in {"wannier", "wannier90"}:
        extra_name = f"{stem}.win"
        generated[extra_name] = generate_wannier_win(stem)
    elif calc_label in {"pw2wannier90"}:
        extra_name = f"{stem}.pw2wan.in"
        generated[extra_name] = (
            f"! Placeholder input for pw2wannier90.x\n"
            f"! Derived from {primary_name}\n\n"
            "&INPUTPP\n    ! specify prefix and outdir matching SCF/NSCF\n/\n"
        )

    for name, content in list(generated.items()):
        generated[name] = ensure_control_verbosity(
            apply_pseudo_settings_to_text(
                content,
                config.pseudo_files,
                pseudo_dir="./",
            )
        )

    return generated


def build_job_commands(config: JobConfig) -> list[str]:
    if not config.input_files:
        return []

    commands: list[str] = []
    calc_label = (config.calculation or "").strip().lower()
    for path in config.input_files:
        input_path = Path(path)
        input_name = input_path.name
        if calc_label in {"matdyn", "matdyn.x"}:
            output_name = _derive_matdyn_output_name(input_name)
        else:
            output_name = f"{input_path.stem}.out"
        command = create_run_command(
            config.calculation,
            input_name,
            output_name,
            config.mpi_procs,
            validate_input=False,
        )
        commands.append(command)

    return commands


def _derive_matdyn_output_name(input_name: str) -> str:
    base = input_name
    lowered = input_name.lower()
    idx = lowered.find(".in")
    if idx == -1:
        return f"{Path(base).stem}.out"
    prefix = base[:idx]
    suffix = base[idx + 3 :]
    return f"{prefix}.out{suffix}"


def generate_job_script(config: JobConfig, commands: Optional[Sequence[str]] = None) -> str:
    if not config.calculation:
        raise ValueError("calculation type must be provided")

    sanitized_name = config.sanitized_name()
    mpi_procs = max(int(config.mpi_procs or 1), 1)
    queue = config.queue or "default"
    working_dir = Path(config.working_directory).expanduser() if config.working_directory else None
    pseudo_dir = Path(config.pseudo_directory).expanduser() if config.pseudo_directory else None
    input_files = [Path(path).expanduser() for path in config.input_files if path]

    header_lines = [
        "#!/bin/bash",
        "# Generated by Quantum ESPRESSO Helper",
        "",
        f"# Job Name    : {config.name or sanitized_name}",
        f"# Calculation : {config.calculation}",
        f"# MPI Procs   : {mpi_procs}",
        f"# Queue       : {queue}",
    ]
    if pseudo_dir:
        header_lines.append(f"# Pseudo Dir   : {pseudo_dir}")
    if working_dir:
        header_lines.append(f"# Working Dir  : {working_dir}")
    if config.pseudo_files:
        header_lines.append("# Pseudopotentials:")
        for element in sorted(config.pseudo_files):
            header_lines.append(f"#   {element}: {config.pseudo_files[element]}")

    body_lines = [
        "",
        "set -e",
        "",
        f'JOB_NAME="{sanitized_name}"',
        f'QUEUE="{queue}"',
        f'MPI_PROCS={mpi_procs}',
    ]

    if working_dir:
        body_lines.append(f'WORK_DIR="{working_dir}"')
        body_lines.append('mkdir -p "$WORK_DIR"')
        body_lines.append('cd "$WORK_DIR"')
    else:
        body_lines.append('# WORK_DIR not set; running in current directory')

    if pseudo_dir:
        body_lines.append(f'PSEUDO_DIR="{pseudo_dir}"')
        body_lines.append('export ESPRESSO_PSEUDO="$PSEUDO_DIR"')
    else:
        body_lines.append('# PSEUDO_DIR not set; ensure pseudopotentials are accessible')

    body_lines.append("")

    if input_files:
        body_lines.append("# Input files selected:")
        for path in input_files:
            body_lines.append(f"#   {path}")
    else:
        body_lines.append('# No input files selected; add at least one file before running.')
        body_lines.append('echo "No input files selected. Nothing to execute."')
        return "\n".join(header_lines + body_lines)

    body_lines.append("")
    command_list: Sequence[str]
    if commands is not None:
        command_list = list(commands)
    else:
        command_list = build_job_commands(config)

    if command_list:
        body_lines.extend(command_list)
    else:
        body_lines.append('echo "No commands generated. Please update the configuration."')

    body_lines.append("")
    body_lines.append('echo "Job script generated by Quantum ESPRESSO Helper."')

    return "\n".join(header_lines + body_lines)
