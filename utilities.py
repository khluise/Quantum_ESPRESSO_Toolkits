"""Compatibility helpers bridging legacy modules to the new qe_helper package."""

from __future__ import annotations

from pathlib import Path

from qe_helper.config import (
    SystemConfig,
    load_system_config,
    resolve_executable_bin_dir,
    resolve_qe_bin_dir,
    resolve_thermo_pw_bin_dir,
    save_system_config,
)
from qe_helper.converter import convert_cif_to_qe
from qe_helper.io import ScfSeedParameters, parse_scf_seed, render_scf_template
from qe_helper.jobs import (
    JobConfig,
    build_job_commands,
    create_run_command,
    generate_dummy_inputs,
    generate_job_script,
    generate_wannier_win,
    apply_pseudo_settings_to_text,
    ensure_control_verbosity,
    sanitize_job_name,
)
from qe_helper.kpoints import (
    HighSymmetryData,
    build_tpiba_b_block,
    generate_crystal_kmesh,
    load_high_symmetry_path,
)
from qe_helper.workflows import (
    CutoffConvergenceSettings,
    KpointConvergenceSettings,
    extract_total_energy,
    save_cutoff_visualizations,
)
from qe_helper.shell import run_command_on_terminal
from qe_helper.text import (
    remove_comments_lines,
    remove_ion_markers,
    remove_lines_from_start_to_end_markers,
    remove_lines_starting_with,
    save_text_to_file,
    trim_qe_input,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "settings.json"

__all__ = [
    "SystemConfig",
    "load_system_config",
    "resolve_executable_bin_dir",
    "resolve_qe_bin_dir",
    "resolve_thermo_pw_bin_dir",
    "save_system_config",
    "JobConfig",
    "build_job_commands",
    "create_run_command",
    "generate_dummy_inputs",
    "generate_job_script",
    "generate_wannier_win",
    "apply_pseudo_settings_to_text",
    "ensure_control_verbosity",
    "sanitize_job_name",
    "HighSymmetryData",
    "generate_crystal_kmesh",
    "load_high_symmetry_path",
    "build_tpiba_b_block",
    "ScfSeedParameters",
    "parse_scf_seed",
    "render_scf_template",
    "convert_cif_to_qe",
    "run_command_on_terminal",
    "remove_comments_lines",
    "remove_ion_markers",
    "remove_lines_from_start_to_end_markers",
    "remove_lines_starting_with",
    "save_text_to_file",
    "trim_qe_input",
    "CutoffConvergenceSettings",
    "KpointConvergenceSettings",
    "extract_total_energy",
    "save_cutoff_visualizations",
]
