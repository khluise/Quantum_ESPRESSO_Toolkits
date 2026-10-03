#!/usr/bin/env python3
"""Run QE Helper workflows from workflow_manifest.json."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from workflow_builder import (
    _generate_workflow_qe_input,
    _finalize_prepared_qe_template,
    BANDS_DIR,
    CHARGE_DENSITY_DIR,
    DATA_PLOTS_DIR,
    DOS_DIR,
    ELASTIC_DIR,
    EPC_MODE,
    FERMI_SURFACE_DIR,
    PHONON_DIR,
    TETRA_EPC_MODE,
    WANNIER90_DIR,
    BOHR_TO_ANGSTROM,
    RY_TO_EV,
    build_cutoff_child,
    build_initial_workflow_state,
    build_kpoint_child,
    create_workflow_directories,
    atomic_species_entries,
    extract_cell_vectors,
    extract_last_qe_card,
    extract_relaxed_structure_from_pw_output,
    extract_relaxed_structure_from_qe_xml,
    force_2d_k_points_automatic,
    kmesh_from_qe_input,
    manual_ecutwfc_from_options,
    parse_namelist_values,
    qe_quote,
    reciprocal_lengths_2pi,
    render_band_calculation_input,
    render_band_scf_input,
    render_bandsx_input,
    render_charge_density_pp_input,
    render_cutoff_convergence_input,
    render_final_relaxation_input,
    render_initial_relaxation_input,
    render_kpoint_convergence_input,
    render_dos_nscf_input,
    render_dos_scf_input,
    render_elastic_scf_input,
    render_epc_scf_input,
    render_fermi_surface_scf_input,
    render_lambda_input,
    render_matdyn_input,
    render_phonon_dos_input,
    render_phonon_input,
    render_phonon_scf_input,
    render_projwfc_input,
    render_q2r_input,
    render_qe_input_with_structure,
    render_tetra_scf_input,
    render_thermo_control,
    remove_namelist_keys,
    replace_qe_card,
    sanitize_job_name,
    set_namelist_values,
    write_json_file,
)
from qe_helper.config import load_system_config
from qe_helper.kpoints import generate_crystal_kmesh


PROJECT_ROOT = Path(__file__).resolve().parent
COMMAND_PREFS_PATH = PROJECT_ROOT / "command_preferences.json"
DEFAULT_QE_ETOT_CONV_THR = "1.0D-4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a QE Helper workflow.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run or resume a workflow.")
    run_parser.add_argument("manifest", help="Path to workflow_manifest.json")
    run_parser.add_argument(
        "--state",
        default="",
        help="Optional workflow_state.json path. Defaults to manifest directory.",
    )
    rerun_lambda_parser = subparsers.add_parser("rerun-lambda", help="Rerun only the phonon EPC lambda.x child.")
    rerun_lambda_parser.add_argument("manifest", help="Path to workflow_manifest.json")
    rerun_lambda_parser.add_argument("--mu", required=True, help="New lambda.x mu value.")
    rerun_lambda_parser.add_argument(
        "--state",
        default="",
        help="Optional workflow_state.json path. Defaults to manifest directory.",
    )

    args = parser.parse_args()
    if args.command == "rerun-lambda":
        return rerun_phonon_lambda(
            Path(args.manifest),
            mu=str(args.mu),
            state_path=Path(args.state) if args.state else None,
        )
    if args.command != "run":
        parser.print_help()
        return 2

    return run_workflow(Path(args.manifest), state_path=Path(args.state) if args.state else None)


def run_workflow(manifest_path: Path, *, state_path: Path | None = None) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_json_file(manifest_path)
    workflow_root = Path(str(manifest.get("workflow_root") or manifest_path.parent)).expanduser().resolve()
    state_path = (state_path.expanduser().resolve() if state_path else workflow_root / "workflow_state.json")

    if state_path.is_file():
        state = load_json_file(state_path)
    else:
        state = build_initial_workflow_state(manifest)
        write_state(state_path, state)

    create_workflow_directories(manifest, state)
    workflow_log = workflow_root / "logs" / "workflow.log"
    append_log(workflow_log, f"Workflow runner started at {timestamp()}\n")
    for notice in workflow_notices(manifest, state):
        append_log(workflow_log, f"Notice: {notice}\n")

    try:
        run_structure_prepare(manifest, state, state_path)
        run_initial_relaxation(manifest, state, state_path)
        run_cutoff_convergence(manifest, state, state_path)
        run_kpoint_convergence(manifest, state, state_path)
        run_final_relaxation(manifest, state, state_path)
        run_single_relaxation(manifest, state, state_path)
        run_bands_workflow(manifest, state, state_path)
        run_fat_bands_workflow(manifest, state, state_path)
        run_fatband_csv_extraction(manifest, state, state_path)
        run_fatband_plotting(manifest, state, state_path)
        run_dos_workflow(manifest, state, state_path)
        run_fermi_surface_workflow(manifest, state, state_path)
        run_charge_density_workflow(manifest, state, state_path)
        run_elastic_workflow(manifest, state, state_path)
        run_phonon_workflow(manifest, state, state_path)
        run_wannier90_workflow(manifest, state, state_path)
    except Exception as exc:  # noqa: BLE001 - runner should persist failure detail
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "workflow.failed", {"message": str(exc)})
        write_state(state_path, state)
        append_log(workflow_log, f"Workflow runner failed: {exc}\n")
        append_log(workflow_log, traceback.format_exc())
        return 1

    if all_tasks_completed(state):
        state["status"] = "completed"
        add_event(state, "workflow.completed", {})
    else:
        state["status"] = "waiting"
        add_event(
            state,
            "workflow.waiting",
            {"message": "Workflow is waiting for the next unimplemented task."},
        )
    state["current_task"] = None
    state["updated_at"] = timestamp()
    write_state(state_path, state)
    append_log(workflow_log, f"Workflow runner stopped at {timestamp()} with status {state['status']}\n")
    return 0


def rerun_phonon_lambda(manifest_path: Path, *, mu: str, state_path: Path | None = None) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_json_file(manifest_path)
    workflow_root = Path(str(manifest.get("workflow_root") or manifest_path.parent)).expanduser().resolve()
    state_path = (state_path.expanduser().resolve() if state_path else workflow_root / "workflow_state.json")
    if not state_path.is_file():
        raise FileNotFoundError(f"Workflow state is missing: {state_path}")
    state = load_json_file(state_path)
    create_workflow_directories(manifest, state)
    workflow_log = workflow_root / "logs" / "workflow.log"
    append_log(workflow_log, f"lambda.x rerun started at {timestamp()} with mu={mu}\n")

    task = workflow_task_state(state, "phonon_workflow")
    if task is None:
        raise RuntimeError("phonon_workflow task is missing from workflow state.")
    params = task.setdefault("params", {})
    if not isinstance(params, dict):
        params = {}
        task["params"] = params
    phonon_options = params.setdefault("phonon_options", {})
    if not isinstance(phonon_options, dict):
        phonon_options = {}
        params["phonon_options"] = phonon_options
    phonon_options["lambda_mu"] = str(mu).strip() or "0.10"

    lambda_child = next(
        (
            child
            for child in task.get("children") or []
            if isinstance(child, dict)
            and str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower() == "lambda"
        ),
        None,
    )
    if not isinstance(lambda_child, dict):
        raise RuntimeError("lambda child is missing from phonon workflow.")

    lambda_ploting_child = next(
        (
            child
            for child in task.get("children") or []
            if isinstance(child, dict)
            and str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower() == "lambda_ploting"
        ),
        None,
    )
    for child in (lambda_child, lambda_ploting_child):
        if isinstance(child, dict):
            child["status"] = "queued"
            child.pop("completed_at", None)
            child.pop("result", None)
    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = "phonon_workflow"
    state["updated_at"] = timestamp()
    write_state(state_path, state)

    try:
        run_phonon_workflow_child(
            manifest,
            state,
            task,
            lambda_child,
            base_text="",
            prefix=str(phonon_options.get("prefix") or "pwscf"),
            outdir=str(phonon_options.get("outdir") or "./out/"),
            scf_kpoint_spacing=0.0,
            converged_kmesh=None,
            ecutwfc=None,
            spin_orbit=False,
            phonon_options=phonon_options,
            state_path=state_path,
        )
        if isinstance(lambda_ploting_child, dict):
            run_phonon_workflow_child(
                manifest,
                state,
                task,
                lambda_ploting_child,
                base_text="",
                prefix=str(phonon_options.get("prefix") or "pwscf"),
                outdir=str(phonon_options.get("outdir") or "./out/"),
                scf_kpoint_spacing=0.0,
                converged_kmesh=None,
                ecutwfc=None,
                spin_orbit=False,
                phonon_options=phonon_options,
                state_path=state_path,
            )
    except Exception as exc:  # noqa: BLE001
        state["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["updated_at"] = timestamp()
        add_event(state, "lambda.rerun_failed", {"message": str(exc)})
        write_state(state_path, state)
        append_log(workflow_log, f"lambda.x rerun failed: {exc}\n")
        append_log(workflow_log, traceback.format_exc())
        return 1

    task["completed"] = count_completed_children(task)
    if task["completed"] >= len(task.get("children") or []):
        task["status"] = "completed"
    state["status"] = "completed" if all_tasks_completed(state) else "waiting"
    state["current_task"] = None
    state["updated_at"] = timestamp()
    add_event(state, "lambda.rerun_completed", {"mu": phonon_options["lambda_mu"]})
    write_state(state_path, state)
    append_log(workflow_log, f"lambda.x rerun finished at {timestamp()} with mu={phonon_options['lambda_mu']}\n")
    return 0


def run_structure_prepare(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    tasks = state.setdefault("tasks", {})
    task = tasks.get("prepare_source")
    if not isinstance(task, dict):
        return
    if str(task.get("status") or "").lower() == "completed":
        return

    children = task.get("children") or []
    if not children or not isinstance(children[0], dict):
        raise RuntimeError("prepare_source task has no child work item.")

    child = children[0]
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    source_info = manifest.get("source") or {}
    conversion_options = manifest.get("conversion") or {}
    requires_conversion = bool(source_info.get("requires_conversion", True))

    task["status"] = "running"
    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = "prepare_source"
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": "prepare_source", "child": child.get("id")})
    write_state(state_path, state)

    child_log_path = workflow_root / str(child.get("log") or "logs/prepare_source.log")

    try:
        stored_input = workflow_root / str(child.get("input") or source_info.get("stored_original") or "")
        prepared_output = workflow_root / str(child.get("output") or source_info.get("prepared_input") or "")
        original = Path(str(source_info.get("original") or "")).expanduser()

        stored_input.parent.mkdir(parents=True, exist_ok=True)
        prepared_output.parent.mkdir(parents=True, exist_ok=True)

        if not stored_input.is_file():
            if not original.is_file():
                raise FileNotFoundError(f"Original workflow source is missing: {original}")
            shutil.copy2(original, stored_input)
            append_log(child_log_path, f"Copied source to {stored_input}\n")

        if requires_conversion:
            append_log(child_log_path, f"Converting {stored_input} to {prepared_output}\n")
            _generate_workflow_qe_input(
                stored_input,
                prepared_output,
                conversion_options,
                log=lambda text: append_log(child_log_path, text + "\n"),
            )
        elif stored_input.resolve() != prepared_output.resolve():
            shutil.copy2(stored_input, prepared_output)
            _finalize_prepared_qe_template(prepared_output, conversion_options)
            append_log(child_log_path, f"Prepared QE input copied to {prepared_output}\n")
        else:
            _finalize_prepared_qe_template(prepared_output, conversion_options)
            append_log(child_log_path, f"Prepared QE input already available at {prepared_output}\n")

        child["status"] = "completed"
        child["completed_at"] = timestamp()
        task["status"] = "completed"
        task["completed"] = 1
        task["running_child"] = None
        task["result"] = {"prepared_input": str(Path(str(child.get("output") or "")).as_posix())}
        state.setdefault("results", {})["prepared_input"] = child.get("output")
        add_event(state, "task.completed", {"task": "prepare_source", "child": child.get("id")})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": "prepare_source", "child": child.get("id")})
        write_state(state_path, state)
        raise


def run_initial_relaxation(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    task = workflow_task_state(state, "initial_relaxation")
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = task.get("children") or []
    if not children or not isinstance(children[0], dict):
        raise RuntimeError("initial_relaxation task has no child work item.")

    child = children[0]
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or "")
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or "logs/initial_relaxation.log")
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    calculation = str((child.get("params") or {}).get("calculation") or task_type_to_calculation(task.get("type")))

    task["status"] = "running"
    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = "initial_relaxation"
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": "initial_relaxation", "child": child.get("id")})
    write_state(state_path, state)

    try:
        if not prepared_input.is_file():
            raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

        child_workdir.mkdir(parents=True, exist_ok=True)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        rendered = render_initial_relaxation_input(
            template_text,
            calculation=calculation,
            workflow_root=workflow_root,
            child_workdir=child.get("workdir") or "",
            child_id=str(child.get("id") or "initial_relaxation"),
            conversion_options=manifest.get("conversion") or {},
            relaxation_options=relaxation_parameters_for_child(manifest, child),
        )
        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote initial relaxation input: {input_path}\n")

        command = build_pw_command(input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        command_result = run_shell_command(
            command,
            cwd=child_workdir,
            log_path=child_log_path,
            allow_nonzero=True,
        )

        relaxed_structure = extract_relaxed_structure_from_qe_run(
            input_path,
            child_workdir,
            child_log_path,
        )
        if not relaxed_structure and output_path.is_file():
            relaxed_structure = extract_relaxed_structure_from_pw_output(
                output_path.read_text(encoding="utf-8", errors="ignore")
            )
        if command_result["fatal_error_found"]:
            raise RuntimeError(f"QE reported a fatal error while running initial relaxation: {command}")
        if command_result["return_code"] != 0 and not relaxed_structure:
            raise RuntimeError(
                f"Initial relaxation command exited with {command_result['return_code']} and no structure was extracted."
            )
        if command_result["return_code"] != 0:
            append_log(
                child_log_path,
                f"Initial relaxation command exited with {command_result['return_code']}; continuing with extracted structure.\n",
            )

        convergence_summary = summarize_relaxation_convergence(relaxed_structure)
        if convergence_summary.get("relaxation_converged") is False:
            append_log(
                child_log_path,
                "Initial relaxation did not fully converge; continuing with the latest extracted structure.\n",
            )

        relaxed_structure_path = workflow_root / "data_and_plots" / "initial_relaxed_structure.json"
        write_json_file(relaxed_structure_path, relaxed_structure)
        archive_child_files(workflow_root, child)

        child["status"] = "completed"
        child["completed_at"] = timestamp()
        task["status"] = "completed"
        task["completed"] = 1
        task["running_child"] = None
        task["result"] = {
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
            "relaxed_structure": "data_and_plots/initial_relaxed_structure.json",
            **convergence_summary,
        }
        child["relaxation_converged"] = convergence_summary.get("relaxation_converged")
        if convergence_summary.get("relaxation_converged") is False:
            child["warning"] = "Initial relaxation did not fully converge; workflow will continue."
        results = state.setdefault("results", {})
        results["initial_relaxation_output"] = child.get("output")
        results["initial_relaxed_structure"] = "data_and_plots/initial_relaxed_structure.json"
        results["initial_relaxation_converged"] = convergence_summary.get("relaxation_converged")
        if convergence_summary.get("relaxation_converged") is False:
            add_event(
                state,
                "task.warning",
                {
                    "task": "initial_relaxation",
                    "child": child.get("id"),
                    "message": "Initial relaxation did not fully converge; workflow will continue.",
                },
            )
        add_event(state, "task.completed", {"task": "initial_relaxation", "child": child.get("id")})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": "initial_relaxation", "child": child.get("id")})
        write_state(state_path, state)
        raise


def run_final_relaxation(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_accuracy_relaxation_task(
        manifest,
        state,
        state_path,
        task_id="final_relaxation",
        structure_result_key="final_relaxed_structure",
        output_result_key="final_relaxation_output",
        structure_filename="final_relaxed_structure.json",
    )


def run_single_relaxation(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_accuracy_relaxation_task(
        manifest,
        state,
        state_path,
        task_id="relaxation",
        structure_result_key="relaxed_structure",
        output_result_key="relaxation_output",
        structure_filename="relaxed_structure.json",
    )


def run_bands_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_band_workflow_task(
        manifest,
        state,
        state_path,
        task_id="bands_workflow",
        require_projwfc=None,
    )


def run_fat_bands_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_band_workflow_task(
        manifest,
        state,
        state_path,
        task_id="fat_bands_workflow",
        require_projwfc=True,
    )


def run_dos_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_dos_workflow_task(
        manifest,
        state,
        state_path,
        task_id="dos_workflow",
    )


def run_wannier90_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_wannier90_workflow_task(
        manifest,
        state,
        state_path,
        task_id="wannier90_workflow",
    )


def run_fermi_surface_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_fermi_surface_workflow_task(
        manifest,
        state,
        state_path,
        task_id="fermi_surface_workflow",
    )


def run_charge_density_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_charge_density_workflow_task(
        manifest,
        state,
        state_path,
        task_id="charge_density_workflow",
    )


def run_elastic_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_elastic_workflow_task(
        manifest,
        state,
        state_path,
        task_id="elastic_workflow",
    )


def run_phonon_workflow(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_phonon_workflow_task(
        manifest,
        state,
        state_path,
        task_id="phonon_workflow",
    )


def run_phonon_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    phonon_options = params.get("phonon_options") if isinstance(params.get("phonon_options"), Mapping) else {}
    scf_kpoint_spacing = positive_float_or_none(phonon_options.get("scf_kpoint_spacing"))
    if scf_kpoint_spacing is None:
        scf_kpoint_spacing = 0.0
    use_converged_kmesh = bool(phonon_options.get("use_converged_kmesh", False))
    prefix = phonon_workflow_prefix(manifest, phonon_options)
    outdir = "./out/"
    fildyn = str(phonon_options.get("fildyn") or "phonon.dyn").strip() or "phonon.dyn"
    flfrc = str(phonon_options.get("flfrc") or "phonon.fc").strip() or "phonon.fc"
    flfrq = str(phonon_options.get("flfrq") or "phonon.freq").strip() or "phonon.freq"
    fldos = str(phonon_options.get("fldos") or "phonon.phdos").strip() or "phonon.phdos"
    epc_fldos = str(phonon_options.get("epc_fldos") or "phonon.dos").strip() or "phonon.dos"
    mode = str(phonon_options.get("mode") or phonon_options.get("calculation_mode") or "dispersion").strip().lower()
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)
    converged_kmesh: list[int] | None = None
    if uses_converged and use_converged_kmesh and isinstance(results, Mapping):
        mesh = results.get("kmesh")
        if isinstance(mesh, list) and len(mesh) >= 3:
            try:
                converged_kmesh = [int(mesh[0]), int(mesh[1]), int(mesh[2])]
            except Exception:
                converged_kmesh = None

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")

            run_phonon_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                scf_kpoint_spacing=float(scf_kpoint_spacing),
                converged_kmesh=converged_kmesh,
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                phonon_options=phonon_options,
                state_path=state_path,
            )

        dyn_file = str((Path(PHONON_DIR) / fildyn).as_posix())
        force_constants_file = str((Path(PHONON_DIR) / flfrc).as_posix())
        frequency_file = str((Path(PHONON_DIR) / flfrq).as_posix())
        dos_file = str((Path(PHONON_DIR) / fldos).as_posix())
        epc_dos_file = str((Path(PHONON_DIR) / epc_fldos).as_posix())
        q2r_child = (
            band_child_by_stage(task, stage="q2r")
            or band_child_by_stage(task, stage="epc_q2r")
            or band_child_by_stage(task, stage="tetra_q2r")
        )
        matdyn_child = (
            band_child_by_stage(task, stage="matdyn")
            or band_child_by_stage(task, stage="epc_matdyn")
            or band_child_by_stage(task, stage="tetra_matdyn")
        )
        matdyn_result = matdyn_child.get("result") if isinstance(matdyn_child, Mapping) else {}
        phdos_child = band_child_by_stage(task, stage="phdos")
        if phdos_child is None:
            phdos_child = band_child_by_stage(task, stage="epc_phdos")
        if phdos_child is None:
            phdos_child = band_child_by_stage(task, stage="tetra_phdos")
        phdos_result = phdos_child.get("result") if isinstance(phdos_child, Mapping) else {}
        projected_child = band_child_by_stage(task, stage="ploting") or band_child_by_stage(task, stage="projected")
        projected_result = projected_child.get("result") if isinstance(projected_child, Mapping) else {}
        lambda_child = band_child_by_stage(task, stage="lambda")
        lambda_result = lambda_child.get("result") if isinstance(lambda_child, Mapping) else {}
        alpha2f_child = band_child_by_stage(task, stage="tetra_alpha2f")
        alpha2f_result = alpha2f_child.get("result") if isinstance(alpha2f_child, Mapping) else {}
        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task_result = {
            "prefix": prefix,
            "outdir": outdir,
            "fildyn": fildyn,
            "flfrc": flfrc,
            "flfrq": flfrq,
            "fldos": fldos,
            "matdyn_flvec": str(phonon_options.get("matdyn_flvec") or "phonon_band.modes"),
            "phdos_flvec": str(phonon_options.get("phdos_flvec") or "phonon_dos.modes"),
            "dyn_file": dyn_file,
            "force_constants_file": force_constants_file if q2r_child else None,
            "band_enabled": mode in {"dispersion", EPC_MODE, TETRA_EPC_MODE} and bool(matdyn_child),
            "frequency_file": frequency_file if matdyn_child else None,
            "dos_enabled": mode in {"dispersion", EPC_MODE, TETRA_EPC_MODE} and bool(phdos_child),
            "dos_file": (epc_dos_file if mode in {EPC_MODE, TETRA_EPC_MODE} else dos_file) if phdos_child else None,
            "dos_data": phdos_result.get("dos_data") if isinstance(phdos_result, Mapping) else None,
            "dos_plot": phdos_result.get("dos_plot") if isinstance(phdos_result, Mapping) else None,
            "projected_band_data": projected_result.get("band_data") if isinstance(projected_result, Mapping) else None,
            "projected_dos_data": projected_result.get("projected_dos_data") if isinstance(projected_result, Mapping) else None,
            "projected_combined_plot": projected_result.get("combined_plot") if isinstance(projected_result, Mapping) else None,
            "projected_replot_script": projected_result.get("replot_script") if isinstance(projected_result, Mapping) else None,
            "mode": mode,
            "q_grid": list(phonon_options.get("q_grid") or phonon_options.get("nq") or [4, 4, 4]),
            "matdyn_q_path": list(phonon_options.get("matdyn_q_path") or []),
            "q_point": list(phonon_options.get("q_point") or [0.0, 0.0, 0.0]),
            "scf_kmesh": list(phonon_options.get("scf_kmesh") or [16, 16, 16]),
            "scf_kpoint_spacing": float(scf_kpoint_spacing),
            "used_converged_kmesh": bool(converged_kmesh),
            "kmesh": converged_kmesh,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        if mode == EPC_MODE:
            task_result.update(
                {
                    "epc_fldos": epc_fldos,
                    "epc_frequency_file": frequency_file,
                    "epc_dos_file": epc_dos_file,
                    "epc_lambda": lambda_result.get("lambda") if isinstance(lambda_result, Mapping) else None,
                    "epc_fit_kmesh": list(phonon_options.get("epc_fit_kmesh") or [24, 24, 24]),
                    "epc_scf_kmesh": list(phonon_options.get("epc_scf_kmesh") or [12, 12, 12]),
                    "epc_q_grid": list(phonon_options.get("epc_q_grid") or [3, 3, 3]),
                }
            )
        if mode == TETRA_EPC_MODE:
            task_result.update(
                {
                    "tetra_scf_kmesh": list(phonon_options.get("tetra_scf_kmesh") or [8, 8, 8]),
                    "tetra_q_grid": list(phonon_options.get("tetra_q_grid") or [4, 4, 4]),
                    "tetra_elph_kmesh": list(phonon_options.get("tetra_elph_kmesh") or [24, 24, 24]),
                    "fildvscf": str(phonon_options.get("fildvscf") or "dvscf"),
                    "tetra_nfreq": int(phonon_options.get("tetra_nfreq") or 500),
                    "tetra_alpha2f": alpha2f_result.get("alpha2f") if isinstance(alpha2f_result, Mapping) else None,
                    "epc_lambda": lambda_result.get("lambda") if isinstance(lambda_result, Mapping) else None,
                }
            )
        task["result"] = task_result

        results_state = state.setdefault("results", {})
        results_state["phonon_dyn_file"] = dyn_file
        if q2r_child:
            results_state["phonon_force_constants_file"] = force_constants_file
        if matdyn_child:
            results_state["phonon_frequency_file"] = frequency_file
        if phdos_child:
            results_state["phonon_dos_file"] = epc_dos_file if mode in {EPC_MODE, TETRA_EPC_MODE} else dos_file
            if isinstance(phdos_result, Mapping):
                results_state["phonon_dos_data"] = phdos_result.get("dos_data")
                results_state["phonon_dos_plot"] = phdos_result.get("dos_plot")
        if projected_child and isinstance(projected_result, Mapping):
            results_state["phonon_projected_band_data"] = projected_result.get("band_data")
            results_state["phonon_projected_dos_data"] = projected_result.get("projected_dos_data")
            results_state["phonon_projected_combined_plot"] = projected_result.get("combined_plot")
            results_state["phonon_projected_replot_script"] = projected_result.get("replot_script")
        if mode == EPC_MODE:
            results_state["phonon_epc_frequency_file"] = frequency_file
            results_state["phonon_epc_dos_file"] = epc_dos_file
            if isinstance(lambda_result, Mapping):
                results_state["phonon_epc_lambda"] = lambda_result.get("lambda")
        if mode == TETRA_EPC_MODE:
            results_state["phonon_tetra_frequency_file"] = frequency_file
            results_state["phonon_tetra_dos_file"] = epc_dos_file
            if isinstance(lambda_result, Mapping):
                results_state["phonon_epc_lambda"] = lambda_result.get("lambda")
            if isinstance(alpha2f_result, Mapping):
                results_state["phonon_tetra_alpha2f"] = alpha2f_result.get("alpha2f")
            results_state["phonon_epc_tetrahedra"] = task_result
        results_state["phonon"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_phonon_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    scf_kpoint_spacing: float,
    converged_kmesh: Sequence[int] | None,
    ecutwfc: Any,
    spin_orbit: bool,
    phonon_options: Mapping[str, Any],
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or PHONON_DIR)
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()
    convert_2d_structure = workflow_convert_2d_structure_enabled(manifest)
    matdyn_asr = str(
        phonon_options.get("matdyn_asr")
        or ("all" if convert_2d_structure else "crystal")
    )

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "phonon_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        handled_execution = False
        precomputed_lambda_outputs: dict[str, Any] = {}
        if stage == "ploting":
            append_log(child_log_path, "Running phonon band + PhDOS ploting post-processing.\n")
            projected_band_outputs = write_phonon_projected_band_outputs(
                workflow_root,
                flvec=str(phonon_options.get("matdyn_flvec") or "phonon_band.modes"),
                q_path_lines=phonon_options.get("matdyn_q_path") or [],
                log_path=child_log_path,
            )
            projected_dos_outputs = write_phonon_projected_dos_outputs(
                workflow_root,
                flvec=str(phonon_options.get("phdos_flvec") or "phonon_dos.modes"),
                log_path=child_log_path,
            )
            projected_combined_outputs = maybe_write_phonon_projected_combined_outputs(
                workflow_root,
                log_path=child_log_path,
            )
            if not projected_band_outputs or not projected_dos_outputs or not projected_combined_outputs:
                raise RuntimeError("Phonon ploting post-processing did not create all expected outputs.")

            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                "band_modes_file": projected_band_outputs.get("band_modes_file"),
                "band_data": projected_band_outputs.get("band_data"),
                "band_csv": projected_band_outputs.get("band_csv"),
                "projected_dos_modes_file": projected_dos_outputs.get("dos_modes_file"),
                "projected_dos_data": projected_dos_outputs.get("projected_dos_data"),
                "projected_dos_csv": projected_dos_outputs.get("projected_dos_csv"),
                "combined_plot": projected_combined_outputs.get("combined_plot"),
                "plot_data": projected_combined_outputs.get("plot_data"),
                "replot_script": projected_combined_outputs.get("replot_script"),
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None

            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["phonon_band_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")).as_posix()
            )
            results["phonon_dos_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("phdos_flvec") or "phonon_dos.modes")).as_posix()
            )
            results["phonon_projected_band_data"] = projected_band_outputs.get("band_data")
            results["phonon_projected_dos_data"] = projected_dos_outputs.get("projected_dos_data")
            results["phonon_projected_combined_plot"] = projected_combined_outputs.get("combined_plot")
            results["phonon_projected_replot_script"] = projected_combined_outputs.get("replot_script")

            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage == "lambda_ploting":
            append_log(child_log_path, "Running EPC lambda ploting post-processing.\n")
            lambda_ploting_outputs = write_epc_lambda_ploting_outputs(
                workflow_root,
                task=task,
                state=state,
                log_path=child_log_path,
            )
            write_json_file(output_path, lambda_ploting_outputs)
            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                "selected_elph_gamma_file": lambda_ploting_outputs.get("selected_elph_gamma_file"),
                "combined_plot": lambda_ploting_outputs.get("combined_plot"),
                "plot_data": lambda_ploting_outputs.get("plot_data"),
                "replot_script": lambda_ploting_outputs.get("replot_script"),
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["phonon_lambda_ploting"] = child["result"]
            results["phonon_epc_linewidth_band_data"] = lambda_ploting_outputs.get("band_data")
            results["phonon_epc_alpha2f_lambda_data"] = lambda_ploting_outputs.get("alpha2f_lambda_data")
            results["phonon_epc_alpha2f_lambda_plot_data"] = lambda_ploting_outputs.get("plot_data")
            results["phonon_epc_alpha2f_lambda_plot"] = lambda_ploting_outputs.get("combined_plot")
            results["phonon_epc_alpha2f_lambda_replot_script"] = lambda_ploting_outputs.get("replot_script")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return

        if stage == "scf":
            rendered = render_phonon_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or PHONON_DIR,
                prefix=prefix,
                kpoint_spacing=float(scf_kpoint_spacing),
                kmesh=phonon_options.get("scf_kmesh") or [16, 16, 16],
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "epc_fit_scf":
            rendered = render_epc_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or PHONON_DIR,
                prefix=prefix,
                kmesh=phonon_options.get("epc_fit_kmesh") or [24, 24, 24],
                la2f=True,
                convert_2d_structure=workflow_convert_2d_structure_enabled(manifest),
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "epc_scf":
            rendered = render_epc_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or PHONON_DIR,
                prefix=prefix,
                kmesh=phonon_options.get("epc_scf_kmesh") or [12, 12, 12],
                la2f=False,
                convert_2d_structure=workflow_convert_2d_structure_enabled(manifest),
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "tetra_scf":
            rendered = render_tetra_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or PHONON_DIR,
                prefix=prefix,
                kmesh=phonon_options.get("tetra_scf_kmesh") or [8, 8, 8],
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "ph":
            rendered = render_phonon_input(
                base_text,
                prefix=prefix,
                outdir=outdir,
                tr2_ph=str(phonon_options.get("tr2_ph") or "1.0d-12"),
                alpha_mix=str(phonon_options.get("alpha_mix") or "0.7"),
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                q_point=phonon_options.get("q_point") or [0.0, 0.0, 0.0],
                ldisp=bool(phonon_options.get("ldisp", False)),
                nq_grid=phonon_options.get("q_grid") or phonon_options.get("nq") or [4, 4, 4],
                trans=bool(phonon_options.get("trans", True)),
                epsil=bool(phonon_options.get("epsil", False)),
                nmix_ph=int(phonon_options.get("nmix_ph") or 4),
                verbosity=str(phonon_options.get("verbosity") or "default"),
            )
            engine = "ph.x"
        elif stage == "epc_ph":
            rendered = render_phonon_input(
                base_text,
                prefix=prefix,
                outdir=outdir,
                tr2_ph=str(phonon_options.get("epc_tr2_ph") or phonon_options.get("tr2_ph") or "1.0d-10"),
                alpha_mix=str(phonon_options.get("alpha_mix") or "0.7"),
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                ldisp=True,
                nq_grid=phonon_options.get("epc_q_grid") or [3, 3, 3],
                trans=bool(phonon_options.get("trans", True)),
                epsil=bool(phonon_options.get("epsil", False)),
                nmix_ph=int(phonon_options.get("nmix_ph") or 4),
                verbosity=str(phonon_options.get("verbosity") or "default"),
                fildvscf=str(phonon_options.get("fildvscf") or "dvscf"),
                electron_phonon=str(phonon_options.get("electron_phonon") or "interpolated"),
            )
            engine = "ph.x"
        elif stage == "tetra_ph":
            rendered = render_phonon_input(
                base_text,
                prefix=prefix,
                outdir=outdir,
                tr2_ph=str(phonon_options.get("tr2_ph") or "1.0d-12"),
                alpha_mix=str(phonon_options.get("alpha_mix") or "0.7"),
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                ldisp=True,
                nq_grid=phonon_options.get("tetra_q_grid") or [4, 4, 4],
                trans=bool(phonon_options.get("trans", True)),
                epsil=bool(phonon_options.get("epsil", False)),
                nmix_ph=int(phonon_options.get("nmix_ph") or 4),
                verbosity=str(phonon_options.get("verbosity") or "default"),
                fildvscf=str(phonon_options.get("fildvscf") or "dvscf"),
                lshift_q=True,
            )
            engine = "ph.x"
        elif stage in {"tetra_elph", "tetra_alpha2f"}:
            rendered = render_phonon_input(
                base_text,
                prefix=prefix,
                outdir=outdir,
                tr2_ph=str(phonon_options.get("tr2_ph") or "1.0d-12"),
                alpha_mix=str(phonon_options.get("alpha_mix") or "0.7"),
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                ldisp=True,
                nq_grid=phonon_options.get("tetra_q_grid") or [4, 4, 4],
                trans=bool(phonon_options.get("trans", True)),
                epsil=bool(phonon_options.get("epsil", False)),
                nmix_ph=int(phonon_options.get("nmix_ph") or 4),
                verbosity=str(phonon_options.get("verbosity") or "default"),
                fildvscf=str(phonon_options.get("fildvscf") or "dvscf"),
                lshift_q=True,
                electron_phonon="lambda_tetra",
                elph_nk_grid=phonon_options.get("tetra_elph_kmesh") or [24, 24, 24],
                inputa2f_nfreq=int(phonon_options.get("tetra_nfreq") or 500),
            )
            engine = "alpha2f.x" if stage == "tetra_alpha2f" else "ph.x"
        elif stage == "q2r":
            rendered = render_q2r_input(
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                zasr=str(phonon_options.get("q2r_zasr") or "crystal"),
                loto_2d=convert_2d_structure,
            )
            engine = "q2r.x"
        elif stage == "epc_q2r":
            rendered = render_q2r_input(
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                zasr=str(phonon_options.get("q2r_zasr") or "crystal"),
                la2f=True,
            )
            engine = "q2r.x"
        elif stage == "tetra_q2r":
            rendered = render_q2r_input(
                fildyn=str(phonon_options.get("fildyn") or "phonon.dyn"),
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                zasr=str(phonon_options.get("q2r_zasr") or "crystal"),
                la2f=True,
                lshift_q=True,
            )
            engine = "q2r.x"
        elif stage == "matdyn":
            rendered = render_matdyn_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                flfrq=str(phonon_options.get("flfrq") or "phonon.freq"),
                flvec=str(phonon_options.get("matdyn_flvec") or "phonon_band.modes"),
                q_path_lines=phonon_options.get("matdyn_q_path") or [],
                asr=matdyn_asr,
                q_in_band_form=bool(phonon_options.get("matdyn_q_in_band_form", True)),
                loto_2d=convert_2d_structure,
            )
            engine = "matdyn.x"
        elif stage == "epc_matdyn":
            rendered = render_matdyn_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                flfrq=str(phonon_options.get("flfrq") or "phonon.freq"),
                flvec=str(phonon_options.get("matdyn_flvec") or "phonon_band.modes"),
                q_path_lines=phonon_options.get("matdyn_q_path") or [],
                asr=matdyn_asr,
                q_in_band_form=bool(phonon_options.get("matdyn_q_in_band_form", True)),
                la2f=True,
                loto_2d=convert_2d_structure,
            )
            engine = "matdyn.x"
        elif stage == "tetra_matdyn":
            rendered = render_matdyn_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                flfrq=str(phonon_options.get("flfrq") or "phonon.freq"),
                flvec=str(phonon_options.get("matdyn_flvec") or "phonon_band.modes"),
                q_path_lines=phonon_options.get("matdyn_q_path") or [],
                asr=str(phonon_options.get("matdyn_asr") or "crystal"),
                q_in_band_form=bool(phonon_options.get("matdyn_q_in_band_form", True)),
                la2f=True,
            )
            engine = "matdyn.x"
        elif stage == "phdos":
            rendered = render_phonon_dos_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                fldos=str(phonon_options.get("fldos") or "phonon.phdos"),
                flvec=str(phonon_options.get("phdos_flvec") or "phonon_dos.modes"),
                nk_grid=phonon_options.get("phdos_nk") or [4, 4, 4],
                asr=matdyn_asr,
                flfrq=str(phonon_options.get("phdos_flfrq") or "phonon.freq.dos"),
                loto_2d=convert_2d_structure,
            )
            engine = "matdyn.x"
        elif stage == "epc_phdos":
            rendered = render_phonon_dos_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                fldos=str(phonon_options.get("epc_fldos") or "phonon.dos"),
                flvec=str(phonon_options.get("phdos_flvec") or "phonon_dos.modes"),
                nk_grid=phonon_options.get("phdos_nk") or [4, 4, 4],
                asr=matdyn_asr,
                flfrq=str(phonon_options.get("phdos_flfrq") or "phonon.freq.dos"),
                ndos=int(phonon_options.get("epc_ndos") or 50),
                la2f=True,
                loto_2d=convert_2d_structure,
            )
            engine = "matdyn.x"
        elif stage == "tetra_phdos":
            rendered = render_phonon_dos_input(
                base_text,
                flfrc=str(phonon_options.get("flfrc") or "phonon.fc"),
                fldos=str(phonon_options.get("epc_fldos") or "phonon.dos"),
                flvec=str(phonon_options.get("phdos_flvec") or "phonon_dos.modes"),
                nk_grid=phonon_options.get("phdos_nk") or [20, 20, 20],
                asr=str(phonon_options.get("matdyn_asr") or "crystal"),
                flfrq=str(phonon_options.get("phdos_flfrq") or "phonon.freq.dos"),
                ndos=int(phonon_options.get("epc_ndos") or 50),
                la2f=True,
            )
            engine = "matdyn.x"
        elif stage == "lambda":
            lambda_q_points = lambda_q_points_from_q2r_outputs(workflow_root, child_log_path)
            if not lambda_q_points:
                lambda_q_points = list(phonon_options.get("lambda_q_points") or [])
            engine = "lambda.x"
            lambda_emax = resolve_lambda_emax(workflow_root, phonon_options, child_log_path)
            lambda_expected_tc = positive_float_or_none(phonon_options.get("lambda_expected_tc"))
            lambda_tc_tolerance = max(0.0, optional_float_or_none(phonon_options.get("lambda_tc_tolerance")) or 1.0)
            lambda_degauss_scan_max = optional_float_or_none(phonon_options.get("lambda_degauss_scan_max")) or 0.8
            if lambda_expected_tc is not None:
                precomputed_lambda_outputs = run_lambda_tc_scan(
                    workflow_root=workflow_root,
                    child_workdir=child_workdir,
                    input_path=input_path,
                    output_path=output_path,
                    emax=lambda_emax,
                    smearing_method=int(phonon_options.get("lambda_smearing_method") or 1),
                    mu=str(phonon_options.get("lambda_mu") or "0.10"),
                    q_points=lambda_q_points,
                    expected_tc=lambda_expected_tc,
                    tolerance=lambda_tc_tolerance,
                    max_degauss=lambda_degauss_scan_max,
                    log_path=child_log_path,
                )
                rendered = input_path.read_text(encoding="utf-8", errors="ignore")
                handled_execution = True
            else:
                rendered = render_lambda_input(
                    emax=lambda_emax,
                    degauss=str(phonon_options.get("lambda_degauss") or "0.1"),
                    smearing_method=int(phonon_options.get("lambda_smearing_method") or 1),
                    mu=str(phonon_options.get("lambda_mu") or "0.10"),
                    q_points=lambda_q_points,
                )
        else:
            raise RuntimeError(f"Unsupported phonon workflow child stage: {stage}")

        if (
            workflow_convert_2d_structure_enabled(manifest)
            and stage in {"scf", "epc_fit_scf", "epc_scf"}
        ):
            rendered = remove_namelist_keys(rendered, "system", {"vdw_corr"})

        if not handled_execution:
            rendered = write_workflow_input(input_path, rendered, manifest)
            append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")
            command = (
                build_qe_stdin_command(engine, input_path.name, output_path.name)
                if engine == "lambda.x"
                else build_qe_command(engine, input_path.name, output_path.name)
            )
            append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
            run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        phdos_outputs: dict[str, Any] = {}
        if stage == "phdos":
            phdos_outputs = write_phonon_dos_outputs(
                workflow_root,
                fldos=str(phonon_options.get("fldos") or "phonon.phdos"),
                log_path=child_log_path,
            )
        lambda_outputs: dict[str, Any] = {}
        if stage == "lambda":
            lambda_outputs = precomputed_lambda_outputs or write_epc_lambda_outputs(
                workflow_root,
                output_relative=str(child.get("output") or Path(PHONON_DIR) / "lambda.out"),
                log_path=child_log_path,
                expected_tc=positive_float_or_none(phonon_options.get("lambda_expected_tc")),
                tolerance=optional_float_or_none(phonon_options.get("lambda_tc_tolerance")),
                input_degauss=optional_float_or_none(phonon_options.get("lambda_degauss")),
            )
        alpha2f_outputs: dict[str, Any] = {}
        if stage == "tetra_alpha2f":
            alpha2f_outputs = write_alpha2f_outputs(
                workflow_root,
                output_relative=str(child.get("output") or Path(PHONON_DIR) / "tetra_alpha2f.out"),
                log_path=child_log_path,
            )

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = {
            "stage": stage,
            "engine": engine,
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        if phdos_outputs:
            child["result"].update(phdos_outputs)
        if lambda_outputs:
            child["result"].update(lambda_outputs)
        if alpha2f_outputs:
            child["result"].update(alpha2f_outputs)
        task["completed"] = count_completed_children(task)
        task["running_child"] = None

        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["phonon_scf_output"] = child.get("output")
        elif stage == "epc_fit_scf":
            results["phonon_epc_fit_scf_output"] = child.get("output")
        elif stage == "epc_scf":
            results["phonon_epc_scf_output"] = child.get("output")
        elif stage == "tetra_scf":
            results["phonon_tetra_scf_output"] = child.get("output")
        elif stage == "ph":
            results["phonon_output"] = child.get("output")
            results["phonon_dyn_file"] = str((Path(PHONON_DIR) / str(phonon_options.get("fildyn") or "phonon.dyn")).as_posix())
        elif stage == "epc_ph":
            results["phonon_epc_ph_output"] = child.get("output")
            results["phonon_dyn_file"] = str((Path(PHONON_DIR) / str(phonon_options.get("fildyn") or "phonon.dyn")).as_posix())
        elif stage == "tetra_ph":
            results["phonon_tetra_ph_output"] = child.get("output")
            results["phonon_dyn_file"] = str((Path(PHONON_DIR) / str(phonon_options.get("fildyn") or "phonon.dyn")).as_posix())
        elif stage == "tetra_elph":
            results["phonon_tetra_elph_output"] = child.get("output")
        elif stage == "q2r":
            results["phonon_q2r_output"] = child.get("output")
            results["phonon_force_constants_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrc") or "phonon.fc")).as_posix()
            )
        elif stage == "epc_q2r":
            results["phonon_epc_q2r_output"] = child.get("output")
            results["phonon_force_constants_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrc") or "phonon.fc")).as_posix()
            )
        elif stage == "tetra_q2r":
            results["phonon_tetra_q2r_output"] = child.get("output")
            results["phonon_force_constants_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrc") or "phonon.fc")).as_posix()
            )
        elif stage == "matdyn":
            results["phonon_matdyn_output"] = child.get("output")
            results["phonon_frequency_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrq") or "phonon.freq")).as_posix()
            )
            results["phonon_band_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")).as_posix()
            )
        elif stage == "epc_matdyn":
            results["phonon_epc_matdyn_output"] = child.get("output")
            results["phonon_epc_frequency_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrq") or "phonon.freq")).as_posix()
            )
            results["phonon_band_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")).as_posix()
            )
        elif stage == "tetra_matdyn":
            results["phonon_tetra_matdyn_output"] = child.get("output")
            results["phonon_tetra_frequency_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("flfrq") or "phonon.freq")).as_posix()
            )
            results["phonon_band_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("matdyn_flvec") or "phonon_band.modes")).as_posix()
            )
        elif stage == "phdos":
            results["phonon_phdos_output"] = child.get("output")
            results["phonon_dos_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("fldos") or "phonon.phdos")).as_posix()
            )
            results["phonon_dos_data"] = phdos_outputs.get("dos_data")
            results["phonon_dos_plot"] = phdos_outputs.get("dos_plot")
            results["phonon_dos_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("phdos_flvec") or "phonon_dos.modes")).as_posix()
            )
        elif stage == "tetra_phdos":
            results["phonon_tetra_phdos_output"] = child.get("output")
            results["phonon_tetra_dos_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("epc_fldos") or "phonon.dos")).as_posix()
            )
            results["phonon_dos_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("phdos_flvec") or "phonon_dos.modes")).as_posix()
            )
        elif stage == "epc_phdos":
            results["phonon_epc_phdos_output"] = child.get("output")
            results["phonon_epc_dos_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("epc_fldos") or "phonon.dos")).as_posix()
            )
            results["phonon_dos_modes_file"] = str(
                (Path(PHONON_DIR) / str(phonon_options.get("phdos_flvec") or "phonon_dos.modes")).as_posix()
            )
        elif stage == "lambda":
            results["phonon_lambda_output"] = child.get("output")
            results["phonon_epc_lambda"] = lambda_outputs.get("lambda")
        elif stage == "tetra_alpha2f":
            results["phonon_tetra_alpha2f_output"] = child.get("output")
            results["phonon_tetra_alpha2f"] = alpha2f_outputs.get("alpha2f")
            results["phonon_epc_tetrahedra"] = {
                "alpha2f": alpha2f_outputs.get("alpha2f"),
                "tetra_scf_kmesh": list(phonon_options.get("tetra_scf_kmesh") or [8, 8, 8]),
                "tetra_q_grid": list(phonon_options.get("tetra_q_grid") or [4, 4, 4]),
                "tetra_elph_kmesh": list(phonon_options.get("tetra_elph_kmesh") or [24, 24, 24]),
            }

        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
        raise


def write_phonon_dos_outputs(
    workflow_root: Path,
    *,
    fldos: str,
    log_path: Path,
) -> dict[str, Any]:
    dos_relative = str(Path(PHONON_DIR) / (fldos or "phonon.phdos"))
    dos_path = workflow_root / dos_relative
    if not dos_path.is_file():
        raise FileNotFoundError(f"Phonon DOS file is missing: {dos_path}")

    frequencies, dos_values = parse_phonon_dos_file(dos_path)
    if not frequencies or not dos_values:
        raise RuntimeError(f"Phonon DOS file is empty or unreadable: {dos_path}")

    csv_relative = str(Path(DATA_PLOTS_DIR) / "phonon_dos.csv")
    json_relative = str(Path(DATA_PLOTS_DIR) / "phonon_dos_data.json")
    plot_relative = str(Path(DATA_PLOTS_DIR) / "phonon_dos.png")

    write_phonon_dos_csv(
        workflow_root / csv_relative,
        frequencies=frequencies,
        dos_values=dos_values,
    )
    write_phonon_dos_plot(
        workflow_root / plot_relative,
        frequencies=frequencies,
        dos_values=dos_values,
    )

    metadata = {
        "source": dos_relative,
        "csv": csv_relative,
        "plot": plot_relative,
        "points": len(frequencies),
        "frequency_units": "cm^-1",
        "frequencies_cm-1": [float(value) for value in frequencies],
        "dos": [float(value) for value in dos_values],
        "data": [
            {
                "frequency_cm-1": float(frequency),
                "dos": float(dos_values[index]) if index < len(dos_values) else None,
            }
            for index, frequency in enumerate(frequencies)
        ],
        "generated_at": timestamp(),
    }
    write_json_file(workflow_root / json_relative, metadata)
    append_log(log_path, f"Saved phonon DOS data: {workflow_root / json_relative}\n")
    append_log(log_path, f"Saved phonon DOS plot: {workflow_root / plot_relative}\n")

    return {
        "dos_file": dos_relative,
        "dos_data": json_relative,
        "dos_csv": csv_relative,
        "dos_plot": plot_relative,
    }


def write_epc_lambda_outputs(
    workflow_root: Path,
    *,
    output_relative: str,
    log_path: Path,
    expected_tc: float | None = None,
    tolerance: float | None = None,
    input_degauss: float | None = None,
) -> dict[str, Any]:
    output = str(Path(output_relative).as_posix())
    output_path = workflow_root / output
    if not output_path.is_file():
        raise FileNotFoundError(f"lambda.x output file is missing: {output_path}")

    text = output_path.read_text(encoding="utf-8", errors="ignore")
    summary = parse_lambda_summary(text)
    tc_rows = parse_lambda_tc_rows(text)
    tc_match = select_lambda_tc_match(
        tc_rows,
        expected_tc=expected_tc,
        tolerance=tolerance,
        workflow_root=workflow_root,
        input_degauss=input_degauss,
    )
    if isinstance(tc_match, Mapping):
        for source_key, summary_key in (("lambda", "lambda"), ("omega_log", "omega_log"), ("tc", "tc")):
            value = optional_float_or_none(tc_match.get(source_key))
            if value is not None:
                summary[summary_key] = value
    payload = {
        "output": output,
        "summary": summary,
        "tc_rows": tc_rows,
        "tc_match": tc_match,
    }
    append_log(log_path, f"Saved EPC lambda summary from: {output_path}\n")
    return {"lambda": payload}


def write_alpha2f_outputs(
    workflow_root: Path,
    *,
    output_relative: str,
    log_path: Path,
) -> dict[str, Any]:
    output = str(Path(output_relative).as_posix())
    output_path = workflow_root / output
    if not output_path.is_file():
        raise FileNotFoundError(f"alpha2f.x output file is missing: {output_path}")

    text = output_path.read_text(encoding="utf-8", errors="ignore")
    payload = {
        "output": output,
        "summary": parse_lambda_summary(text),
    }
    append_log(log_path, f"Saved tetrahedra alpha2F summary from: {output_path}\n")
    return {"alpha2f": payload}


def parse_lambda_summary(text: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    patterns = {
        "lambda": r"(?:lambda|lambda_epc)\s*[=:]\s*([-+]?\d+(?:\.\d*)?(?:[deE][-+]?\d+)?)",
        "omega_log": r"(?:omega[_\s-]*log|wlog)\s*[=:]\s*([-+]?\d+(?:\.\d*)?(?:[deE][-+]?\d+)?)",
        "tc": r"(?:t_c|tc)\s*[=:]\s*([-+]?\d+(?:\.\d*)?(?:[deE][-+]?\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            summary[key] = float(match.group(1).replace("d", "e").replace("D", "e"))
        except ValueError:
            continue
    return summary


LAMBDA_NUMBER_PATTERN = r"[-+]?\d+(?:\.\d*)?(?:[deDE][-+]?\d+)?"


def parse_lambda_tc_rows(text: str) -> list[dict[str, float | int]]:
    """Parse lambda.x row table, preserving row index for elph.gamma.N lookup."""

    rows: list[dict[str, float | int]] = []
    lambda_pattern = re.compile(
        rf"lambda\s*=\s*({LAMBDA_NUMBER_PATTERN})"
        rf"(?:\s*\(\s*({LAMBDA_NUMBER_PATTERN})\s*\))?"
        rf".*?<log\s+w>\s*=\s*({LAMBDA_NUMBER_PATTERN})\s*K"
        rf".*?N\(Ef\)\s*=\s*({LAMBDA_NUMBER_PATTERN})"
        rf".*?degauss\s*=\s*({LAMBDA_NUMBER_PATTERN})",
        flags=re.IGNORECASE,
    )
    for match in lambda_pattern.finditer(text):
        row: dict[str, float | int] = {"index": len(rows) + 1}
        keys = ["lambda", "lambda_parenthesized", "omega_log", "n_ef", "degauss"]
        for key, raw in zip(keys, match.groups()):
            if raw is None:
                continue
            try:
                row[key] = float(raw.replace("D", "E").replace("d", "e"))
            except ValueError:
                pass
        rows.append(row)

    table_values: list[tuple[float, float, float]] = []
    in_table = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        compact = re.sub(r"\s+", " ", stripped.lower())
        if compact.startswith("lambda ") and "omega_log" in compact and "t_c" in compact:
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped:
            if table_values:
                break
            continue
        parts = stripped.split()
        if len(parts) < 3:
            if table_values:
                break
            continue
        try:
            table_values.append(
                (
                    float(parts[0].replace("D", "E").replace("d", "e")),
                    float(parts[1].replace("D", "E").replace("d", "e")),
                    float(parts[2].replace("D", "E").replace("d", "e")),
                )
            )
        except ValueError:
            if table_values:
                break

    for index, (lambda_value, omega_log, tc_value) in enumerate(table_values, start=1):
        if index <= len(rows):
            rows[index - 1]["table_lambda"] = lambda_value
            rows[index - 1]["table_omega_log"] = omega_log
            rows[index - 1]["tc"] = tc_value
        else:
            rows.append(
                {
                    "index": index,
                    "lambda": lambda_value,
                    "omega_log": omega_log,
                    "tc": tc_value,
                }
            )
    return rows


def select_lambda_tc_match(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_tc: float | None,
    tolerance: float | None,
    workflow_root: Path,
    input_degauss: float | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        tc = optional_float_or_none(row.get("tc"))
        index_value = row.get("index")
        try:
            row_index = int(index_value)
        except (TypeError, ValueError):
            continue
        if tc is None:
            continue
        elph_gamma_relative = str((Path(PHONON_DIR) / f"elph.gamma.{row_index}").as_posix())
        row_degauss = optional_float_or_none(row.get("degauss"))
        difference = abs(tc - float(expected_tc)) if expected_tc is not None else None
        candidate = {
            "index": row_index,
            "lambda": optional_float_or_none(row.get("lambda")),
            "omega_log": optional_float_or_none(row.get("omega_log")),
            "tc": tc,
            "expected_tc": float(expected_tc) if expected_tc is not None else None,
            "difference": difference,
            "within_tolerance": (
                bool(difference <= float(tolerance if tolerance is not None else 1.0))
                if difference is not None
                else None
            ),
            "lambda_output_degauss": row_degauss,
            "input_degauss": input_degauss,
            "elph_gamma_file": elph_gamma_relative,
            "elph_gamma_exists": (workflow_root / elph_gamma_relative).is_file(),
        }
        candidates.append(candidate)
    if not candidates:
        return None
    if expected_tc is None:
        return min(candidates, key=lambda item: int(item["index"]))
    return min(candidates, key=lambda item: (float(item["difference"]), int(item["index"])))


def parse_phonon_dos_file(path: Path) -> tuple[list[float], list[float]]:
    frequencies: list[float] = []
    dos_values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            frequency = float(parts[0].replace("d", "e").replace("D", "E"))
            dos_value = float(parts[1].replace("d", "e").replace("D", "E"))
        except ValueError:
            continue
        frequencies.append(frequency)
        dos_values.append(dos_value)
    return frequencies, dos_values


def write_phonon_dos_csv(
    path: Path,
    *,
    frequencies: Sequence[float],
    dos_values: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frequency(cm^-1)", "DOS"])
        for index, frequency in enumerate(frequencies):
            writer.writerow([float(frequency), float(dos_values[index]) if index < len(dos_values) else ""])


def write_phonon_dos_plot(
    path: Path,
    *,
    frequencies: Sequence[float],
    dos_values: Sequence[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(frequencies, dos_values, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("Phonon DOS")
    ax.set_title("Phonon Density of States")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


PHONON_PROJECTION_COLOR_PAIRS: list[tuple[str, str]] = [
    ("#8B0000", "#FF69B4"),
    ("#00008B", "#87CEFA"),
    ("#006400", "#90EE90"),
    ("#B26A00", "#FFD54F"),
    ("#4B0082", "#B39DDB"),
    ("#5D4037", "#BCAAA4"),
]
PHONON_PROJECTED_CSV_DIR = Path(DATA_PLOTS_DIR) / "phonon_projected"
CM1_TO_THZ = 0.0299792458


def phonon_projected_csv_relative(filename: str) -> str:
    return str((PHONON_PROJECTED_CSV_DIR / filename).as_posix())


def data_and_plots_relative(filename: str) -> str:
    return str((Path(DATA_PLOTS_DIR) / filename).as_posix())


def epc_alpha2f_csv_relative(broadening: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(broadening or "selected")).strip("_") or "selected"
    return data_and_plots_relative(f"epc_alpha2f_lambda_broadening_{clean}.csv")


def parse_phonon_freq_gp(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"phonon frequency file is missing: {path}")
    distances: list[float] = []
    frequencies: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            values = [float(part.replace("D", "E").replace("d", "e")) for part in parts]
        except ValueError:
            continue
        distances.append(values[0])
        frequencies.append(values[1:])
    if not distances or not frequencies:
        raise RuntimeError(f"Could not parse phonon frequencies from {path}")
    return {"distances": distances, "frequencies_cm-1": frequencies}


def parse_elph_gamma_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"elph.gamma file is missing: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise RuntimeError(f"elph.gamma file is empty: {path}")
    header = lines[0]
    nbnd_match = re.search(r"\bnbnd\s*=\s*(\d+)", header, flags=re.IGNORECASE)
    nks_match = re.search(r"\bnks\s*=\s*(\d+)", header, flags=re.IGNORECASE)
    if not nbnd_match:
        raise RuntimeError(f"Could not find nbnd in elph.gamma header: {path}")
    nbnd = int(nbnd_match.group(1))
    nks = int(nks_match.group(1)) if nks_match else None
    gamma_rows: list[list[float]] = []
    q_coords: list[list[float]] = []
    current: list[float] = []
    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "!")) or stripped == "/":
            continue
        try:
            values = [float(part.replace("D", "E").replace("d", "e")) for part in stripped.replace(",", " ").split()]
        except ValueError:
            continue
        if not current and len(values) == 3:
            q_coords.append(values[:3])
            continue
        current.extend(values)
        while len(current) >= nbnd:
            gamma_rows.append(current[:nbnd])
            current = current[nbnd:]
    if not gamma_rows:
        raise RuntimeError(f"Could not parse gamma rows from {path}")
    if nks is not None and len(gamma_rows) != nks:
        # QE can still provide usable data if the header count is stale; keep the rows and record both counts.
        pass
    return {"nbnd": nbnd, "nks": nks, "q_coords": q_coords, "gamma": gamma_rows}


def select_alpha2f_column(headers: Sequence[str], selected_index: int, log_path: Path) -> tuple[int, str]:
    broadening_count = max(0, len(headers) - 1)
    if broadening_count <= 0:
        raise RuntimeError("alpha2F.dat does not contain any broadening columns.")
    requested = max(1, int(selected_index))
    selected = requested
    if selected > broadening_count:
        selected = broadening_count
        append_log(
            log_path,
            f"Requested alpha2F broadening column {requested}, but only {broadening_count} columns are available; using column {selected}.\n",
        )
    return selected, str(headers[selected])


def parse_alpha2f_dat(path: Path, *, selected_index: int, log_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"alpha2F.dat is missing: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#"):
        raise RuntimeError(f"alpha2F.dat header must start with '#': {path}")
    headers = lines[0].lstrip("#").strip().split()
    if not headers or headers[0] != "E(THz)":
        raise RuntimeError(f"alpha2F.dat first header must be E(THz): {path}")
    column_index, broadening = select_alpha2f_column(headers, selected_index, log_path)
    frequencies_thz: list[float] = []
    alpha2f: list[float] = []
    for raw_line in lines[1:]:
        if raw_line.startswith(("#", "!")):
            continue
        parts = raw_line.replace(",", " ").split()
        if len(parts) <= column_index:
            continue
        try:
            freq = float(parts[0].replace("D", "E").replace("d", "e"))
            value = float(parts[column_index].replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        frequencies_thz.append(freq)
        alpha2f.append(max(0.0, value))
    if not frequencies_thz:
        raise RuntimeError(f"Could not parse selected alpha2F data from {path}")
    lambdas = cumulative_lambda_from_alpha2f(frequencies_thz, alpha2f)
    return {
        "source": str((Path(PHONON_DIR) / path.name).as_posix()) if not path.is_absolute() else str(path),
        "headers": headers,
        "selected_broadening_index": column_index,
        "selected_broadening_number": column_index,
        "selected_broadening": broadening,
        "frequency_thz": frequencies_thz,
        "frequency_cm-1": [value * 33.35641 for value in frequencies_thz],
        "alpha2f": alpha2f,
        "lambda": lambdas,
    }


def cumulative_lambda_from_alpha2f(frequencies_thz: Sequence[float], alpha2f: Sequence[float]) -> list[float]:
    eps = 1.0e-10
    lambdas: list[float] = [0.0]
    for index in range(1, min(len(frequencies_thz), len(alpha2f))):
        f0 = float(frequencies_thz[index - 1])
        f1 = float(frequencies_thz[index])
        a0 = max(0.0, float(alpha2f[index - 1]))
        a1 = max(0.0, float(alpha2f[index]))
        y0 = a0 / (f0 if abs(f0) > eps else eps)
        y1 = a1 / (f1 if abs(f1) > eps else eps)
        area = 2.0 * 0.5 * (y0 + y1) * (f1 - f0)
        lambdas.append(max(lambdas[-1], lambdas[-1] + area))
    return lambdas


def build_epc_linewidth_band_payload(
    *,
    freq_source: str,
    gamma_source: str,
    distances: Sequence[float],
    frequencies: Sequence[Sequence[float]],
    gamma_rows: Sequence[Sequence[float]],
    ticks: Mapping[str, Sequence[Any]],
    width_scale: float = 13.0,
) -> dict[str, Any]:
    q_count = min(len(distances), len(frequencies), len(gamma_rows))
    band_count = min(
        min((len(row) for row in frequencies[:q_count]), default=0),
        min((len(row) for row in gamma_rows[:q_count]), default=0),
    )
    linewidth: list[list[float]] = []
    max_value = 0.0
    for q_index in range(q_count):
        row: list[float] = []
        for band_index in range(band_count):
            frequency = max(float(frequencies[q_index][band_index]), 2.0)
            value = max(0.0, float(gamma_rows[q_index][band_index])) / (frequency * frequency)
            row.append(value)
            max_value = max(max_value, value)
        linewidth.append(row)
    normalized = [
        [(value / max_value if max_value > 0 else 0.0) for value in row]
        for row in linewidth
    ]
    return {
        "freq_source": freq_source,
        "gamma_source": gamma_source,
        "distances": [float(value) for value in distances[:q_count]],
        "frequencies_cm-1": [[float(value) for value in row[:band_count]] for row in frequencies[:q_count]],
        "linewidth_raw": linewidth,
        "linewidth_normalized": normalized,
        "width_scale_cm-1": float(width_scale),
        "ticks": {
            "positions": [float(value) for value in ticks.get("positions", [])],
            "labels": [str(value) for value in ticks.get("labels", [])],
        },
        "generated_at": timestamp(),
    }


def nice_axis_ceiling(value: float, *, fallback: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(number) or number <= 0:
        return float(fallback)
    exponent = math.floor(math.log10(number))
    base = 10.0 ** exponent
    scaled = number / base
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= step:
            return step * base
    return 10.0 * base


def integer_axis_ceiling(value: float, *, fallback: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(number) or number <= 0:
        return float(fallback)
    return float(max(1, math.ceil(number)))


def endpoint_inclusive_ticks(left: float, right: float) -> list[float]:
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        return [left, right]
    span = right - left
    if span <= 1.0:
        step = 0.5
    elif span <= 5.0:
        step = 1.0
    else:
        step = max(1.0, math.ceil(span / 5.0))
    ticks: list[float] = []
    value = left
    guard = 0
    while value < right and guard < 100:
        ticks.append(round(value, 10))
        value += step
        guard += 1
    if not ticks or not math.isclose(ticks[-1], right, rel_tol=1e-9, abs_tol=1e-9):
        ticks.append(round(right, 10))
    return ticks


def write_epc_linewidth_band_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    distances = payload.get("distances") or []
    frequencies = payload.get("frequencies_cm-1") or []
    linewidth_raw = payload.get("linewidth_raw") or []
    linewidth_normalized = payload.get("linewidth_normalized") or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_index", "distance", "mode_index", "frequency_cm-1", "linewidth_raw", "linewidth_normalized"])
        for q_index, row in enumerate(frequencies):
            for mode_index, frequency in enumerate(row):
                writer.writerow(
                    [
                        q_index,
                        distances[q_index] if q_index < len(distances) else "",
                        mode_index,
                        frequency,
                        linewidth_raw[q_index][mode_index] if q_index < len(linewidth_raw) and mode_index < len(linewidth_raw[q_index]) else "",
                        linewidth_normalized[q_index][mode_index]
                        if q_index < len(linewidth_normalized) and mode_index < len(linewidth_normalized[q_index])
                        else "",
                    ]
                )


def write_alpha2f_lambda_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frequencies_thz = payload.get("frequency_thz") or []
    frequencies_cm1 = payload.get("frequency_cm-1") or []
    alpha2f = payload.get("alpha2f") or []
    lambdas = payload.get("lambda") or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frequency(THz)", "Frequency(cm-1)", "Alpha2F", "Lambda"])
        for index, frequency in enumerate(frequencies_thz):
            writer.writerow(
                [
                    frequency,
                    frequencies_cm1[index] if index < len(frequencies_cm1) else "",
                    alpha2f[index] if index < len(alpha2f) else "",
                    lambdas[index] if index < len(lambdas) else "",
                ]
            )


def write_epc_lambda_ploting_outputs(
    workflow_root: Path,
    *,
    task: Mapping[str, Any],
    state: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    lambda_payload = results.get("phonon_epc_lambda") if isinstance(results.get("phonon_epc_lambda"), Mapping) else {}
    tc_match = lambda_payload.get("tc_match") if isinstance(lambda_payload.get("tc_match"), Mapping) else {}
    selected_index = int(tc_match.get("index") or 1)
    selected_gamma_rel = str(tc_match.get("elph_gamma_file") or (Path(PHONON_DIR) / f"elph.gamma.{selected_index}").as_posix())
    gamma_path = workflow_root / selected_gamma_rel
    freq_rel, freq_path = resolve_epc_frequency_gp(workflow_root, task=task, state=state, log_path=log_path)
    alpha2f_rel = str((Path(PHONON_DIR) / "alpha2F.dat").as_posix())
    alpha2f_path = workflow_root / alpha2f_rel

    freq_payload = parse_phonon_freq_gp(freq_path)
    gamma_payload = parse_elph_gamma_file(gamma_path)
    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    phonon_options = params.get("phonon_options") if isinstance(params.get("phonon_options"), Mapping) else {}
    target_max = float((freq_payload.get("distances") or [0.0])[-1])
    ticks = phonon_q_path_ticks(phonon_options.get("matdyn_q_path") or [], target_max)
    band_payload = build_epc_linewidth_band_payload(
        freq_source=freq_rel,
        gamma_source=selected_gamma_rel,
        distances=freq_payload["distances"],
        frequencies=freq_payload["frequencies_cm-1"],
        gamma_rows=gamma_payload["gamma"],
        ticks=ticks,
    )
    alpha_payload = parse_alpha2f_dat(alpha2f_path, selected_index=selected_index, log_path=log_path)

    band_data_rel = data_and_plots_relative("epc_phonon_linewidth_band_data.json")
    band_csv_rel = data_and_plots_relative("epc_phonon_linewidth_band_data.csv")
    alpha_csv_rel = epc_alpha2f_csv_relative(str(alpha_payload.get("selected_broadening") or selected_index))
    plot_data_rel = data_and_plots_relative("epc_alpha2f_lambda_plot_data.json")
    plot_rel = data_and_plots_relative("epc_phonon_linewidth_alpha2f_lambda.png")
    replot_rel = data_and_plots_relative("replot_epc_phonon_alpha2f_lambda.py")

    combined_payload = {
        "stage": "lambda_ploting",
        "selected_index": selected_index,
        "selected_elph_gamma_file": selected_gamma_rel,
        "selected_alpha2f_broadening": alpha_payload.get("selected_broadening"),
        "selected_alpha2f_broadening_index": alpha_payload.get("selected_broadening_number"),
        "band_data": band_data_rel,
        "band_csv": band_csv_rel,
        "alpha2f_lambda_data": alpha_csv_rel,
        "combined_plot": plot_rel,
        "replot_script": replot_rel,
        "lambda": lambda_payload,
        "generated_at": timestamp(),
    }

    write_json_file(workflow_root / band_data_rel, band_payload)
    write_epc_linewidth_band_csv(workflow_root / band_csv_rel, band_payload)
    write_alpha2f_lambda_csv(workflow_root / alpha_csv_rel, alpha_payload)
    write_json_file(
        workflow_root / plot_data_rel,
        {
            "band": band_payload,
            "alpha2f_lambda": alpha_payload,
            "metadata": combined_payload,
            "generated_at": timestamp(),
        },
    )
    write_epc_phonon_alpha2f_lambda_plot(
        workflow_root / plot_rel,
        band_payload=band_payload,
        alpha_payload=alpha_payload,
    )
    write_epc_lambda_replot_script(workflow_root / replot_rel)
    append_log(log_path, f"Saved EPC phonon linewidth + alpha2F/lambda plot: {workflow_root / plot_rel}\n")
    return {**combined_payload, "plot_data": plot_data_rel}


def resolve_epc_frequency_gp(
    workflow_root: Path,
    *,
    task: Mapping[str, Any],
    state: Mapping[str, Any],
    log_path: Path,
) -> tuple[str, Path]:
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    phonon_options = params.get("phonon_options") if isinstance(params.get("phonon_options"), Mapping) else {}

    candidates: list[str] = []

    def add_candidate(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        path = Path(text)
        if not path.is_absolute() and path.parts and path.parts[0] != PHONON_DIR:
            path = Path(PHONON_DIR) / path
        base = str(path.as_posix())
        if not base.endswith(".gp"):
            candidates.append(f"{base}.gp")
        candidates.append(base)

    if isinstance(results, Mapping):
        add_candidate(results.get("phonon_epc_frequency_file"))
        add_candidate(results.get("phonon_frequency_file"))
    add_candidate(phonon_options.get("flfrq") or "phonon.freq.band")
    for fallback in (
        Path(PHONON_DIR) / "phonon.freq.band.gp",
        Path(PHONON_DIR) / "phonon.freq.gp",
        Path(PHONON_DIR) / "matdyn.freq.gp",
    ):
        candidates.append(str(fallback.as_posix()))

    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        absolute = path if path.is_absolute() else workflow_root / path
        checked.append(str(absolute))
        if absolute.is_file():
            if path.is_absolute():
                try:
                    rel = str(absolute.relative_to(workflow_root).as_posix())
                except ValueError:
                    rel = str(absolute.as_posix())
            else:
                rel = str(path.as_posix())
            append_log(log_path, f"Using EPC phonon frequency file: {absolute}\n")
            return rel, absolute

    raise FileNotFoundError(f"phonon frequency file is missing; checked: {', '.join(checked)}")


def write_epc_phonon_alpha2f_lambda_plot(
    path: Path,
    *,
    band_payload: Mapping[str, Any],
    alpha_payload: Mapping[str, Any],
    y_min: float | None = None,
    y_max: float | None = None,
    ribbon_scale: float = 13.0,
    ribbon_color: str = "#D00000",
    band_color: str = "black",
    alpha2f_color: str = "#00008B",
    lambda_color: str = "#C00000",
    alpha2f_x_min: float | None = None,
    alpha2f_x_max: float | None = None,
    lambda_x_min: float | None = None,
    lambda_x_max: float | None = None,
    dpi: int = 300,
    figsize: Sequence[float] | None = None,
    border_width: float = 2.3,
    label_font_size: float = 17.0,
    tick_font_size: float = 13.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    distances = [float(value) for value in band_payload.get("distances") or []]
    frequencies = band_payload.get("frequencies_cm-1") or []
    widths = band_payload.get("linewidth_normalized") or []
    width_scale = float(ribbon_scale)
    freq_alpha = [float(value) for value in alpha_payload.get("frequency_cm-1") or []]
    alpha2f = [float(value) for value in alpha_payload.get("alpha2f") or []]
    lambdas = [float(value) for value in alpha_payload.get("lambda") or []]

    fig, (ax_band, ax_spec) = plt.subplots(
        1,
        2,
        figsize=tuple(float(value) for value in (figsize or (8.8, 6.2))),
        dpi=int(dpi),
        gridspec_kw={"width_ratios": [3.5, 1.25], "wspace": 0.08},
        sharey=True,
    )

    band_count = max((len(row) for row in frequencies), default=0)
    for band_index in range(band_count):
        y_values = [float(row[band_index]) for row in frequencies if band_index < len(row)]
        x_values = distances[: len(y_values)]
        if not x_values or not y_values:
            continue
        band_widths = [
            float(widths[q_index][band_index]) * width_scale
            for q_index in range(min(len(widths), len(y_values)))
            if band_index < len(widths[q_index])
        ]
        if len(band_widths) == len(y_values):
            lower = [y_values[index] - band_widths[index] for index in range(len(y_values))]
            upper = [y_values[index] + band_widths[index] for index in range(len(y_values))]
            ax_band.fill_between(x_values, lower, upper, color=str(ribbon_color), alpha=0.9, linewidth=0, zorder=2)
        ax_band.plot(x_values, y_values, color=str(band_color), linewidth=1.15, zorder=3)

    ticks = band_payload.get("ticks") if isinstance(band_payload.get("ticks"), Mapping) else {}
    tick_positions = [float(value) for value in ticks.get("positions", [])]
    tick_labels = [str(value) for value in ticks.get("labels", [])]
    if tick_positions and len(tick_positions) == len(tick_labels):
        ax_band.set_xticks(tick_positions)
        ax_band.set_xticklabels(tick_labels, fontweight="bold")
        for value in tick_positions:
            ax_band.axvline(value, color="#cccccc", linewidth=0.8, zorder=1)
    if distances:
        ax_band.set_xlim(min(distances), max(distances) if max(distances) > min(distances) else min(distances) + 1.0)
    all_freqs = [float(value) for row in frequencies for value in row]
    if freq_alpha:
        all_freqs.extend(freq_alpha)
    lower = 0.0 if y_min is None else float(y_min)
    upper = rounded_phonon_frequency_ceiling(max(all_freqs), fallback=400.0) if y_max is None and all_freqs else float(y_max or 400.0)
    ax_band.set_ylim(lower, upper)
    ax_band.set_ylabel(r"Frequency (cm$^{-1}$)", fontsize=label_font_size, fontweight="bold")
    ax_band.set_xlabel("")
    ax_band.tick_params(axis="both", which="major", width=border_width, length=5, labelsize=tick_font_size)
    for label in ax_band.get_yticklabels():
        label.set_fontweight("bold")

    ax_spec.plot(alpha2f, freq_alpha, color=str(alpha2f_color), linewidth=1.1)
    alpha_max = max(alpha2f) if alpha2f else 1.0
    alpha_left = 0.0 if alpha2f_x_min is None else float(alpha2f_x_min)
    alpha_right = integer_axis_ceiling(alpha_max, fallback=1.0) if alpha2f_x_max is None else float(alpha2f_x_max)
    ax_spec.set_xlim(
        alpha_left,
        alpha_right,
    )
    ax_spec.xaxis.set_major_locator(FixedLocator(endpoint_inclusive_ticks(alpha_left, alpha_right)))
    ax_spec.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax_spec.set_xlabel("")
    ax_spec.tick_params(axis="x", colors=str(alpha2f_color), width=border_width, labelsize=tick_font_size)
    ax_spec.tick_params(axis="y", width=border_width, length=5, labelleft=False)
    for label in ax_spec.get_xticklabels():
        label.set_fontweight("bold")

    ax_lambda = ax_spec.twiny()
    ax_lambda.plot(lambdas, freq_alpha[: len(lambdas)], color=str(lambda_color), linewidth=1.2)
    lambda_max = max(lambdas) if lambdas else 1.0
    lambda_left = 0.0 if lambda_x_min is None else float(lambda_x_min)
    lambda_right = integer_axis_ceiling(lambda_max, fallback=1.0) if lambda_x_max is None else float(lambda_x_max)
    ax_lambda.set_xlim(
        lambda_left,
        lambda_right,
    )
    ax_lambda.xaxis.set_major_locator(FixedLocator(endpoint_inclusive_ticks(lambda_left, lambda_right)))
    ax_lambda.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax_lambda.set_xlabel("")
    ax_lambda.tick_params(axis="x", colors=str(lambda_color), width=border_width, labelsize=tick_font_size)
    for label in ax_lambda.get_xticklabels():
        label.set_fontweight("bold")

    handles = [
        Line2D([0], [0], color=str(lambda_color), linewidth=2.0, label="λ"),
        Line2D([0], [0], color=str(alpha2f_color), linewidth=2.0, label="α²F(ω)"),
    ]
    legend = ax_spec.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        prop={"weight": "bold", "size": max(8.0, tick_font_size - 1.0)},
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for axis in (ax_band, ax_spec, ax_lambda):
        for spine in axis.spines.values():
            spine.set_linewidth(border_width)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.90, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def write_epc_lambda_replot_script(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
"""Replot EPC phonon linewidth + alpha2F/lambda from saved JSON data.

Edit the USER CONFIGURATION block, then run:

    python replot_epc_phonon_alpha2f_lambda.py
"""
import json
import sys
from pathlib import Path

ROOT_CANDIDATES = [
    Path(__PROJECT_ROOT__),
    Path(__file__).resolve().parents[2],
    Path.cwd(),
]
for candidate in ROOT_CANDIDATES:
    if (candidate / "workflow_runner.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from workflow_runner import write_epc_phonon_alpha2f_lambda_plot


# ================= USER CONFIGURATION =================

PLOT_DATA_FILE = "epc_alpha2f_lambda_plot_data.json"
OUTPUT_FILE = "epc_phonon_linewidth_alpha2f_lambda.png"
Y_MIN = None
Y_MAX = None
ALPHA2F_X_MIN = None
ALPHA2F_X_MAX = None
LAMBDA_X_MIN = None
LAMBDA_X_MAX = None
RIBBON_SCALE = 13.0
RIBBON_COLOR = "#D00000"
BAND_COLOR = "black"
ALPHA2F_COLOR = "#00008B"
LAMBDA_COLOR = "#C00000"
DPI = 300
FIGSIZE = (8.8, 6.2)

# ================= END USER CONFIGURATION =================


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def main() -> int:
    payload = json.loads(resolve_path(PLOT_DATA_FILE).read_text(encoding="utf-8"))
    write_epc_phonon_alpha2f_lambda_plot(
        resolve_path(OUTPUT_FILE),
        band_payload=payload["band"],
        alpha_payload=payload["alpha2f_lambda"],
        y_min=Y_MIN,
        y_max=Y_MAX,
        ribbon_scale=RIBBON_SCALE,
        ribbon_color=RIBBON_COLOR,
        band_color=BAND_COLOR,
        alpha2f_color=ALPHA2F_COLOR,
        lambda_color=LAMBDA_COLOR,
        alpha2f_x_min=ALPHA2F_X_MIN,
        alpha2f_x_max=ALPHA2F_X_MAX,
        lambda_x_min=LAMBDA_X_MIN,
        lambda_x_max=LAMBDA_X_MAX,
        dpi=DPI,
        figsize=FIGSIZE,
    )
    print(f"Saved {resolve_path(OUTPUT_FILE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.replace("__PROJECT_ROOT__", json.dumps(str(PROJECT_ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def lambda_emax_from_projected_band_csv(workflow_root: Path) -> int | None:
    """Return ceil(max frequency_cm-1 converted to THz) from saved phonon band CSV."""

    csv_path = workflow_root / PHONON_PROJECTED_CSV_DIR / "phonon_projected_band_data.csv"
    if not csv_path.is_file():
        return None
    max_frequency_cm1: float | None = None
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_value = row.get("frequency_cm-1")
            if raw_value is None:
                continue
            try:
                value = float(str(raw_value).strip())
            except ValueError:
                continue
            if max_frequency_cm1 is None or value > max_frequency_cm1:
                max_frequency_cm1 = value
    if max_frequency_cm1 is None:
        return None
    return max(1, int(math.ceil(max_frequency_cm1 * CM1_TO_THZ)))


def lambda_emax_from_phonon_freq_gp(workflow_root: Path) -> int | None:
    freq_path = workflow_root / PHONON_DIR / "phonon.freq.gp"
    if not freq_path.is_file():
        return None
    try:
        payload = parse_phonon_freq_gp(freq_path)
    except Exception:
        return None
    max_frequency: float | None = None
    for row in payload.get("frequencies") or []:
        if not isinstance(row, Sequence):
            continue
        for value in row:
            try:
                frequency = float(value)
            except (TypeError, ValueError):
                continue
            if max_frequency is None or frequency > max_frequency:
                max_frequency = frequency
    if max_frequency is None:
        return None
    return max(1, int(math.ceil(max_frequency * CM1_TO_THZ)))


def resolve_lambda_emax(workflow_root: Path, phonon_options: Mapping[str, Any], log_path: Path) -> str:
    requested = str(phonon_options.get("lambda_emax") or "auto").strip()
    if requested and requested.lower() not in {"auto", "automatic"}:
        return requested
    computed = lambda_emax_from_projected_band_csv(workflow_root)
    if computed is not None:
        append_log(log_path, f"Using lambda.x emax={computed} from projected phonon band CSV.\n")
        return str(computed)
    computed = lambda_emax_from_phonon_freq_gp(workflow_root)
    if computed is not None:
        append_log(log_path, f"Using lambda.x emax={computed} from phonon.freq.gp.\n")
        return str(computed)
    fallback = "10"
    append_log(log_path, f"Could not compute lambda.x emax from phonon frequency data; using {fallback}.\n")
    return fallback


def parse_lambda_q_points_from_q2r_output(q2r_output: Path) -> list[list[float]]:
    """Parse lambda.x representative q-points and nqs weights from q2r.x output."""

    if not q2r_output.is_file():
        return []
    lines = q2r_output.read_text(encoding="utf-8", errors="ignore").splitlines()
    points: list[list[float]] = []
    for index, line in enumerate(lines):
        weight_match = re.search(r"\bnqs\s*=\s*([0-9]+)", line, flags=re.IGNORECASE)
        if not weight_match:
            continue
        weight = float(weight_match.group(1))
        q_line = lines[index + 1] if index + 1 < len(lines) else ""
        q_match = re.search(
            r"\bq\s*=\s*([0-9.DEdEe+\-]+)\s+([0-9.DEdEe+\-]+)\s+([0-9.DEdEe+\-]+)",
            q_line,
            flags=re.IGNORECASE,
        )
        if not q_match:
            continue
        try:
            qx, qy, qz = [float(item.replace("D", "E").replace("d", "e")) for item in q_match.groups()]
        except ValueError:
            continue
        points.append([qx, qy, qz, weight])
    return points


def lambda_q_points_from_q2r_outputs(workflow_root: Path, log_path: Path) -> list[list[float]]:
    candidates = [
        Path(PHONON_DIR) / "tetra_q2r.out",
        Path(PHONON_DIR) / "epc_q2r.out",
        Path(PHONON_DIR) / "q2r.out",
    ]
    for relative in candidates:
        points = parse_lambda_q_points_from_q2r_output(workflow_root / relative)
        if points:
            append_log(log_path, f"Using {len(points)} lambda.x q-points from {relative.as_posix()}.\n")
            return points
    checked = ", ".join(str(relative.as_posix()) for relative in candidates)
    append_log(log_path, f"Could not parse lambda.x q-points from q2r output ({checked}); using defaults.\n")
    return []


def lambda_degauss_scan_values(max_degauss: float = 0.8) -> list[float]:
    upper = min(0.8, max(0.1, float(max_degauss)))
    count = int(math.floor((upper + 1.0e-12) / 0.1))
    return [round(0.1 * index, 10) for index in range(1, count + 1)]


def lambda_scan_stem(degauss: float) -> str:
    return f"lambda_degauss_{degauss:.3f}".replace(".", "p")


def run_lambda_tc_scan(
    *,
    workflow_root: Path,
    child_workdir: Path,
    input_path: Path,
    output_path: Path,
    emax: str,
    smearing_method: int,
    mu: str,
    q_points: Sequence[Sequence[Any]],
    expected_tc: float,
    tolerance: float,
    max_degauss: float,
    log_path: Path,
) -> dict[str, Any]:
    best_payload: dict[str, Any] | None = None
    selected_within_tolerance = False
    selected_input: str | None = None
    selected_output_relative: str | None = None
    scan_outputs: list[str] = []

    for degauss in lambda_degauss_scan_values(max_degauss):
        rendered = render_lambda_input(
            emax=emax,
            degauss=f"{degauss:.1f}",
            smearing_method=smearing_method,
            mu=mu,
            q_points=q_points,
        )
        input_path.write_text(rendered, encoding="utf-8")
        trial_stem = lambda_scan_stem(degauss)
        trial_output_name = f"{trial_stem}.out"
        trial_output_relative = str((Path(PHONON_DIR) / trial_output_name).as_posix())
        scan_outputs.append(trial_output_relative)
        append_log(log_path, f"Running lambda.x T_c scan at degauss={degauss:.1f}\n")
        command = build_qe_stdin_command("lambda.x", input_path.name, trial_output_name)
        append_log(log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=log_path)
        payload = write_epc_lambda_outputs(
            workflow_root,
            output_relative=trial_output_relative,
            log_path=log_path,
            expected_tc=expected_tc,
            tolerance=tolerance,
            input_degauss=degauss,
        ).get("lambda") or {}
        match = payload.get("tc_match") if isinstance(payload, Mapping) else None
        if isinstance(match, Mapping):
            payload["scan_input"] = str((Path(PHONON_DIR) / f"{trial_stem}.in").as_posix())
            payload["scan_output"] = trial_output_relative
            payload["scan_outputs"] = list(scan_outputs)
            trial_input_path = child_workdir / f"{trial_stem}.in"
            trial_input_path.write_text(rendered, encoding="utf-8")
            if best_payload is None:
                best_payload = dict(payload)
                selected_input = rendered
                selected_output_relative = trial_output_relative
            else:
                best_match = best_payload.get("tc_match") if isinstance(best_payload.get("tc_match"), Mapping) else {}
                if float(match.get("difference", float("inf"))) < float(best_match.get("difference", float("inf"))):
                    best_payload = dict(payload)
                    selected_input = rendered
                    selected_output_relative = trial_output_relative
            if bool(match.get("within_tolerance")):
                selected_within_tolerance = True
                break

    if best_payload is None or selected_input is None or selected_output_relative is None:
        raise RuntimeError("lambda.x T_c scan did not produce any parseable T_c rows.")

    selected_output_path = workflow_root / selected_output_relative
    shutil.copyfile(selected_output_path, output_path)
    input_path.write_text(selected_input, encoding="utf-8")
    final_payload = write_epc_lambda_outputs(
        workflow_root,
        output_relative=str((Path(PHONON_DIR) / output_path.name).as_posix()),
        log_path=log_path,
        expected_tc=expected_tc,
        tolerance=tolerance,
        input_degauss=(
            best_payload.get("tc_match", {}).get("input_degauss")
            if isinstance(best_payload.get("tc_match"), Mapping)
            else None
        ),
    ).get("lambda") or {}
    final_payload["scan_outputs"] = scan_outputs
    final_payload["selected_scan_output"] = selected_output_relative
    final_payload["selected_within_tolerance"] = selected_within_tolerance
    append_log(log_path, f"Selected lambda.x scan output: {selected_output_relative}\n")
    return {"lambda": final_payload}


def cleanup_legacy_projected_layout(workflow_root: Path) -> None:
    data_dir = workflow_root / DATA_PLOTS_DIR
    for filename in ("phonon_projected_band_data.csv", "phonon_projected_dos_data.csv"):
        path = data_dir / filename
        try:
            if path.is_file():
                path.unlink()
        except Exception:
            pass

    legacy_dir = workflow_root / PHONON_PROJECTED_CSV_DIR
    if not legacy_dir.is_dir():
        return
    for path in legacy_dir.iterdir():
        if path.suffix.lower() == ".csv":
            continue
        try:
            if path.is_file():
                path.unlink()
        except Exception:
            pass


def write_phonon_projected_band_outputs(
    workflow_root: Path,
    *,
    flvec: str,
    q_path_lines: Sequence[str],
    log_path: Path,
) -> dict[str, Any]:
    cleanup_legacy_projected_layout(workflow_root)
    modes_relative = str(Path(PHONON_DIR) / (flvec or "phonon_band.modes"))
    modes_path = workflow_root / modes_relative
    if not modes_path.is_file():
        append_log(log_path, f"Phonon band modes file not found; skipping ploting data: {modes_path}\n")
        return {}

    layout = infer_phonon_atom_layout_from_scf(workflow_root)
    parsed = parse_matdyn_modes_file(modes_path, total_atoms=layout["total_atoms"])
    sorted_freqs, sorted_eigs = sort_phonon_modes_by_overlap(parsed["frequencies"], parsed["eigenvectors"])
    distances = cumulative_q_distances(parsed["q_coords"])
    ticks = phonon_q_path_ticks(q_path_lines, distances[-1] if distances else 0.0)
    colors = phonon_projection_colors(layout["elements"])

    payload = build_projected_band_payload(
        source=modes_relative,
        layout=layout,
        colors=colors,
        q_coords=parsed["q_coords"],
        distances=distances,
        frequencies=sorted_freqs,
        eigenvectors=sorted_eigs,
        ticks=ticks,
    )

    data_relative = data_and_plots_relative("phonon_projected_band_data.json")
    csv_relative = phonon_projected_csv_relative("phonon_projected_band_data.csv")
    write_json_file(workflow_root / data_relative, payload)
    write_projected_band_csv(workflow_root / csv_relative, payload)
    append_log(log_path, f"Saved projected phonon band data: {workflow_root / data_relative}\n")
    return {
        "band_modes_file": modes_relative,
        "band_data": data_relative,
        "band_csv": csv_relative,
        "band_payload": payload,
    }


def write_phonon_projected_dos_outputs(
    workflow_root: Path,
    *,
    flvec: str,
    log_path: Path,
) -> dict[str, Any]:
    modes_relative = str(Path(PHONON_DIR) / (flvec or "phonon_dos.modes"))
    modes_path = workflow_root / modes_relative
    if not modes_path.is_file():
        append_log(log_path, f"Phonon DOS modes file not found; skipping ploting data: {modes_path}\n")
        return {}

    layout = infer_phonon_atom_layout_from_scf(workflow_root)
    parsed = parse_matdyn_modes_file(modes_path, total_atoms=layout["total_atoms"])
    colors = phonon_projection_colors(layout["elements"])
    payload = build_projected_dos_payload(
        source=modes_relative,
        layout=layout,
        colors=colors,
        frequencies=parsed["frequencies"],
        eigenvectors=parsed["eigenvectors"],
    )

    data_relative = data_and_plots_relative("phonon_projected_dos_data.json")
    csv_relative = phonon_projected_csv_relative("phonon_projected_dos_data.csv")
    write_json_file(workflow_root / data_relative, payload)
    write_projected_dos_csv(workflow_root / csv_relative, payload)
    append_log(log_path, f"Saved projected phonon DOS data: {workflow_root / data_relative}\n")
    return {
        "dos_modes_file": modes_relative,
        "projected_dos_data": data_relative,
        "projected_dos_csv": csv_relative,
        "dos_payload": payload,
    }


def maybe_write_phonon_projected_combined_outputs(workflow_root: Path, *, log_path: Path) -> dict[str, Any]:
    band_relative = data_and_plots_relative("phonon_projected_band_data.json")
    dos_relative = data_and_plots_relative("phonon_projected_dos_data.json")
    band_path = workflow_root / band_relative
    dos_path = workflow_root / dos_relative
    if not band_path.is_file() or not dos_path.is_file():
        return {}

    band_payload = load_json_file(band_path)
    dos_payload = load_json_file(dos_path)
    combined = {
        "band_data": band_relative,
        "dos_data": dos_relative,
        "elements": band_payload.get("layout", {}).get("elements") or dos_payload.get("layout", {}).get("elements") or [],
        "colors": band_payload.get("colors") or dos_payload.get("colors") or {},
        "generated_at": timestamp(),
    }
    combined_relative = data_and_plots_relative("phonon_projected_plot_data.json")
    plot_relative = data_and_plots_relative("phonon_projected_band_dos.png")
    replot_relative = data_and_plots_relative("replot_phonon_projected.py")
    write_json_file(workflow_root / combined_relative, combined)
    write_phonon_projected_combined_plot(
        workflow_root / plot_relative,
        band_payload=band_payload,
        dos_payload=dos_payload,
    )
    write_phonon_projected_replot_script(workflow_root / replot_relative)
    append_log(log_path, f"Saved projected phonon band+DOS plot: {workflow_root / plot_relative}\n")
    return {
        "combined_plot": plot_relative,
        "plot_data": combined_relative,
        "replot_script": replot_relative,
    }


def infer_phonon_atom_layout_from_scf(workflow_root: Path) -> dict[str, Any]:
    candidates = [
        Path(PHONON_DIR) / "scf.in",
        Path(PHONON_DIR) / "epc_scf.in",
        Path(PHONON_DIR) / "tetra_scf.in",
    ]
    scf_path: Path | None = None
    scf_relative: Path | None = None
    for relative in candidates:
        candidate = workflow_root / relative
        if candidate.is_file():
            scf_path = candidate
            scf_relative = relative
            break
    if scf_path is None or scf_relative is None:
        checked = ", ".join(str(workflow_root / relative) for relative in candidates)
        raise FileNotFoundError(f"Phonon SCF input is missing; cannot infer atom layout. Checked: {checked}")
    elements = parse_atomic_positions_element_order(scf_path.read_text(encoding="utf-8", errors="ignore"))
    if not elements:
        raise RuntimeError(f"Could not infer atom order from ATOMIC_POSITIONS in {scf_path}")
    counts: dict[str, int] = {}
    order: list[str] = []
    indices: dict[str, list[int]] = {}
    for atom_index, element in enumerate(elements):
        if element not in counts:
            counts[element] = 0
            order.append(element)
            indices[element] = []
        counts[element] += 1
        indices[element].append(atom_index)
    return {
        "source": scf_relative.as_posix(),
        "elements": order,
        "counts": counts,
        "indices": indices,
        "total_atoms": len(elements),
    }


def parse_atomic_positions_element_order(text: str) -> list[str]:
    lines = text.splitlines()
    elements: list[str] = []
    in_card = False
    for raw_line in lines:
        stripped = raw_line.strip()
        upper = stripped.upper()
        if not in_card:
            if upper.startswith("ATOMIC_POSITIONS"):
                in_card = True
            continue
        if not stripped or stripped.startswith(("!", "#")):
            continue
        if upper.startswith(("K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES")) or stripped.startswith("&"):
            break
        parts = stripped.split()
        if parts:
            elements.append(parts[0])
    return elements


def parse_matdyn_modes_file(path: Path, *, total_atoms: int) -> dict[str, Any]:
    num_pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[dDeE][-+]?\d+)?")
    q_coords: list[list[float]] = []
    frequencies: list[list[float]] = []
    eigenvectors: list[list[list[list[complex]]]] = []
    current_freqs: list[float] = []
    current_eigs: list[list[list[complex]]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()
        if lower.startswith("q") and "=" in stripped:
            if current_freqs:
                frequencies.append(current_freqs)
                eigenvectors.append(current_eigs)
                current_freqs = []
                current_eigs = []
            numbers = [parse_fortran_float(value) for value in num_pattern.findall(stripped.split("=", 1)[-1])]
            if len(numbers) >= 3:
                q_coords.append([float(numbers[0]), float(numbers[1]), float(numbers[2])])
        elif lower.startswith("freq") and "=" in stripped:
            current_freqs.append(parse_matdyn_frequency_cm1(stripped, num_pattern))
            vectors: list[list[complex]] = []
            while len(vectors) < total_atoms and i + 1 < len(lines):
                i += 1
                values = [parse_fortran_float(value) for value in num_pattern.findall(lines[i])]
                if len(values) >= 6:
                    vectors.append(
                        [
                            complex(values[0], values[1]),
                            complex(values[2], values[3]),
                            complex(values[4], values[5]),
                        ]
                    )
            if len(vectors) == total_atoms:
                current_eigs.append(vectors)
        i += 1
    if current_freqs:
        frequencies.append(current_freqs)
        eigenvectors.append(current_eigs)
    if not q_coords or not frequencies or not eigenvectors:
        raise RuntimeError(f"No readable phonon modes found in {path}")
    return {"q_coords": q_coords, "frequencies": frequencies, "eigenvectors": eigenvectors}


def parse_matdyn_frequency_cm1(line: str, num_pattern: re.Pattern[str]) -> float:
    """Return the cm-1 frequency from a QE matdyn frequency line."""

    text = str(line or "").strip()
    lower = text.lower()
    if "[cm-1]" in lower:
        before_unit = text[: lower.index("[cm-1]")]
        segment = before_unit.rsplit("=", 1)[-1]
    else:
        segment = text.split("=", 1)[-1]
    numbers = [parse_fortran_float(value) for value in num_pattern.findall(segment)]
    if numbers:
        return float(numbers[-1])
    raise RuntimeError(f"Could not parse matdyn frequency line: {line}")


def parse_fortran_float(value: str) -> float:
    return float(str(value).replace("D", "E").replace("d", "e"))


def sort_phonon_modes_by_overlap(
    frequencies: Sequence[Sequence[float]],
    eigenvectors: Sequence[Sequence[Sequence[Sequence[complex]]]],
) -> tuple[list[list[float]], list[list[list[list[complex]]]]]:
    sorted_freqs = [[float(value) for value in row] for row in frequencies]
    sorted_eigs = [
        [
            [[complex(component) for component in atom] for atom in mode]
            for mode in q_modes
        ]
        for q_modes in eigenvectors
    ]
    if len(sorted_freqs) <= 1:
        return sorted_freqs, sorted_eigs
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return sorted_freqs, sorted_eigs

    for q_index in range(1, len(sorted_freqs)):
        prev_flat = [flatten_mode_vector(mode) for mode in sorted_eigs[q_index - 1]]
        curr_flat = [flatten_mode_vector(mode) for mode in sorted_eigs[q_index]]
        n_prev = len(prev_flat)
        n_curr = len(curr_flat)
        if n_prev != n_curr or n_prev == 0:
            continue
        cost: list[list[float]] = []
        for prev in prev_flat:
            row: list[float] = []
            for curr in curr_flat:
                row.append(1.0 - abs(complex_dot(prev, curr)))
            cost.append(row)
        _rows, cols = linear_sum_assignment(cost)
        sorted_freqs[q_index] = [sorted_freqs[q_index][int(col)] for col in cols]
        sorted_eigs[q_index] = [sorted_eigs[q_index][int(col)] for col in cols]
    return sorted_freqs, sorted_eigs


def flatten_mode_vector(mode: Sequence[Sequence[complex]]) -> list[complex]:
    return [complex(component) for atom in mode for component in atom]


def complex_dot(left: Sequence[complex], right: Sequence[complex]) -> complex:
    return sum(complex(a).conjugate() * complex(b) for a, b in zip(left, right))


def cumulative_q_distances(q_coords: Sequence[Sequence[float]]) -> list[float]:
    if not q_coords:
        return []
    distances = [0.0]
    for index in range(1, len(q_coords)):
        prev = q_coords[index - 1]
        curr = q_coords[index]
        step = math.sqrt(sum((float(curr[axis]) - float(prev[axis])) ** 2 for axis in range(3)))
        distances.append(distances[-1] + step)
    return distances


def phonon_q_path_ticks(q_path_lines: Sequence[str], target_max: float) -> dict[str, list[Any]]:
    raw_lines = [str(line).strip() for line in q_path_lines if str(line).strip()]
    if raw_lines:
        try:
            int(float(raw_lines[0].split()[0]))
            raw_lines = raw_lines[1:]
        except Exception:
            pass
    points: list[tuple[list[float], str]] = []
    for line in raw_lines:
        clean, _, comment = line.partition("!")
        parts = clean.split()
        if len(parts) < 3:
            continue
        try:
            coord = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            continue
        label = comment.strip() or ""
        points.append((coord, normalize_phonon_tick_label(label)))
    if not points:
        return {"positions": [0.0, float(target_max)], "labels": ["", ""]}
    distances = [0.0]
    for index in range(1, len(points)):
        prev = points[index - 1][0]
        curr = points[index][0]
        distances.append(distances[-1] + math.sqrt(sum((curr[axis] - prev[axis]) ** 2 for axis in range(3))))
    if distances[-1] > 0 and target_max > 0:
        scale = float(target_max) / distances[-1]
        distances = [value * scale for value in distances]
    labels = [label for _coord, label in points]
    return {"positions": [float(value) for value in distances], "labels": labels}


def normalize_phonon_tick_label(label: str) -> str:
    text = str(label or "").strip()
    if text.lower() in {"gamma", "g", "gm", "Γ".lower()}:
        return "Γ"
    return text


def phonon_projection_colors(elements: Sequence[str]) -> dict[str, dict[str, str]]:
    colors: dict[str, dict[str, str]] = {}
    for index, element in enumerate(elements):
        xy, z = PHONON_PROJECTION_COLOR_PAIRS[index % len(PHONON_PROJECTION_COLOR_PAIRS)]
        colors[str(element)] = {"XY": xy, "Z": z}
    return colors


def build_projected_band_payload(
    *,
    source: str,
    layout: Mapping[str, Any],
    colors: Mapping[str, Mapping[str, str]],
    q_coords: Sequence[Sequence[float]],
    distances: Sequence[float],
    frequencies: Sequence[Sequence[float]],
    eigenvectors: Sequence[Sequence[Sequence[Sequence[complex]]]],
    ticks: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    elements = [str(element) for element in layout.get("elements") or []]
    indices_by_element = layout.get("indices") if isinstance(layout.get("indices"), Mapping) else {}
    projections: dict[str, dict[str, list[list[float]]]] = {
        element: {"XY": [], "Z": []} for element in elements
    }
    for q_index, q_modes in enumerate(eigenvectors):
        for element in elements:
            projections[element]["XY"].append([])
            projections[element]["Z"].append([])
        for mode in q_modes:
            for element in elements:
                atom_indices = [int(value) for value in indices_by_element.get(element, [])]
                xy_weight = 0.0
                z_weight = 0.0
                for atom_index in atom_indices:
                    if atom_index >= len(mode):
                        continue
                    vec = mode[atom_index]
                    xy_weight += abs(vec[0]) ** 2 + abs(vec[1]) ** 2
                    z_weight += abs(vec[2]) ** 2
                projections[element]["XY"][q_index].append(float(xy_weight))
                projections[element]["Z"][q_index].append(float(z_weight))
    return {
        "source": source,
        "layout": layout,
        "colors": colors,
        "q_coords": [[float(component) for component in row[:3]] for row in q_coords],
        "distances": [float(value) for value in distances],
        "frequencies_cm-1": [[float(value) for value in row] for row in frequencies],
        "projections": projections,
        "ticks": {
            "positions": [float(value) for value in ticks.get("positions", [])],
            "labels": [str(value) for value in ticks.get("labels", [])],
        },
        "generated_at": timestamp(),
    }


def build_projected_dos_payload(
    *,
    source: str,
    layout: Mapping[str, Any],
    colors: Mapping[str, Mapping[str, str]],
    frequencies: Sequence[Sequence[float]],
    eigenvectors: Sequence[Sequence[Sequence[Sequence[complex]]]],
    sigma: float = 2.0,
    points: int = 4000,
) -> dict[str, Any]:
    collected_frequencies = [
        float(value)
        for row in frequencies
        for value in row
    ]
    if collected_frequencies:
        margin = 4.0 * max(float(sigma), 0.0)
        f_min = min(0.0, min(collected_frequencies) - margin)
        f_max = max(collected_frequencies) + margin
        if f_max <= f_min:
            f_max = f_min + 1.0
    else:
        f_min = 0.0
        f_max = 400.0
    grid = linspace(f_min, f_max, points)
    total = [0.0 for _ in grid]
    elements = [str(element) for element in layout.get("elements") or []]
    indices_by_element = layout.get("indices") if isinstance(layout.get("indices"), Mapping) else {}
    projections: dict[str, dict[str, list[float]]] = {
        element: {"XY": [0.0 for _ in grid], "Z": [0.0 for _ in grid]} for element in elements
    }
    mode_samples: list[dict[str, Any]] = []
    for q_index, q_modes in enumerate(eigenvectors):
        for mode_index, mode in enumerate(q_modes):
            try:
                frequency = float(frequencies[q_index][mode_index])
            except Exception:
                continue
            broadening = gaussian_values(grid, frequency, sigma)
            total_weight = 0.0
            weights_by_element: dict[str, tuple[float, float]] = {}
            for element in elements:
                atom_indices = [int(value) for value in indices_by_element.get(element, [])]
                xy_weight = 0.0
                z_weight = 0.0
                for atom_index in atom_indices:
                    if atom_index >= len(mode):
                        continue
                    vec = mode[atom_index]
                    xy_weight += abs(vec[0]) ** 2 + abs(vec[1]) ** 2
                    z_weight += abs(vec[2]) ** 2
                weights_by_element[element] = (float(xy_weight), float(z_weight))
                total_weight += xy_weight + z_weight
            mode_samples.append(
                {
                    "frequency_cm-1": float(frequency),
                    "total": float(total_weight),
                    "projections": {
                        element: {"XY": weights[0], "Z": weights[1]}
                        for element, weights in weights_by_element.items()
                    },
                }
            )
            for index, value in enumerate(broadening):
                total[index] += total_weight * value
                for element, (xy_weight, z_weight) in weights_by_element.items():
                    projections[element]["XY"][index] += xy_weight * value
                    projections[element]["Z"][index] += z_weight * value
    return {
        "source": source,
        "layout": layout,
        "colors": colors,
        "sigma_cm-1": float(sigma),
        "frequency_range_cm-1": {"min": float(f_min), "max": float(f_max)},
        "frequency_grid_cm-1": [float(value) for value in grid],
        "total_dos": [float(value) for value in total],
        "projections": projections,
        "mode_samples": mode_samples,
        "generated_at": timestamp(),
    }


def linspace(start: float, stop: float, count: int) -> list[float]:
    count = max(2, int(count))
    step = (float(stop) - float(start)) / (count - 1)
    return [float(start) + index * step for index in range(count)]


def gaussian_values(grid: Sequence[float], center: float, sigma: float) -> list[float]:
    sigma = max(float(sigma), 1.0e-9)
    prefactor = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return [prefactor * math.exp(-0.5 * ((float(value) - center) / sigma) ** 2) for value in grid]


def write_projected_band_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = [str(element) for element in (payload.get("layout") or {}).get("elements") or []]
    frequencies = payload.get("frequencies_cm-1") or []
    distances = payload.get("distances") or []
    projections = payload.get("projections") if isinstance(payload.get("projections"), Mapping) else {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = ["q_index", "distance", "mode_index", "frequency_cm-1"]
        for element in elements:
            headers.extend([f"{element}_XY", f"{element}_Z"])
        writer.writerow(headers)
        for q_index, row in enumerate(frequencies):
            for mode_index, frequency in enumerate(row):
                values: list[Any] = [q_index, distances[q_index] if q_index < len(distances) else "", mode_index, frequency]
                for element in elements:
                    element_data = projections.get(element) if isinstance(projections.get(element), Mapping) else {}
                    xy_rows = element_data.get("XY") or []
                    z_rows = element_data.get("Z") or []
                    values.append(xy_rows[q_index][mode_index] if q_index < len(xy_rows) and mode_index < len(xy_rows[q_index]) else "")
                    values.append(z_rows[q_index][mode_index] if q_index < len(z_rows) and mode_index < len(z_rows[q_index]) else "")
                writer.writerow(values)


def write_projected_dos_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = [str(element) for element in (payload.get("layout") or {}).get("elements") or []]
    grid = payload.get("frequency_grid_cm-1") or []
    total = payload.get("total_dos") or []
    projections = payload.get("projections") if isinstance(payload.get("projections"), Mapping) else {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = ["frequency_cm-1", "total_dos"]
        for element in elements:
            headers.extend([f"{element}_XY", f"{element}_Z"])
        writer.writerow(headers)
        for index, frequency in enumerate(grid):
            values: list[Any] = [frequency, total[index] if index < len(total) else ""]
            for element in elements:
                element_data = projections.get(element) if isinstance(projections.get(element), Mapping) else {}
                xy = element_data.get("XY") or []
                z = element_data.get("Z") or []
                values.append(xy[index] if index < len(xy) else "")
                values.append(z[index] if index < len(z) else "")
            writer.writerow(values)


def rounded_phonon_frequency_ceiling(frequency: float, *, step: float = 100.0, fallback: float = 400.0) -> float:
    try:
        value = float(frequency)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(value) or value <= 0:
        return float(fallback)
    step = max(float(step), 1.0)
    return float(math.ceil(value / step) * step)


def write_phonon_projected_combined_plot(
    path: Path,
    *,
    band_payload: Mapping[str, Any],
    dos_payload: Mapping[str, Any],
    y_min: float | None = None,
    y_max: float | None = None,
    fatness: float = 3.0,
    dpi: int = 300,
    figsize: Sequence[float] | None = None,
    legend_columns: int | None = None,
    ribbon_alpha: float = 0.9,
    skeleton_width: float = 1.2,
    dos_line_width: float = 3.2,
    total_dos_alpha: float = 1.0,
    border_width: float = 2.2,
    label_font_size: float = 18.0,
    tick_font_size: float = 14.0,
    legend_font_size: float = 13.0,
    legend_marker_size: float = 8.0,
    band_width_ratio: float = 3.8,
    dos_width_ratio: float = 1.05,
    panel_wspace: float = 0.04,
    subplot_left: float = 0.08,
    subplot_right: float = 0.985,
    subplot_bottom: float = 0.12,
    subplot_top: float = 0.89,
    vertical_line_color: str = "#999999",
    vertical_line_width: float = 0.7,
    vertical_line_alpha: float = 0.45,
    show_vertical_lines: bool = True,
    skeleton_color: str = "black",
    show_skeleton: bool = True,
    total_dos_color: str = "#E0E0E0",
    show_total_dos: bool = True,
    dos_xmax: float | None = None,
    show_legend: bool = True,
    legend_anchor: Sequence[float] = (0.5, 0.945),
    phdos_label: str = "PhDOS",
    phdos_label_x: float = 0.78,
    phdos_label_y: float = 0.085,
    phdos_label_font_size: float = 13.0,
    transparent: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    elements = [str(element) for element in (band_payload.get("layout") or {}).get("elements") or []]
    if not elements:
        elements = [str(element) for element in (dos_payload.get("layout") or {}).get("elements") or []]
    colors = band_payload.get("colors") if isinstance(band_payload.get("colors"), Mapping) else {}
    distances = [float(value) for value in band_payload.get("distances") or []]
    frequencies = band_payload.get("frequencies_cm-1") or []
    band_proj = band_payload.get("projections") if isinstance(band_payload.get("projections"), Mapping) else {}
    grid = [float(value) for value in dos_payload.get("frequency_grid_cm-1") or []]
    total_dos = [float(value) for value in dos_payload.get("total_dos") or []]
    dos_proj = dos_payload.get("projections") if isinstance(dos_payload.get("projections"), Mapping) else {}

    fig_width = 7.8 + 1.35 * max(1, len(elements))
    fig_size = tuple(float(value) for value in (figsize or (fig_width, 6.6)))
    fig, axes = plt.subplots(
        1,
        1 + max(1, len(elements)),
        figsize=fig_size,
        dpi=int(dpi),
        gridspec_kw={
            "width_ratios": [float(band_width_ratio), *([float(dos_width_ratio)] * max(1, len(elements)))],
            "wspace": float(panel_wspace),
        },
        sharey=True,
    )
    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = [axes]
    ax_band = axes_list[0]

    fatness = float(fatness)
    ribbon_alpha = max(0.0, min(1.0, float(ribbon_alpha)))
    skeleton_width = max(0.0, float(skeleton_width))
    dos_line_width = max(0.0, float(dos_line_width))
    total_dos_alpha = max(0.0, min(1.0, float(total_dos_alpha)))
    border_width = max(0.0, float(border_width))
    label_font_size = max(1.0, float(label_font_size))
    tick_font_size = max(1.0, float(tick_font_size))
    legend_font_size = max(1.0, float(legend_font_size))
    legend_marker_size = max(1.0, float(legend_marker_size))
    vertical_line_width = max(0.0, float(vertical_line_width))
    vertical_line_alpha = max(0.0, min(1.0, float(vertical_line_alpha)))
    phdos_label_font_size = max(1.0, float(phdos_label_font_size))
    for mode_index in range(max((len(row) for row in frequencies), default=0)):
        y_path = [float(row[mode_index]) for row in frequencies if mode_index < len(row)]
        x_path = distances[: len(y_path)]
        if not x_path or not y_path:
            continue
        for element in elements:
            element_colors = colors.get(element, {}) if isinstance(colors.get(element), Mapping) else {}
            element_proj = band_proj.get(element, {}) if isinstance(band_proj.get(element), Mapping) else {}
            for direction in ("XY", "Z"):
                rows = element_proj.get(direction) or []
                weights = [
                    float(rows[q_index][mode_index])
                    for q_index in range(min(len(rows), len(y_path)))
                    if mode_index < len(rows[q_index])
                ]
                if len(weights) != len(y_path):
                    continue
                color = str(element_colors.get(direction) or "#666666")
                lower = [y_path[index] - weights[index] * fatness for index in range(len(y_path))]
                upper = [y_path[index] + weights[index] * fatness for index in range(len(y_path))]
                ax_band.fill_between(x_path, lower, upper, color=color, alpha=ribbon_alpha, linewidth=0, zorder=3)
        if show_skeleton and skeleton_width > 0:
            ax_band.plot(x_path, y_path, color=str(skeleton_color), linewidth=skeleton_width, zorder=4)

    ticks = band_payload.get("ticks") if isinstance(band_payload.get("ticks"), Mapping) else {}
    tick_positions = [float(value) for value in ticks.get("positions", [])]
    tick_labels = [str(value) for value in ticks.get("labels", [])]
    if tick_positions and tick_labels and len(tick_positions) == len(tick_labels):
        ax_band.set_xticks(tick_positions)
        ax_band.set_xticklabels(tick_labels, fontweight="bold")
        if show_vertical_lines:
            for value in tick_positions:
                ax_band.axvline(
                    float(value),
                    color=str(vertical_line_color),
                    linewidth=vertical_line_width,
                    alpha=vertical_line_alpha,
                    zorder=1,
                )
    if distances:
        ax_band.set_xlim(min(distances), max(distances) if max(distances) > min(distances) else min(distances) + 1.0)
    ax_band.set_ylabel(r"Frequency (cm$^{-1}$)", fontsize=label_font_size, fontweight="bold")
    ax_band.set_xlabel("")
    ax_band.grid(False)

    all_freqs = [float(value) for row in frequencies for value in row]
    if not all_freqs:
        for sample in dos_payload.get("mode_samples") or []:
            if isinstance(sample, Mapping) and sample.get("frequency_cm-1") is not None:
                all_freqs.append(float(sample.get("frequency_cm-1")))
    if all_freqs or y_min is not None or y_max is not None:
        lower = 0.0 if y_min is None else float(y_min)
        upper = rounded_phonon_frequency_ceiling(max(all_freqs), fallback=400.0) if y_max is None else float(y_max)
        ax_band.set_ylim(lower, upper)
    for spine in ax_band.spines.values():
        spine.set_linewidth(border_width)
    ax_band.tick_params(axis="both", which="major", width=border_width, length=6, labelsize=tick_font_size)
    for tick_label in ax_band.get_yticklabels():
        tick_label.set_fontweight("bold")

    for panel_index, element in enumerate(elements, start=1):
        ax = axes_list[panel_index]
        if show_total_dos and grid and total_dos:
            ax.fill_betweenx(grid, total_dos, color=str(total_dos_color), alpha=total_dos_alpha, linewidth=0)
        element_colors = colors.get(element, {}) if isinstance(colors.get(element), Mapping) else {}
        element_proj = dos_proj.get(element, {}) if isinstance(dos_proj.get(element), Mapping) else {}
        for direction in ("XY", "Z"):
            values = [float(value) for value in element_proj.get(direction) or []]
            if len(values) == len(grid):
                ax.plot(values, grid, color=str(element_colors.get(direction) or "#444444"), linewidth=dos_line_width)
        if dos_xmax is None:
            ax.set_xlim(left=0.0)
        else:
            ax.set_xlim(0.0, float(dos_xmax))
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", width=border_width, length=5, labelleft=False)
        ax.set_xlabel("")
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
    if len(axes_list) > 1 and phdos_label:
        fig.text(
            float(phdos_label_x),
            float(phdos_label_y),
            str(phdos_label),
            ha="center",
            fontweight="bold",
            fontsize=phdos_label_font_size,
        )

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for element in elements:
        element_colors = colors.get(element, {}) if isinstance(colors.get(element), Mapping) else {}
        for direction in ("XY", "Z"):
            color = str(element_colors.get(direction) or "#444444")
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=legend_marker_size,
                )
            )
            legend_labels.append(f"{element} {direction}")
    if legend_handles and show_legend:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=int(legend_columns or min(len(legend_handles), 6)),
            prop={"weight": "bold", "size": legend_font_size},
            frameon=False,
            bbox_to_anchor=tuple(float(value) for value in legend_anchor[:2]),
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.subplots_adjust(
        left=float(subplot_left),
        right=float(subplot_right),
        bottom=float(subplot_bottom),
        top=float(subplot_top),
        wspace=float(panel_wspace),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), transparent=bool(transparent))
    plt.close(fig)


def write_phonon_projected_replot_script(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
"""Replot projected phonon band + PhDOS from saved JSON data.

Edit the USER CONFIGURATION block, then run:

    python replot_phonon_projected.py
"""
import json
import math
import sys
from pathlib import Path

ROOT_CANDIDATES = [
    Path(__PROJECT_ROOT__),
    Path(__file__).resolve().parents[2],
    Path.cwd(),
]
for candidate in ROOT_CANDIDATES:
    if (candidate / "workflow_runner.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from workflow_runner import write_phonon_projected_combined_plot


# ================= USER CONFIGURATION =================

# Input/output files. Paths are relative to this script unless absolute.
BAND_DATA_FILE = "phonon_projected_band_data.json"
DOS_DATA_FILE = "phonon_projected_dos_data.json"
OUTPUT_FILE = "phonon_projected_band_dos.png"

# Figure and save settings.
DPI = 300
FIGSIZE = (10.5, 6.6)       # inches, or None for automatic size
TRANSPARENT_BACKGROUND = False

# Frequency axis limits in cm^-1. Y_MAX=None auto-rounds the max frequency upward to the nearest 100.
Y_MIN = 0.0
Y_MAX = None

# Recompute the projected PhDOS with a different Gaussian sigma.
# Set DOS_SIGMA_CM1 = None to keep the saved DOS curves as-is.
DOS_SIGMA_CM1 = None

# Fat-band ribbon styling.
RIBBON_SCALE = 3.0          # bigger = thicker projected ribbons
RIBBON_ALPHA = 0.90         # 0 transparent, 1 opaque
SHOW_SKELETON = True
SKELETON_COLOR = "black"
SKELETON_WIDTH = 1.2

# PhDOS styling.
DOS_LINE_WIDTH = 3.2
SHOW_TOTAL_DOS = True
TOTAL_DOS_COLOR = "#E0E0E0"
TOTAL_DOS_ALPHA = 1.0
DOS_XMAX = None             # e.g. 0.25 to force all PhDOS panels to same x max

# Axes, ticks, and panel layout.
BORDER_WIDTH = 2.2
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
BAND_PANEL_WIDTH = 3.8
DOS_PANEL_WIDTH = 1.05
PANEL_WSPACE = 0.04
SUBPLOT_LEFT = 0.08
SUBPLOT_RIGHT = 0.985
SUBPLOT_BOTTOM = 0.12
SUBPLOT_TOP = 0.89

# High-symmetry guide lines.
SHOW_VERTICAL_LINES = True
VERTICAL_LINE_COLOR = "#999999"
VERTICAL_LINE_WIDTH = 0.7
VERTICAL_LINE_ALPHA = 0.45

# Shared figure legend. The workflow keeps filled-circle markers by default.
SHOW_LEGEND = True
LEGEND_COLUMNS = None       # None = automatic
LEGEND_FONT_SIZE = 13
LEGEND_MARKER_SIZE = 8
LEGEND_ANCHOR = (0.5, 0.945)

# Bottom label below PhDOS panels.
PHDOS_LABEL = "PhDOS"
PHDOS_LABEL_X = 0.78
PHDOS_LABEL_Y = 0.085
PHDOS_LABEL_FONT_SIZE = 13

# Optional manual colors. Leave empty to use saved/default colors.
# Example:
# COLORS = {
#     "Mg": {"XY": "#8B0000", "Z": "#FF69B4"},
#     "B":  {"XY": "#00008B", "Z": "#87CEFA"},
# }
COLORS = {}

# Optional external colors JSON. Set to "" to ignore.
# Format: {"Mg": {"XY": "#8B0000", "Z": "#FF69B4"}, ...}
COLORS_JSON_FILE = ""

# ================= END USER CONFIGURATION =================


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def gaussian(grid, center, sigma):
    sigma = max(float(sigma), 1.0e-9)
    prefactor = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return [prefactor * math.exp(-0.5 * ((float(value) - center) / sigma) ** 2) for value in grid]


def recompute_dos_with_sigma(dos, sigma):
    samples = dos.get("mode_samples") or []
    grid = dos.get("frequency_grid_cm-1") or []
    elements = (dos.get("layout") or {}).get("elements") or []
    total = [0.0 for _ in grid]
    projections = {element: {"XY": [0.0 for _ in grid], "Z": [0.0 for _ in grid]} for element in elements}
    for sample in samples:
        freq = float(sample.get("frequency_cm-1") or 0.0)
        broadening = gaussian(grid, freq, sigma)
        total_weight = float(sample.get("total") or 0.0)
        sample_proj = sample.get("projections") or {}
        for index, value in enumerate(broadening):
            total[index] += total_weight * value
            for element in elements:
                element_proj = sample_proj.get(element) or {}
                projections[element]["XY"][index] += float(element_proj.get("XY") or 0.0) * value
                projections[element]["Z"][index] += float(element_proj.get("Z") or 0.0) * value
    dos["sigma_cm-1"] = float(sigma)
    dos["total_dos"] = total
    dos["projections"] = projections


def main() -> int:
    output_path = resolve_path(OUTPUT_FILE)
    band = json.loads(resolve_path(BAND_DATA_FILE).read_text(encoding="utf-8"))
    dos = json.loads(resolve_path(DOS_DATA_FILE).read_text(encoding="utf-8"))
    colors = dict(COLORS)
    if COLORS_JSON_FILE:
        colors.update(json.loads(resolve_path(COLORS_JSON_FILE).read_text(encoding="utf-8")))
    if colors:
        band["colors"] = colors
        dos["colors"] = colors
    if DOS_SIGMA_CM1 is not None:
        recompute_dos_with_sigma(dos, DOS_SIGMA_CM1)
    write_phonon_projected_combined_plot(
        output_path,
        band_payload=band,
        dos_payload=dos,
        y_min=Y_MIN,
        y_max=Y_MAX,
        fatness=RIBBON_SCALE,
        dpi=DPI,
        figsize=FIGSIZE,
        legend_columns=LEGEND_COLUMNS,
        ribbon_alpha=RIBBON_ALPHA,
        skeleton_width=SKELETON_WIDTH,
        dos_line_width=DOS_LINE_WIDTH,
        total_dos_alpha=TOTAL_DOS_ALPHA,
        border_width=BORDER_WIDTH,
        label_font_size=LABEL_FONT_SIZE,
        tick_font_size=TICK_FONT_SIZE,
        legend_font_size=LEGEND_FONT_SIZE,
        legend_marker_size=LEGEND_MARKER_SIZE,
        band_width_ratio=BAND_PANEL_WIDTH,
        dos_width_ratio=DOS_PANEL_WIDTH,
        panel_wspace=PANEL_WSPACE,
        subplot_left=SUBPLOT_LEFT,
        subplot_right=SUBPLOT_RIGHT,
        subplot_bottom=SUBPLOT_BOTTOM,
        subplot_top=SUBPLOT_TOP,
        vertical_line_color=VERTICAL_LINE_COLOR,
        vertical_line_width=VERTICAL_LINE_WIDTH,
        vertical_line_alpha=VERTICAL_LINE_ALPHA,
        show_vertical_lines=SHOW_VERTICAL_LINES,
        skeleton_color=SKELETON_COLOR,
        show_skeleton=SHOW_SKELETON,
        total_dos_color=TOTAL_DOS_COLOR,
        show_total_dos=SHOW_TOTAL_DOS,
        dos_xmax=DOS_XMAX,
        show_legend=SHOW_LEGEND,
        legend_anchor=LEGEND_ANCHOR,
        phdos_label=PHDOS_LABEL,
        phdos_label_x=PHDOS_LABEL_X,
        phdos_label_y=PHDOS_LABEL_Y,
        phdos_label_font_size=PHDOS_LABEL_FONT_SIZE,
        transparent=TRANSPARENT_BACKGROUND,
    )
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.replace("__PROJECT_ROOT__", json.dumps(str(PROJECT_ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def run_fatband_csv_extraction(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_fatband_postprocess_task(
        manifest,
        state,
        state_path,
        task_id="fatband_csv_extraction",
        stage="fatband_csv",
    )


def run_fatband_plotting(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    run_fatband_postprocess_task(
        manifest,
        state,
        state_path,
        task_id="fatband_plotting",
        stage="fatband_plot",
    )


def run_fatband_postprocess_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
    stage: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return

    status = str(task.get("status") or "").lower()
    if status in {"completed", "continued"}:
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")
    child = children[0]

    task_params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    source_task_id = str(task_params.get("source_task") or "bands_workflow")
    source_task = workflow_task_state(state, source_task_id)
    if source_task is None:
        raise RuntimeError(f"{task_id} requires source task '{source_task_id}'.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or BANDS_DIR)
    child_log_path = workflow_root / str(child.get("log") or f"logs/{task_id}.log")

    task["status"] = "running"
    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id, "child": child.get("id"), "stage": stage})
    write_state(state_path, state)

    child_workdir.mkdir(parents=True, exist_ok=True)

    try:
        if stage == "fatband_csv":
            csv_outputs = write_fatband_projection_csv_outputs(
                workflow_root,
                source_task,
                fermi_ev=optional_float_or_none((state.get("results") or {}).get("band_structure_fermi_ev")),
                log_path=child_log_path,
            )
            child["result"] = {
                "stage": stage,
                **csv_outputs,
            }
            task["result"] = {
                "source_task": source_task_id,
                **csv_outputs,
            }
            results_state = state.setdefault("results", {})
            results_state["fatband_projection_csv_dir"] = csv_outputs.get("csv_dir")
            results_state["fatband_projection_csv"] = csv_outputs.get("all_states_csv")
            results_state["fatband_csv_metadata"] = csv_outputs.get("metadata")
            results_state["fatband_fermi_ev"] = csv_outputs.get("fermi_ev")
            results_state["fatband_state_range"] = csv_outputs.get("state_range")
        elif stage == "fatband_plot":
            source_params = source_task.get("params") if isinstance(source_task.get("params"), Mapping) else {}
            band_options = source_params.get("band_options") if isinstance(source_params.get("band_options"), Mapping) else {}
            color_mode = normalize_fatband_color_mode(
                task_params.get("fatband_color_mode") or band_options.get("fatband_color_mode")
            )
            filband = str(task_params.get("filband") or band_options.get("filband") or "bands.out")
            kpoints_crystal_b = task_params.get("kpoints_crystal_b")
            if not isinstance(kpoints_crystal_b, Sequence) or isinstance(kpoints_crystal_b, (str, bytes)):
                kpoints_crystal_b = band_options.get("kpoints_crystal_b") or []
            plot_outputs = write_fatband_plot_outputs(
                workflow_root,
                source_task,
                filband=filband,
                kpoints_crystal_b=[str(line).rstrip() for line in kpoints_crystal_b if str(line).strip()],
                color_mode=color_mode,
                log_path=child_log_path,
            )
            child["result"] = {
                "stage": stage,
                **plot_outputs,
            }
            task["result"] = {
                "source_task": source_task_id,
                **plot_outputs,
            }
            results_state = state.setdefault("results", {})
            results_state["fatband"] = {"status": "completed", "color_mode": color_mode, **plot_outputs}
            results_state["fatband_plots"] = plot_outputs.get("plots")
            results_state["fatband_color_mode"] = color_mode
        else:
            raise RuntimeError(f"Unsupported fat-band stage: {stage}")

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        state["updated_at"] = timestamp()
        add_event(state, "task.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception as exc:
        child["status"] = "continued"
        child["completed_at"] = timestamp()
        child["result"] = {
            "stage": stage,
            "continued_on_failure": True,
            "error": str(exc),
        }
        task["status"] = "continued"
        task["completed"] = 1
        task["planned_total"] = len(children) if children else 1
        task["running_child"] = None
        task["result"] = {
            "source_task": source_task_id,
            "continued_on_failure": True,
            "error": str(exc),
        }
        results_state = state.setdefault("results", {})
        if stage == "fatband_csv":
            results_state["fatband_projection_error"] = str(exc)
        elif stage == "fatband_plot":
            results_state["fatband_error"] = str(exc)
        add_event(
            state,
            "task.warning",
            {
                "task": task_id,
                "child": child.get("id"),
                "warning": "optional_stage_failed",
                "message": str(exc),
            },
        )
        append_log(workflow_root / "logs" / "workflow.log", f"{task_id} continued after failure: {exc}\n")
        state["updated_at"] = timestamp()
        write_state(state_path, state)


def run_dos_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    dos_options = params.get("dos_options") if isinstance(params.get("dos_options"), Mapping) else {}
    scf_kpoint_spacing = positive_float_or_none(dos_options.get("scf_kpoint_spacing"))
    if scf_kpoint_spacing is None:
        scf_kpoint_spacing = 0.30
    nscf_kpoint_spacing = positive_float_or_none(dos_options.get("nscf_kpoint_spacing"))
    if nscf_kpoint_spacing is None:
        nscf_kpoint_spacing = 0.10
    use_converged_kmesh = bool(dos_options.get("use_converged_kmesh", True))
    # PDOS SCF/NSCF must use tetrahedra occupations (no smearing), so always run
    # a dedicated DOS SCF stage and do not reuse band SCF artifacts.
    prefix = dos_workflow_prefix(manifest, dos_options)
    outdir = "./out/"
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)
    converged_kmesh: list[int] | None = None
    if uses_converged and use_converged_kmesh and isinstance(results, Mapping):
        mesh = results.get("kmesh")
        if isinstance(mesh, list) and len(mesh) >= 3:
            try:
                converged_kmesh = [int(mesh[0]), int(mesh[1]), int(mesh[2])]
            except Exception:
                converged_kmesh = None

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")

            run_dos_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                scf_kpoint_spacing=float(scf_kpoint_spacing),
                nscf_kpoint_spacing=float(nscf_kpoint_spacing),
                converged_kmesh=converged_kmesh,
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                state_path=state_path,
            )

        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = {
            "prefix": prefix,
            "outdir": outdir,
            "scf_kpoint_spacing": float(scf_kpoint_spacing),
            "nscf_kpoint_spacing": float(nscf_kpoint_spacing),
            "used_converged_kmesh": bool(converged_kmesh),
            "kmesh": converged_kmesh,
            "reused_band_scf": False,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        pdos_child = (
            band_child_by_stage(task, stage="pdos_plotting")
            or band_child_by_stage(task, stage="pdos_postprocess")
        )
        pdos_result = pdos_child.get("result") if isinstance(pdos_child, Mapping) else None
        if isinstance(pdos_result, Mapping):
            task["result"]["pdos_postprocess"] = {
                "metadata": pdos_result.get("metadata"),
                "csv_files": pdos_result.get("csv_files"),
                "plot_files": pdos_result.get("plot_files"),
                "dos_summary_plot": pdos_result.get("dos_summary_plot"),
                "combined_band_dos_plot": pdos_result.get("combined_band_dos_plot"),
                "combined_plot_mode": pdos_result.get("combined_plot_mode"),
                "replot_script": pdos_result.get("replot_script"),
                "fermi_ev": pdos_result.get("fermi_ev"),
            }
        sumpdos_child = band_child_by_stage(task, stage="pdos_sumpdos")
        sumpdos_result = sumpdos_child.get("result") if isinstance(sumpdos_child, Mapping) else None
        if isinstance(sumpdos_result, Mapping):
            task["result"]["pdos_sumpdos"] = {
                "data": sumpdos_result.get("data"),
                "fermi_ev": sumpdos_result.get("fermi_ev"),
                "atom_types": sumpdos_result.get("atom_types"),
                "summed_dat_files": sumpdos_result.get("summed_dat_files"),
            }

        results_state = state.setdefault("results", {})
        results_state["dos"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_wannier90_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    band_options = params.get("band_options") if isinstance(params.get("band_options"), Mapping) else {}
    wannier_options = params.get("wannier_options") if isinstance(params.get("wannier_options"), Mapping) else {}
    merged_options = {**dict(band_options), **dict(wannier_options)}
    kpoint_spacing = positive_float_or_none(merged_options.get("kpoint_spacing")) or 0.10
    wannier_scf_kmesh = normalize_wannier90_kmesh(
        merged_options.get("scf_kmesh")
        or merged_options.get("kmesh")
        or merged_options.get("mp_grid")
    )
    kpoints_crystal_b = [
        str(line).rstrip()
        for line in (merged_options.get("kpoints_crystal_b") or [])
        if str(line).strip()
    ]
    if len(kpoints_crystal_b) < 2:
        raise RuntimeError(f"{task_id} is missing K_POINTS crystal_b data.")

    filband = str(merged_options.get("filband") or "bands.out")
    write_unk = truthy_runtime_option(merged_options.get("write_unk", False))
    win_num_iter = optional_positive_int(
        merged_options.get("num_iter") or merged_options.get("num_inter")
    ) or 10000
    win_dis_num_iter = optional_nonnegative_int(merged_options.get("dis_num_iter"))
    if win_dis_num_iter is None:
        win_dis_num_iter = 0
    prefix = wannier90_workflow_prefix(manifest, merged_options)
    filproj = str(merged_options.get("filproj") or f"{prefix}.proj").strip() or f"{prefix}.proj"
    outdir = "./out/"
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")

            run_wannier90_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                kpoint_spacing=kpoint_spacing,
                scf_kmesh=wannier_scf_kmesh,
                kpoints_crystal_b=kpoints_crystal_b,
                write_unk=write_unk,
                win_num_iter=win_num_iter,
                win_dis_num_iter=win_dis_num_iter,
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                filband=filband,
                filproj=filproj,
                state_path=state_path,
            )

        derived_nbnd = resolve_wannier90_nbnd(
            workflow_root,
            task,
            spin_orbit=spin_orbit,
        )
        number_of_electrons = resolve_wannier90_number_of_electrons(workflow_root, task)
        natomwfc = resolve_wannier90_natomwfc(workflow_root, task)

        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = {
            "prefix": prefix,
            "outdir": outdir,
            "kpoint_spacing": kpoint_spacing,
            "scf_kmesh": wannier_scf_kmesh,
            "nbnd": derived_nbnd,
            "wannier_nbnd": derived_nbnd,
            "number_of_electrons": number_of_electrons,
            "natomwfc": natomwfc,
            "kpoints_crystal_b": kpoints_crystal_b,
            "filband": filband,
            "filproj": filproj,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        nscf_child = band_child_by_stage(task, stage="nscf")
        nscf_result = nscf_child.get("result") if isinstance(nscf_child, Mapping) else None
        if isinstance(nscf_result, Mapping):
            task["result"]["nscf_kmesh"] = nscf_result.get("kmesh")
            task["result"]["nscf_kpoints"] = nscf_result.get("kpoints")

        results_state = state.setdefault("results", {})
        results_state["wannier90"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_wannier90_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    kpoint_spacing: float,
    scf_kmesh: Sequence[int] | None,
    kpoints_crystal_b: Sequence[str],
    write_unk: bool,
    win_num_iter: int,
    win_dis_num_iter: int,
    ecutwfc: Any,
    spin_orbit: bool,
    filband: str,
    filproj: str,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or WANNIER90_DIR)
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "wannier90_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        nscf_kmesh: list[int] | None = None
        nscf_kpoint_count: int | None = None
        filproj_relative: str | None = None
        effective_nbnd: int | None = None
        number_of_electrons: float | None = None
        natomwfc: int | None = None
        if stage in {"wannier90_preprocess", "wannier_preprocess", "win_pp"}:
            command = build_wannier90_preprocess_command("wannier_control.win")
            run_wannier90_command_child(
                workflow_root=workflow_root,
                task=task,
                child=child,
                state=state,
                state_path=state_path,
                task_id=task_id,
                stage=stage,
                command=command,
                child_workdir=child_workdir,
                child_log_path=child_log_path,
                result_updates={
                    "wannier90_preprocess_output": child.get("output"),
                    "wannier90_nnkp": child.get("output"),
                },
            )
            return
        if stage in {"pw2wan_run", "pw2wannier90", "pw2wannier90_run"}:
            command = build_pw2wannier90_run_command("pw2wan.in", "pw2wan.out")
            run_wannier90_command_child(
                workflow_root=workflow_root,
                task=task,
                child=child,
                state=state,
                state_path=state_path,
                task_id=task_id,
                stage=stage,
                command=command,
                child_workdir=child_workdir,
                child_log_path=child_log_path,
                result_updates={
                    "wannier90_pw2wan_output": child.get("output"),
                    "wannier90_pw2wannier90_output": child.get("output"),
                },
            )
            return
        if stage in {"wannier90_run", "wannier_run"}:
            command = build_wannier90_run_command("wannier_control")
            run_wannier90_command_child(
                workflow_root=workflow_root,
                task=task,
                child=child,
                state=state,
                state_path=state_path,
                task_id=task_id,
                stage=stage,
                command=command,
                child_workdir=child_workdir,
                child_log_path=child_log_path,
                result_updates={
                    "wannier90_run_output": child.get("output"),
                    "wannier90_wout": child.get("output"),
                },
            )
            return
        if stage in {"wannier_control", "win_input", "wannier90_input"}:
            win_outputs = write_wannier90_win_input(
                output_path=output_path,
                workflow_root=workflow_root,
                task=task,
                base_text=base_text,
                kpoints_crystal_b=kpoints_crystal_b,
                nscf_kmesh=wannier90_nscf_kmesh(task, base_text),
                log_path=child_log_path,
                spin_orbit=spin_orbit,
                num_iter=win_num_iter,
                dis_num_iter=win_dis_num_iter,
            )
            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                **win_outputs,
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["wannier90_win_input"] = child.get("output")
            results["wannier90_win_num_bands"] = win_outputs.get("num_bands")
            results["wannier90_win_num_wann"] = win_outputs.get("num_wann")
            results["wannier90_win_conv_tol"] = win_outputs.get("conv_tol")
            results["wannier90_win_mp_grid"] = win_outputs.get("mp_grid")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage in {"pw2wan_input", "pw2wannier90_input"}:
            pw2wan_outputs = write_wannier90_pw2wan_input(
                input_path=input_path,
                output_path=output_path,
                prefix=prefix,
                outdir=outdir,
                seedname="wannier_control",
                write_unk=write_unk,
                log_path=child_log_path,
            )
            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                **pw2wan_outputs,
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["wannier90_pw2wan_input"] = child.get("output")
            results["wannier90_pw2wan_scdm_entanglement"] = pw2wan_outputs.get("scdm_entanglement")
            results["wannier90_pw2wan_scdm_mu"] = pw2wan_outputs.get("scdm_mu")
            results["wannier90_pw2wan_scdm_sigma"] = pw2wan_outputs.get("scdm_sigma")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage in {"scdm_fitting", "scdm_fit", "scdm"}:
            p_vs_e_path = workflow_root / WANNIER90_DIR / "p_vs_e.dat"
            postprocess_outputs = write_wannier90_projectability_weights(
                input_path=input_path,
                output_path=p_vs_e_path,
                log_path=child_log_path,
            )
            fit_outputs = write_wannier90_projectability_fit(
                input_path=p_vs_e_path,
                output_path=output_path,
                log_path=child_log_path,
            )
            archive_child_files(workflow_root, child)
            archive_path = workflow_root / "inputs_and_outputs" / f"{WANNIER90_DIR}_p_vs_e.dat"
            if p_vs_e_path.is_file():
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p_vs_e_path, archive_path)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                "p_vs_e": str(Path(WANNIER90_DIR) / "p_vs_e.dat"),
                "projectability_rows": postprocess_outputs.get("rows"),
                **fit_outputs,
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["wannier90_projectability_output"] = str(Path(WANNIER90_DIR) / "p_vs_e.dat")
            results["wannier90_projectability_fit_output"] = child.get("output")
            results["wannier90_projectability_fit"] = fit_outputs
            results["wannier90_projectability_fit_model"] = fit_outputs.get("best_model")
            results["wannier90_projectability_mu"] = fit_outputs.get("mu")
            results["wannier90_projectability_sigma"] = fit_outputs.get("sigma")
            results["wannier90_scdm_fitting_output"] = child.get("output")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage in {"projectability_fit", "fit"}:
            fit_outputs = write_wannier90_projectability_fit(
                input_path=input_path,
                output_path=output_path,
                log_path=child_log_path,
            )
            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                **fit_outputs,
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["wannier90_projectability_fit_output"] = child.get("output")
            results["wannier90_projectability_fit"] = fit_outputs
            results["wannier90_projectability_fit_model"] = fit_outputs.get("best_model")
            results["wannier90_projectability_mu"] = fit_outputs.get("mu")
            results["wannier90_projectability_sigma"] = fit_outputs.get("sigma")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage in {"projectability", "weights", "p_vs_e"}:
            postprocess_outputs = write_wannier90_projectability_weights(
                input_path=input_path,
                output_path=output_path,
                log_path=child_log_path,
            )
            archive_child_files(workflow_root, child)
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                "input": str(Path(str(child.get("input") or "")).as_posix()),
                "output": str(Path(str(child.get("output") or "")).as_posix()),
                **postprocess_outputs,
            }
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results[f"{task_id}_{stage}_output"] = child.get("output")
            results["wannier90_projectability_output"] = child.get("output")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        if stage == "scf":
            rendered = render_band_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or WANNIER90_DIR,
                prefix=prefix,
                kpoint_spacing=kpoint_spacing,
                kmesh=scf_kmesh,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
                outdir=outdir,
            )
            if not str(
                parse_namelist_values(rendered, "control").get("etot_conv_thr")
                or ""
            ).strip():
                rendered = set_namelist_values(
                    rendered,
                    "control",
                    {"etot_conv_thr": DEFAULT_QE_ETOT_CONV_THR},
                )
            rendered = remove_namelist_keys(rendered, "system", {"nbnd"})
            if spin_orbit:
                rendered = set_namelist_values(
                    rendered,
                    "system",
                    {"noncolin": ".true.", "lspinorb": ".true."},
                )
            engine = "pw.x"
        elif stage in {"pwbands", "bands"}:
            number_of_electrons = resolve_wannier90_number_of_electrons(
                workflow_root,
                task,
            )
            effective_nbnd = wannier90_nbnd_from_electron_count(
                number_of_electrons,
                spin_orbit=spin_orbit,
            )
            rendered = render_band_calculation_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or WANNIER90_DIR,
                prefix=prefix,
                kpoints_crystal_b=kpoints_crystal_b,
                nbnd=effective_nbnd,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            rendered = set_namelist_values(
                rendered,
                "electrons",
                {
                    "diago_full_acc": ".true.",
                    "diagonalization": "'cg'",
                },
            )
            if spin_orbit:
                rendered = set_namelist_values(
                    rendered,
                    "system",
                    {"noncolin": ".true.", "lspinorb": ".true."},
                )
            engine = "pw.x"
        elif stage == "bandx":
            rendered = render_bandsx_input(
                prefix=prefix,
                outdir=outdir,
                filband=filband,
            )
            engine = "bands.x"
        elif stage == "nscf":
            number_of_electrons = resolve_wannier90_number_of_electrons(
                workflow_root,
                task,
            )
            effective_nbnd = wannier90_nbnd_from_electron_count(
                number_of_electrons,
                spin_orbit=spin_orbit,
            )
            rendered, nscf_kmesh, nscf_kpoint_count = render_wannier90_nscf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or WANNIER90_DIR,
                prefix=prefix,
                kpoint_spacing=kpoint_spacing,
                kmesh=scf_kmesh,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
                outdir=outdir,
            )
            rendered = set_namelist_values(
                rendered,
                "system",
                {"nbnd": str(effective_nbnd), "nosym": ".true."},
            )
            rendered = set_namelist_values(
                rendered,
                "electrons",
                {"diagonalization": "'cg'"},
            )
            if spin_orbit:
                rendered = set_namelist_values(
                    rendered,
                    "system",
                    {"noncolin": ".true.", "lspinorb": ".true."},
                )
            engine = "pw.x"
        elif stage == "projwfc":
            rendered = render_projwfc_input(
                prefix=prefix,
                outdir=outdir,
                lsym=True,
                filproj=filproj,
                deltae="2.0000000000d-01",
                kresolveddos=False,
                lbinary_data=False,
                lwrite_overlaps=False,
                plotboxes=False,
                tdosinboxes=False,
            )
            filproj_relative = str((Path(str(child.get("workdir") or WANNIER90_DIR)) / filproj).as_posix())
            engine = "projwfc.x"
        else:
            raise RuntimeError(f"Unsupported Wannier90 + SCDM workflow child stage: {stage}")

        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")

        command = build_qe_command(engine, input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        output_text = (
            output_path.read_text(encoding="utf-8", errors="ignore")
            if output_path.is_file()
            else ""
        )
        if stage == "scf":
            number_of_electrons = extract_number_of_electrons(output_text)
        elif stage == "projwfc":
            natomwfc = extract_natomwfc(output_text)
            if natomwfc is None:
                raise RuntimeError(
                    f"Could not find a positive 'natomwfc =' value in Wannier90 "
                    f"projwfc output: {output_path}"
                )

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child_result = {
            "stage": stage,
            "engine": engine,
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        if nscf_kmesh is not None:
            child_result["kmesh"] = nscf_kmesh
        if nscf_kpoint_count is not None:
            child_result["kpoints"] = nscf_kpoint_count
        if filproj_relative is not None:
            child_result["filproj"] = filproj_relative
        if effective_nbnd is not None:
            child_result["nbnd"] = effective_nbnd
        if number_of_electrons is not None:
            child_result["number_of_electrons"] = number_of_electrons
        if natomwfc is not None:
            child_result["natomwfc"] = natomwfc
        child["result"] = child_result
        task["completed"] = count_completed_children(task)
        task["running_child"] = None

        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["wannier90_scf_output"] = child.get("output")
        elif stage in {"pwbands", "bands"}:
            results["wannier90_pw_output"] = child.get("output")
        elif stage == "bandx":
            results["wannier90_bandx_output"] = child.get("output")
        elif stage == "nscf":
            results["wannier90_nscf_output"] = child.get("output")
        elif stage == "projwfc":
            results["wannier90_projwfc_output"] = child.get("output")
            if filproj_relative is not None:
                results["wannier90_filproj"] = filproj_relative
        elif stage in {"projectability", "weights", "p_vs_e"}:
            results["wannier90_projectability_output"] = child.get("output")
        elif stage in {"projectability_fit", "fit"}:
            results["wannier90_projectability_fit_output"] = child.get("output")
        elif stage in {"pw2wan_input", "pw2wannier90_input"}:
            results["wannier90_pw2wan_input"] = child.get("output")
        elif stage in {"wannier_control", "win_input", "wannier90_input"}:
            results["wannier90_win_input"] = child.get("output")

        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
        raise


def write_wannier90_projectability_weights(
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Create p_vs_e.dat from projwfc.x output, matching generate_weights.sh."""

    if not input_path.is_file():
        raise FileNotFoundError(f"projwfc output not found: {input_path}")

    energies: list[float] = []
    projectabilities: list[float] = []
    for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if "==" in line and len(parts) >= 5:
            try:
                energies.append(float(parts[4]))
            except ValueError:
                continue
        if "|psi|^2" in line and len(parts) >= 3:
            try:
                projectabilities.append(float(parts[2]))
            except ValueError:
                continue

    pair_count = min(len(energies), len(projectabilities))
    if pair_count <= 0:
        raise RuntimeError(f"No energy/projectability rows could be parsed from {input_path}")
    if len(energies) != len(projectabilities):
        append_log(
            log_path,
            "Warning: projwfc energy/projectability counts differ "
            f"({len(energies)} energies, {len(projectabilities)} projectabilities); using {pair_count} pairs.\n",
        )

    rows = sorted(
        (energies[index], projectabilities[index])
        for index in range(pair_count)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{energy:.10g}\t{projectability:.10g}\n" for energy, projectability in rows),
        encoding="utf-8",
    )
    append_log(log_path, f"Saved Wannier90 projectability weights: {output_path}\n")
    return {
        "rows": len(rows),
        "energies": len(energies),
        "projectabilities": len(projectabilities),
    }


def run_wannier90_command_child(
    *,
    workflow_root: Path,
    task: dict[str, Any],
    child: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    task_id: str,
    stage: str,
    command: str,
    child_workdir: Path,
    child_log_path: Path,
    result_updates: Mapping[str, Any],
) -> None:
    """Run a Wannier90 command-only child and update workflow state."""

    append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
    run_shell_command(command, cwd=child_workdir, log_path=child_log_path)
    archive_child_files(workflow_root, child)
    child["status"] = "completed"
    child["completed_at"] = timestamp()
    child["result"] = {
        "stage": stage,
        "engine": str((child.get("params") or {}).get("engine") or ""),
        "input": str(Path(str(child.get("input") or "")).as_posix()),
        "output": str(Path(str(child.get("output") or "")).as_posix()),
        "command": command,
    }
    task["completed"] = count_completed_children(task)
    task["running_child"] = None
    results = state.setdefault("results", {})
    results[f"{task_id}_{stage}_output"] = child.get("output")
    for key, value in result_updates.items():
        results[str(key)] = value
    state["updated_at"] = timestamp()
    add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
    write_state(state_path, state)


def write_wannier90_projectability_fit(
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Fit p_vs_e.dat to ERFC and Gaussian models and store the best mu/sigma."""

    if not input_path.is_file():
        raise FileNotFoundError(f"projectability weights file not found: {input_path}")

    try:
        import numpy as np  # type: ignore
        from scipy.optimize import curve_fit  # type: ignore
        from scipy.special import erfc  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Projectability fitting requires numpy and scipy.") from exc

    data = np.loadtxt(str(input_path), dtype=float)
    if data.ndim == 1:
        if data.size != 2:
            raise RuntimeError(f"Expected two columns in {input_path}")
        data = data.reshape(1, 2)
    if data.shape[1] < 2:
        raise RuntimeError(f"Expected two columns in {input_path}")

    x = data[:, 0]
    y = data[:, 1]
    if x.size < 2:
        raise RuntimeError(f"At least two projectability rows are required for fitting: {input_path}")

    def erfc_model(values: Any, a0: float, sigma: float) -> Any:
        return 0.5 * erfc((values - a0) / sigma)

    def gaussian_model(values: Any, mu: float, sigma: float) -> Any:
        return np.exp(-((values - mu) ** 2) / sigma**2)

    sigma_guess = float((x.max() - x.min()) / 10.0) or 1.0
    a0_guess = float(x[np.argmin(np.abs(y - 0.5))])
    mu_guess = float(x[np.argmax(y)])

    fits: dict[str, Any] = {}
    rmse_erfc = math.inf
    rmse_gaussian = math.inf
    mu_erfc: float | None = None
    sigma_erfc: float | None = None
    mu_gaussian: float | None = None
    sigma_gaussian: float | None = None

    try:
        popt_erfc, _ = curve_fit(
            erfc_model,
            x,
            y,
            p0=[a0_guess, sigma_guess],
            maxfev=10000,
        )
        a0 = float(popt_erfc[0])
        sigma_erfc = float(popt_erfc[1])
        mu_erfc = float(a0 - 3.0 * sigma_erfc)
        yfit_erfc = erfc_model(x, *popt_erfc)
        rmse_erfc = float(np.sqrt(np.mean((y - yfit_erfc) ** 2)))
        fits["erfc"] = {
            "A0": a0,
            "sigma": sigma_erfc,
            "mu": mu_erfc,
            "mui": mu_erfc,
            "rmse": rmse_erfc,
        }
    except Exception as exc:  # noqa: BLE001 - keep fitting robust and report failure
        fits["erfc"] = {"error": str(exc), "rmse": None}

    try:
        popt_gaussian, _ = curve_fit(
            gaussian_model,
            x,
            y,
            p0=[mu_guess, sigma_guess],
            maxfev=10000,
        )
        mu_gaussian = float(popt_gaussian[0])
        sigma_gaussian = float(popt_gaussian[1])
        yfit_gaussian = gaussian_model(x, *popt_gaussian)
        rmse_gaussian = float(np.sqrt(np.mean((y - yfit_gaussian) ** 2)))
        fits["gaussian"] = {
            "mu": mu_gaussian,
            "mui": mu_gaussian,
            "sigma": sigma_gaussian,
            "rmse": rmse_gaussian,
        }
    except Exception as exc:  # noqa: BLE001 - keep fitting robust and report failure
        fits["gaussian"] = {"error": str(exc), "rmse": None}

    if not math.isfinite(rmse_erfc) and not math.isfinite(rmse_gaussian):
        raise RuntimeError(f"Could not fit projectability data from {input_path}")

    if rmse_erfc < rmse_gaussian and mu_erfc is not None and sigma_erfc is not None:
        best_model = "erfc"
        best_mu = mu_erfc
        best_sigma = sigma_erfc
        best_rmse = rmse_erfc
    elif mu_gaussian is not None and sigma_gaussian is not None:
        best_model = "gaussian"
        best_mu = mu_gaussian
        best_sigma = sigma_gaussian
        best_rmse = rmse_gaussian
    else:
        best_model = "erfc"
        best_mu = float(mu_erfc)
        best_sigma = float(sigma_erfc)
        best_rmse = rmse_erfc

    payload = {
        "best_model": best_model,
        "fit_used": best_model,
        "mu": float(best_mu),
        "mui": float(best_mu),
        "sigma": float(best_sigma),
        "rmse": float(best_rmse),
        "rows": int(x.size),
        "fits": fits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    append_log(
        log_path,
        f"Saved Wannier90 projectability fit: {output_path} "
        f"(best={best_model}, mu={best_mu:.6f}, sigma={best_sigma:.6f})\n",
    )
    return payload


def write_wannier90_pw2wan_input(
    *,
    input_path: Path,
    output_path: Path,
    prefix: str,
    outdir: str,
    seedname: str,
    write_unk: bool,
    log_path: Path,
) -> dict[str, Any]:
    """Render pw2wannier90.x input using SCDM fit parameters."""

    if not input_path.is_file():
        raise FileNotFoundError(f"projectability fit file not found: {input_path}")

    fit_payload = load_json_file(input_path)
    fit_used = str(
        fit_payload.get("fit_used")
        or fit_payload.get("best_model")
        or fit_payload.get("model")
        or ""
    ).strip().lower()
    if fit_used not in {"erfc", "gaussian"}:
        raise RuntimeError(f"Unsupported SCDM fit model in {input_path}: {fit_used or '<missing>'}")

    mu_value = fit_payload.get("mu", fit_payload.get("mui"))
    sigma_value = fit_payload.get("sigma")
    mu = optional_float_or_none(mu_value)
    sigma = optional_float_or_none(sigma_value)
    if mu is None:
        raise RuntimeError(f"Missing numeric mu/mui value in {input_path}")
    if sigma is None:
        raise RuntimeError(f"Missing numeric sigma value in {input_path}")

    lines = [
        "&inputpp",
        f"   outdir = {qe_quote(outdir or './out/')}",
        f"   prefix = {qe_quote(prefix)}",
        f"   seedname = {qe_quote(seedname)}",
        "   write_unk  =  .true." if write_unk else "   !write_unk  =  .true.",
        "   scdm_proj = .true.",
        f"   scdm_entanglement = {qe_quote(fit_used)}",
        f"   scdm_mu = {format_qe_float(mu)}",
        f"   scdm_sigma = {format_qe_float(sigma)}",
        "/",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    append_log(
        log_path,
        f"Saved pw2wannier90 input: {output_path} "
        f"(scdm_entanglement={fit_used}, scdm_mu={mu:.6f}, scdm_sigma={sigma:.6f})\n",
    )
    return {
        "scdm_entanglement": fit_used,
        "scdm_mu": float(mu),
        "scdm_sigma": float(sigma),
        "seedname": seedname,
        "write_unk": bool(write_unk),
    }


def write_wannier90_win_input(
    *,
    output_path: Path,
    workflow_root: Path,
    task: Mapping[str, Any],
    base_text: str,
    kpoints_crystal_b: Sequence[str],
    nscf_kmesh: Sequence[int] | None,
    log_path: Path,
    spin_orbit: bool,
    num_iter: int,
    dis_num_iter: int,
) -> dict[str, Any]:
    """Render wannier_control.win for the Wannier90 + SCDM workflow."""

    nbnd = resolve_wannier90_nbnd(
        workflow_root,
        task,
        spin_orbit=spin_orbit,
    )
    num_wann = resolve_wannier90_natomwfc(workflow_root, task)
    conv_tol = resolve_wannier90_conv_tol(workflow_root, task)

    kmesh = [int(value) for value in (nscf_kmesh or [])[:3]]
    if len(kmesh) < 3:
        extracted = extract_automatic_kmesh(base_text)
        if extracted:
            kmesh = [int(value) for value in extracted[:3]]
    if len(kmesh) < 3:
        raise RuntimeError("Could not determine NSCF kmesh for wannier_control.win mp_grid.")

    rendered = render_wannier90_win_input(
        base_text=base_text,
        kpoints_crystal_b=kpoints_crystal_b,
        nbnd=nbnd,
        num_wann=num_wann,
        conv_tol=conv_tol,
        mp_grid=kmesh,
        spin_orbit=spin_orbit,
        num_iter=num_iter,
        dis_num_iter=dis_num_iter,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    append_log(
        log_path,
        f"Saved Wannier90 control input: {output_path} "
        f"(num_bands={nbnd}, num_wann={num_wann}, conv_tol={conv_tol}, "
        f"mp_grid={kmesh[0]} {kmesh[1]} {kmesh[2]})\n",
    )
    return {
        "num_bands": nbnd,
        "num_wann": num_wann,
        "conv_tol": conv_tol,
        "mp_grid": kmesh,
        "kpoints": int(kmesh[0]) * int(kmesh[1]) * int(kmesh[2]),
        "num_iter": int(num_iter),
        "dis_num_iter": int(dis_num_iter),
    }


def extract_number_of_electrons(text: str) -> float | None:
    matches = list(
        re.finditer(
            r"number\s+of\s+electrons\s*=\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
            text or "",
            re.IGNORECASE,
        )
    )
    if not matches:
        return None
    try:
        value = float(matches[-1].group(1).replace("D", "E").replace("d", "E"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def wannier90_nbnd_from_electron_count(
    value: float,
    spin_orbit: bool = False,
) -> int:
    try:
        electron_count = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wannier90 electron count must be a positive finite number.") from exc
    if not math.isfinite(electron_count) or electron_count <= 0:
        raise ValueError("Wannier90 electron count must be a positive finite number.")
    multiplier = 3.0 if spin_orbit else 1.5
    return int(math.ceil(multiplier * electron_count))


def extract_natomwfc(text: str) -> int | None:
    matches = list(re.finditer(r"\bnatomwfc\s*=\s*(\d+)", text or "", re.IGNORECASE))
    if not matches:
        return None
    try:
        value = int(matches[-1].group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_wannier90_number_of_electrons(
    workflow_root: Path,
    task: Mapping[str, Any],
) -> float:
    scf_output = band_child_output_path(workflow_root, task, stage="scf")
    if scf_output is None:
        scf_output = workflow_root / WANNIER90_DIR / "scf.out"
    if not scf_output.is_file():
        raise FileNotFoundError(f"Wannier90 SCF output is missing: {scf_output}")
    value = extract_number_of_electrons(
        scf_output.read_text(encoding="utf-8", errors="ignore")
    )
    if value is None:
        raise RuntimeError(
            f"Could not find a positive 'number of electrons =' value in "
            f"Wannier90 SCF output: {scf_output}"
        )
    return value


def resolve_wannier90_nbnd(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    spin_orbit: bool = False,
) -> int:
    expected_nbnd = wannier90_nbnd_from_electron_count(
        resolve_wannier90_number_of_electrons(workflow_root, task),
        spin_orbit=spin_orbit,
    )
    mismatches: list[tuple[str, int]] = []
    for stage in ("pwbands", "bands", "nscf"):
        child = band_child_by_stage(task, stage=stage)
        if not isinstance(child, Mapping):
            continue
        if str(child.get("status") or "").strip().lower() != "completed":
            continue

        actual_nbnd: int | None = None
        input_path = band_child_input_path(workflow_root, task, stage=stage)
        if input_path is not None and input_path.is_file():
            actual_nbnd = optional_positive_int(
                parse_namelist_values(
                    input_path.read_text(encoding="utf-8", errors="ignore"),
                    "system",
                ).get("nbnd")
            )
        if actual_nbnd is None:
            result = child.get("result") if isinstance(child.get("result"), Mapping) else {}
            actual_nbnd = optional_positive_int(result.get("nbnd"))
        if actual_nbnd is not None and actual_nbnd != expected_nbnd:
            mismatches.append((stage, actual_nbnd))

    if mismatches:
        details = ", ".join(f"{stage}={value}" for stage, value in mismatches)
        raise RuntimeError(
            "Completed Wannier90 child inputs use stale nbnd values "
            f"({details}); expected {expected_nbnd} from the SCF electron count"
            f"{' with spin-orbit coupling' if spin_orbit else ''}. "
            "Reset the listed child stages to queued and rerun the workflow."
        )
    return expected_nbnd


def resolve_wannier90_natomwfc(workflow_root: Path, task: Mapping[str, Any]) -> int:
    projwfc_output = band_child_output_path(workflow_root, task, stage="projwfc")
    if projwfc_output is None:
        projwfc_output = workflow_root / WANNIER90_DIR / "projwfc.out"
    if not projwfc_output.is_file():
        raise FileNotFoundError(f"Wannier90 projwfc output is missing: {projwfc_output}")
    value = extract_natomwfc(
        projwfc_output.read_text(encoding="utf-8", errors="ignore")
    )
    if value is None:
        raise RuntimeError(
            f"Could not find a positive 'natomwfc =' value in Wannier90 "
            f"projwfc output: {projwfc_output}"
        )
    return value


def resolve_wannier90_conv_tol(workflow_root: Path, task: Mapping[str, Any]) -> str:
    scf_input = band_child_input_path(workflow_root, task, stage="scf")
    if scf_input is None:
        scf_input = workflow_root / WANNIER90_DIR / "scf.in"
    if not scf_input.is_file():
        raise FileNotFoundError(f"Wannier90 SCF input is missing: {scf_input}")
    value = parse_namelist_values(
        scf_input.read_text(encoding="utf-8", errors="ignore"),
        "control",
    ).get("etot_conv_thr", "").strip()
    if not value:
        # Older/imported SCF inputs may rely on QE's documented effective default.
        value = DEFAULT_QE_ETOT_CONV_THR
    try:
        numeric_value = float(value.replace("D", "E").replace("d", "E"))
    except (TypeError, ValueError):
        numeric_value = float("nan")
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise RuntimeError(
            f"Could not find a positive '&CONTROL etot_conv_thr' value in "
            f"Wannier90 SCF input: {scf_input}"
        )
    return value


def resolve_wannier90_fermi_ev(workflow_root: Path, task: Mapping[str, Any]) -> float | None:
    scf_output = band_child_output_path(workflow_root, task, stage="scf")
    candidates = [scf_output] if scf_output is not None else []
    candidates.append(workflow_root / WANNIER90_DIR / "scf.out")
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        value = extract_fermi_energy_ev_from_scf_output(
            candidate.read_text(encoding="utf-8", errors="ignore")
        )
        if value is not None:
            return float(value)
    return None


def wannier90_nscf_kmesh(task: Mapping[str, Any], base_text: str) -> list[int] | None:
    child = band_child_by_stage(task, stage="nscf")
    result = child.get("result") if isinstance(child, Mapping) and isinstance(child.get("result"), Mapping) else None
    mesh = result.get("kmesh") if isinstance(result, Mapping) else None
    if isinstance(mesh, Sequence) and not isinstance(mesh, (str, bytes)) and len(mesh) >= 3:
        try:
            return [max(1, int(mesh[0])), max(1, int(mesh[1])), max(1, int(mesh[2]))]
        except (TypeError, ValueError):
            pass
    return extract_automatic_kmesh(base_text)


def render_wannier90_win_input(
    *,
    base_text: str,
    kpoints_crystal_b: Sequence[str],
    nbnd: int,
    num_wann: int,
    conv_tol: str,
    mp_grid: Sequence[int],
    spin_orbit: bool = False,
    num_iter: int = 10000,
    dis_num_iter: int = 0,
) -> str:
    nk1, nk2, nk3 = [int(value) for value in mp_grid[:3]]
    cell_rows = wannier90_cell_rows_from_qe_input(base_text)
    atom_rows = wannier90_atom_frac_rows_from_qe_input(base_text, cell_rows)
    path_rows = wannier90_kpoint_path_rows(kpoints_crystal_b)
    kpoint_rows = generate_wannier90_kmesh_points(nk1, nk2, nk3)

    lines = [
        f"num_bands       = {int(nbnd)}",
        f"num_wann        = {int(num_wann)}",
        f"num_iter        = {int(num_iter)}",
        f"dis_num_iter = {int(dis_num_iter)}",
        f"conv_tol = {conv_tol}",
        "conv_window = 3",
        "kmesh_tol = 1.0000000000d-05",
        "use_ws_distance = .true.",
        "wannier_plot = .false.",
        "write_xyz = .true.",
        "iprint = 3",
        "",
        "! Crucial for generating the hr.dat file",
        "write_hr        = .true.",
        "",
        "! **Automatic generation of initial projections**",
        "",
        "Begin Kpoint_Path",
        *path_rows,
        "End Kpoint_Path",
        " ",
        "bands_plot = .true.",
        "",
        "! SYSTEM",
        "",
        "begin unit_cell_cart",
        "angstrom",
        *[f"{row[0]:16.10f}{row[1]:17.10f}{row[2]:17.10f}" for row in cell_rows],
        "end unit_cell_cart",
        "",
        "begin atoms_frac",
        *[f"{symbol:<4}{coords[0]:16.10f}{coords[1]:16.10f}{coords[2]:16.10f}" for symbol, coords in atom_rows],
        "end atoms_frac",
        "auto_projections = .true.",
        "",
        " ",
        *(["spinors = .true.", ""] if spin_orbit else []),
        "! KPOINTS",
        "",
        f"mp_grid : {nk1} {nk2} {nk3}",
        "",
        "begin kpoints",
        *kpoint_rows,
        "end kpoints",
        "",
    ]
    return "\n".join(lines)


def wannier90_cell_rows_from_qe_input(text: str) -> list[list[float]]:
    cell = extract_last_qe_card(text, "CELL_PARAMETERS", max_data_lines=3)
    if not cell:
        raise RuntimeError("wannier_control.win requires CELL_PARAMETERS.")
    unit = qe_card_unit(str(cell.get("header") or "CELL_PARAMETERS angstrom"), default="angstrom")
    rows: list[list[float]] = []
    for raw in cell.get("lines") or []:
        parts = str(raw).split("!", 1)[0].split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise RuntimeError("CELL_PARAMETERS is incomplete for wannier_control.win.")
    if unit in {"bohr", "a.u.", "au"}:
        return [[component * BOHR_TO_ANGSTROM for component in row] for row in rows]
    if unit not in {"angstrom", "ang"}:
        raise RuntimeError(f"Unsupported CELL_PARAMETERS unit for wannier_control.win: {unit}")
    return rows


def wannier90_atom_frac_rows_from_qe_input(
    text: str,
    cell_rows: Sequence[Sequence[float]],
) -> list[tuple[str, list[float]]]:
    atomic = extract_last_qe_card(text, "ATOMIC_POSITIONS")
    if not atomic:
        raise RuntimeError("wannier_control.win requires ATOMIC_POSITIONS.")
    unit = qe_card_unit(str(atomic.get("header") or "ATOMIC_POSITIONS crystal"), default="crystal")
    rows: list[tuple[str, list[float]]] = []
    for raw in atomic.get("lines") or []:
        parts = str(raw).split("!", 1)[0].split()
        if len(parts) < 4:
            continue
        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
        if unit in {"angstrom", "ang"}:
            coords = cartesian_to_fractional(coords, cell_rows)
        elif unit in {"bohr", "a.u.", "au"}:
            coords = cartesian_to_fractional([value * BOHR_TO_ANGSTROM for value in coords], cell_rows)
        elif unit not in {"crystal", "cryst", "alat"}:
            raise RuntimeError(f"Unsupported ATOMIC_POSITIONS unit for wannier_control.win: {unit}")
        if unit == "alat":
            raise RuntimeError("ATOMIC_POSITIONS alat is not supported for wannier_control.win.")
        rows.append((parts[0], coords))
    if not rows:
        raise RuntimeError("ATOMIC_POSITIONS is empty for wannier_control.win.")
    return rows


def cartesian_to_fractional(
    position: Sequence[float],
    cell_rows: Sequence[Sequence[float]],
) -> list[float]:
    a, b, c = [[float(value) for value in row[:3]] for row in cell_rows[:3]]
    px, py, pz = [float(value) for value in position[:3]]
    matrix = [
        [a[0], b[0], c[0]],
        [a[1], b[1], c[1]],
        [a[2], b[2], c[2]],
    ]
    inv = invert_3x3(matrix)
    return [
        inv[0][0] * px + inv[0][1] * py + inv[0][2] * pz,
        inv[1][0] * px + inv[1][1] * py + inv[1][2] * pz,
        inv[2][0] * px + inv[2][1] * py + inv[2][2] * pz,
    ]


def invert_3x3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) <= 1.0e-12:
        raise RuntimeError("CELL_PARAMETERS matrix is singular for wannier_control.win.")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def wannier90_kpoint_path_rows(kpoints_crystal_b: Sequence[str]) -> list[str]:
    points = parse_wannier90_crystal_b_points(kpoints_crystal_b)
    rows: list[str] = []
    for start, end in zip(points, points[1:]):
        sx, sy, sz = start["coords"]
        ex, ey, ez = end["coords"]
        rows.append(
            f"{start['label']:<4}{sx:14.10f}{sy:14.10f}{sz:14.10f}   "
            f"{end['label']:<4}{ex:14.10f}{ey:14.10f}{ez:14.10f}"
        )
    return rows


def parse_wannier90_crystal_b_points(lines: Sequence[str]) -> list[dict[str, Any]]:
    rows = [str(line).rstrip() for line in lines if str(line).strip()]
    if rows and rows[0].strip().upper().startswith("K_POINTS"):
        rows = rows[1:]
    if rows and re.fullmatch(r"\d+", rows[0].strip()):
        rows = rows[1:]
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        body, _, comment = raw.partition("!")
        parts = body.split()
        if len(parts) < 3:
            continue
        try:
            coords = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            continue
        label = wannier90_path_label(comment.strip() or f"P{index + 1}")
        points.append({"label": label, "coords": coords})
    if len(points) < 2:
        raise RuntimeError("wannier_control.win requires at least two high-symmetry K_POINTS crystal_b points.")
    return points


def wannier90_path_label(label: str) -> str:
    text = str(label or "").strip().lstrip("!").strip()
    lowered = text.lower()
    if lowered in {"gamma", "gam", "g", "Γ"}:
        return "G"
    return text.upper()


def generate_wannier90_kmesh_points(n1: int, n2: int, n3: int) -> list[str]:
    if min(int(n1), int(n2), int(n3)) <= 0:
        raise ValueError("Wannier90 mp_grid values must be positive integers.")
    rows: list[str] = []
    for ix in range(int(n1)):
        fx = ix / int(n1)
        for iy in range(int(n2)):
            fy = iy / int(n2)
            for iz in range(int(n3)):
                fz = iz / int(n3)
                rows.append(f"{fx:12.8f}{fy:12.8f}{fz:12.8f}")
    return rows


def format_qe_float(value: float) -> str:
    return f"{float(value):.10g}"


def format_window_float(value: float) -> str:
    return f"{float(value):.6f}"


def render_wannier90_nscf_input(
    template_text: str,
    *,
    workflow_root: str | Path,
    child_workdir: str | Path,
    prefix: str,
    kpoint_spacing: float,
    kmesh: Sequence[int] | None = None,
    ecutwfc: int | float | None,
    spin_orbit_required: bool | None,
    outdir: str,
) -> tuple[str, list[int], int]:
    """Render NSCF input with an explicit K_POINTS crystal mesh."""

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
    rendered = set_namelist_values(rendered, "control", {"calculation": "'nscf'"})
    rendered = set_namelist_values(rendered, "electrons", {"diago_full_acc": ".true."})
    kmesh = extract_automatic_kmesh(rendered)
    if not kmesh:
        raise RuntimeError("Wannier90 NSCF input requires a K_POINTS automatic mesh to expand.")

    crystal_lines = generate_crystal_kmesh(int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))
    rendered = replace_qe_card(rendered, "K_POINTS", crystal_lines[0], crystal_lines[1:])
    return rendered, [int(kmesh[0]), int(kmesh[1]), int(kmesh[2])], int(kmesh[0]) * int(kmesh[1]) * int(kmesh[2])


def run_dos_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    scf_kpoint_spacing: float,
    nscf_kpoint_spacing: float,
    converged_kmesh: Sequence[int] | None,
    ecutwfc: Any,
    spin_orbit: bool,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or DOS_DIR)
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "dos_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        rendered = ""
        engine = ""

        if stage == "scf":
            engine = "pw.x"
            rendered = render_dos_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or DOS_DIR,
                prefix=prefix,
                kpoint_spacing=float(scf_kpoint_spacing),
                kmesh=converged_kmesh,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
        elif stage == "nscf":
            rendered = render_dos_nscf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or DOS_DIR,
                prefix=prefix,
                kpoint_spacing=float(nscf_kpoint_spacing),
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "projwfc":
            rendered = render_projwfc_input(
                prefix=prefix,
                outdir=outdir,
            )
            engine = "projwfc.x"
        elif stage == "pdos_sumpdos":
            postprocess_outputs = run_pdos_sumpdos_from_projwfc(
                workflow_root,
                dos_task=task,
                dos_child=child,
                prefix=prefix,
                state=state,
                log_path=child_log_path,
            )
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                **postprocess_outputs,
            }
            archive_child_files(workflow_root, child)
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results["pdos_sumpdos_output"] = child.get("output")
            results["pdos_sumpdos_data"] = postprocess_outputs.get("data")
            results["pdos_sumpdos_fermi_ev"] = postprocess_outputs.get("fermi_ev")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        elif stage in {"pdos_plotting", "pdos_postprocess"}:
            postprocess_outputs = (
                run_pdos_plotting_from_sumpdos_data(
                    workflow_root,
                    dos_task=task,
                    dos_child=child,
                    prefix=prefix,
                    state=state,
                    log_path=child_log_path,
                )
                if stage == "pdos_plotting"
                else run_pdos_postprocess_from_projwfc(
                    workflow_root,
                    dos_task=task,
                    dos_child=child,
                    prefix=prefix,
                    state=state,
                    log_path=child_log_path,
                )
            )
            child["status"] = "completed"
            child["completed_at"] = timestamp()
            child["result"] = {
                "stage": stage,
                "engine": "python",
                **postprocess_outputs,
            }
            archive_child_files(workflow_root, child)
            task["completed"] = count_completed_children(task)
            task["running_child"] = None
            results = state.setdefault("results", {})
            results["pdos_postprocess_output"] = child.get("output")
            results["pdos_postprocess_metadata"] = postprocess_outputs.get("metadata")
            results["pdos_csv_files"] = postprocess_outputs.get("csv_files")
            results["pdos_plot_files"] = postprocess_outputs.get("plot_files")
            results["dos_summary_csv"] = postprocess_outputs.get("dos_summary_csv")
            results["dos_summary_plot"] = postprocess_outputs.get("dos_summary_plot")
            results["pdos_combined_plot"] = postprocess_outputs.get("combined_band_dos_plot")
            results["pdos_combined_plot_mode"] = postprocess_outputs.get("combined_plot_mode")
            results["pdos_replot_script"] = postprocess_outputs.get("replot_script")
            state["updated_at"] = timestamp()
            add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
            write_state(state_path, state)
            return
        else:
            raise RuntimeError(f"Unsupported DOS workflow child stage: {stage}")

        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")
        command = build_qe_command(engine, input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child_result = {
            "stage": stage,
            "engine": engine or "pw.x",
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        child["result"] = child_result
        task["completed"] = count_completed_children(task)
        task["running_child"] = None

        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["dos_scf_output"] = child.get("output")
        elif stage == "nscf":
            results["dos_nscf_output"] = child.get("output")
        elif stage == "projwfc":
            results["dos_projwfc_output"] = child.get("output")

        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
        raise


def run_fermi_surface_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    fermi_options = (
        params.get("fermi_surface_options")
        if isinstance(params.get("fermi_surface_options"), Mapping)
        else {}
    )
    kpoint_spacing = optional_float_or_none(fermi_options.get("kpoint_spacing")) or 0.10
    prefix = fermi_surface_workflow_prefix(manifest, fermi_options)
    outdir = "./out/"
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")
            run_fermi_surface_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                kpoint_spacing=float(kpoint_spacing),
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                state_path=state_path,
            )

        fermi_child = band_child_by_stage(task, stage="fermi_velocity")
        fermi_result = fermi_child.get("result") if isinstance(fermi_child, Mapping) else {}
        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = {
            "prefix": prefix,
            "outdir": outdir,
            "kpoint_spacing": kpoint_spacing,
            "frmsf": fermi_result.get("frmsf") if isinstance(fermi_result, Mapping) else None,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        results_state = state.setdefault("results", {})
        results_state["fermi_surface"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_fermi_surface_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    kpoint_spacing: float,
    ecutwfc: Any,
    spin_orbit: bool,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or FERMI_SURFACE_DIR)
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "fermi_surface_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        engine = ""
        if stage == "scf":
            rendered = render_fermi_surface_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or FERMI_SURFACE_DIR,
                prefix=prefix,
                kpoint_spacing=float(kpoint_spacing),
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
                outdir=outdir,
            )
            rendered = write_workflow_input(input_path, rendered, manifest)
            append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")
            engine = "pw.x"
        elif stage == "fermi_velocity":
            if not input_path.is_file():
                raise FileNotFoundError(f"Fermi velocity input is missing: {input_path}")
            engine = "fermi_velocity.x"
        else:
            raise RuntimeError(f"Unsupported Fermi surface workflow child stage: {stage}")

        command = build_qe_command(engine, input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        child_result = {
            "stage": stage,
            "engine": engine,
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["fermi_surface_scf_output"] = child.get("output")
        elif stage == "fermi_velocity":
            frmsf_path = collect_fermi_surface_frmsf(
                child_workdir,
                workflow_root,
                prefix=prefix,
                outdir=outdir,
                log_path=child_log_path,
            )
            relative_frmsf = str(frmsf_path.relative_to(workflow_root).as_posix())
            child_result["frmsf"] = relative_frmsf
            results["fermi_surface_velocity_output"] = child.get("output")
            results["fermi_surface_frmsf"] = relative_frmsf

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = child_result
        task["completed"] = count_completed_children(task)
        task["running_child"] = None
        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
        raise


def collect_fermi_surface_frmsf(
    child_workdir: Path,
    workflow_root: Path,
    *,
    prefix: str,
    outdir: str,
    log_path: Path,
) -> Path:
    outdir_path = Path(outdir).expanduser()
    if not outdir_path.is_absolute():
        outdir_path = child_workdir / outdir_path
    candidates = [
        outdir_path / f"{prefix}_vfermi.frmsf",
        child_workdir / f"{prefix}_vfermi.frmsf",
    ]
    candidates.extend(sorted(outdir_path.glob("*vfermi*.frmsf")) if outdir_path.is_dir() else [])
    candidates.extend(sorted(child_workdir.glob("*vfermi*.frmsf")))
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"fermi_velocity.x did not create a {prefix}_vfermi.frmsf file.")

    target = workflow_root / DATA_PLOTS_DIR / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    append_log(log_path, f"Collected FermiSurfer file: {target}\n")
    return target


def run_charge_density_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    charge_options = (
        params.get("charge_density_options")
        if isinstance(params.get("charge_density_options"), Mapping)
        else {}
    )
    kpoint_spacing = positive_float_or_none(charge_options.get("kpoint_spacing")) or 0.10
    prefix = charge_density_workflow_prefix(manifest, charge_options)
    outdir = "./out/"
    fileout = str(charge_options.get("fileout") or "{prefix}.cube").strip() or "{prefix}.cube"
    fileout = fileout.replace("{prefix}", prefix)
    try:
        plot_num = int(charge_options.get("plot_num", 8))
    except (TypeError, ValueError):
        plot_num = 8
    plot_num = min(25, max(0, plot_num))
    nx = optional_positive_int(charge_options.get("nx")) or 64
    ny = optional_positive_int(charge_options.get("ny")) or 64
    nz = optional_positive_int(charge_options.get("nz")) or 64
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")

            run_charge_density_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                fileout=fileout,
                plot_num=plot_num,
                nx=nx,
                ny=ny,
                nz=nz,
                kpoint_spacing=float(kpoint_spacing),
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                state_path=state_path,
            )

        pp_child = band_child_by_stage(task, stage="pp")
        pp_result = pp_child.get("result") if isinstance(pp_child, Mapping) else {}
        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = {
            "prefix": prefix,
            "outdir": outdir,
            "kpoint_spacing": float(kpoint_spacing),
            "fileout": fileout,
            "plot_num": plot_num,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "cube": pp_result.get("cube") if isinstance(pp_result, Mapping) else None,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        results_state = state.setdefault("results", {})
        results_state["charge_density"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_charge_density_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    fileout: str,
    plot_num: int,
    nx: int,
    ny: int,
    nz: int,
    kpoint_spacing: float,
    ecutwfc: Any,
    spin_orbit: bool,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or CHARGE_DENSITY_DIR)
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "charge_density_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        if stage == "scf":
            rendered = render_band_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or CHARGE_DENSITY_DIR,
                prefix=prefix,
                kpoint_spacing=float(kpoint_spacing),
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
                outdir=outdir,
            )
            engine = "pw.x"
        elif stage == "pp":
            rendered = render_charge_density_pp_input(
                prefix=prefix,
                outdir=outdir,
                fileout=fileout,
                plot_num=plot_num,
                nx=nx,
                ny=ny,
                nz=nz,
            )
            engine = "pp.x"
        else:
            raise RuntimeError(f"Unsupported post-processing workflow child stage: {stage}")

        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")
        command = build_qe_command(engine, input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        child_result = {
            "stage": stage,
            "engine": engine,
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["charge_density_scf_output"] = child.get("output")
        elif stage == "pp":
            cube_path = collect_charge_density_cube(
                child_workdir,
                workflow_root,
                fileout=fileout,
                log_path=child_log_path,
            )
            relative_cube = str(cube_path.relative_to(workflow_root).as_posix())
            child_result["cube"] = relative_cube
            results["charge_density_pp_output"] = child.get("output")
            results["charge_density_cube"] = relative_cube

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = child_result
        task["completed"] = count_completed_children(task)
        task["running_child"] = None
        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
        raise


def collect_charge_density_cube(
    child_workdir: Path,
    workflow_root: Path,
    *,
    fileout: str,
    log_path: Path,
) -> Path:
    requested = Path(str(fileout or "").strip()).expanduser()
    candidates: list[Path] = []
    if str(requested):
        candidates.append(requested if requested.is_absolute() else child_workdir / requested)
    candidates.extend(sorted(child_workdir.glob("*rho*.cube")))
    candidates.extend(sorted(child_workdir.glob("*.cube")))
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"pp.x did not create the requested post-processing cube file: {fileout}")

    target = workflow_root / DATA_PLOTS_DIR / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    append_log(log_path, f"Collected post-processing cube file: {target}\n")
    return target


def run_elastic_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")

    child = children[0]
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    elastic_options = params.get("elastic_options") if isinstance(params.get("elastic_options"), Mapping) else {}
    uses_converged = bool(params.get("uses_converged_values", True))
    ecutwfc = workflow_manual_ecutwfc(manifest)
    if ecutwfc is None and uses_converged:
        ecutwfc = resolve_elastic_ecutwfc(workflow_root, state)
    kmesh = resolve_elastic_kmesh(workflow_root, state) if uses_converged else None
    spin_orbit = False

    child_workdir = workflow_root / str(child.get("workdir") or ELASTIC_DIR)
    input_path = workflow_root / str(child.get("input") or Path(ELASTIC_DIR) / "scf.in")
    control_path = workflow_root / str(child.get("control") or Path(ELASTIC_DIR) / "thermo_control")
    output_path = workflow_root / str(child.get("output") or Path(ELASTIC_DIR) / "thermo_pw.out")
    child_log_path = workflow_root / str(child.get("log") or "logs/elastic.log")

    task["status"] = "running"
    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        child_workdir.mkdir(parents=True, exist_ok=True)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        rendered_scf = render_elastic_scf_input(
            base_text,
            workflow_root=workflow_root,
            child_workdir=child.get("workdir") or ELASTIC_DIR,
            prefix="elastic",
            kmesh=kmesh,
            ecutwfc=optional_float_or_none(ecutwfc),
            spin_orbit_required=spin_orbit,
        )
        rendered_scf = write_workflow_input(input_path, rendered_scf, manifest)
        control_path.write_text(render_thermo_control(elastic_options), encoding="utf-8")
        append_log(child_log_path, f"Wrote elastic SCF input: {input_path}\n")
        append_log(child_log_path, f"Wrote thermo_pw control file: {control_path}\n")

        command = build_qe_command("thermo_pw.x", input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        archive_elastic_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = {
            "stage": "elastic",
            "engine": "thermo_pw.x",
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "control": str(Path(str(child.get("control") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
            "ecutwfc": optional_float_or_none(ecutwfc),
            "kmesh": list(kmesh) if kmesh is not None else None,
            "elastic_options": dict(elastic_options or {}),
        }
        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = dict(child["result"])

        results_state = state.setdefault("results", {})
        results_state["elastic_scf_input"] = child.get("input")
        results_state["elastic_thermo_control"] = child.get("control")
        results_state["elastic_output"] = child.get("output")
        results_state["elastic"] = task["result"]
        add_event(state, "task.completed", {"task": task_id, "child": child.get("id")})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": task_id, "child": child.get("id")})
        write_state(state_path, state)
        raise


def resolve_elastic_ecutwfc(workflow_root: Path, state: Mapping[str, Any]) -> float | None:
    results = state.get("results") if isinstance(state, Mapping) else {}
    if isinstance(results, Mapping):
        value = optional_float_or_none(results.get("ecutwfc"))
        if value is not None:
            return value
    point = last_convergence_json_point(workflow_root, "cutoff_convergence_energies.json")
    return optional_float_or_none(point.get("ecutwfc")) if point else None


def resolve_elastic_kmesh(workflow_root: Path, state: Mapping[str, Any]) -> list[int] | None:
    results = state.get("results") if isinstance(state, Mapping) else {}
    if isinstance(results, Mapping):
        mesh = normalize_kmesh_value(results.get("kmesh"))
        if mesh is not None:
            return mesh
    point = last_convergence_json_point(workflow_root, "kpoint_convergence_energies.json")
    return normalize_kmesh_value(point.get("kmesh")) if point else None


def last_convergence_json_point(workflow_root: Path, filename: str) -> Mapping[str, Any]:
    path = workflow_root / DATA_PLOTS_DIR / filename
    if not path.is_file():
        return {}
    try:
        payload = load_json_file(path)
    except Exception:
        return {}
    points = payload.get("points") if isinstance(payload.get("points"), list) else []
    for point in reversed(points):
        if isinstance(point, Mapping):
            return point
    return {}


def normalize_kmesh_value(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        return [max(1, int(float(value[0]))), max(1, int(float(value[1]))), max(1, int(float(value[2])))]
    except Exception:
        return None


def archive_elastic_child_files(workflow_root: Path, child: Mapping[str, Any]) -> None:
    source_paths = [
        workflow_root / str(child.get("input") or ""),
        workflow_root / str(child.get("control") or ""),
        workflow_root / str(child.get("output") or ""),
    ]
    archive_paths = [workflow_root / str(path) for path in (child.get("archive_inputs_outputs") or [])]
    for source, archive in zip(source_paths, archive_paths):
        if not source.is_file():
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archive)


def run_pdos_postprocess_from_projwfc(
    workflow_root: Path,
    *,
    dos_task: Mapping[str, Any],
    dos_child: Mapping[str, Any],
    prefix: str,
    state: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    """Legacy one-shot PDOS postprocessing (sumpdos + plotting)."""

    sumpdos_outputs = run_pdos_sumpdos_from_projwfc(
        workflow_root,
        dos_task=dos_task,
        dos_child=dos_child,
        prefix=prefix,
        state=state,
        log_path=log_path,
        output_relative=str(Path("data_and_plots") / "pdos_sumpdos_data.json"),
    )
    plotting_outputs = run_pdos_plotting_from_sumpdos_data(
        workflow_root,
        dos_task=dos_task,
        dos_child=dos_child,
        prefix=prefix,
        state=state,
        log_path=log_path,
        sumpdos_data_relative=str(sumpdos_outputs.get("data") or ""),
    )
    return {
        **plotting_outputs,
        "sumpdos_data": sumpdos_outputs.get("data"),
    }


def run_pdos_sumpdos_from_projwfc(
    workflow_root: Path,
    *,
    dos_task: Mapping[str, Any],
    dos_child: Mapping[str, Any],
    prefix: str,
    state: Mapping[str, Any],
    log_path: Path,
    output_relative: str | None = None,
) -> dict[str, Any]:
    """Run sumpdos.x and save intermediate PDOS data for plotting."""

    dos_workdir = workflow_root / str(dos_child.get("workdir") or DOS_DIR)
    if not dos_workdir.is_dir():
        raise FileNotFoundError(f"DOS workdir is missing: {dos_workdir}")

    scf_child = band_child_by_stage(dos_task, stage="scf")
    scf_input = workflow_root / str(scf_child.get("input") or "") if scf_child is not None else None
    if scf_input is None or not scf_input.is_file():
        raise FileNotFoundError("DOS SCF input file is missing; cannot extract ATOMIC_SPECIES for PDOS.")

    scf_output = workflow_root / str(scf_child.get("output") or "") if scf_child is not None else None
    if scf_output is None or not scf_output.is_file():
        raise FileNotFoundError("DOS SCF output is missing; cannot determine Fermi energy for PDOS.")

    scf_text = scf_output.read_text(encoding="utf-8", errors="ignore")
    fermi_ev = extract_fermi_energy_ev_from_scf_output(scf_text)
    if fermi_ev is None:
        results = state.get("results") if isinstance(state, Mapping) else {}
        if isinstance(results, Mapping):
            fermi_ev = optional_float_or_none(results.get("band_structure_fermi_ev"))
    if fermi_ev is None:
        raise RuntimeError("Could not determine Fermi energy for PDOS post-processing.")

    input_text = scf_input.read_text(encoding="utf-8", errors="ignore")
    species_entries = atomic_species_entries(input_text)
    atom_types: list[str] = []
    seen: set[str] = set()
    for entry in species_entries:
        symbol = str(entry.get("element") or "").strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        atom_types.append(symbol)
    if not atom_types:
        raise RuntimeError("ATOMIC_SPECIES entries could not be parsed from DOS SCF input.")

    orbitals = ["s", "p", "d", "f"]
    atom_data: dict[str, dict[str, Any]] = {}
    summed_dat_files: list[str] = []
    for atom in atom_types:
        available_orbitals: list[str] = []
        orbital_data: dict[str, list[float]] = {}
        energy_values: list[float] = []

        for orbital in orbitals:
            pattern_glob = f"*({atom})*({orbital}*"
            matched_inputs = glob.glob(str(dos_workdir / pattern_glob))
            if not matched_inputs:
                continue
            input_pattern = f"*\\({atom}\\)*\\({orbital}*"
            output_name = f"atom_{atom}_{orbital}.dat"
            run_sumpdos_for_pattern(
                input_pattern=input_pattern,
                output_name=output_name,
                workdir=dos_workdir,
                log_path=log_path,
            )
            dat_path = dos_workdir / output_name
            if not dat_path.is_file():
                raise RuntimeError(
                    f"sumpdos.x completed but output is missing for atom {atom} orbital {orbital}: {dat_path}"
                )
            x_vals, y_vals = parse_sumpdos_dat(dat_path)
            if not x_vals or not y_vals:
                raise RuntimeError(
                    f"sumpdos.x output is empty/unreadable for atom {atom} orbital {orbital}: {dat_path}"
                )
            if not energy_values:
                energy_values = list(x_vals)
            else:
                shared_len = min(len(energy_values), len(x_vals), len(y_vals))
                if shared_len <= 0:
                    raise RuntimeError(
                        f"sumpdos.x produced incompatible data lengths for atom {atom} orbital {orbital}."
                    )
                energy_values = energy_values[:shared_len]
                x_vals = x_vals[:shared_len]
                y_vals = y_vals[:shared_len]
                for key in list(orbital_data):
                    orbital_data[key] = orbital_data[key][:shared_len]
            if len(energy_values) != len(y_vals):
                shared_len = min(len(energy_values), len(y_vals))
                energy_values = energy_values[:shared_len]
                y_vals = y_vals[:shared_len]
                for key in list(orbital_data):
                    orbital_data[key] = orbital_data[key][:shared_len]
            available_orbitals.append(orbital)
            orbital_data[orbital] = list(y_vals)
            summed_dat_files.append(str(dat_path.relative_to(workflow_root)))

        total_pattern = f"*\\({atom}\\)*"
        total_output = f"atom_{atom}_total.dat"
        total_matches = glob.glob(str(dos_workdir / f"*({atom})*"))
        if total_matches:
            run_sumpdos_for_pattern(
                input_pattern=total_pattern,
                output_name=total_output,
                workdir=dos_workdir,
                log_path=log_path,
            )
            total_path = dos_workdir / total_output
            if not total_path.is_file():
                raise RuntimeError(
                    f"sumpdos.x completed but total output is missing for atom {atom}: {total_path}"
                )
            x_vals, y_vals = parse_sumpdos_dat(total_path)
            if not x_vals or not y_vals:
                raise RuntimeError(
                    f"sumpdos.x total output is empty/unreadable for atom {atom}: {total_path}"
                )
            if not energy_values:
                energy_values = list(x_vals)
            else:
                shared_len = min(len(energy_values), len(x_vals), len(y_vals))
                if shared_len <= 0:
                    raise RuntimeError(f"sumpdos.x total output is incompatible for atom {atom}.")
                energy_values = energy_values[:shared_len]
                y_vals = y_vals[:shared_len]
                for key in list(orbital_data):
                    orbital_data[key] = orbital_data[key][:shared_len]
            if len(energy_values) != len(y_vals):
                shared_len = min(len(energy_values), len(y_vals))
                energy_values = energy_values[:shared_len]
                y_vals = y_vals[:shared_len]
                for key in list(orbital_data):
                    orbital_data[key] = orbital_data[key][:shared_len]
            available_orbitals.append("total")
            orbital_data["total"] = list(y_vals)
            summed_dat_files.append(str(total_path.relative_to(workflow_root)))

        if not energy_values or not orbital_data:
            raise RuntimeError(f"No PDOS data was generated for atom type {atom}.")

        shifted_energy = [float(value) - float(fermi_ev) for value in energy_values]
        atom_data[atom] = {
            "energy_ev_shifted": shifted_energy,
            "orbitals": list(available_orbitals),
            "pdos": {name: [float(value) for value in values] for name, values in orbital_data.items()},
        }

    if not atom_data:
        raise RuntimeError("No PDOS data was generated from projwfc outputs.")

    data_relative = output_relative or str(dos_child.get("output") or "").strip()
    if not data_relative:
        data_relative = str(Path("data_and_plots") / "pdos_sumpdos_data.json")
    data_path = workflow_root / data_relative
    data_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "prefix": prefix,
        "fermi_ev": float(fermi_ev),
        "dos_workdir": str(dos_workdir.relative_to(workflow_root)),
        "atom_types": atom_types,
        "summed_dat_files": summed_dat_files,
        "atom_data": atom_data,
        "generated_at": timestamp(),
    }
    write_json_file(data_path, metadata)
    append_log(log_path, f"Saved PDOS sumpdos intermediate data: {data_path}\n")

    return {
        "data": str(Path(data_relative).as_posix()),
        "fermi_ev": float(fermi_ev),
        "atom_types": atom_types,
        "summed_dat_files": summed_dat_files,
    }


def run_pdos_plotting_from_sumpdos_data(
    workflow_root: Path,
    *,
    dos_task: Mapping[str, Any],
    dos_child: Mapping[str, Any],
    prefix: str,
    state: Mapping[str, Any],
    log_path: Path,
    sumpdos_data_relative: str | None = None,
) -> dict[str, Any]:
    """Build PDOS CSV/plots from intermediate sumpdos data."""

    input_relative = (sumpdos_data_relative or str(dos_child.get("input") or "")).strip()
    if not input_relative:
        raise RuntimeError("PDOS plotting stage is missing its sumpdos input file.")
    sumpdos_data_path = workflow_root / input_relative
    if not sumpdos_data_path.is_file():
        raise FileNotFoundError(f"PDOS sumpdos data file is missing: {sumpdos_data_path}")

    payload = load_json_file(sumpdos_data_path)
    fermi_ev = optional_float_or_none(payload.get("fermi_ev"))
    if fermi_ev is None:
        raise RuntimeError(f"PDOS sumpdos data is missing fermi_ev: {sumpdos_data_path}")
    atom_types_raw = payload.get("atom_types")
    atom_data_raw = payload.get("atom_data")
    if not isinstance(atom_types_raw, Sequence) or isinstance(atom_types_raw, (str, bytes)):
        raise RuntimeError(f"PDOS sumpdos data is missing atom_types: {sumpdos_data_path}")
    if not isinstance(atom_data_raw, Mapping):
        raise RuntimeError(f"PDOS sumpdos data is missing atom_data: {sumpdos_data_path}")

    atom_types = [str(item).strip() for item in atom_types_raw if str(item).strip()]
    csv_files: dict[str, str] = {}
    plot_files: dict[str, list[str]] = {}
    aggregate_energy: list[float] = []
    aggregate_pdos: dict[str, list[float]] = {}
    aggregate_labels: list[str] = []

    data_plots_dir = workflow_root / "data_and_plots"
    data_plots_dir.mkdir(parents=True, exist_ok=True)
    pdos_csv_dir = data_plots_dir / "pdos_csv"
    pdos_csv_dir.mkdir(parents=True, exist_ok=True)

    for atom in atom_types:
        atom_entry = atom_data_raw.get(atom) if isinstance(atom_data_raw, Mapping) else None
        if not isinstance(atom_entry, Mapping):
            raise RuntimeError(f"PDOS sumpdos data is missing atom entry for {atom}.")
        shifted_energy_raw = atom_entry.get("energy_ev_shifted")
        orbitals_raw = atom_entry.get("orbitals")
        pdos_raw = atom_entry.get("pdos")
        if not isinstance(shifted_energy_raw, Sequence) or isinstance(shifted_energy_raw, (str, bytes)):
            raise RuntimeError(f"PDOS sumpdos atom entry for {atom} is missing energy data.")
        if not isinstance(orbitals_raw, Sequence) or isinstance(orbitals_raw, (str, bytes)):
            raise RuntimeError(f"PDOS sumpdos atom entry for {atom} is missing orbital labels.")
        if not isinstance(pdos_raw, Mapping):
            raise RuntimeError(f"PDOS sumpdos atom entry for {atom} is missing orbital PDOS values.")

        shifted_energy = [float(value) for value in shifted_energy_raw]
        available_orbitals = [str(name) for name in orbitals_raw if str(name).strip()]
        orbital_data: dict[str, list[float]] = {}
        for orbital in available_orbitals:
            values_raw = pdos_raw.get(orbital)
            if not isinstance(values_raw, Sequence) or isinstance(values_raw, (str, bytes)):
                raise RuntimeError(f"PDOS sumpdos atom entry for {atom} is missing values for orbital {orbital}.")
            values = [float(item) for item in values_raw]
            shared_len = min(len(shifted_energy), len(values))
            if shared_len <= 0:
                raise RuntimeError(f"PDOS sumpdos atom entry for {atom} has empty values for orbital {orbital}.")
            shifted_energy = shifted_energy[:shared_len]
            values = values[:shared_len]
            for existing_key in list(orbital_data):
                orbital_data[existing_key] = orbital_data[existing_key][:shared_len]
            orbital_data[orbital] = values

        if not shifted_energy or not orbital_data:
            raise RuntimeError(f"PDOS sumpdos atom entry for {atom} has no usable data.")

        csv_name = f"pdos_atom_{atom}.csv"
        csv_relative = str(Path("data_and_plots") / "pdos_csv" / csv_name)
        write_atom_pdos_csv(
            workflow_root / csv_relative,
            shifted_energy=shifted_energy,
            available_orbitals=available_orbitals,
            orbital_data=orbital_data,
        )
        csv_files[atom] = csv_relative

        plot_orbital = str(Path("data_and_plots") / f"{atom}_pdos.png")
        plot_total = str(Path("data_and_plots") / f"{atom}_pdos_total.png")
        write_atom_pdos_plot(
            workflow_root / plot_orbital,
            shifted_energy=shifted_energy,
            available_orbitals=available_orbitals,
            orbital_data=orbital_data,
            atom=atom,
            include_total=False,
        )
        write_atom_pdos_plot(
            workflow_root / plot_total,
            shifted_energy=shifted_energy,
            available_orbitals=available_orbitals,
            orbital_data=orbital_data,
            atom=atom,
            include_total=True,
        )
        plot_files[atom] = [plot_orbital, plot_total]

        if not aggregate_energy:
            aggregate_energy = list(shifted_energy)
        shared_len = min(len(aggregate_energy), len(shifted_energy))
        if shared_len <= 0:
            continue
        if shared_len != len(aggregate_energy):
            aggregate_energy = aggregate_energy[:shared_len]
            for key in list(aggregate_pdos):
                aggregate_pdos[key] = aggregate_pdos[key][:shared_len]
        for orbital in available_orbitals:
            values = list(orbital_data.get(orbital) or [])[:shared_len]
            if orbital not in aggregate_pdos:
                aggregate_pdos[orbital] = [0.0] * shared_len
                aggregate_labels.append(orbital)
            aggregate_pdos[orbital] = [
                float(aggregate_pdos[orbital][idx]) + float(values[idx] if idx < len(values) else 0.0)
                for idx in range(shared_len)
            ]

    if not aggregate_energy or not aggregate_pdos:
        raise RuntimeError("PDOS plotting could not build aggregate DOS data from sumpdos results.")

    dos_summary_csv_relative = str(Path("data_and_plots") / "density_of_states.csv")
    write_atom_pdos_csv(
        workflow_root / dos_summary_csv_relative,
        shifted_energy=aggregate_energy,
        available_orbitals=aggregate_labels,
        orbital_data=aggregate_pdos,
    )
    dos_summary_plot_relative = str(Path("data_and_plots") / "density_of_states.png")
    write_total_dos_plot(
        workflow_root / dos_summary_plot_relative,
        shifted_energy=aggregate_energy,
        available_orbitals=aggregate_labels,
        orbital_data=aggregate_pdos,
    )

    combined_plot_relative = ""
    combined_plot_mode = ""
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    band_shifted_relative = ""
    fatband_all_states_relative = ""
    fatband_color_mode = "atomic"
    results = state.get("results") if isinstance(state, Mapping) else {}
    if isinstance(results, Mapping):
        candidate_band_shifted = str(results.get("band_structure_shifted_gnu") or "")
        if candidate_band_shifted and (workflow_root / candidate_band_shifted).is_file():
            band_shifted_relative = candidate_band_shifted

        band_meta_rel = ""
        band_state = results.get("band_structure")
        if isinstance(band_state, Mapping):
            band_meta_rel = str(band_state.get("metadata") or "")
        if band_meta_rel and (workflow_root / band_meta_rel).is_file():
            try:
                band_meta = load_json_file(workflow_root / band_meta_rel)
            except Exception:
                band_meta = {}
            ticks = band_meta.get("kpoint_ticks") if isinstance(band_meta, Mapping) else []
            if isinstance(ticks, Sequence):
                for tick in ticks:
                    if not isinstance(tick, Mapping):
                        continue
                    x_value = optional_float_or_none(tick.get("x"))
                    label = str(tick.get("label") or "").strip()
                    if x_value is None or not label:
                        continue
                    tick_positions.append(float(x_value))
                    tick_labels.append(label)

        fatband_all_states_rel = str(results.get("fatband_projection_csv") or "").strip()
        fatband_csv_dir_rel = str(results.get("fatband_projection_csv_dir") or "").strip()
        fatband_all_states_path = Path()
        if fatband_all_states_rel:
            fatband_all_states_path = workflow_root / fatband_all_states_rel
        elif fatband_csv_dir_rel:
            fatband_all_states_path = workflow_root / fatband_csv_dir_rel / "all_states.csv"

        fatband_color_mode = normalize_fatband_color_mode(results.get("fatband_color_mode"))
        if fatband_all_states_path and fatband_all_states_path.is_file():
            try:
                fatband_all_states_relative = str(fatband_all_states_path.relative_to(workflow_root))
            except ValueError:
                fatband_all_states_relative = str(fatband_all_states_path)
            combined_plot_relative = str(Path("data_and_plots") / "fatband_plus_dos.png")
            write_fatband_orbital_dos_side_by_side_plot(
                workflow_root / combined_plot_relative,
                fatband_all_states_csv=fatband_all_states_path,
                shifted_energy=aggregate_energy,
                available_orbitals=aggregate_labels,
                orbital_data=aggregate_pdos,
                color_mode=fatband_color_mode,
                tick_positions=tick_positions,
                tick_labels=tick_labels,
            )
            combined_plot_mode = "fatband"
        else:
            if band_shifted_relative and (workflow_root / band_shifted_relative).is_file():
                curves = parse_bands_gnu_curves(workflow_root / band_shifted_relative)
                if curves:
                    combined_plot_relative = str(Path("data_and_plots") / "band_plus_dos.png")
                    write_band_dos_side_by_side_plot(
                        workflow_root / combined_plot_relative,
                        curves=curves,
                        tick_positions=tick_positions,
                        tick_labels=tick_labels,
                        shifted_energy=aggregate_energy,
                        available_orbitals=aggregate_labels,
                        orbital_data=aggregate_pdos,
                        )
                    combined_plot_mode = "bands"

    # Generate band structure + per-atom/orbital PDOS plot
    atomic_dos_plot_relative = ""
    try:
        if band_shifted_relative and (workflow_root / band_shifted_relative).is_file():
            curves = parse_bands_gnu_curves(workflow_root / band_shifted_relative)
            if curves and aggregate_energy:
                atom_orbital_plot_data = atom_orbital_pdos_from_sumpdos_payload(payload, atom_types)
                if atom_orbital_plot_data:
                    atomic_dos_plot_relative = str(Path("data_and_plots") / "band_plus_atomic_dos.png")
                    write_band_atomic_orbital_dos_side_by_side_plot(
                        workflow_root / atomic_dos_plot_relative,
                        curves=curves,
                        tick_positions=tick_positions,
                        tick_labels=tick_labels,
                        atom_orbital_pdos_data=atom_orbital_plot_data,
                    )
                    append_log(log_path, f"Generated band structure + atomic orbital DOS plot: {atomic_dos_plot_relative}\n")
    except Exception as exc:
        append_log(log_path, f"Warning: Could not generate band structure + atomic orbital DOS plot: {exc}\n")

    metadata_relative = str(Path("data_and_plots") / "pdos_postprocess.json")
    replot_script_relative = write_pdos_replot_helper_script(workflow_root)
    metadata = {
        "prefix": prefix,
        "fermi_ev": float(fermi_ev),
        "source_data": str(Path(input_relative).as_posix()),
        "atom_types": atom_types,
        "csv_dir": str(Path("data_and_plots") / "pdos_csv"),
        "csv_files": csv_files,
        "plot_files": plot_files,
        "dos_summary_csv": dos_summary_csv_relative,
        "dos_summary_plot": dos_summary_plot_relative,
        "combined_band_dos_plot": combined_plot_relative or None,
        "combined_plot_mode": combined_plot_mode or None,
        "atomic_dos_band_plot": atomic_dos_plot_relative or None,
        "band_shifted_gnu": band_shifted_relative or None,
        "fatband_all_states_csv": fatband_all_states_relative or None,
        "fatband_color_mode": fatband_color_mode,
        "kpoint_ticks": [{"x": float(x), "label": str(label)} for x, label in zip(tick_positions, tick_labels)],
        "replot_script": replot_script_relative,
        "generated_at": timestamp(),
    }
    write_json_file(workflow_root / metadata_relative, metadata)
    output_relative = str(dos_child.get("output") or "").strip()
    if output_relative:
        target = workflow_root / output_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_meta = (workflow_root / metadata_relative).resolve()
        target_meta = target.resolve()
        if source_meta != target_meta:
            shutil.copy2(source_meta, target_meta)

    append_log(log_path, f"Saved PDOS postprocess metadata: {workflow_root / metadata_relative}\n")
    return {
        "metadata": metadata_relative,
        "csv_files": csv_files,
        "plot_files": plot_files,
        "dos_summary_csv": dos_summary_csv_relative,
        "dos_summary_plot": dos_summary_plot_relative,
        "combined_band_dos_plot": combined_plot_relative or None,
        "combined_plot_mode": combined_plot_mode or None,
        "atomic_dos_band_plot": atomic_dos_plot_relative or None,
        "replot_script": replot_script_relative,
        "fermi_ev": float(fermi_ev),
    }


def write_pdos_replot_helper_script(workflow_root: Path) -> str:
    script_relative = str(Path("data_and_plots") / "replot_pdos.py")
    script_path = workflow_root / script_relative
    script_body = r'''#!/usr/bin/env python3
"""Replot saved PDOS workflow outputs.

Edit the USER CONFIGURATION block, then run:

    python replot_pdos.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_CANDIDATES = [
    Path(__PROJECT_ROOT__),
    Path(__file__).resolve().parents[2],
    Path.cwd(),
]
for candidate in ROOT_CANDIDATES:
    if (candidate / "workflow_runner.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from workflow_runner import (
    atom_orbital_pdos_from_sumpdos_payload,
    parse_bands_gnu_curves,
    write_atom_pdos_plot,
    write_band_atomic_orbital_dos_side_by_side_plot,
    write_band_dos_side_by_side_plot,
    write_fatband_orbital_dos_side_by_side_plot,
    write_total_dos_plot,
)


# ================= USER CONFIGURATION =================

METADATA_FILE = "pdos_postprocess.json"
OUTPUT_SUFFIX = ""
REPLOT_TOTAL_DOS = True
REPLOT_ATOM_PLOTS = True
REPLOT_COMBINED = True
REPLOT_ATOMIC_ORBITAL_DOS = True
ATOMS_TO_PLOT = []             # empty = all atoms from metadata
COMBINED_MODE = "auto"         # auto, fatband, bands, none

# Axis and output controls. Use None to let the renderer choose.
ENERGY_MIN = -10.0             # Energy - E_F lower limit, eV
ENERGY_MAX = 10.0              # Energy - E_F upper limit, eV
DOS_X_MIN = None               # DOS panel x-axis lower limit
DOS_X_MAX = None               # DOS panel x-axis upper limit
BAND_X_MIN = None              # band/fat-band x-axis lower limit
BAND_X_MAX = None              # band/fat-band x-axis upper limit
DPI = 300
TOTAL_DOS_FIGURE_SIZE = (5.0, 6.4)
COMBINED_FIGURE_SIZE = (11.0, 5.6)

# Plain band + DOS fallback controls.
BAND_LINE_COLOR = "#1f4ad6"
BAND_LINE_WIDTH = 0.8

# Band + atom-orbital DOS controls.
ATOMIC_ORBITAL_DOS_OUTPUT = ""  # empty = use metadata name with OUTPUT_SUFFIX
ATOMIC_ORBITAL_DOS_FIGURE_SIZE = (11.0, 5.6)
ATOMIC_ORBITAL_DOS_LEGEND_COLUMNS = 0        # 0 = auto
ATOMIC_ORBITAL_DOS_LEGEND_Y_OFFSET = 0.004   # distance above plot border
ATOMIC_ORBITAL_DOS_LEGEND_FONT_SIZE = 9.0
ATOMIC_ORBITAL_DOS_LEGEND_MARKER_SIZE = 7.0
ATOMIC_ORBITAL_DOS_LEGEND_COLUMN_SPACING = 1.1
ATOMIC_ORBITAL_DOS_LEGEND_TEXT_PAD = 0.35
ATOMIC_ORBITAL_DOS_FONT_FAMILY = "Times New Roman"
ATOMIC_ORBITAL_DOS_COLORS = {
    # "B - s": "#1f77b4",
    # "B - p": "#ff7f0e",
    # "Mg - s": "#2ca02c",
    # "Mg - p": "#d62728",
}

# Fat-band + DOS controls.
FATBAND_CONTRIBUTION_MODE = "orbitals"  # orbitals, atoms, atom-orbitals
FATBAND_COLOR_MODE = ""        # empty = use metadata
FATBAND_COLOR_SCHEME = "rgb"   # rgb or cym
FATBAND_MINIMUM_WEIGHT = 0.25
FATBAND_POINT_SCALE = 10.0
FATBAND_SIZE_POWER = 2.0
FATBAND_RIBBON_MIN_WIDTH = 0.6
FATBAND_RIBBON_SCALE = 12.0
FATBAND_RIBBON_ALPHA = 0.78
FATBAND_RIBBON_MIN_CONTRIBUTION = 0.04
FATBAND_RIBBON_HALO_WIDTH = 3.2
FATBAND_RIBBON_HALO_COLOR = "white"
FATBAND_RIBBON_HALO_ALPHA = 1.0
FATBAND_CENTER_LINE_WIDTH = 0.75
FATBAND_CENTER_LINE_ALPHA = 0.82
FATBAND_CENTER_LINE_COLOR = "#111111"

# ================= END USER CONFIGURATION =================


def data_path(value):
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def workflow_path(value):
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def read_pdos_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        orbitals = [col.replace("PDOS_", "", 1) for col in header[1:]]
        energy = []
        orbital_data = {orbital: [] for orbital in orbitals}
        for row in reader:
            if not row:
                continue
            try:
                energy.append(float(row[0]))
            except (TypeError, ValueError):
                continue
            for index, orbital in enumerate(orbitals, start=1):
                try:
                    orbital_data[orbital].append(float(row[index]))
                except (IndexError, TypeError, ValueError):
                    orbital_data[orbital].append(0.0)
    return energy, orbitals, orbital_data


def output_with_suffix(path_text):
    path = data_path(path_text)
    return path.with_name(path.stem + str(OUTPUT_SUFFIX) + path.suffix)


def selected_atoms(metadata):
    atoms = [str(atom) for atom in metadata.get("atom_types") or []]
    requested = [str(atom).strip() for atom in (ATOMS_TO_PLOT or []) if str(atom).strip()]
    return [atom for atom in atoms if not requested or atom in requested]


def main() -> int:
    metadata = json.loads(data_path(METADATA_FILE).read_text(encoding="utf-8"))
    dos_csv = metadata.get("dos_summary_csv") or "data_and_plots/density_of_states.csv"
    dos_plot = metadata.get("dos_summary_plot") or "data_and_plots/density_of_states.png"
    energy, orbitals, orbital_data = read_pdos_csv(workflow_path(dos_csv))

    if REPLOT_TOTAL_DOS:
        output = output_with_suffix(Path(dos_plot).name)
        write_total_dos_plot(
            output,
            shifted_energy=energy,
            available_orbitals=orbitals,
            orbital_data=orbital_data,
            energy_min=ENERGY_MIN,
            energy_max=ENERGY_MAX,
            dos_min=DOS_X_MIN,
            dos_max=DOS_X_MAX,
            figsize=TOTAL_DOS_FIGURE_SIZE,
            dpi=DPI,
        )
        print("Saved " + str(output))

    if REPLOT_ATOM_PLOTS:
        csv_files = metadata.get("csv_files") or {}
        plot_files = metadata.get("plot_files") or {}
        for atom in selected_atoms(metadata):
            csv_path = csv_files.get(atom)
            if not csv_path:
                continue
            atom_energy, atom_orbitals, atom_data = read_pdos_csv(workflow_path(csv_path))
            outputs = list(plot_files.get(atom) or [])
            orbital_output = output_with_suffix(Path(outputs[0]).name if outputs else f"{atom}_pdos.png")
            total_output = output_with_suffix(Path(outputs[1]).name if len(outputs) > 1 else f"{atom}_pdos_total.png")
            write_atom_pdos_plot(
                orbital_output,
                shifted_energy=atom_energy,
                available_orbitals=atom_orbitals,
                orbital_data=atom_data,
                atom=atom,
                include_total=False,
                energy_min=ENERGY_MIN,
                energy_max=ENERGY_MAX,
                dos_min=DOS_X_MIN,
                dos_max=DOS_X_MAX,
                dpi=DPI,
            )
            write_atom_pdos_plot(
                total_output,
                shifted_energy=atom_energy,
                available_orbitals=atom_orbitals,
                orbital_data=atom_data,
                atom=atom,
                include_total=True,
                energy_min=ENERGY_MIN,
                energy_max=ENERGY_MAX,
                dos_min=DOS_X_MIN,
                dos_max=DOS_X_MAX,
                dpi=DPI,
            )
            print("Saved " + str(orbital_output))
            print("Saved " + str(total_output))

    mode = str(COMBINED_MODE or "auto").strip().lower()
    if REPLOT_COMBINED and mode != "none":
        ticks = metadata.get("kpoint_ticks") or []
        tick_positions = [float(item.get("x")) for item in ticks if isinstance(item, dict) and item.get("x") is not None]
        tick_labels = [str(item.get("label") or "") for item in ticks if isinstance(item, dict) and item.get("x") is not None]
        fatband_csv = metadata.get("fatband_all_states_csv")
        band_gnu = metadata.get("band_shifted_gnu")
        if mode in {"auto", "fatband"} and fatband_csv:
            output = data_path("fatband_plus_dos" + str(OUTPUT_SUFFIX) + ".png")
            write_fatband_orbital_dos_side_by_side_plot(
                output,
                fatband_all_states_csv=workflow_path(fatband_csv),
                shifted_energy=energy,
                available_orbitals=orbitals,
                orbital_data=orbital_data,
                color_mode=FATBAND_COLOR_MODE or metadata.get("fatband_color_mode") or "atomic",
                tick_positions=tick_positions,
                tick_labels=tick_labels,
                contribution_mode=FATBAND_CONTRIBUTION_MODE,
                color_scheme=FATBAND_COLOR_SCHEME,
                energy_min=ENERGY_MIN,
                energy_max=ENERGY_MAX,
                band_x_min=BAND_X_MIN,
                band_x_max=BAND_X_MAX,
                dos_min=DOS_X_MIN,
                dos_max=DOS_X_MAX,
                minimum_weight=FATBAND_MINIMUM_WEIGHT,
                point_scale=FATBAND_POINT_SCALE,
                size_power=FATBAND_SIZE_POWER,
                ribbon_min_width=FATBAND_RIBBON_MIN_WIDTH,
                ribbon_scale=FATBAND_RIBBON_SCALE,
                ribbon_alpha=FATBAND_RIBBON_ALPHA,
                ribbon_min_contribution=FATBAND_RIBBON_MIN_CONTRIBUTION,
                ribbon_halo_width=FATBAND_RIBBON_HALO_WIDTH,
                ribbon_halo_color=FATBAND_RIBBON_HALO_COLOR,
                ribbon_halo_alpha=FATBAND_RIBBON_HALO_ALPHA,
                center_line_width=FATBAND_CENTER_LINE_WIDTH,
                center_line_alpha=FATBAND_CENTER_LINE_ALPHA,
                center_line_color=FATBAND_CENTER_LINE_COLOR,
                figsize=COMBINED_FIGURE_SIZE,
                dpi=DPI,
            )
            print("Saved " + str(output))
        elif mode in {"auto", "bands"} and band_gnu:
            curves = parse_bands_gnu_curves(workflow_path(band_gnu))
            output = data_path("band_plus_dos" + str(OUTPUT_SUFFIX) + ".png")
            write_band_dos_side_by_side_plot(
                output,
                curves=curves,
                tick_positions=tick_positions,
                tick_labels=tick_labels,
                shifted_energy=energy,
                available_orbitals=orbitals,
                orbital_data=orbital_data,
                energy_min=ENERGY_MIN,
                energy_max=ENERGY_MAX,
                band_x_min=BAND_X_MIN,
                band_x_max=BAND_X_MAX,
                dos_min=DOS_X_MIN,
                dos_max=DOS_X_MAX,
                band_line_color=BAND_LINE_COLOR,
                band_line_width=BAND_LINE_WIDTH,
                figsize=COMBINED_FIGURE_SIZE,
                dpi=DPI,
            )
            print("Saved " + str(output))

    if REPLOT_ATOMIC_ORBITAL_DOS:
        ticks = metadata.get("kpoint_ticks") or []
        tick_positions = [float(item.get("x")) for item in ticks if isinstance(item, dict) and item.get("x") is not None]
        tick_labels = [str(item.get("label") or "") for item in ticks if isinstance(item, dict) and item.get("x") is not None]
        band_gnu = metadata.get("band_shifted_gnu")
        source_data = metadata.get("source_data") or "data_and_plots/pdos_sumpdos_data.json"
        if not band_gnu:
            print("Skipped atomic-orbital DOS replot: metadata is missing band_shifted_gnu")
        elif not source_data:
            print("Skipped atomic-orbital DOS replot: metadata is missing source_data")
        else:
            sumpdos_payload = json.loads(workflow_path(source_data).read_text(encoding="utf-8"))
            atom_orbital_data = atom_orbital_pdos_from_sumpdos_payload(sumpdos_payload, selected_atoms(metadata))
            if not atom_orbital_data:
                print("Skipped atomic-orbital DOS replot: no atom/orbital PDOS channels found")
            else:
                curves = parse_bands_gnu_curves(workflow_path(band_gnu))
                if not curves:
                    print("Skipped atomic-orbital DOS replot: no band curves found in " + str(band_gnu))
                else:
                    if ATOMIC_ORBITAL_DOS_OUTPUT:
                        output = output_with_suffix(ATOMIC_ORBITAL_DOS_OUTPUT)
                    else:
                        atomic_output = metadata.get("atomic_dos_band_plot") or "data_and_plots/band_plus_atomic_dos.png"
                        output = output_with_suffix(Path(atomic_output).name)
                    write_band_atomic_orbital_dos_side_by_side_plot(
                        output,
                        curves=curves,
                        tick_positions=tick_positions,
                        tick_labels=tick_labels,
                        atom_orbital_pdos_data=atom_orbital_data,
                        energy_min=ENERGY_MIN,
                        energy_max=ENERGY_MAX,
                        band_x_min=BAND_X_MIN,
                        band_x_max=BAND_X_MAX,
                        dos_min=DOS_X_MIN,
                        dos_max=DOS_X_MAX,
                        band_line_color=BAND_LINE_COLOR,
                        band_line_width=BAND_LINE_WIDTH,
                        figsize=ATOMIC_ORBITAL_DOS_FIGURE_SIZE,
                        dpi=DPI,
                        atom_orbital_colors=ATOMIC_ORBITAL_DOS_COLORS,
                        legend_columns=ATOMIC_ORBITAL_DOS_LEGEND_COLUMNS,
                        legend_y_offset=ATOMIC_ORBITAL_DOS_LEGEND_Y_OFFSET,
                        legend_fontsize=ATOMIC_ORBITAL_DOS_LEGEND_FONT_SIZE,
                        legend_marker_size=ATOMIC_ORBITAL_DOS_LEGEND_MARKER_SIZE,
                        legend_column_spacing=ATOMIC_ORBITAL_DOS_LEGEND_COLUMN_SPACING,
                        legend_handletextpad=ATOMIC_ORBITAL_DOS_LEGEND_TEXT_PAD,
                        font_family=ATOMIC_ORBITAL_DOS_FONT_FAMILY,
                    )
                    print("Saved " + str(output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.replace("__PROJECT_ROOT__", json.dumps(str(PROJECT_ROOT)))
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o755)
    return script_relative


def run_sumpdos_for_pattern(
    *,
    input_pattern: str,
    output_name: str,
    workdir: Path,
    log_path: Path,
) -> None:
    command_base = load_qe_command_base("sumpdos.x")
    command = f"{command_base} {input_pattern} > {shlex.quote(output_name)}"
    append_log(log_path, f"Running sumpdos.x in {workdir}: {command}\n")
    run_shell_command(command, cwd=workdir, log_path=log_path)


def parse_sumpdos_dat(path: Path) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        x_value = parse_fortran_float(parts[0])
        y_value = parse_fortran_float(parts[1])
        if x_value is None or y_value is None:
            continue
        x_values.append(float(x_value))
        y_values.append(float(y_value))
    return x_values, y_values


def parse_fortran_float(text: str) -> float | None:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return float(value.replace("D", "E").replace("d", "E"))
    except ValueError:
        return None


def write_atom_pdos_csv(
    path: Path,
    *,
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["Energy(eV)"] + [f"PDOS_{orb}" for orb in available_orbitals]
        writer.writerow(header)
        for index, energy in enumerate(shifted_energy):
            row: list[Any] = [float(energy)]
            for orbital in available_orbitals:
                values = orbital_data.get(orbital) or []
                row.append(values[index] if index < len(values) else "")
            writer.writerow(row)


def write_atom_pdos_plot(
    path: Path,
    *,
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
    atom: str,
    include_total: bool,
    energy_min: float | None = None,
    energy_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    dpi: int = 300,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = {
        "s": "#1f77b4",
        "p": "#ff7f0e",
        "d": "#2ca02c",
        "f": "#d62728",
        "total": "#000000",
    }

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for orbital in available_orbitals:
        if orbital == "total" and not include_total:
            continue
        values = orbital_data.get(orbital) or []
        if not values:
            continue
        ax.plot(
            shifted_energy,
            values,
            label=orbital,
            color=color_map.get(orbital),
            linewidth=1.0,
        )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density of States (electrons/eV)")
    ax.set_title(f"PDOS for {atom}")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    if energy_min is not None or energy_max is not None:
        ax.set_xlim(energy_min, energy_max)
    if dos_min is not None or dos_max is not None:
        ax.set_ylim(dos_min, dos_max)
    ax.legend(title="Orbital", loc="upper right", frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def write_total_dos_plot(
    path: Path,
    *,
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
    energy_min: float | None = None,
    energy_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    figsize: tuple[float, float] = (5.0, 6.4),
    dpi: int = 300,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    draw_dos_panel(
        ax,
        shifted_energy=shifted_energy,
        available_orbitals=available_orbitals,
        orbital_data=orbital_data,
        show_ylabel=True,
        energy_min=energy_min,
        energy_max=energy_max,
        dos_min=dos_min,
        dos_max=dos_max,
    )
    ax.set_title("Density of States")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def draw_dos_panel(
    ax: Any,
    *,
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
    show_ylabel: bool,
    legend_loc: str = "best",
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    dos_min: float | None = None,
    dos_max: float | None = None,
) -> None:
    color_map = {
        "s": "#1f77b4",
        "p": "#2ca02c",
        "d": "#d62728",
        "f": "#9467bd",
        "total": "#000000",
    }

    ordered_labels = [label for label in available_orbitals if label == "total"] + [
        label for label in available_orbitals if label != "total"
    ]
    for orbital in ordered_labels:
        values = orbital_data.get(orbital) or []
        if not values:
            continue
        ax.plot(
            values,
            shifted_energy,
            label="Total" if orbital == "total" else orbital,
            color=color_map.get(orbital, None),
            linewidth=1.2 if orbital == "total" else 1.0,
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("DOS (states/eV)")
    if show_ylabel:
        ax.set_ylabel("Energy - E_F (eV)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    if energy_min is not None or energy_max is not None:
        ax.set_ylim(energy_min, energy_max)
    if dos_min is not None or dos_max is not None:
        ax.set_xlim(dos_min, dos_max)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc=legend_loc, fontsize=8)


def write_band_dos_side_by_side_plot(
    path: Path,
    *,
    curves: Sequence[Sequence[tuple[float, float]]],
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    band_x_min: float | None = None,
    band_x_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    band_line_color: str = "#1f4ad6",
    band_line_width: float = 0.8,
    figsize: tuple[float, float] = (11.0, 5.6),
    dpi: int = 300,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=int(dpi),
        gridspec_kw={"width_ratios": [3.4, 1.3], "wspace": 0.03},
        sharey=True,
    )

    for curve in curves:
        if not curve:
            continue
        x_vals = [float(point[0]) for point in curve]
        y_vals = [float(point[1]) for point in curve]
        ax_band.plot(x_vals, y_vals, color=band_line_color, linewidth=float(band_line_width))

    ax_band.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.8)
    for x_value in tick_positions:
        ax_band.axvline(float(x_value), color="#666666", linewidth=0.7, alpha=0.65)
    if tick_positions and tick_labels and len(tick_positions) == len(tick_labels):
        ax_band.set_xticks([float(value) for value in tick_positions])
        ax_band.set_xticklabels([str(label) for label in tick_labels])

    all_kpoints = [float(point[0]) for curve in curves for point in curve]
    if all_kpoints:
        x_min = min(all_kpoints)
        x_max = max(all_kpoints)
        if x_max > x_min:
            ax_band.set_xlim(x_min, x_max)
        ax_band.margins(x=0.0)
    if band_x_min is not None or band_x_max is not None:
        ax_band.set_xlim(band_x_min, band_x_max)

    ax_band.set_xlabel("k-path")
    ax_band.set_ylabel("Energy - E_F (eV)")
    if energy_min is not None or energy_max is not None:
        ax_band.set_ylim(energy_min, energy_max)
    ax_band.grid(False)
    ax_band.set_title("Band Structure")

    draw_dos_panel(
        ax_dos,
        shifted_energy=shifted_energy,
        available_orbitals=available_orbitals,
        orbital_data=orbital_data,
        show_ylabel=False,
        legend_loc="lower right",
        energy_min=energy_min,
        energy_max=energy_max,
        dos_min=dos_min,
        dos_max=dos_max,
    )
    ax_dos.set_title("Density of States")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def write_fatband_orbital_dos_side_by_side_plot(
    path: Path,
    *,
    fatband_all_states_csv: Path,
    shifted_energy: Sequence[float],
    available_orbitals: Sequence[str],
    orbital_data: Mapping[str, Sequence[float]],
    color_mode: str,
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    contribution_mode: str = "orbitals",
    color_scheme: str = "rgb",
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    band_x_min: float | None = None,
    band_x_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    minimum_weight: float = 0.25,
    point_scale: float = 10.0,
    size_power: float = 2.0,
    ribbon_min_width: float = 0.6,
    ribbon_scale: float = 12.0,
    ribbon_alpha: float = 0.78,
    ribbon_min_contribution: float = 0.04,
    ribbon_halo_width: float = 3.2,
    ribbon_halo_color: str = "white",
    ribbon_halo_alpha: float = 1.0,
    center_line_width: float = 0.75,
    center_line_alpha: float = 0.82,
    center_line_color: str = "#111111",
    figsize: tuple[float, float] = (11.0, 5.6),
    dpi: int = 300,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_projwfc_fatbands import (
        build_contribution_channels,
        parse_projection_csv,
        render_fatbands_axis,
    )

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=int(dpi),
        gridspec_kw={"width_ratios": [3.4, 1.3], "wspace": 0.03},
        sharey=True,
    )

    proj = parse_projection_csv(fatband_all_states_csv)
    channels = build_contribution_channels(
        proj,
        contribution_mode=str(contribution_mode or "orbitals"),
        elements=[],
        atoms=[],
        color_scheme=str(color_scheme or "rgb"),
    )

    fatband_args = argparse.Namespace(
        size_source="dominant",
        minimum_weight=float(minimum_weight),
        point_scale=float(point_scale),
        size_power=float(size_power),
        color_mode=normalize_fatband_color_mode(color_mode),
        color_scheme=str(color_scheme or "rgb"),
        plot_style="ribbons",
        ribbon_min_width=float(ribbon_min_width),
        ribbon_scale=float(ribbon_scale),
        ribbon_alpha=float(ribbon_alpha),
        ribbon_min_contribution=float(ribbon_min_contribution),
        ribbon_crossing_mode="overpass",
        ribbon_halo_width=float(ribbon_halo_width),
        ribbon_halo_color=str(ribbon_halo_color),
        ribbon_halo_alpha=float(ribbon_halo_alpha),
        center_line_width=float(center_line_width),
        center_line_alpha=float(center_line_alpha),
        center_line_color=str(center_line_color),
        line_width=1.0,
        line_alpha=0.4,
        line_color="gray",
        point_alpha=0.85,
        point_edge_width=0.55,
        filled=False,
        xlabel="k-path",
        ylabel="Energy - E_F (eV)",
        ymin=energy_min,
        ymax=energy_max,
    )
    render_fatbands_axis(
        ax_band,
        proj,
        channels,
        fatband_args,
        title="Orbital Contributions",
        k_ticks=[float(value) for value in tick_positions],
        k_labels=[str(label) for label in tick_labels],
        show_xlabel=True,
        show_ylabel=True,
    )
    ax_band.set_xlim(float(ax_band.get_xlim()[0]), float(ax_band.get_xlim()[1]))
    ax_band.margins(x=0.0)
    if band_x_min is not None or band_x_max is not None:
        ax_band.set_xlim(band_x_min, band_x_max)

    band_legend = ax_band.get_legend()
    if band_legend is not None:
        handles = list(getattr(band_legend, "legend_handles", []))
        if not handles and hasattr(band_legend, "legendHandles"):
            handles = list(getattr(band_legend, "legendHandles"))
        labels = [text.get_text() for text in band_legend.get_texts()]
        title_text = band_legend.get_title().get_text() if band_legend.get_title() else ""
        band_legend.remove()
        if handles and labels:
            ax_band.legend(
                handles,
                labels,
                title=title_text,
                loc="lower right",
                frameon=True,
                shadow=True,
                fontsize=8,
                title_fontsize=9,
            )

    draw_dos_panel(
        ax_dos,
        shifted_energy=shifted_energy,
        available_orbitals=available_orbitals,
        orbital_data=orbital_data,
        show_ylabel=False,
        legend_loc="lower right",
        energy_min=energy_min,
        energy_max=energy_max,
        dos_min=dos_min,
        dos_max=dos_max,
    )
    ax_dos.set_title("Density of States")
    if energy_min is not None or energy_max is not None:
        ax_dos.set_ylim(energy_min, energy_max)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def atom_orbital_pdos_from_sumpdos_payload(
    payload: Mapping[str, Any],
    atom_types: Sequence[str],
) -> dict[str, tuple[list[float], list[float]]]:
    """Return label -> (energy, PDOS) for each atom/orbital channel."""

    atom_data_raw = payload.get("atom_data")
    if not isinstance(atom_data_raw, Mapping):
        return {}

    plot_data: dict[str, tuple[list[float], list[float]]] = {}
    for atom in atom_types:
        atom_label = str(atom).strip()
        if not atom_label:
            continue
        atom_entry = atom_data_raw.get(atom_label)
        if not isinstance(atom_entry, Mapping):
            continue

        energy_raw = atom_entry.get("energy_ev_shifted")
        orbitals_raw = atom_entry.get("orbitals")
        pdos_raw = atom_entry.get("pdos")
        if not isinstance(energy_raw, Sequence) or isinstance(energy_raw, (str, bytes)):
            continue
        if not isinstance(orbitals_raw, Sequence) or isinstance(orbitals_raw, (str, bytes)):
            continue
        if not isinstance(pdos_raw, Mapping):
            continue

        try:
            energy = [float(value) for value in energy_raw]
        except (TypeError, ValueError):
            continue

        for orbital in orbitals_raw:
            orbital_label = str(orbital).strip()
            if not orbital_label or orbital_label.lower() == "total":
                continue
            values_raw = pdos_raw.get(orbital)
            if values_raw is None:
                values_raw = pdos_raw.get(orbital_label)
            if not isinstance(values_raw, Sequence) or isinstance(values_raw, (str, bytes)):
                continue
            try:
                values = [float(value) for value in values_raw]
            except (TypeError, ValueError):
                continue
            shared_len = min(len(energy), len(values))
            if shared_len <= 0:
                continue
            plot_data[f"{atom_label} - {orbital_label}"] = (
                energy[:shared_len],
                values[:shared_len],
            )

    return plot_data


def _bold_plot_axes(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="both",
        width=1.6,
        labelsize=10,
        direction="in",
        top=True,
        right=True,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def italic_atom_orbital_legend_label(label: str) -> str:
    text = str(label)
    if " - " not in text:
        return text
    atom, orbital = text.rsplit(" - ", 1)
    orbital = orbital.strip()
    if not orbital:
        return text
    return f"{atom} - $\\boldsymbol{{\\mathit{{{orbital}}}}}$"


def draw_atomic_orbital_dos_panel(
    ax: Any,
    *,
    atom_orbital_pdos_data: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    show_ylabel: bool,
    atom_orbital_colors: Mapping[str, str] | None = None,
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    dos_min: float | None = None,
    dos_max: float | None = None,
) -> None:
    """Draw DOS panel showing individual atom/orbital contributions."""

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    color_map = atom_orbital_colors or {}
    for label, data_pair in atom_orbital_pdos_data.items():
        if not isinstance(data_pair, Sequence) or len(data_pair) != 2:
            continue
        shifted_energy, values = data_pair
        shared_len = min(len(shifted_energy), len(values))
        if shared_len <= 0:
            continue
        ax.plot(
            [float(value) for value in values[:shared_len]],
            [float(value) for value in shifted_energy[:shared_len]],
            label=str(label),
            color=color_map.get(str(label)) or None,
            linewidth=1.2,
            alpha=0.9,
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("DOS (states/eV)", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(r"$\boldsymbol{\mathit{E - E}_{\mathit{F}}}$ (eV)", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    if energy_min is not None or energy_max is not None:
        ax.set_ylim(energy_min, energy_max)
    if dos_min is not None or dos_max is not None:
        ax.set_xlim(dos_min, dos_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))


def write_band_atomic_orbital_dos_side_by_side_plot(
    path: Path,
    *,
    curves: Sequence[Sequence[tuple[float, float]]],
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    atom_orbital_pdos_data: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    band_x_min: float | None = None,
    band_x_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    band_line_color: str = "#1f4ad6",
    band_line_width: float = 1.0,
    figsize: tuple[float, float] = (11.0, 5.6),
    dpi: int = 300,
    atom_orbital_colors: Mapping[str, str] | None = None,
    legend_columns: int | None = None,
    legend_y_offset: float = 0.004,
    legend_fontsize: float = 9.0,
    legend_marker_size: float = 7.0,
    legend_column_spacing: float = 1.1,
    legend_handletextpad: float = 0.35,
    font_family: str = "Times New Roman",
) -> None:
    """Create side-by-side plot of band structure and atom/orbital-projected DOS."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    font_name = str(font_family or "").strip() or "Times New Roman"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name, "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=int(dpi),
        gridspec_kw={"width_ratios": [3.4, 1.3], "wspace": 0.03},
        sharey=True,
    )

    for curve in curves:
        if not curve:
            continue
        x_vals = [float(point[0]) for point in curve]
        y_vals = [float(point[1]) for point in curve]
        ax_band.plot(x_vals, y_vals, color=band_line_color, linewidth=float(band_line_width))

    ax_band.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--", alpha=0.8)
    for x_value in tick_positions:
        ax_band.axvline(float(x_value), color="#666666", linewidth=0.8, alpha=0.65)
    if tick_positions and tick_labels and len(tick_positions) == len(tick_labels):
        ax_band.set_xticks([float(value) for value in tick_positions])
        ax_band.set_xticklabels([str(label) for label in tick_labels], fontweight="bold")

    all_kpoints = [float(point[0]) for curve in curves for point in curve]
    if all_kpoints:
        x_min = min(all_kpoints)
        x_max = max(all_kpoints)
        if x_max > x_min:
            ax_band.set_xlim(x_min, x_max)
        ax_band.margins(x=0.0)
    if band_x_min is not None or band_x_max is not None:
        ax_band.set_xlim(band_x_min, band_x_max)

    ax_band.set_xlabel(r"$\boldsymbol{\mathit{k}}$-path", fontweight="bold")
    ax_band.set_ylabel(r"$\boldsymbol{\mathit{E - E}_{\mathit{F}}}$ (eV)", fontweight="bold")
    if energy_min is not None or energy_max is not None:
        ax_band.set_ylim(energy_min, energy_max)
    ax_band.grid(False)

    draw_atomic_orbital_dos_panel(
        ax_dos,
        atom_orbital_pdos_data=atom_orbital_pdos_data,
        show_ylabel=False,
        atom_orbital_colors=atom_orbital_colors,
        energy_min=energy_min,
        energy_max=energy_max,
        dos_min=dos_min,
        dos_max=dos_max,
    )

    _bold_plot_axes(ax_band)
    _bold_plot_axes(ax_dos)
    dos_ticklabels = ax_dos.get_xticklabels()
    if dos_ticklabels:
        dos_ticklabels[0].set_ha("left")
        dos_ticklabels[-1].set_ha("right")

    fig.subplots_adjust(top=0.88)
    handles, labels = ax_dos.get_legend_handles_labels()
    if handles:
        circle_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=float(legend_marker_size),
                markerfacecolor=handle.get_color(),
                markeredgecolor=handle.get_color(),
                label=label,
            )
            for handle, label in zip(handles, labels)
        ]
        axes_top = max(ax_band.get_position().y1, ax_dos.get_position().y1)
        ncol = int(legend_columns) if legend_columns else min(len(labels), 6)
        ncol = max(1, min(ncol, len(labels)))
        display_labels = [italic_atom_orbital_legend_label(label) for label in labels]
        legend = fig.legend(
            circle_handles,
            display_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, axes_top + float(legend_y_offset)),
            ncol=ncol,
            frameon=False,
            fontsize=float(legend_fontsize),
            columnspacing=float(legend_column_spacing),
            handlelength=0.8,
            handletextpad=float(legend_handletextpad),
            borderaxespad=0.0,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def draw_atomic_dos_panel(
    ax: Any,
    *,
    shifted_energy: Sequence[float],
    atom_types: Sequence[str],
    atom_pdos_data: Mapping[str, Sequence[float]],
    show_ylabel: bool,
    legend_loc: str = "best",
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    dos_min: float | None = None,
    dos_max: float | None = None,
) -> None:
    """Draw DOS panel showing individual atom contributions."""
    import matplotlib.pyplot as plt

    color_map = {
        "s": "#1f77b4", "p": "#2ca02c", "d": "#d62728", "f": "#9467bd",
        "total": "#000000", "H": "#FF0000", "C": "#909090", "N": "#0055FF",
        "O": "#FF5500", "S": "#FFFF00", "Cl": "#00FF00", "Br": "#A52A2A",
        "I": "#9932CC", "F": "#00FFFF", "P": "#FFA500", "Si": "#C0C0C0",
        "Mg": "#808000", "Ca": "#808080", "Na": "#FFB6C1", "K": "#FFD700",
        "Fe": "#B22222", "Cu": "#B87333", "Zn": "#A8A9AD", "B": "#8B4513",
        "Al": "#D3D3D3", "Ni": "#CFCFCF", "Ti": "#E8E8E8", "V": "#A020F0",
        "Cr": "#C0C0C0", "Mn": "#808080", "Co": "#A9A9A9", "Ru": "#C0C0C0",
        "Zr": "#E0E0E0", "As": "#FFD700", "Be": "#C6E2FF", "Li": "#FFB6C1",
        "Rb": "#FFD700", "Cs": "#FFE4B5"
    }

    for atom in atom_types:
        values = atom_pdos_data.get(atom) or []
        if not values:
            continue
        # Use atom-specific color if available, otherwise use cycle colors
        color = color_map.get(atom, None)
        ax.plot(
            values,
            shifted_energy,
            label=str(atom),
            color=color,
            linewidth=1.0,
            alpha=0.8,
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("DOS (states/eV)")
    if show_ylabel:
        ax.set_ylabel("Energy - E_F (eV)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    if energy_min is not None or energy_max is not None:
        ax.set_ylim(energy_min, energy_max)
    if dos_min is not None or dos_max is not None:
        ax.set_xlim(dos_min, dos_max)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc=legend_loc, fontsize=7, ncol=1)


def write_band_atomic_dos_side_by_side_plot(
    path: Path,
    *,
    curves: Sequence[Sequence[tuple[float, float]]],
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    shifted_energy: Sequence[float],
    atom_types: Sequence[str],
    atom_pdos_data: Mapping[str, Sequence[float]],
    energy_min: float | None = -10.0,
    energy_max: float | None = 10.0,
    band_x_min: float | None = None,
    band_x_max: float | None = None,
    dos_min: float | None = None,
    dos_max: float | None = None,
    band_line_color: str = "#1f4ad6",
    band_line_width: float = 0.8,
    figsize: tuple[float, float] = (11.0, 5.6),
    dpi: int = 300,
) -> None:
    """Create side-by-side plot of band structure and per-atom projected DOS."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=int(dpi),
        gridspec_kw={"width_ratios": [3.4, 1.3], "wspace": 0.03},
        sharey=True,
    )

    # Plot band structure on left
    for curve in curves:
        if not curve:
            continue
        x_vals = [float(point[0]) for point in curve]
        y_vals = [float(point[1]) for point in curve]
        ax_band.plot(x_vals, y_vals, color=band_line_color, linewidth=float(band_line_width))

    ax_band.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.8)
    for x_value in tick_positions:
        ax_band.axvline(float(x_value), color="#666666", linewidth=0.7, alpha=0.65)
    if tick_positions and tick_labels and len(tick_positions) == len(tick_labels):
        ax_band.set_xticks([float(value) for value in tick_positions])
        ax_band.set_xticklabels([str(label) for label in tick_labels])

    all_kpoints = [float(point[0]) for curve in curves for point in curve]
    if all_kpoints:
        x_min = min(all_kpoints)
        x_max = max(all_kpoints)
        if x_max > x_min:
            ax_band.set_xlim(x_min, x_max)
        ax_band.margins(x=0.0)
    if band_x_min is not None or band_x_max is not None:
        ax_band.set_xlim(band_x_min, band_x_max)

    ax_band.set_xlabel("k-path")
    ax_band.set_ylabel("Energy - E_F (eV)")
    if energy_min is not None or energy_max is not None:
        ax_band.set_ylim(energy_min, energy_max)
    ax_band.grid(False)
    ax_band.set_title("Band Structure")

    # Plot per-atom DOS on right
    draw_atomic_dos_panel(
        ax_dos,
        shifted_energy=shifted_energy,
        atom_types=atom_types,
        atom_pdos_data=atom_pdos_data,
        show_ylabel=False,
        legend_loc="upper left",
        energy_min=energy_min,
        energy_max=energy_max,
        dos_min=dos_min,
        dos_max=dos_max,
    )
    ax_dos.set_title("Atomic DOS")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def run_band_workflow_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
    require_projwfc: bool | None,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
    if not children:
        raise RuntimeError(f"{task_id} has no child jobs.")
    has_projwfc_child = any(
        str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower() == "projwfc"
        for child in children
    )
    if require_projwfc is None:
        require_projwfc = has_projwfc_child
    if require_projwfc and not has_projwfc_child:
        raise RuntimeError(f"{task_id} requires a projwfc child stage.")

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    params = task.get("params") if isinstance(task.get("params"), Mapping) else {}
    band_options = params.get("band_options") if isinstance(params.get("band_options"), Mapping) else {}
    kpoint_spacing = positive_float_or_none(band_options.get("kpoint_spacing")) or 0.10
    kpoints_crystal_b = [
        str(line).rstrip()
        for line in (band_options.get("kpoints_crystal_b") or [])
        if str(line).strip()
    ]
    if len(kpoints_crystal_b) < 2:
        raise RuntimeError(f"{task_id} is missing K_POINTS crystal_b data.")

    nbnd = optional_positive_int(band_options.get("nbnd"))
    filband = str(band_options.get("filband") or "bands.out")
    prefix = band_workflow_prefix(manifest, band_options)
    outdir = "./out/"
    spin_orbit = workflow_spin_orbit_enabled(manifest)
    uses_converged = bool(params.get("uses_converged_values", True))
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=uses_converged)

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_band_workflow(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        for child in children:
            child_status = str(child.get("status") or "queued").lower()
            if child_status == "completed":
                continue
            if child_status == "failed":
                raise RuntimeError(f"{task_id} child is already failed: {child.get('id')}")

            run_band_workflow_child(
                manifest,
                state,
                task,
                child,
                base_text=base_text,
                prefix=prefix,
                outdir=outdir,
                kpoint_spacing=kpoint_spacing,
                kpoints_crystal_b=kpoints_crystal_b,
                nbnd=nbnd,
                ecutwfc=ecutwfc,
                spin_orbit=spin_orbit,
                filband=filband,
                state_path=state_path,
            )

        band_plot_files = write_band_structure_outputs(
            workflow_root,
            task,
            filband=filband,
            kpoints_crystal_b=kpoints_crystal_b,
        )
        fatband_files: dict[str, Any] = {}
        fatband_error = ""
        fatband_status = "skipped"
        inline_fatband_post = bool(require_projwfc) and not workflow_task_exists(
            state, "fatband_csv_extraction"
        ) and not workflow_task_exists(state, "fatband_plotting")
        if inline_fatband_post:
            fatband_color_mode = normalize_fatband_color_mode(band_options.get("fatband_color_mode"))
            try:
                fatband_files = write_fatband_outputs(
                    workflow_root,
                    task,
                    filband=filband,
                    kpoints_crystal_b=kpoints_crystal_b,
                    color_mode=fatband_color_mode,
                    fermi_ev=optional_float_or_none(band_plot_files.get("fermi_ev")),
                )
                fatband_status = "completed"
            except Exception as exc:
                # Fat-band plotting is optional post-processing; do not fail the whole band workflow.
                fatband_status = "failed"
                fatband_error = str(exc)
                append_log(workflow_root / "logs" / "workflow.log", f"Fat-band post-processing failed: {exc}\n")
                add_event(
                    state,
                    "task.warning",
                    {
                        "task": task_id,
                        "warning": "fatband_postprocessing_failed",
                        "message": str(exc),
                    },
                )

        task["status"] = "completed"
        task["completed"] = count_completed_children(task)
        task["planned_total"] = len(children)
        task["running_child"] = None
        task["result"] = {
            "prefix": prefix,
            "outdir": outdir,
            "kpoint_spacing": kpoint_spacing,
            "nbnd": nbnd,
            "kpoints_crystal_b": kpoints_crystal_b,
            "filband": filband,
            "child_outputs": {
                str(child.get("id") or ""): str(child.get("output") or "")
                for child in children
            },
        }
        if band_plot_files:
            task["result"]["band_structure"] = band_plot_files
        if inline_fatband_post:
            task["result"]["fatband"] = {
                "status": fatband_status,
                "color_mode": fatband_color_mode,
                **fatband_files,
            }
            if fatband_error:
                task["result"]["fatband"]["error"] = fatband_error

        results_state = state.setdefault("results", {})
        if band_plot_files:
            results_state["band_structure"] = band_plot_files
            results_state["band_structure_plot"] = band_plot_files.get("plot")
            results_state["band_structure_shifted_gnu"] = band_plot_files.get("shifted_gnu")
            results_state["band_structure_fermi_ev"] = band_plot_files.get("fermi_ev")
        if inline_fatband_post:
            results_state["fatband"] = {
                "status": fatband_status,
                "color_mode": fatband_color_mode,
                **fatband_files,
            }
            results_state["fatband_projection_csv_dir"] = fatband_files.get("csv_dir")
            results_state["fatband_plots"] = fatband_files.get("plots")
            results_state["fatband_color_mode"] = fatband_color_mode
            if fatband_error:
                results_state["fatband_error"] = fatband_error
        add_event(state, "task.completed", {"task": task_id, "prefix": prefix})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id})
        write_state(state_path, state)
        raise


def run_band_workflow_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    *,
    base_text: str,
    prefix: str,
    outdir: str,
    kpoint_spacing: float,
    kpoints_crystal_b: Sequence[str],
    nbnd: int | None,
    ecutwfc: Any,
    spin_orbit: bool,
    filband: str,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or "")
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{child.get('id')}.log")
    stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "bands_workflow")
    task["running_child"] = child.get("id")
    task["status"] = "running"
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        if stage == "scf":
            rendered = render_band_scf_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or "",
                prefix=prefix,
                kpoint_spacing=kpoint_spacing,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage in {"pwbands", "bands"}:
            rendered = render_band_calculation_input(
                base_text,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or "",
                prefix=prefix,
                kpoints_crystal_b=kpoints_crystal_b,
                nbnd=nbnd,
                ecutwfc=optional_float_or_none(ecutwfc),
                spin_orbit_required=spin_orbit,
            )
            engine = "pw.x"
        elif stage == "bandx":
            rendered = render_bandsx_input(
                prefix=prefix,
                outdir=outdir,
                filband=filband,
            )
            engine = "bands.x"
        elif stage == "projwfc":
            rendered = render_projwfc_input(
                prefix=prefix,
                outdir=outdir,
            )
            engine = "projwfc.x"
        else:
            raise RuntimeError(f"Unsupported band workflow child stage: {stage}")

        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote {task_id} {stage} input: {input_path}\n")

        command = build_qe_command(engine, input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        archive_child_files(workflow_root, child)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = {
            "stage": stage,
            "engine": engine,
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
        }
        task["completed"] = count_completed_children(task)
        task["running_child"] = None

        results = state.setdefault("results", {})
        results[f"{task_id}_{stage}_output"] = child.get("output")
        if stage == "scf":
            results["band_scf_output"] = child.get("output")
        elif stage in {"pwbands", "bands"}:
            results["band_pw_output"] = child.get("output")
        elif stage == "bandx":
            results["bandx_output"] = child.get("output")
        elif stage == "projwfc":
            results["projwfc_output"] = child.get("output")

        state["updated_at"] = timestamp()
        add_event(state, "child.completed", {"task": task_id, "child": child.get("id"), "stage": stage})
        write_state(state_path, state)
    except Exception as exc:
        archive_child_files(workflow_root, child)
        error_message = str(exc)
        child["status"] = "failed"
        child["error"] = error_message
        task["status"] = "failed"
        task["error"] = error_message
        task["running_child"] = None
        state["status"] = "failed"
        state["last_error"] = error_message
        state["updated_at"] = timestamp()
        add_event(
            state,
            "child.failed",
            {"task": task_id, "child": child.get("id"), "stage": stage, "error": error_message},
        )
        write_state(state_path, state)
        raise


def write_band_structure_outputs(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    filband: str,
    kpoints_crystal_b: Sequence[str],
) -> dict[str, Any]:
    scf_output = band_child_output_path(workflow_root, task, stage="scf")
    if scf_output is None or not scf_output.is_file():
        raise FileNotFoundError("Band workflow SCF output is missing; cannot extract Fermi level.")

    scf_text = scf_output.read_text(encoding="utf-8", errors="ignore")
    fermi_ev = extract_fermi_energy_ev_from_scf_output(scf_text)
    if fermi_ev is None:
        raise RuntimeError(
            "Could not find Fermi level in SCF output. Expected "
            "'the Fermi energy is ... ev' or 'highest occupied level (ev): ...'."
        )

    gnu_path = resolve_band_gnu_path(workflow_root, task, filband)
    if gnu_path is None or not gnu_path.is_file():
        raise FileNotFoundError(f"Band data file is missing: expected {filband}.gnu")

    curves = parse_bands_gnu_curves(gnu_path)
    if not curves:
        raise RuntimeError(f"No band data could be parsed from {gnu_path}.")
    shifted_curves = [[(k_value, energy - fermi_ev) for k_value, energy in curve] for curve in curves]

    x_reference = [point[0] for point in shifted_curves[0]] if shifted_curves and shifted_curves[0] else []
    tick_positions, tick_labels = crystal_b_ticks_from_lines(kpoints_crystal_b, x_reference)

    plot_relative = str(Path("data_and_plots") / "band_structure.png")
    shifted_relative = str(Path("data_and_plots") / "band_structure_shifted.gnu")
    metadata_relative = str(Path("data_and_plots") / "band_structure_data.json")
    plot_path = workflow_root / plot_relative
    shifted_path = workflow_root / shifted_relative
    metadata_path = workflow_root / metadata_relative

    write_shifted_band_gnu(shifted_path, shifted_curves)
    write_band_structure_plot(
        plot_path,
        shifted_curves,
        title="Band Structure",
        tick_positions=tick_positions,
        tick_labels=tick_labels,
    )
    replot_script_relative = write_band_structure_replot_helper_script(workflow_root)

    metadata = {
        "fermi_energy_ev": fermi_ev,
        "source_files": {
            "scf_output": str(scf_output.relative_to(workflow_root)),
            "bands_gnu": str(gnu_path.relative_to(workflow_root)),
        },
        "shifted_gnu": shifted_relative,
        "plot": plot_relative,
        "num_bands": len(shifted_curves),
        "num_kpoints_per_band": len(x_reference),
        "kpoint_ticks": [{"x": float(x), "label": str(label)} for x, label in zip(tick_positions, tick_labels)],
        "replot_script": replot_script_relative,
        "generated_at": timestamp(),
    }
    write_json_file(metadata_path, metadata)

    return {
        "plot": plot_relative,
        "shifted_gnu": shifted_relative,
        "metadata": metadata_relative,
        "replot_script": replot_script_relative,
        "fermi_ev": fermi_ev,
    }


def write_band_structure_replot_helper_script(workflow_root: Path) -> str:
    script_relative = str(Path("data_and_plots") / "replot_band_structure.py")
    script_path = workflow_root / script_relative
    script_body = r'''#!/usr/bin/env python3
"""Replot the saved electronic band structure.

Edit the USER CONFIGURATION block, then run:

    python replot_band_structure.py
"""
import json
import sys
from pathlib import Path

ROOT_CANDIDATES = [
    Path(__PROJECT_ROOT__),
    Path(__file__).resolve().parents[2],
    Path.cwd(),
]
for candidate in ROOT_CANDIDATES:
    if (candidate / "workflow_runner.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from workflow_runner import parse_bands_gnu_curves, write_band_structure_plot


# ================= USER CONFIGURATION =================

METADATA_FILE = "band_structure_data.json"
SHIFTED_GNU_FILE = ""              # empty = use value from metadata
OUTPUT_FILE = "band_structure.png"
TITLE = "Band Structure"

# ================= END USER CONFIGURATION =================


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def resolve_workflow_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    workflow_root = Path(__file__).resolve().parent.parent
    return workflow_root / path


def main() -> int:
    metadata = json.loads(resolve_path(METADATA_FILE).read_text(encoding="utf-8"))
    shifted_gnu = SHIFTED_GNU_FILE or metadata.get("shifted_gnu") or "band_structure_shifted.gnu"
    curves = parse_bands_gnu_curves(resolve_workflow_path(shifted_gnu))
    ticks = metadata.get("kpoint_ticks") or []
    tick_positions = [float(item.get("x")) for item in ticks if isinstance(item, dict) and item.get("x") is not None]
    tick_labels = [str(item.get("label") or "") for item in ticks if isinstance(item, dict) and item.get("x") is not None]
    write_band_structure_plot(
        resolve_path(OUTPUT_FILE),
        curves,
        title=TITLE,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
    )
    print(f"Saved {resolve_path(OUTPUT_FILE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.replace("__PROJECT_ROOT__", json.dumps(str(PROJECT_ROOT)))
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o755)
    return script_relative


def write_fatband_outputs(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    filband: str,
    kpoints_crystal_b: Sequence[str],
    color_mode: str,
    fermi_ev: float | None = None,
) -> dict[str, Any]:
    csv_outputs = write_fatband_projection_csv_outputs(
        workflow_root,
        task,
        fermi_ev=fermi_ev,
    )
    plot_outputs = write_fatband_plot_outputs(
        workflow_root,
        task,
        filband=filband,
        kpoints_crystal_b=kpoints_crystal_b,
        color_mode=color_mode,
    )
    return {
        "csv_dir": csv_outputs.get("csv_dir"),
        "all_states_csv": csv_outputs.get("all_states_csv"),
        "csv_metadata": csv_outputs.get("metadata"),
        "plots": plot_outputs.get("plots"),
        "metadata": plot_outputs.get("metadata"),
        "replot_script": plot_outputs.get("replot_script"),
        "fermi_ev": csv_outputs.get("fermi_ev"),
        "state_range": csv_outputs.get("state_range"),
        "color_mode": normalize_fatband_color_mode(color_mode),
    }


def write_fatband_projection_csv_outputs(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    fermi_ev: float | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    projwfc_output = band_child_output_path(workflow_root, task, stage="projwfc")
    if projwfc_output is None or not projwfc_output.is_file():
        raise FileNotFoundError("projwfc output is missing; cannot build fat-band data.")
    if fermi_ev is None:
        scf_output = band_child_output_path(workflow_root, task, stage="scf")
        if scf_output is None or not scf_output.is_file():
            raise FileNotFoundError("Band SCF output is missing; cannot extract Fermi level for fat-band plotting.")
        fermi_ev = extract_fermi_energy_ev_from_scf_output(
            scf_output.read_text(encoding="utf-8", errors="ignore")
        )
        if fermi_ev is None:
            raise RuntimeError(
                "Could not find Fermi level in SCF output while preparing fat-band plots."
            )

    converter_script = resolve_local_script_path(["projwfc_to_bandsd.py", "projwfc_to_bands.py"])

    state_min, state_max = projwfc_state_bounds(
        projwfc_output.read_text(encoding="utf-8", errors="ignore")
    )
    if state_min is None or state_max is None:
        raise RuntimeError("Could not detect projection state range from projwfc output.")

    csv_dir_relative = str(Path("data_and_plots") / "fatband_projection_csv")
    csv_dir = workflow_root / csv_dir_relative
    csv_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        proj_child = band_child_by_stage(task, stage="projwfc")
        log_relative = (
            str(proj_child.get("log") or "logs/bands_projwfc.log")
            if isinstance(proj_child, Mapping)
            else "logs/bands_projwfc.log"
        )
        fatband_log_path = workflow_root / log_relative
    else:
        fatband_log_path = log_path

    run_python_script(
        converter_script,
        [
            str(projwfc_output),
            "--ef",
            str(float(fermi_ev)),
            "--firststate",
            str(int(state_min)),
            "--laststate",
            str(int(state_max)),
            "--group-by-band",
            "--csv-dir",
            str(csv_dir),
        ],
        cwd=workflow_root,
        log_path=fatband_log_path,
    )

    all_states = csv_dir / "all_states.csv"
    if not all_states.is_file():
        raise RuntimeError(f"CSV extraction did not produce all_states.csv in {csv_dir}")

    metadata_relative = str(Path("data_and_plots") / "fatband_projection_data.json")
    metadata = {
        "converter_script": str(converter_script.relative_to(PROJECT_ROOT)),
        "projwfc_output": str(projwfc_output.relative_to(workflow_root)),
        "fermi_ev": float(fermi_ev),
        "state_range": {"first": int(state_min), "last": int(state_max)},
        "csv_dir": csv_dir_relative,
        "all_states_csv": str(all_states.relative_to(workflow_root)),
        "generated_at": timestamp(),
    }
    write_json_file(workflow_root / metadata_relative, metadata)

    return {
        "csv_dir": csv_dir_relative,
        "all_states_csv": str(all_states.relative_to(workflow_root)),
        "metadata": metadata_relative,
        "fermi_ev": float(fermi_ev),
        "state_range": {"first": int(state_min), "last": int(state_max)},
    }


def write_fatband_plot_outputs(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    filband: str,
    kpoints_crystal_b: Sequence[str],
    color_mode: str,
    csv_dir_relative: str = str(Path("data_and_plots") / "fatband_projection_csv"),
    log_path: Path | None = None,
) -> dict[str, Any]:
    plotter_script = resolve_local_script_path(["plot_projwfc_fatbands.py"])
    color_mode = normalize_fatband_color_mode(color_mode)
    csv_dir = workflow_root / csv_dir_relative
    all_states = csv_dir / "all_states.csv"
    if not all_states.is_file():
        raise FileNotFoundError(f"Fat-band CSV input is missing: {all_states}")
    plot_ymin = -10.0
    plot_ymax = 10.0

    if log_path is None:
        proj_child = band_child_by_stage(task, stage="projwfc")
        log_relative = (
            str(proj_child.get("log") or "logs/bands_projwfc.log")
            if isinstance(proj_child, Mapping)
            else "logs/bands_projwfc.log"
        )
        fatband_log_path = workflow_root / log_relative
    else:
        fatband_log_path = log_path

    # Use the same high-symmetry ticks as the band plot where possible.
    k_ticks_values: list[float] = []
    k_labels_values: list[str] = []
    band_gnu = resolve_band_gnu_path(workflow_root, task, filband)
    if band_gnu is not None and band_gnu.is_file():
        curves = parse_bands_gnu_curves(band_gnu)
        x_reference = [point[0] for point in curves[0]] if curves and curves[0] else []
        k_ticks_values, k_labels_values = crystal_b_ticks_from_lines(kpoints_crystal_b, x_reference)

    k_ticks_arg = ",".join(f"{float(value):.10g}" for value in k_ticks_values)
    k_labels_arg = ",".join(str(label) for label in k_labels_values)
    append_log(fatband_log_path, "Using fixed fat-band y-limits: -10.0 to 10.0 eV.\n")

    contribution_modes = [
        ("orbitals", "fatband_orbital_contributions", "Fat Bands: Orbital Contributions"),
        ("atoms", "fatband_atomic_contributions", "Fat Bands: Atomic Contributions"),
        ("atom-orbitals", "fatband_atom_orbital_contributions", "Fat Bands: Atom-Orbital Contributions"),
        ("atom-orbital-panels", "fatband_atom_orbital_panels", "Fat Bands: Atom-Orbital Panels"),
    ]
    plot_relatives: dict[str, str] = {}
    total_plots = len(contribution_modes)
    for index, (contribution_mode, stem, title) in enumerate(contribution_modes, start=1):
        plot_relative = str(Path("data_and_plots") / f"{stem}.png")
        plot_path = workflow_root / plot_relative
        append_log(
            fatband_log_path,
            f"[{index}/{total_plots}] Starting fat-band plot: mode={contribution_mode}, color={color_mode}, output={plot_relative}\n",
        )
        args = [
            str(csv_dir),
            "-o",
            str(plot_path),
            "--contribution-mode",
            contribution_mode,
            "--color-mode",
            color_mode,
            "--title",
            title,
            "--ymin",
            f"{plot_ymin:.10g}",
            "--ymax",
            f"{plot_ymax:.10g}",
            "--ribbon-scale",
            "12.0",
            "--ribbon-min-width",
            "0.6",
        ]
        if k_ticks_arg and k_labels_arg:
            args.extend(["--k-ticks", k_ticks_arg, "--k-labels", k_labels_arg])
        run_python_script(
            plotter_script,
            args,
            cwd=workflow_root,
            log_path=fatband_log_path,
        )
        append_log(
            fatband_log_path,
            f"[{index}/{total_plots}] Completed fat-band plot: {plot_relative}\n",
        )
        plot_relatives[contribution_mode] = plot_relative

    metadata_relative = str(Path("data_and_plots") / "fatband_plot_data.json")
    replot_script_relative = write_fatband_replot_helper_script(
        workflow_root,
        plotter_script=plotter_script,
        color_mode=color_mode,
        csv_dir_relative=csv_dir_relative,
        k_ticks_values=k_ticks_values,
        k_labels_values=k_labels_values,
        default_ymin=plot_ymin,
        default_ymax=plot_ymax,
    )
    metadata = {
        "plotter_script": str(plotter_script.relative_to(PROJECT_ROOT)),
        "color_mode": color_mode,
        "csv_dir": csv_dir_relative,
        "all_states_csv": str(all_states.relative_to(workflow_root)),
        "y_limits": {
            "ymin": plot_ymin,
            "ymax": plot_ymax,
            "adaptive": False,
            "source": "fixed_default",
        },
        "plots": plot_relatives,
        "k_ticks": k_ticks_values,
        "k_labels": k_labels_values,
        "replot_script": replot_script_relative,
        "generated_at": timestamp(),
    }
    write_json_file(workflow_root / metadata_relative, metadata)

    return {
        "plots": plot_relatives,
        "metadata": metadata_relative,
        "y_limits": {"ymin": plot_ymin, "ymax": plot_ymax},
        "replot_script": replot_script_relative,
    }


def write_fatband_replot_helper_script(
    workflow_root: Path,
    *,
    plotter_script: Path,
    color_mode: str,
    csv_dir_relative: str,
    k_ticks_values: Sequence[float],
    k_labels_values: Sequence[str],
    default_ymin: float,
    default_ymax: float,
) -> str:
    script_relative = str(Path("data_and_plots") / "replot_fatbands.py")
    script_path = workflow_root / script_relative
    data_dir = workflow_root / "data_and_plots"
    csv_dir = workflow_root / csv_dir_relative
    k_ticks_csv = ",".join(f"{float(value):.10g}" for value in k_ticks_values)
    k_labels_csv = ",".join(str(label) for label in k_labels_values)
    script_body = r'''#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ================= USER CONFIGURATION =================

PLOTTER = Path(__PLOTTER__)
CSV_DIR = Path(__CSV_DIR__)
DATA_DIR = Path(__DATA_DIR__)
COLOR_MODE = __COLOR_MODE__
YMIN = __YMIN__
YMAX = __YMAX__
DEFAULT_K_TICKS = __K_TICKS__
DEFAULT_K_LABELS = __K_LABELS__
OUTPUT_SUFFIX = ""
MODES_TO_PLOT = []          # empty = all; options: orbitals, atoms, atom-orbitals, atom-orbital-panels
RIBBON_SCALE = 12.0
RIBBON_MIN_WIDTH = 0.6

# ================= END USER CONFIGURATION =================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_plotter_path(configured: Path) -> Path:
    candidates = [configured.expanduser()]
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(parent / "plot_projwfc_fatbands.py")
    candidates.append(Path.cwd() / "plot_projwfc_fatbands.py")
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return configured


PLOTTER = resolve_plotter_path(PLOTTER)
if not DATA_DIR.is_dir():
    DATA_DIR = SCRIPT_DIR
if not CSV_DIR.is_dir():
    CSV_DIR = DATA_DIR / "fatband_projection_csv"

MODES = [
    ("orbitals", "fatband_orbital_contributions", "Fat Bands: Orbital Contributions"),
    ("atoms", "fatband_atomic_contributions", "Fat Bands: Atomic Contributions"),
    ("atom-orbitals", "fatband_atom_orbital_contributions", "Fat Bands: Atom-Orbital Contributions"),
    ("atom-orbital-panels", "fatband_atom_orbital_panels", "Fat Bands: Atom-Orbital Panels"),
]


def selected_modes(raw) -> list[tuple[str, str, str]]:
    if isinstance(raw, str):
        requested = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in (raw or []) if str(item).strip()]
    if not requested:
        return list(MODES)
    allowed = {mode for mode, _, _ in MODES}
    invalid = [mode for mode in requested if mode not in allowed]
    if invalid:
        raise ValueError("Unknown mode(s): " + ", ".join(invalid))
    return [item for item in MODES if item[0] in requested]


def main() -> int:
    if not PLOTTER.is_file():
        raise FileNotFoundError("Plot helper is missing: " + str(PLOTTER))
    if not CSV_DIR.is_dir():
        raise FileNotFoundError("Projection CSV directory is missing: " + str(CSV_DIR))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = selected_modes(MODES_TO_PLOT)
    total = len(targets)
    for index, (mode, stem, title) in enumerate(targets, start=1):
        output = DATA_DIR / (stem + str(OUTPUT_SUFFIX) + ".png")
        cmd = [
            sys.executable,
            str(PLOTTER),
            str(CSV_DIR),
            "-o",
            str(output),
            "--contribution-mode",
            mode,
            "--color-mode",
            str(COLOR_MODE),
            "--title",
            title,
            "--ribbon-scale",
            str(RIBBON_SCALE),
            "--ribbon-min-width",
            str(RIBBON_MIN_WIDTH),
        ]
        if DEFAULT_K_TICKS and DEFAULT_K_LABELS:
            cmd.extend(["--k-ticks", DEFAULT_K_TICKS, "--k-labels", DEFAULT_K_LABELS])
        cmd.extend(["--ymin", str(YMIN), "--ymax", str(YMAX)])

        print("[{}/{}] Replotting {} -> {}".format(index, total, mode, output.name))
        subprocess.run(cmd, check=True)

    print("Fat-band replot complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.replace("__PLOTTER__", json.dumps(str(plotter_script))).replace(
        "__CSV_DIR__", json.dumps(str(csv_dir))
    ).replace("__DATA_DIR__", json.dumps(str(data_dir))).replace(
        "__COLOR_MODE__", json.dumps(str(color_mode))
    ).replace("__YMIN__", f"{default_ymin:.10g}").replace(
        "__YMAX__", f"{default_ymax:.10g}"
    ).replace("__K_TICKS__", json.dumps(k_ticks_csv)).replace(
        "__K_LABELS__", json.dumps(k_labels_csv)
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o755)
    return script_relative


def band_child_output_path(workflow_root: Path, task: Mapping[str, Any], *, stage: str) -> Path | None:
    stage_key = str(stage or "").strip().lower()
    for child in task.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        child_stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()
        if child_stage != stage_key:
            continue
        relative_output = str(child.get("output") or "").strip()
        if not relative_output:
            continue
        return workflow_root / relative_output
    return None


def band_child_input_path(
    workflow_root: Path,
    task: Mapping[str, Any],
    *,
    stage: str,
) -> Path | None:
    stage_key = str(stage or "").strip().lower()
    for child in task.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        child_stage = str(
            (child.get("params") or {}).get("stage") or child.get("id") or ""
        ).strip().lower()
        if child_stage != stage_key:
            continue
        relative_input = str(child.get("input") or "").strip()
        if relative_input:
            return workflow_root / relative_input
    return None


def band_child_by_stage(task: Mapping[str, Any], *, stage: str) -> Mapping[str, Any] | None:
    stage_key = str(stage or "").strip().lower()
    for child in task.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        child_stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()
        if child_stage == stage_key:
            return child
    return None


def resolve_band_gnu_path(workflow_root: Path, task: Mapping[str, Any], filband: str) -> Path | None:
    bandx_output = band_child_output_path(workflow_root, task, stage="bandx")
    if bandx_output is not None:
        band_dir = bandx_output.parent
    else:
        band_dir = workflow_root

    filband_name = str(filband or "").strip() or "bands.out"
    filband_path = Path(filband_name).expanduser()
    base = filband_path if filband_path.is_absolute() else (band_dir / filband_path)

    candidates: list[Path] = []
    if base.suffix.lower() == ".gnu":
        candidates.append(base)
    else:
        candidates.append(Path(str(base) + ".gnu"))
        if base.suffix:
            candidates.append(base.with_suffix(base.suffix + ".gnu"))
        else:
            candidates.append(base.with_suffix(".gnu"))

    # Keep order, remove duplicates.
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate
    return unique_candidates[0] if unique_candidates else None


def resolve_local_script_path(candidates: Sequence[str]) -> Path:
    for name in candidates:
        path = (PROJECT_ROOT / str(name)).resolve()
        if path.is_file():
            return path
    joined = ", ".join(str(PROJECT_ROOT / str(name)) for name in candidates)
    raise FileNotFoundError(f"Required helper script was not found. Checked: {joined}")


def run_python_script(
    script_path: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> None:
    command_parts = [shlex.quote(sys.executable), shlex.quote(str(script_path))]
    command_parts.extend(shlex.quote(str(value)) for value in args)
    command = " ".join(command_parts)
    append_log(log_path, f"Running helper script in {cwd}: {command}\n")
    run_shell_command(command, cwd=cwd, log_path=log_path)


def projwfc_state_bounds(text: str) -> tuple[int | None, int | None]:
    matches = [int(match.group(1)) for match in re.finditer(r"state\s+#\s*(\d+)\s*:", text or "", re.IGNORECASE)]
    if not matches:
        return None, None
    return min(matches), max(matches)


def normalize_fatband_color_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"v2-basis", "v2basis", "blend", "blended", "v2-element-mix", "element-mix"}:
        return "blended"
    return "dominant"


def parse_bands_gnu_curves(path: Path) -> list[list[tuple[float, float]]]:
    curves: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                curves.append(current)
                current = []
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            k_value = float(parts[0].replace("D", "E").replace("d", "E"))
            energy = float(parts[1].replace("D", "E").replace("d", "E"))
        except ValueError:
            continue
        current.append((k_value, energy))
    if current:
        curves.append(current)
    return curves


def extract_fermi_energy_ev_from_scf_output(text: str) -> float | None:
    patterns = [
        re.compile(
            r"the\s+Fermi\s+energy\s+is\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s*ev",
            re.IGNORECASE,
        ),
        re.compile(
            r"highest\s+occupied\s+level\s*\(ev\)\s*:\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
            re.IGNORECASE,
        ),
    ]
    matches: list[tuple[int, float]] = []
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            try:
                value = float(match.group(1).replace("D", "E").replace("d", "E"))
            except (TypeError, ValueError):
                continue
            matches.append((match.start(), value))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def crystal_b_ticks_from_lines(
    lines: Sequence[str],
    x_values: Sequence[float],
) -> tuple[list[float], list[str]]:
    if not x_values:
        return [], []
    points = parse_crystal_b_points(lines)
    if len(points) < 2:
        return [], []

    tick_positions = [float(x_values[0])]
    tick_labels = [points[0]["label"]]
    index = 0
    for point_index in range(1, len(points)):
        segment_weight = max(1, int(points[point_index - 1]["weight"]))
        index = min(index + segment_weight, len(x_values) - 1)
        tick_positions.append(float(x_values[index]))
        tick_labels.append(points[point_index]["label"])
    return tick_positions, tick_labels


def parse_crystal_b_points(lines: Sequence[str]) -> list[dict[str, Any]]:
    rows = [str(line).rstrip() for line in lines if str(line).strip()]
    if not rows:
        return []
    if rows and rows[0].strip().upper().startswith("K_POINTS"):
        rows = rows[1:]
    if rows and re.fullmatch(r"\d+", rows[0].strip()):
        rows = rows[1:]

    points: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        body, _, comment = raw.partition("!")
        parts = body.split()
        if len(parts) < 4:
            continue
        try:
            weight = int(float(parts[3].replace("D", "E").replace("d", "E")))
        except ValueError:
            weight = 1
        label = normalize_kpoint_label(comment.strip()) or f"P{index + 1}"
        points.append({"label": label, "weight": max(1, weight)})
    return points


def normalize_kpoint_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    greek_map = {
        "gamma": "Γ",
        "g": "Γ",
        "delta": "Δ",
        "sigma": "Σ",
        "lambda": "Λ",
        "omega": "Ω",
        "phi": "Φ",
    }
    return greek_map.get(lowered, text.upper())


def write_shifted_band_gnu(path: Path, curves: Sequence[Sequence[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for curve_index, curve in enumerate(curves):
            for k_value, energy in curve:
                handle.write(f"{k_value:12.6f} {energy:16.8f}\n")
            if curve_index != len(curves) - 1:
                handle.write("\n")


def write_band_structure_plot(
    path: Path,
    curves: Sequence[Sequence[tuple[float, float]]],
    *,
    title: str,
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    for curve in curves:
        if not curve:
            continue
        x_vals = [float(point[0]) for point in curve]
        y_vals = [float(point[1]) for point in curve]
        ax.plot(x_vals, y_vals, color="#222222", linewidth=0.7)

    ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="-")
    for x_value in tick_positions:
        ax.axvline(float(x_value), color="#666666", linewidth=0.7, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy - E_F (eV)")
    ax.grid(False)
    if tick_positions and tick_labels and len(tick_positions) == len(tick_labels):
        ax.set_xticks([float(value) for value in tick_positions])
        ax.set_xticklabels([str(label) for label in tick_labels])

    all_kpoints = [float(point[0]) for curve in curves for point in curve]
    if all_kpoints:
        x_min = min(all_kpoints)
        x_max = max(all_kpoints)
        if x_max > x_min:
            ax.set_xlim(x_min, x_max)
        else:
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.margins(x=0.0)

    ax.set_ylim(-10.0, 10.0)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_accuracy_relaxation_task(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    task_id: str,
    structure_result_key: str,
    output_result_key: str,
    structure_filename: str,
) -> None:
    task = workflow_task_state(state, task_id)
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if not dependencies_completed(state, task):
        return

    children = task.get("children") or []
    if not children or not isinstance(children[0], dict):
        raise RuntimeError(f"{task_id} task has no child work item.")

    child = children[0]
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or "")
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or f"logs/{task_id}.log")
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    params = child.get("params") or {}
    calculation = str(params.get("calculation") or task_type_to_calculation(task.get("type")))
    uses_converged_values = bool(params.get("uses_converged_values", False))

    task["status"] = "running"
    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["status"] = "running"
    state["current_task"] = task_id
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": task_id, "child": child.get("id")})
    write_state(state_path, state)

    try:
        if not prepared_input.is_file():
            raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

        child_workdir.mkdir(parents=True, exist_ok=True)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_structure_for_relaxation(workflow_root, state, params)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
        ecutwfc = effective_workflow_ecutwfc(
            manifest,
            state,
            uses_converged_values=uses_converged_values,
        )
        kmesh = results.get("kmesh") if uses_converged_values and isinstance(results, Mapping) else None
        rendered = render_final_relaxation_input(
            base_text,
            calculation=calculation,
            workflow_root=workflow_root,
            child_workdir=child.get("workdir") or "",
            child_id=str(child.get("id") or task_id),
            ecutwfc=ecutwfc,
            kmesh=kmesh if isinstance(kmesh, list) else None,
            conversion_options=manifest.get("conversion") or {},
            relaxation_options=relaxation_parameters_for_child(manifest, child),
        )
        attempt = run_relaxation_attempt(
            rendered,
            manifest=manifest,
            input_path=input_path,
            output_path=output_path,
            child_workdir=child_workdir,
            child_log_path=child_log_path,
            task_id=task_id,
            attempt_number=1,
        )
        attempts = [relaxation_attempt_summary(attempt)]
        child["attempts"] = attempts

        if should_retry_final_relaxation(task_id, attempt):
            preserve_attempt_files(input_path, output_path, attempt_number=1, log_path=child_log_path)
            child["status"] = "retrying"
            state["updated_at"] = timestamp()
            add_event(
                state,
                "task.retry",
                {
                    "task": task_id,
                    "child": child.get("id"),
                    "message": "Final relaxation did not converge; retrying once with the latest extracted structure.",
                },
            )
            write_state(state_path, state)

            retry_base_text = render_qe_input_with_structure(rendered, attempt["relaxed_structure"])
            retry_rendered = render_final_relaxation_input(
                retry_base_text,
                calculation=calculation,
                workflow_root=workflow_root,
                child_workdir=child.get("workdir") or "",
                child_id=str(child.get("id") or task_id),
                ecutwfc=ecutwfc,
                kmesh=kmesh if isinstance(kmesh, list) else None,
                conversion_options=manifest.get("conversion") or {},
                relaxation_options=relaxation_parameters_for_child(manifest, child),
            )
            attempt = run_relaxation_attempt(
                retry_rendered,
                manifest=manifest,
                input_path=input_path,
                output_path=output_path,
                child_workdir=child_workdir,
                child_log_path=child_log_path,
                task_id=task_id,
                attempt_number=2,
            )
            attempts.append(relaxation_attempt_summary(attempt))
            child["attempts"] = attempts

        command_result = attempt["command_result"]
        relaxed_structure = attempt["relaxed_structure"]
        convergence_summary = attempt["convergence_summary"]

        if command_result["fatal_error_found"]:
            raise RuntimeError(f"QE reported a fatal error while running {task_id}.")
        if command_result["return_code"] != 0 and not relaxed_structure:
            raise RuntimeError(
                f"{task_id} command exited with {command_result['return_code']} and no structure was extracted."
            )
        if not relaxed_structure:
            raise RuntimeError(f"Could not extract relaxed structure for {task_id}.")
        if convergence_summary.get("relaxation_converged") is False:
            error_message = relaxation_failure_message(task_id, convergence_summary, attempts)
            if task_id == "final_relaxation":
                record_structure_relaxation_failure(
                    workflow_root,
                    state,
                    task,
                    child,
                    relaxed_structure,
                    convergence_summary,
                    attempts,
                    error_message,
                    manifest,
                )
            else:
                failed_structure_relative = str(Path("data_and_plots") / structure_filename)
                write_json_file(workflow_root / failed_structure_relative, relaxed_structure)
                archive_child_files(workflow_root, child)
                child["status"] = "failed"
                child["error"] = error_message
                child["relaxation_converged"] = False
                child["failure_reason"] = "structure_relaxation_failed"
                task["status"] = "failed"
                task["error"] = error_message
                task["running_child"] = None
                task["result"] = {
                    "structure_relaxation_failed": True,
                    "continued_on_failure": True,
                    "continued_structure": failed_structure_relative,
                    "relaxed_structure": failed_structure_relative,
                    "input": str(Path(str(child.get("input") or "")).as_posix()),
                    "output": str(Path(str(child.get("output") or "")).as_posix()),
                    "attempts": attempts,
                    "error": error_message,
                    **convergence_summary,
                }
                results_state = state.setdefault("results", {})
                results_state[output_result_key] = child.get("output")
                results_state[structure_result_key] = failed_structure_relative
                results_state[f"{task_id}_converged"] = False
                results_state[f"{task_id}_continued_on_failure"] = True
            state["updated_at"] = timestamp()
            add_event(
                state,
                "task.failed",
                {
                    "task": task_id,
                    "child": child.get("id"),
                    "error": error_message,
                    "continued_on_failure": True,
                },
            )
            add_event(
                state,
                "task.warning",
                {
                    "task": task_id,
                    "child": child.get("id"),
                    "message": "Relaxation did not converge; continuing with the latest extracted structure.",
                },
            )
            write_state(state_path, state)
            return
        if command_result["return_code"] != 0 and convergence_summary.get("relaxation_converged") is not True:
            raise RuntimeError(f"{task_id} command exited with {command_result['return_code']}.")

        relaxed_structure_path = workflow_root / "data_and_plots" / structure_filename
        write_json_file(relaxed_structure_path, relaxed_structure)
        structure_template_relative = ""
        if task_id != "final_relaxation":
            structure_template_relative = structure_template_path_for_task(task_id, success=True)
            write_structure_template(
                template_text,
                relaxed_structure,
                workflow_root / structure_template_relative,
                manifest=manifest,
            )
            append_log(
                child_log_path,
                f"Wrote relaxed-structure QE template: {workflow_root / structure_template_relative}\n",
            )
        cif_relative = str(Path("data_and_plots") / Path(structure_filename).with_suffix(".cif").name)
        cif_error = ""
        try:
            write_relaxed_structure_cif(relaxed_structure, workflow_root / cif_relative)
            append_log(child_log_path, f"Wrote relaxed structure CIF: {workflow_root / cif_relative}\n")
        except Exception as exc:  # noqa: BLE001 - CIF export should not invalidate a completed QE run
            cif_relative = ""
            cif_error = str(exc)
            append_log(child_log_path, f"Could not write relaxed structure CIF: {exc}\n")
        archive_child_files(workflow_root, child)

        relative_structure = str(Path("data_and_plots") / structure_filename)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["relaxation_converged"] = convergence_summary.get("relaxation_converged")
        task["status"] = "completed"
        task["completed"] = 1
        task["running_child"] = None
        task["result"] = {
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
            "relaxed_structure": relative_structure,
            "ecutwfc": ecutwfc,
            "kmesh": kmesh if isinstance(kmesh, list) else None,
            "attempts": attempts,
            **convergence_summary,
        }
        if structure_template_relative:
            task["result"]["relaxed_structure_template"] = structure_template_relative
        if cif_relative:
            task["result"]["relaxed_structure_cif"] = cif_relative
        if cif_error:
            task["result"]["relaxed_structure_cif_error"] = cif_error
        results_state = state.setdefault("results", {})
        results_state[output_result_key] = child.get("output")
        results_state[structure_result_key] = relative_structure
        if structure_template_relative:
            results_state[f"{structure_result_key}_template"] = structure_template_relative
        results_state[f"{task_id}_converged"] = convergence_summary.get("relaxation_converged")
        if cif_relative:
            results_state[f"{structure_result_key}_cif"] = cif_relative
        add_event(state, "task.completed", {"task": task_id, "child": child.get("id")})
        state["updated_at"] = timestamp()
        write_state(state_path, state)
    except Exception as exc:
        error_message = str(exc)
        if task_id == "final_relaxation":
            archive_terminal_relaxation_attempt(workflow_root, child)
        child["status"] = "failed"
        child["error"] = error_message
        task["status"] = "failed"
        task["error"] = error_message
        task["running_child"] = None
        state["status"] = "failed"
        state["last_error"] = error_message
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": task_id, "child": child.get("id"), "error": error_message})
        write_state(state_path, state)
        raise


def run_relaxation_attempt(
    rendered_input: str,
    *,
    manifest: Mapping[str, Any],
    input_path: Path,
    output_path: Path,
    child_workdir: Path,
    child_log_path: Path,
    task_id: str,
    attempt_number: int,
) -> dict[str, Any]:
    rendered_input = write_workflow_input(input_path, rendered_input, manifest)
    append_log(child_log_path, f"Wrote {task_id} attempt {attempt_number} input: {input_path}\n")

    command = build_pw_command(input_path.name, output_path.name)
    append_log(child_log_path, f"Running {task_id} attempt {attempt_number} in {child_workdir}: {command}\n")
    command_result = run_shell_command(
        command,
        cwd=child_workdir,
        log_path=child_log_path,
        allow_nonzero=True,
    )

    relaxed_structure = extract_relaxed_structure_from_qe_run(
        input_path,
        child_workdir,
        child_log_path,
    )
    if not relaxed_structure and output_path.is_file():
        relaxed_structure = extract_relaxed_structure_from_pw_output(
            output_path.read_text(encoding="utf-8", errors="ignore")
        )
    convergence_summary = summarize_relaxation_convergence(relaxed_structure)
    return {
        "attempt": attempt_number,
        "command_result": command_result,
        "relaxed_structure": relaxed_structure,
        "convergence_summary": convergence_summary,
    }


def should_retry_final_relaxation(task_id: str, attempt: Mapping[str, Any]) -> bool:
    if task_id != "final_relaxation":
        return False
    command_result = attempt.get("command_result") if isinstance(attempt.get("command_result"), Mapping) else {}
    if command_result.get("fatal_error_found"):
        return False
    relaxed_structure = attempt.get("relaxed_structure")
    if not isinstance(relaxed_structure, Mapping) or not relaxed_structure:
        return False
    convergence = attempt.get("convergence_summary")
    if not isinstance(convergence, Mapping):
        return False
    return convergence.get("relaxation_converged") is False


def preserve_attempt_files(input_path: Path, output_path: Path, *, attempt_number: int, log_path: Path) -> None:
    for path in (input_path, output_path):
        if not path.is_file():
            continue
        archived = path.with_name(f"{path.stem}.attempt{attempt_number}{path.suffix}")
        shutil.copy2(path, archived)
        append_log(log_path, f"Preserved failed attempt file: {archived}\n")


def relaxation_attempt_summary(attempt: Mapping[str, Any]) -> dict[str, Any]:
    command_result = attempt.get("command_result") if isinstance(attempt.get("command_result"), Mapping) else {}
    convergence = attempt.get("convergence_summary") if isinstance(attempt.get("convergence_summary"), Mapping) else {}
    relaxed_structure = attempt.get("relaxed_structure")
    return {
        "attempt": attempt.get("attempt"),
        "return_code": command_result.get("return_code"),
        "fatal_error_found": command_result.get("fatal_error_found"),
        "extracted_structure": bool(relaxed_structure),
        **dict(convergence),
    }


def relaxation_failure_message(
    task_id: str,
    convergence_summary: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> str:
    scf = convergence_summary.get("scf_converged")
    opt = convergence_summary.get("optimization_converged")
    details = [f"scf_converged={scf}", f"optimization_converged={opt}"]
    if attempts:
        details.append(f"attempts={len(attempts)}")
    return (
        f"{task_id} did not fully converge "
        f"({', '.join(details)}). Both scf_conv and opt_conv convergence_achieved must be true."
    )


def record_structure_relaxation_failure(
    workflow_root: Path,
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    relaxed_structure: Mapping[str, Any],
    convergence_summary: Mapping[str, Any],
    attempts: list[dict[str, Any]],
    error_message: str,
    manifest: Mapping[str, Any],
) -> None:
    failed_structure_relative = str(Path("data_and_plots") / "final_relaxation_failed_structure.json")
    write_json_file(workflow_root / failed_structure_relative, relaxed_structure)
    failed_template_relative = structure_template_path_for_task("final_relaxation", success=False)
    prepared_template = workflow_root / "source" / "QE_input.template"
    if prepared_template.is_file():
        write_structure_template(
            prepared_template.read_text(encoding="utf-8", errors="ignore"),
            relaxed_structure,
            workflow_root / failed_template_relative,
            manifest=manifest,
        )
    archive_terminal_relaxation_attempt(workflow_root, child)

    child["status"] = "failed"
    child["relaxation_converged"] = False
    child["failure_reason"] = "structure_relaxation_failed"
    child["error"] = error_message
    task["status"] = "failed"
    task["error"] = error_message
    task["running_child"] = None
    task["result"] = {
        "structure_relaxation_failed": True,
        "continued_on_failure": True,
        "continued_structure": failed_structure_relative,
        "relaxed_structure": failed_structure_relative,
        "failed_structure": failed_structure_relative,
        "failed_structure_template": failed_template_relative,
        "attempts": attempts,
        "error": error_message,
        **dict(convergence_summary),
    }
    results = state.setdefault("results", {})
    results["final_relaxation_converged"] = False
    results["final_relaxation_failed"] = True
    results["structure_relaxation_failed"] = True
    results["final_relaxed_structure"] = failed_structure_relative
    results["final_relaxation_continued_on_failure"] = True
    results["final_relaxation_output"] = child.get("output")
    results["final_relaxation_failed_structure"] = failed_structure_relative
    results["final_relaxation_failed_structure_template"] = failed_template_relative


def archive_terminal_relaxation_attempt(workflow_root: Path, child: Mapping[str, Any]) -> None:
    try:
        archive_child_files(workflow_root, child)
    except Exception:
        return


def structure_template_path_for_task(task_id: str, *, success: bool) -> str:
    if task_id == "final_relaxation":
        name = "QE_input.final_relaxed.template" if success else "QE_input.final_attempt.template"
    else:
        name = "QE_input.relaxed.template" if success else "QE_input.relaxation_attempt.template"
    return str(Path("source") / name)


def write_structure_template(
    template_text: str,
    structure_data: Mapping[str, Any],
    output_path: Path,
    *,
    manifest: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_qe_input_with_structure(template_text, structure_data)
    write_workflow_input(output_path, rendered, manifest)


def write_relaxed_structure_cif(structure_data: Mapping[str, Any], cif_path: Path) -> None:
    atoms = ase_atoms_from_relaxed_structure(structure_data)
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    from ase.io import write  # type: ignore

    write(str(cif_path), atoms, format="cif")


def ase_atoms_from_relaxed_structure(structure_data: Mapping[str, Any]) -> Any:
    from ase import Atoms  # type: ignore

    symbols, positions, position_unit = parse_structure_positions(structure_data)
    cell = parse_structure_cell(structure_data)
    if position_unit in {"crystal", "crystal_sg"}:
        return Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    if position_unit in {"bohr", "a.u.", "au"}:
        positions = [[component * BOHR_TO_ANGSTROM for component in row] for row in positions]
    elif position_unit not in {"angstrom", "ang"}:
        raise ValueError(f"Unsupported ATOMIC_POSITIONS unit for CIF export: {position_unit}")
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def parse_structure_positions(structure_data: Mapping[str, Any]) -> tuple[list[str], list[list[float]], str]:
    atomic = structure_data.get("atomic_positions") if isinstance(structure_data, Mapping) else None
    if not isinstance(atomic, Mapping):
        raise ValueError("Relaxed structure has no atomic_positions block.")
    unit = qe_card_unit(str(atomic.get("header") or "ATOMIC_POSITIONS angstrom"), default="angstrom")
    symbols: list[str] = []
    positions: list[list[float]] = []
    for raw in atomic.get("lines") or []:
        parts = str(raw).split("!", 1)[0].split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not symbols:
        raise ValueError("Relaxed structure has no atomic positions.")
    return symbols, positions, unit


def parse_structure_cell(structure_data: Mapping[str, Any]) -> list[list[float]]:
    cell = structure_data.get("cell_parameters") if isinstance(structure_data, Mapping) else None
    if not isinstance(cell, Mapping):
        raise ValueError("Relaxed structure has no cell_parameters block.")
    unit = qe_card_unit(str(cell.get("header") or "CELL_PARAMETERS angstrom"), default="angstrom")
    vectors: list[list[float]] = []
    for raw in cell.get("lines") or []:
        parts = str(raw).split("!", 1)[0].split()
        if len(parts) < 3:
            continue
        vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(vectors) == 3:
            break
    if len(vectors) != 3:
        raise ValueError("Relaxed structure cell is incomplete.")
    if unit in {"bohr", "a.u.", "au"}:
        return [[component * BOHR_TO_ANGSTROM for component in row] for row in vectors]
    if unit not in {"angstrom", "ang"}:
        raise ValueError(f"Unsupported CELL_PARAMETERS unit for CIF export: {unit}")
    return vectors


def qe_card_unit(header: str, *, default: str) -> str:
    match = re.search(r"\(([^)]+)\)", header)
    if match:
        return match.group(1).strip().lower()
    parts = header.split()
    if len(parts) > 1:
        return parts[1].strip().lower()
    return default


def run_cutoff_convergence(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    task = workflow_task_state(state, "cutoff_convergence")
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if task_satisfies_dependency(task):
        return
    if not dependencies_completed(state, task):
        return

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = "cutoff_convergence"
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": "cutoff_convergence"})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_initial_structure_for_convergence(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text

        while True:
            children = task.setdefault("children", [])
            if not children:
                raise RuntimeError("cutoff_convergence task has no child jobs.")

            progressed = False
            for child in children:
                if not isinstance(child, dict):
                    continue
                status = str(child.get("status") or "queued").lower()
                if status == "completed":
                    continue
                if status == "failed":
                    raise RuntimeError(f"Cutoff child is already failed: {child.get('id')}")

                run_cutoff_child(manifest, state, task, child, base_text, state_path)
                progressed = True

                convergence = evaluate_cutoff_convergence(task)
                if convergence.get("converged"):
                    finish_cutoff_convergence(workflow_root, task, state, convergence)
                    write_state(state_path, state)
                    add_event(
                        state,
                        "task.completed",
                        {"task": "cutoff_convergence", "ecutwfc": convergence.get("ecutwfc")},
                    )
                    write_state(state_path, state)
                    return

            convergence = evaluate_cutoff_convergence(task)
            if convergence.get("converged"):
                finish_cutoff_convergence(workflow_root, task, state, convergence)
                add_event(
                    state,
                    "task.completed",
                    {"task": "cutoff_convergence", "ecutwfc": convergence.get("ecutwfc")},
                )
                write_state(state_path, state)
                return

            if expand_cutoff_children(task):
                task["status"] = "running"
                task["planned_total"] = len(task.get("children") or [])
                state["updated_at"] = timestamp()
                add_event(
                    state,
                    "task.expanded",
                    {"task": "cutoff_convergence", "planned_total": task["planned_total"]},
                )
                write_state(state_path, state)
                continue

            if finish_cutoff_convergence_unconverged(
                workflow_root,
                task,
                state,
                convergence,
                reason="max_ecutwfc_reached",
            ):
                add_event(
                    state,
                    "task.failed",
                    {
                        "task": "cutoff_convergence",
                        "reason": "max_ecutwfc_reached",
                        "continued_with_last_tried": True,
                        "ecutwfc": convergence.get("ecutwfc"),
                    },
                )
                write_state(state_path, state)
                return

            raise RuntimeError("Cutoff convergence did not converge before reaching the maximum ecutwfc.")
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": "cutoff_convergence"})
        write_state(state_path, state)
        raise


def run_cutoff_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    base_text: str,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or "")
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or "logs/cutoff_convergence.log")
    params = child.get("params") or {}
    ecutwfc = params.get("ecutwfc")
    if ecutwfc is None:
        raise RuntimeError(f"Cutoff child has no ecutwfc value: {child.get('id')}")

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["current_task"] = "cutoff_convergence"
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": "cutoff_convergence", "child": child.get("id")})
    write_state(state_path, state)

    try:
        rendered = render_cutoff_convergence_input(
            base_text,
            ecutwfc=ecutwfc,
            workflow_root=workflow_root,
            child_workdir=child.get("workdir") or "",
            child_id=str(child.get("id") or f"ecut{ecutwfc}"),
        )
        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote cutoff convergence input: {input_path}\n")

        command = build_pw_command(input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        run_data = extract_relaxed_structure_from_qe_run(input_path, child_workdir, child_log_path)
        energy = run_data.get("total_energy_ry") if isinstance(run_data, Mapping) else None
        if energy is None and output_path.is_file():
            energy = extract_total_energy_ry_from_pw_output(output_path.read_text(encoding="utf-8", errors="ignore"))
        if energy is None:
            raise RuntimeError(f"Could not extract total energy for {child.get('id')}")

        child["energy_ry"] = float(energy)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = {
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
            "total_energy_ry": float(energy),
        }
        archive_child_files(workflow_root, child)
        task["completed"] = count_completed_children(task)
        task["running_child"] = None
        write_convergence_energy_data(workflow_root, "cutoff", task)
        state["updated_at"] = timestamp()
        add_event(
            state,
            "child.completed",
            {"task": "cutoff_convergence", "child": child.get("id"), "total_energy_ry": float(energy)},
        )
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": "cutoff_convergence", "child": child.get("id")})
        write_state(state_path, state)
        raise


def load_initial_structure_for_convergence(workflow_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    results = state.get("results") if isinstance(state, Mapping) else {}
    relative = ""
    if isinstance(results, Mapping):
        relative = str(results.get("initial_relaxed_structure") or "")
    if not relative:
        relative = "data_and_plots/initial_relaxed_structure.json"
    path = workflow_root / relative
    if not path.is_file():
        return {}
    try:
        return load_json_file(path)
    except Exception:
        return {}


def load_structure_for_relaxation(
    workflow_root: Path,
    state: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    source = str(params.get("structure_source") or "").strip().lower()
    if source == "initial_relaxation":
        return load_initial_structure_for_convergence(workflow_root, state)

    results = state.get("results") if isinstance(state, Mapping) else {}
    if isinstance(results, Mapping):
        key = f"{source}_relaxed_structure" if source else ""
        relative = str(results.get(source) or results.get(key) or "")
        if relative:
            path = workflow_root / relative
            if path.is_file():
                try:
                    return load_json_file(path)
                except Exception:
                    return {}
    return {}


def load_structure_for_band_workflow(workflow_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Load the best available relaxed structure for electronic-structure workflows."""

    results = state.get("results") if isinstance(state, Mapping) else {}
    if not isinstance(results, Mapping):
        return {}
    candidates = [
        str(results.get("final_relaxed_structure") or ""),
        str(results.get("relaxed_structure") or ""),
        str(results.get("initial_relaxed_structure") or ""),
    ]
    for relative in candidates:
        if not relative:
            continue
        path = workflow_root / relative
        if not path.is_file():
            continue
        try:
            return load_json_file(path)
        except Exception:
            continue
    return {}


def band_workflow_prefix(manifest: Mapping[str, Any], band_options: Mapping[str, Any]) -> str:
    prefix = str(band_options.get("prefix") or "").strip()
    if prefix:
        return sanitize_job_name(prefix).replace("-", "_").replace(".", "_")

    source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
    stored = str(source.get("stored_original") or "") if isinstance(source, Mapping) else ""
    source_stem = Path(stored).stem if stored else ""
    if not source_stem:
        source_stem = str(manifest.get("name") or "bands")
    return sanitize_job_name(source_stem).replace("-", "_").replace(".", "_")


def dos_workflow_prefix(
    manifest: Mapping[str, Any],
    dos_options: Mapping[str, Any],
) -> str:
    explicit = str(dos_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, dos_options)


def wannier90_workflow_prefix(
    manifest: Mapping[str, Any],
    wannier_options: Mapping[str, Any],
) -> str:
    explicit = str(wannier_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, wannier_options)


def charge_density_workflow_prefix(
    manifest: Mapping[str, Any],
    charge_density_options: Mapping[str, Any],
) -> str:
    explicit = str(charge_density_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, charge_density_options)


def fermi_surface_workflow_prefix(
    manifest: Mapping[str, Any],
    fermi_surface_options: Mapping[str, Any],
) -> str:
    explicit = str(fermi_surface_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, fermi_surface_options)


def elastic_workflow_prefix(
    manifest: Mapping[str, Any],
    elastic_options: Mapping[str, Any],
) -> str:
    explicit = str(elastic_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, elastic_options)


def phonon_workflow_prefix(
    manifest: Mapping[str, Any],
    phonon_options: Mapping[str, Any],
) -> str:
    explicit = str(phonon_options.get("prefix") or "").strip()
    if explicit:
        return sanitize_job_name(explicit).replace("-", "_").replace(".", "_")

    return band_workflow_prefix(manifest, phonon_options)


def workflow_spin_orbit_enabled(manifest: Mapping[str, Any]) -> bool:
    conversion = manifest.get("conversion") if isinstance(manifest.get("conversion"), Mapping) else {}
    value = conversion.get("spin_orbit") if isinstance(conversion, Mapping) else None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def workflow_convert_2d_structure_enabled(manifest: Mapping[str, Any]) -> bool:
    conversion = (
        manifest.get("conversion")
        if isinstance(manifest.get("conversion"), Mapping)
        else {}
    )
    value = conversion.get("convert_2d_structure") if isinstance(conversion, Mapping) else None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def write_workflow_input(
    path: Path,
    text: str,
    manifest: Mapping[str, Any],
) -> str:
    """Write a generated input after enforcing workflow-wide invariants."""

    finalized = force_2d_k_points_automatic(
        text,
        enabled=workflow_convert_2d_structure_enabled(manifest),
    )
    path.write_text(finalized, encoding="utf-8")
    return finalized


def evaluate_cutoff_convergence(task: Mapping[str, Any]) -> dict[str, Any]:
    params = task.get("params") or {}
    threshold = float(params.get("energy_delta_ry") or 0.0)
    completed = []
    for child in task.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        if str(child.get("status") or "").lower() != "completed":
            continue
        child_params = child.get("params") or {}
        ecut = child_params.get("ecutwfc")
        energy = child.get("energy_ry")
        if ecut is None or energy is None:
            continue
        completed.append((int(ecut), float(energy), str(child.get("id") or "")))
    completed.sort(key=lambda item: item[0])
    if not completed:
        return {"converged": False, "threshold_ry": threshold, "completed": completed}

    current = completed[-1]
    result = {
        "converged": False,
        "ecutwfc": current[0],
        "total_energy_ry": current[1],
        "threshold_ry": threshold,
        "completed": completed,
    }
    if len(completed) < 2:
        return result

    previous = completed[-2]
    delta = abs(current[1] - previous[1])
    result.update(
        {
            "converged": delta <= threshold,
            "previous_ecutwfc": previous[0],
            "previous_total_energy_ry": previous[1],
            "energy_delta_ry": delta,
        }
    )
    return result


def finish_cutoff_convergence(
    workflow_root: Path,
    task: dict[str, Any],
    state: dict[str, Any],
    convergence: Mapping[str, Any],
) -> None:
    completed_children = [
        child
        for child in task.get("children") or []
        if isinstance(child, dict) and str(child.get("status") or "").lower() == "completed"
    ]
    task["children"] = completed_children
    task["completed"] = len(completed_children)
    task["planned_total"] = len(completed_children)
    task["running_child"] = None
    task["status"] = "completed"
    energy_data_files = write_convergence_energy_data(workflow_root, "cutoff", task)
    task["result"] = {
        "converged": True,
        "ecutwfc": convergence.get("ecutwfc"),
        "energy_delta_ry": convergence.get("energy_delta_ry"),
        "threshold_ry": convergence.get("threshold_ry"),
        "total_energy_ry": convergence.get("total_energy_ry"),
    }
    if energy_data_files:
        task["result"]["energy_data"] = energy_data_files
    results = state.setdefault("results", {})
    results["ecutwfc"] = convergence.get("ecutwfc")
    results["cutoff_convergence"] = task["result"]
    results["cutoff_convergence_converged"] = True
    results["cutoff_convergence_failed"] = False
    state["updated_at"] = timestamp()


def finish_cutoff_convergence_unconverged(
    workflow_root: Path,
    task: dict[str, Any],
    state: dict[str, Any],
    convergence: Mapping[str, Any],
    *,
    reason: str,
) -> bool:
    ecutwfc = convergence.get("ecutwfc")
    if ecutwfc is None:
        return False

    energy_data_files = write_convergence_energy_data(workflow_root, "cutoff", task)
    task["completed"] = count_completed_children(task)
    task["planned_total"] = len(task.get("children") or [])
    task["running_child"] = None
    task["status"] = "failed"
    task["failure_reason"] = reason
    task["result"] = {
        "converged": False,
        "failure_reason": reason,
        "continued_with_last_tried": True,
        "continuation_policy": "use_last_tried_value",
        "fallback_ecutwfc": ecutwfc,
        "previous_ecutwfc": convergence.get("previous_ecutwfc"),
        "energy_delta_ry": convergence.get("energy_delta_ry"),
        "threshold_ry": convergence.get("threshold_ry"),
        "total_energy_ry": convergence.get("total_energy_ry"),
        "previous_total_energy_ry": convergence.get("previous_total_energy_ry"),
    }
    if energy_data_files:
        task["result"]["energy_data"] = energy_data_files

    results = state.setdefault("results", {})
    results["ecutwfc"] = ecutwfc
    results["cutoff_convergence"] = task["result"]
    results["cutoff_convergence_converged"] = False
    results["cutoff_convergence_failed"] = True
    state["updated_at"] = timestamp()
    return True


def write_convergence_energy_data(
    workflow_root: Path,
    convergence_kind: str,
    task: Mapping[str, Any],
) -> dict[str, str]:
    kind = str(convergence_kind or "").strip().lower()
    if kind not in {"cutoff", "kpoints"}:
        return {}

    points = convergence_energy_points(kind, task)
    if not points:
        return {}

    stem = "cutoff_convergence_energies" if kind == "cutoff" else "kpoint_convergence_energies"
    json_relative = str(Path("data_and_plots") / f"{stem}.json")
    csv_relative = str(Path("data_and_plots") / f"{stem}.csv")
    plot_relative = str(Path("data_and_plots") / f"{stem}.png")
    json_path = workflow_root / json_relative
    csv_path = workflow_root / csv_relative
    plot_path = workflow_root / plot_relative
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        "convergence_kind": kind,
        "threshold_ry": convergence_threshold_ry(task),
        "updated_at": timestamp(),
        "points": points,
    }
    write_json_file(json_path, payload)
    write_convergence_energy_csv(csv_path, kind, points)
    write_convergence_energy_plot(plot_path, kind, points, threshold_ry=payload["threshold_ry"])
    return {"json": json_relative, "csv": csv_relative, "plot": plot_relative}


def convergence_threshold_ry(task: Mapping[str, Any]) -> float | None:
    result = task.get("result") if isinstance(task.get("result"), Mapping) else {}
    for source in (result, task.get("params") if isinstance(task.get("params"), Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        value = source.get("threshold_ry")
        if value is None:
            value = source.get("energy_delta_ry")
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def convergence_energy_points(kind: str, task: Mapping[str, Any]) -> list[dict[str, Any]]:
    completed: list[tuple[Any, Mapping[str, Any]]] = []
    for index, child in enumerate(task.get("children") or []):
        if not isinstance(child, Mapping):
            continue
        if str(child.get("status") or "").lower() != "completed":
            continue
        if child.get("energy_ry") is None:
            continue
        params = child.get("params") if isinstance(child.get("params"), Mapping) else {}
        if kind == "cutoff":
            ecut = params.get("ecutwfc")
            if ecut is None:
                continue
            sort_key = int(ecut)
        else:
            kmesh = params.get("kmesh")
            if not isinstance(kmesh, list) or len(kmesh) < 3:
                continue
            sort_key = index
        completed.append((sort_key, child))

    completed.sort(key=lambda item: item[0])
    points: list[dict[str, Any]] = []
    previous_energy: float | None = None
    for step, (_, child) in enumerate(completed, start=1):
        params = child.get("params") if isinstance(child.get("params"), Mapping) else {}
        energy_ry = float(child.get("energy_ry"))
        point: dict[str, Any] = {
            "step": step,
            "child_id": child.get("id"),
            "label": child.get("label"),
            "total_energy_ry": energy_ry,
            "total_energy_ev": energy_ry * RY_TO_EV,
            "energy_delta_from_previous_ry": None,
            "energy_delta_from_previous_ev": None,
            "input": child.get("input"),
            "output": child.get("output"),
        }
        if previous_energy is not None:
            delta_ry = abs(energy_ry - previous_energy)
            point["energy_delta_from_previous_ry"] = delta_ry
            point["energy_delta_from_previous_ev"] = delta_ry * RY_TO_EV
        previous_energy = energy_ry

        if kind == "cutoff":
            point["ecutwfc"] = int(params.get("ecutwfc"))
        else:
            kmesh = [int(value) for value in (params.get("kmesh") or [])[:3]]
            point["kmesh"] = kmesh
            point["nk1"] = kmesh[0]
            point["nk2"] = kmesh[1]
            point["nk3"] = kmesh[2]
            point["total_kpoints"] = kmesh[0] * kmesh[1] * kmesh[2]
            point["kmesh_label"] = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
        points.append(point)
    return points


def write_convergence_energy_csv(path: Path, kind: str, points: list[Mapping[str, Any]]) -> None:
    if kind == "cutoff":
        headers = [
            "step",
            "child_id",
            "ecutwfc",
            "total_energy_ry",
            "total_energy_ev",
            "energy_delta_from_previous_ry",
            "energy_delta_from_previous_ev",
            "input",
            "output",
        ]
    else:
        headers = [
            "step",
            "child_id",
            "kmesh_label",
            "nk1",
            "nk2",
            "nk3",
            "total_kpoints",
            "total_energy_ry",
            "total_energy_ev",
            "energy_delta_from_previous_ry",
            "energy_delta_from_previous_ev",
            "input",
            "output",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(headers) + "\n")
        for point in points:
            values = [csv_value(point.get(header)) for header in headers]
            handle.write(",".join(values) + "\n")


def write_convergence_energy_plot(
    path: Path,
    kind: str,
    points: list[Mapping[str, Any]],
    *,
    threshold_ry: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # type: ignore

    if kind == "kpoints":
        write_kpoint_convergence_energy_plot(path, points, threshold_ry=threshold_ry)
        return

    if kind == "cutoff":
        x_values = [point.get("ecutwfc") for point in points]
        x_labels = [str(value) for value in x_values]
        x_axis = [float(value) for value in x_values]
        x_label = "ecutwfc (Ry)"
        title = "Cutoff Energy Convergence"
    else:
        return

    y_values = [float(point.get("total_energy_ry")) for point in points]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=140)
    ax.plot(x_axis, y_values, marker="o", linewidth=1.8, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Total energy (Ry)")
    ax.grid(True, alpha=0.28)
    if kind == "kpoints":
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_labels)
    for x_value, y_value in zip(x_axis, y_values):
        ax.annotate(
            f"{y_value:.6f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_kpoint_convergence_energy_plot(
    path: Path,
    points: list[Mapping[str, Any]],
    *,
    threshold_ry: float | None = None,
) -> None:
    from matplotlib import pyplot as plt  # type: ignore

    x_axis = [float(point.get("total_kpoints") or 0.0) for point in points]
    labels = [str(point.get("kmesh_label") or point.get("child_id") or "") for point in points]
    energies = [float(point.get("total_energy_ry")) for point in points]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    ax.plot(x_axis, energies, marker="o", linewidth=1.8, color="#1f77b4")
    ax.set_title("K-point Convergence")
    ax.set_xlabel("K-point mesh")
    ax.set_ylabel("Total energy (Ry)")
    ax.grid(True, alpha=0.28)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    apply_convergence_energy_ylim(ax, energies, threshold_ry=threshold_ry)
    for x_value, y_value in zip(x_axis, energies):
        ax.annotate(
            f"{y_value:.6f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

    positive_x = [value for value in x_axis if value > 0]
    if positive_x and max(positive_x) / min(positive_x) >= 4.0:
        ax.set_xscale("log")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def apply_convergence_energy_ylim(ax: Any, energies: Sequence[float], *, threshold_ry: float | None = None) -> None:
    if not energies:
        return
    minimum = min(float(value) for value in energies)
    maximum = max(float(value) for value in energies)
    center = 0.5 * (minimum + maximum)
    data_span = max(maximum - minimum, 1.0e-12)
    threshold_span = abs(float(threshold_ry or 0.0)) * 20.0
    span = max(data_span * 5.0, threshold_span, 1.0e-4)
    ax.set_ylim(center - 0.5 * span, center + 0.5 * span)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(char in text for char in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def expand_cutoff_children(task: dict[str, Any]) -> bool:
    expansion = task.get("expansion") if isinstance(task.get("expansion"), dict) else {}
    if not expansion:
        return False
    next_value = expansion.get("next_value")
    maximum = expansion.get("maximum")
    if next_value is None:
        return False

    expand_by = max(1, int(expansion.get("expand_by") or 2))
    existing = {
        int((child.get("params") or {}).get("ecutwfc"))
        for child in task.get("children") or []
        if isinstance(child, Mapping) and (child.get("params") or {}).get("ecutwfc") is not None
    }
    added = 0
    current = int(next_value)
    step = int((task.get("params") or {}).get("step") or 10)
    maximum_int = int(maximum) if maximum is not None else None
    while added < expand_by and (maximum_int is None or current <= maximum_int):
        if current not in existing:
            task.setdefault("children", []).append(build_cutoff_child(current))
            existing.add(current)
            added += 1
        current += step

    if maximum_int is not None and current > maximum_int:
        expansion["next_value"] = None
    else:
        expansion["next_value"] = current
    return added > 0


def count_completed_children(task: Mapping[str, Any]) -> int:
    return sum(
        1
        for child in task.get("children") or []
        if isinstance(child, Mapping) and str(child.get("status") or "").lower() in {"completed", "continued"}
    )


def extract_total_energy_ry_from_pw_output(text: str) -> float | None:
    matches = list(
        re.finditer(
            r"!\s+total energy\s+=\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+Ry",
            text,
        )
    )
    if not matches:
        return None
    return float(matches[-1].group(1).replace("d", "e").replace("D", "e"))


def run_kpoint_convergence(manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path) -> None:
    task = workflow_task_state(state, "kpoint_convergence")
    if task is None:
        return
    if str(task.get("status") or "").lower() == "completed":
        return
    if task_satisfies_dependency(task):
        return
    if not dependencies_completed(state, task):
        return

    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    prepared_input = workflow_root / str((manifest.get("source") or {}).get("prepared_input") or "")
    if not prepared_input.is_file():
        raise FileNotFoundError(f"Prepared workflow template is missing: {prepared_input}")

    task["status"] = "running"
    task["running_child"] = None
    state["status"] = "running"
    state["current_task"] = "kpoint_convergence"
    state["updated_at"] = timestamp()
    add_event(state, "task.started", {"task": "kpoint_convergence"})
    write_state(state_path, state)

    try:
        template_text = prepared_input.read_text(encoding="utf-8", errors="ignore")
        structure_data = load_initial_structure_for_convergence(workflow_root, state)
        base_text = render_qe_input_with_structure(template_text, structure_data) if structure_data else template_text
        ensure_kpoint_children(task, base_text)
        write_state(state_path, state)

        while True:
            children = task.setdefault("children", [])
            if not children:
                raise RuntimeError("kpoint_convergence task has no child jobs.")

            progressed = False
            for child in children:
                if not isinstance(child, dict):
                    continue
                status = str(child.get("status") or "queued").lower()
                if status == "completed":
                    continue
                if status == "failed":
                    raise RuntimeError(f"K-point child is already failed: {child.get('id')}")

                run_kpoint_child(manifest, state, task, child, base_text, state_path)
                progressed = True

                convergence = evaluate_kpoint_convergence(task)
                if convergence.get("converged"):
                    finish_kpoint_convergence(workflow_root, task, state, convergence)
                    add_event(
                        state,
                        "task.completed",
                        {"task": "kpoint_convergence", "kmesh": convergence.get("kmesh")},
                    )
                    write_state(state_path, state)
                    return

            convergence = evaluate_kpoint_convergence(task)
            if convergence.get("converged"):
                finish_kpoint_convergence(workflow_root, task, state, convergence)
                add_event(
                    state,
                    "task.completed",
                    {"task": "kpoint_convergence", "kmesh": convergence.get("kmesh")},
                )
                write_state(state_path, state)
                return

            if expand_kpoint_children(task, base_text):
                task["status"] = "running"
                task["planned_total"] = len(task.get("children") or [])
                state["updated_at"] = timestamp()
                add_event(
                    state,
                    "task.expanded",
                    {"task": "kpoint_convergence", "planned_total": task["planned_total"]},
                )
                write_state(state_path, state)
                continue

            if finish_kpoint_convergence_unconverged(
                workflow_root,
                task,
                state,
                convergence,
                reason="kpoint_mesh_limit_reached",
            ):
                add_event(
                    state,
                    "task.failed",
                    {
                        "task": "kpoint_convergence",
                        "reason": "kpoint_mesh_limit_reached",
                        "continued_with_last_tried": True,
                        "kmesh": convergence.get("kmesh"),
                    },
                )
                write_state(state_path, state)
                return

            raise RuntimeError("K-point convergence did not converge before reaching the mesh limit.")
    except Exception:
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "task.failed", {"task": "kpoint_convergence"})
        write_state(state_path, state)
        raise


def run_kpoint_child(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    task: dict[str, Any],
    child: dict[str, Any],
    base_text: str,
    state_path: Path,
) -> None:
    workflow_root = Path(str(manifest.get("workflow_root") or "")).expanduser().resolve()
    child_workdir = workflow_root / str(child.get("workdir") or "")
    input_path = workflow_root / str(child.get("input") or "")
    output_path = workflow_root / str(child.get("output") or "")
    child_log_path = workflow_root / str(child.get("log") or "logs/kpoint_convergence.log")
    params = child.get("params") or {}
    kmesh = params.get("kmesh")
    if not isinstance(kmesh, list) or len(kmesh) < 3:
        raise RuntimeError(f"K-point child has no kmesh value: {child.get('id')}")

    child_workdir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task["running_child"] = child.get("id")
    child["status"] = "running"
    state["current_task"] = "kpoint_convergence"
    state["updated_at"] = timestamp()
    add_event(state, "child.started", {"task": "kpoint_convergence", "child": child.get("id")})
    write_state(state_path, state)

    try:
        ecutwfc = effective_workflow_ecutwfc(manifest, state, uses_converged_values=True)
        rendered = render_kpoint_convergence_input(
            base_text,
            kmesh=[int(value) for value in kmesh[:3]],
            workflow_root=workflow_root,
            child_workdir=child.get("workdir") or "",
            child_id=str(child.get("id") or "kmesh"),
            ecutwfc=ecutwfc,
        )
        rendered = write_workflow_input(input_path, rendered, manifest)
        append_log(child_log_path, f"Wrote k-point convergence input: {input_path}\n")

        command = build_pw_command(input_path.name, output_path.name)
        append_log(child_log_path, f"Running in {child_workdir}: {command}\n")
        run_shell_command(command, cwd=child_workdir, log_path=child_log_path)

        run_data = extract_relaxed_structure_from_qe_run(input_path, child_workdir, child_log_path)
        energy = run_data.get("total_energy_ry") if isinstance(run_data, Mapping) else None
        if energy is None and output_path.is_file():
            energy = extract_total_energy_ry_from_pw_output(output_path.read_text(encoding="utf-8", errors="ignore"))
        if energy is None:
            raise RuntimeError(f"Could not extract total energy for {child.get('id')}")

        child["energy_ry"] = float(energy)
        child["status"] = "completed"
        child["completed_at"] = timestamp()
        child["result"] = {
            "input": str(Path(str(child.get("input") or "")).as_posix()),
            "output": str(Path(str(child.get("output") or "")).as_posix()),
            "total_energy_ry": float(energy),
        }
        archive_child_files(workflow_root, child)
        task["completed"] = count_completed_children(task)
        task["running_child"] = None
        write_convergence_energy_data(workflow_root, "kpoints", task)
        state["updated_at"] = timestamp()
        add_event(
            state,
            "child.completed",
            {"task": "kpoint_convergence", "child": child.get("id"), "total_energy_ry": float(energy)},
        )
        write_state(state_path, state)
    except Exception:
        child["status"] = "failed"
        task["status"] = "failed"
        task["running_child"] = None
        state["status"] = "failed"
        state["updated_at"] = timestamp()
        add_event(state, "child.failed", {"task": "kpoint_convergence", "child": child.get("id")})
        write_state(state_path, state)
        raise


def ensure_kpoint_children(task: dict[str, Any], base_text: str) -> None:
    children = task.setdefault("children", [])
    if children:
        task["planned_total"] = len(children)
        return
    meshes, next_spacing = plan_kpoint_meshes(task, base_text, count=int((task.get("params") or {}).get("window_size") or 5))
    children.extend(build_kpoint_child(mesh) for mesh in meshes)
    task["planned_total"] = len(children)
    expansion = task.setdefault("expansion", {})
    if isinstance(expansion, dict):
        expansion["next_spacing"] = next_spacing


def plan_kpoint_meshes(
    task: Mapping[str, Any],
    base_text: str,
    *,
    count: int,
) -> tuple[list[list[int]], float | None]:
    params = task.get("params") or {}
    expansion = task.get("expansion") if isinstance(task.get("expansion"), Mapping) else {}
    spacing = positive_float_or_none(expansion.get("next_spacing")) or positive_float_or_none(params.get("start_spacing"))
    if spacing is None:
        spacing = infer_kpoint_spacing_from_qe_input(base_text) or 0.30
    step = max(float(params.get("spacing_step") or 0.025), 1.0e-8)
    max_mesh_size = int(params.get("max_mesh_size") or 14000)
    existing = {
        tuple((child.get("params") or {}).get("kmesh") or [])
        for child in task.get("children") or []
        if isinstance(child, Mapping)
    }
    meshes: list[list[int]] = []
    attempts = 0
    while len(meshes) < count and spacing > 0 and attempts < 10000:
        attempts += 1
        mesh = kmesh_from_qe_input(base_text, spacing)
        if mesh is None:
            break
        mesh_tuple = tuple(int(value) for value in mesh[:3])
        mesh_size = mesh_tuple[0] * mesh_tuple[1] * mesh_tuple[2]
        if mesh_size > max_mesh_size:
            return meshes, None
        if mesh_tuple not in existing:
            meshes.append([mesh_tuple[0], mesh_tuple[1], mesh_tuple[2]])
            existing.add(mesh_tuple)
        spacing -= step
    return meshes, spacing if spacing > 0 else None


def expand_kpoint_children(task: dict[str, Any], base_text: str) -> bool:
    expansion = task.get("expansion") if isinstance(task.get("expansion"), dict) else {}
    if task.get("children") and expansion.get("next_spacing") is None:
        return False
    expand_by = max(1, int((task.get("expansion") or {}).get("expand_by") or (task.get("params") or {}).get("expand_by") or 2))
    meshes, next_spacing = plan_kpoint_meshes(task, base_text, count=expand_by)
    if not meshes:
        if isinstance(task.get("expansion"), dict):
            task["expansion"]["next_spacing"] = next_spacing
        return False
    task.setdefault("children", []).extend(build_kpoint_child(mesh) for mesh in meshes)
    expansion = task.setdefault("expansion", {})
    if isinstance(expansion, dict):
        expansion["next_spacing"] = next_spacing
    return True


def infer_kpoint_spacing_from_qe_input(text: str) -> float | None:
    mesh = extract_automatic_kmesh(text)
    if mesh is None:
        return None
    cell = extract_cell_vectors(text)
    reciprocal = reciprocal_lengths_2pi(cell) if cell is not None else None
    if reciprocal is None:
        return None
    return max(float(length) / max(int(n), 1) for length, n in zip(reciprocal, mesh))


def extract_automatic_kmesh(text: str) -> list[int] | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.strip().upper().startswith("K_POINTS"):
            continue
        if "automatic" not in line.lower() or idx + 1 >= len(lines):
            return None
        parts = lines[idx + 1].split()
        if len(parts) < 3:
            return None
        try:
            return [max(1, int(parts[0])), max(1, int(parts[1])), max(1, int(parts[2]))]
        except ValueError:
            return None
    return None


def evaluate_kpoint_convergence(task: Mapping[str, Any]) -> dict[str, Any]:
    params = task.get("params") or {}
    threshold = float(params.get("energy_delta_ry") or 0.0)
    completed = []
    for child in task.get("children") or []:
        if not isinstance(child, Mapping):
            continue
        if str(child.get("status") or "").lower() != "completed":
            continue
        child_params = child.get("params") or {}
        kmesh = child_params.get("kmesh")
        energy = child.get("energy_ry")
        if not isinstance(kmesh, list) or len(kmesh) < 3 or energy is None:
            continue
        completed.append(([int(kmesh[0]), int(kmesh[1]), int(kmesh[2])], float(energy), str(child.get("id") or "")))
    if not completed:
        return {"converged": False, "threshold_ry": threshold, "completed": completed}

    current = completed[-1]
    result = {
        "converged": False,
        "kmesh": current[0],
        "total_energy_ry": current[1],
        "threshold_ry": threshold,
        "completed": completed,
    }
    if len(completed) < 2:
        return result

    previous = completed[-2]
    delta = abs(current[1] - previous[1])
    result.update(
        {
            "converged": delta <= threshold,
            "previous_kmesh": previous[0],
            "previous_total_energy_ry": previous[1],
            "energy_delta_ry": delta,
        }
    )
    return result


def finish_kpoint_convergence(
    workflow_root: Path,
    task: dict[str, Any],
    state: dict[str, Any],
    convergence: Mapping[str, Any],
) -> None:
    completed_children = [
        child
        for child in task.get("children") or []
        if isinstance(child, dict) and str(child.get("status") or "").lower() == "completed"
    ]
    task["children"] = completed_children
    task["completed"] = len(completed_children)
    task["planned_total"] = len(completed_children)
    task["running_child"] = None
    task["status"] = "completed"
    energy_data_files = write_convergence_energy_data(workflow_root, "kpoints", task)
    task["result"] = {
        "converged": True,
        "kmesh": convergence.get("kmesh"),
        "energy_delta_ry": convergence.get("energy_delta_ry"),
        "threshold_ry": convergence.get("threshold_ry"),
        "total_energy_ry": convergence.get("total_energy_ry"),
    }
    if energy_data_files:
        task["result"]["energy_data"] = energy_data_files
    results = state.setdefault("results", {})
    results["kmesh"] = convergence.get("kmesh")
    results["kpoint_convergence"] = task["result"]
    results["kpoint_convergence_converged"] = True
    results["kpoint_convergence_failed"] = False
    state["updated_at"] = timestamp()


def finish_kpoint_convergence_unconverged(
    workflow_root: Path,
    task: dict[str, Any],
    state: dict[str, Any],
    convergence: Mapping[str, Any],
    *,
    reason: str,
) -> bool:
    kmesh = convergence.get("kmesh")
    if not isinstance(kmesh, list) or len(kmesh) < 3:
        return False

    energy_data_files = write_convergence_energy_data(workflow_root, "kpoints", task)
    task["completed"] = count_completed_children(task)
    task["planned_total"] = len(task.get("children") or [])
    task["running_child"] = None
    task["status"] = "failed"
    task["failure_reason"] = reason
    task["result"] = {
        "converged": False,
        "failure_reason": reason,
        "continued_with_last_tried": True,
        "continuation_policy": "use_last_tried_value",
        "fallback_kmesh": kmesh,
        "previous_kmesh": convergence.get("previous_kmesh"),
        "energy_delta_ry": convergence.get("energy_delta_ry"),
        "threshold_ry": convergence.get("threshold_ry"),
        "total_energy_ry": convergence.get("total_energy_ry"),
        "previous_total_energy_ry": convergence.get("previous_total_energy_ry"),
    }
    if energy_data_files:
        task["result"]["energy_data"] = energy_data_files

    results = state.setdefault("results", {})
    results["kmesh"] = kmesh
    results["kpoint_convergence"] = task["result"]
    results["kpoint_convergence_converged"] = False
    results["kpoint_convergence_failed"] = True
    state["updated_at"] = timestamp()
    return True


def positive_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def optional_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def workflow_manual_ecutwfc(manifest: Mapping[str, Any]) -> float | None:
    conversion = manifest.get("conversion") if isinstance(manifest.get("conversion"), Mapping) else {}
    return manual_ecutwfc_from_options(conversion)


def effective_workflow_ecutwfc(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    uses_converged_values: bool = True,
) -> float | None:
    manual = workflow_manual_ecutwfc(manifest)
    if manual is not None:
        return manual
    if not uses_converged_values:
        return None
    results = state.get("results") if isinstance(state.get("results"), Mapping) else {}
    return optional_float_or_none(results.get("ecutwfc"))


def workflow_notices(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    notices: list[str] = []
    for source in (manifest.get("notices"), state.get("notices")):
        if not isinstance(source, list):
            continue
        for notice in source:
            text = str(notice or "").strip()
            if text and text not in notices:
                notices.append(text)
    return notices


def optional_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def optional_nonnegative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def relaxation_parameters_for_child(
    manifest: Mapping[str, Any],
    child: Mapping[str, Any],
) -> Mapping[str, Any]:
    params = child.get("params") if isinstance(child.get("params"), Mapping) else {}
    child_options = params.get("relaxation_parameters") if isinstance(params, Mapping) else None
    if isinstance(child_options, Mapping) and child_options:
        return child_options
    relaxation = manifest.get("relaxation") if isinstance(manifest.get("relaxation"), Mapping) else {}
    manifest_options = relaxation.get("parameters") if isinstance(relaxation, Mapping) else None
    return manifest_options if isinstance(manifest_options, Mapping) else {}


def workflow_task_state(state: Mapping[str, Any], task_id: str) -> dict[str, Any] | None:
    tasks = state.get("tasks") or {}
    if not isinstance(tasks, Mapping):
        return None
    task = tasks.get(task_id)
    return task if isinstance(task, dict) else None


def workflow_task_exists(state: Mapping[str, Any], task_id: str) -> bool:
    return workflow_task_state(state, task_id) is not None


def dependencies_completed(state: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    for dependency in task.get("depends_on") or []:
        dep_task = workflow_task_state(state, str(dependency))
        if dep_task is None or not task_satisfies_dependency(dep_task):
            return False
    return True


def task_satisfies_dependency(task: Mapping[str, Any]) -> bool:
    status = str(task.get("status") or "").lower()
    if status in {"completed", "continued"}:
        return True
    if status != "failed":
        return False

    result = task.get("result") if isinstance(task.get("result"), Mapping) else {}
    if result.get("continued_with_last_tried") or result.get("continued_on_failure"):
        return True

    task_type = str(task.get("type") or "")
    if task_type in {"convergence.cutoff", "convergence.kpoints"}:
        return bool(result.get("continued_with_last_tried"))
    return False


def task_type_to_calculation(task_type: Any) -> str:
    return "vc-relax" if str(task_type or "").strip() == "qe.vc_relax" else "relax"


def summarize_relaxation_convergence(relaxed_structure: Mapping[str, Any]) -> dict[str, Any]:
    convergence = relaxed_structure.get("convergence") if isinstance(relaxed_structure, Mapping) else {}
    if not isinstance(convergence, Mapping):
        return {}

    scf = convergence.get("scf") if isinstance(convergence.get("scf"), Mapping) else {}
    opt = convergence.get("optimization") if isinstance(convergence.get("optimization"), Mapping) else {}
    scf_converged = scf.get("convergence_achieved") if isinstance(scf, Mapping) else None
    opt_converged = opt.get("convergence_achieved") if isinstance(opt, Mapping) else None
    summary = {
        "scf_converged": scf_converged,
        "optimization_converged": opt_converged,
    }
    if scf_converged is not None and opt_converged is not None:
        summary["relaxation_converged"] = bool(scf_converged and opt_converged)
    else:
        summary["relaxation_converged"] = None
    return summary


def extract_relaxed_structure_from_qe_run(
    input_path: Path,
    workdir: Path,
    log_path: Path,
) -> dict[str, Any]:
    for xml_path in qe_xml_candidates(input_path, workdir):
        if not xml_path.is_file():
            continue
        try:
            data = extract_relaxed_structure_from_qe_xml(xml_path)
            append_log(log_path, f"Extracted relaxed structure from QE XML: {xml_path}\n")
            return data
        except Exception as exc:  # noqa: BLE001 - fall back to next XML or .out parsing
            append_log(log_path, f"Could not parse QE XML {xml_path}: {exc}\n")
    return {}


def qe_xml_candidates(input_path: Path, workdir: Path) -> list[Path]:
    prefix, outdir = qe_output_location_from_input(input_path, workdir)
    return [
        outdir / f"{prefix}.xml",
        outdir / f"{prefix}.save" / "data-file-schema.xml",
    ]


def qe_output_location_from_input(input_path: Path, workdir: Path) -> tuple[str, Path]:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    control = parse_namelist_values(text, "control")
    prefix = clean_qe_string(control.get("prefix")) or "calc"
    outdir_text = clean_qe_string(control.get("outdir")) or "."
    outdir = Path(outdir_text).expanduser()
    if not outdir.is_absolute():
        outdir = workdir / outdir
    return prefix, outdir.resolve()


def clean_qe_string(value: Any) -> str:
    text = str(value or "").strip()
    while text.endswith(","):
        text = text[:-1].strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text.strip()


def build_pw_command(input_name: str, output_name: str) -> str:
    return build_qe_command("pw.x", input_name, output_name)


def load_pw_command_base() -> str:
    return load_qe_command_base("pw.x")


def build_qe_command(executable: str, input_name: str, output_name: str) -> str:
    command_base = load_qe_command_base(executable)
    return f"{command_base} -in {shlex.quote(input_name)} | tee {shlex.quote(output_name)}"


def build_qe_stdin_command(executable: str, input_name: str, output_name: str) -> str:
    # lambda.x reads an interactive-style stdin stream and is not MPI-safe here:
    # with mpirun, only one rank consumes the input while the others hit EOF.
    command_base = (
        load_nonmpi_command_base(executable)
        if str(executable or "").strip() == "lambda.x"
        else load_qe_command_base(executable)
    )
    return f"{command_base} < {shlex.quote(input_name)} | tee {shlex.quote(output_name)}"


def build_wannier90_preprocess_command(win_name: str) -> str:
    command_base = load_nonmpi_command_base("wannier90.x")
    return f"{command_base} -pp {shlex.quote(win_name)}"


def build_pw2wannier90_run_command(input_name: str, output_name: str) -> str:
    command_base = load_mpi_command_base("pw2wannier90.x")
    return f"{command_base} -in {shlex.quote(input_name)} | tee {shlex.quote(output_name)}"


def build_wannier90_run_command(seedname: str) -> str:
    command_base = load_nonmpi_command_base("wannier90.x")
    return f"{command_base} {shlex.quote(seedname)}"


def load_qe_command_base(executable: str) -> str:
    exe = str(executable or "").strip() or "pw.x"
    prefs = load_json_file(COMMAND_PREFS_PATH) if COMMAND_PREFS_PATH.is_file() else {}
    commands = prefs.get("commands") or {}
    if isinstance(commands, Mapping):
        command = str(commands.get(exe) or "").strip()
        if command:
            return command
    try:
        system_config = load_system_config()
        if exe == "thermo_pw.x":
            bin_dir_text = str(getattr(system_config, "thermo_pw_path", "") or "").strip()
        else:
            bin_dir_text = str(getattr(system_config, "qe_path", "") or "").strip()
        if bin_dir_text:
            candidate = Path(bin_dir_text).expanduser() / exe
            return shlex.quote(str(candidate))
    except Exception:
        pass
    return exe


def truthy_runtime_option(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", ".true.", "yes", "y", "on", "enabled"}


def normalize_wannier90_kmesh(value: Any) -> list[int] | None:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = list(value)
    else:
        return None
    if len(parts) < 3:
        return None
    try:
        return [max(1, int(float(parts[index]))) for index in range(3)]
    except (TypeError, ValueError):
        return None


def load_command_preferences() -> dict[str, Any]:
    return load_json_file(COMMAND_PREFS_PATH) if COMMAND_PREFS_PATH.is_file() else {}


def load_mpi_prefix() -> str:
    prefs = load_command_preferences()
    prefix = str(prefs.get("mpirun_suffix") or "mpirun").strip()
    return prefix or "mpirun"


def load_mpi_command_base(executable: str) -> str:
    command = load_qe_command_base(executable)
    if command_starts_with_mpi(command):
        return command
    return f"{load_mpi_prefix()} {command}"


def load_nonmpi_command_base(executable: str) -> str:
    command = load_qe_command_base(executable)
    return strip_mpi_prefix(command)


def command_starts_with_mpi(command: str) -> bool:
    parts = shlex.split(str(command or ""))
    if not parts:
        return False
    return Path(parts[0]).name in {"mpirun", "mpiexec"}


def strip_mpi_prefix(command: str) -> str:
    parts = shlex.split(str(command or ""))
    if not parts:
        return command
    if Path(parts[0]).name not in {"mpirun", "mpiexec"}:
        return command
    index = 1
    options_with_values = {"-n", "-np", "--np", "--n", "-host", "--host", "-machinefile", "--machinefile"}
    while index < len(parts):
        token = parts[index]
        if token in options_with_values and index + 1 < len(parts):
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    return " ".join(shlex.quote(part) for part in parts[index:]) or command


def run_shell_command(
    command: str,
    *,
    cwd: Path,
    log_path: Path,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    bash_command = f"set -o pipefail; {command}"
    process = subprocess.Popen(
        ["/bin/bash", "-lc", bash_command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    fatal_error_found = False
    if process.stdout is not None:
        for line in process.stdout:
            append_log(log_path, line)
            if "Error in routine" in line or "MPI_ABORT" in line:
                fatal_error_found = True
    return_code = process.wait()
    result = {"return_code": return_code, "fatal_error_found": fatal_error_found}
    if not allow_nonzero and (return_code != 0 or fatal_error_found):
        raise RuntimeError(f"Command failed with exit code {return_code}: {command}")
    return result


def archive_child_files(workflow_root: Path, child: Mapping[str, Any]) -> None:
    source_paths = [
        workflow_root / str(child.get("input") or ""),
        workflow_root / str(child.get("output") or ""),
    ]
    archive_paths = [workflow_root / str(path) for path in (child.get("archive_inputs_outputs") or [])]
    for source, archive in zip(source_paths, archive_paths):
        if not source.is_file():
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archive)


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    write_json_file(path, state)


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def add_event(state: dict[str, Any], event_type: str, payload: Mapping[str, Any]) -> None:
    events = state.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        state["events"] = events
    events.append({"time": timestamp(), "type": event_type, "payload": dict(payload)})


def all_tasks_completed(state: Mapping[str, Any]) -> bool:
    tasks = state.get("tasks") or {}
    if not isinstance(tasks, Mapping):
        return True
    return all(task_satisfies_dependency(task) for task in tasks.values() if isinstance(task, Mapping))


def timestamp() -> str:
    return datetime.now().isoformat(timespec="microseconds")


if __name__ == "__main__":
    raise SystemExit(main())
