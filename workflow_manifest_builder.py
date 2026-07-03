#!/usr/bin/env python3
"""Create workflow_manifest.json and workflow_state.json from CLI options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from workflow_builder import WorkflowBuildRequest, create_workflow_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a QE Helper workflow workspace with manifest/state JSON files.",
    )
    parser.add_argument("--name", required=True, help="Workflow name.")
    parser.add_argument("--source", required=True, help="Uploaded structure or QE input file.")
    parser.add_argument("--output-dir", required=True, help="Parent directory where the workflow folder is created.")
    parser.add_argument(
        "--selected",
        action="append",
        default=[],
        help="Selected Step 3 workflow key. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--relaxation",
        default="as_is",
        choices=["as_is", "atomic_positions", "full_geometry"],
        help="Step 2 relaxation mode.",
    )
    parser.add_argument("--description", default="", help="Optional workflow description.")
    parser.add_argument("--queue", default="default", help="Queue label stored in the manifest.")
    parser.add_argument(
        "--conversion-json",
        default="",
        help="Optional JSON object with conversion settings from the Conversion tab.",
    )
    parser.add_argument(
        "--convergence-json",
        default="",
        help="Optional JSON object with full convergence settings.",
    )
    parser.add_argument(
        "--relaxation-json",
        default="",
        help="Optional JSON object with relaxation parameter settings.",
    )
    parser.add_argument(
        "--band-json",
        default="",
        help="Optional JSON object with band/fat-band settings.",
    )
    parser.add_argument(
        "--phonon-json",
        default="",
        help="Optional JSON object with phonon workflow settings.",
    )
    parser.add_argument(
        "--elastic-json",
        default="",
        help="Optional JSON object with elastic workflow settings.",
    )

    parser.add_argument("--cutoff", action="store_true", help="Enable cutoff energy convergence.")
    parser.add_argument("--ecut-start", type=int, default=None, help="Starting ecutwfc in Ry.")
    parser.add_argument("--ecut-step", type=int, default=10, help="ecutwfc increment in Ry.")
    parser.add_argument("--ecut-max", type=int, default=120, help="Maximum ecutwfc in Ry.")
    parser.add_argument("--ecut-window", type=int, default=5, help="Initial planned cutoff job window.")
    parser.add_argument("--ecut-expand-by", type=int, default=2, help="How many cutoff jobs to append when extending.")
    parser.add_argument("--ecut-energy-delta-ry", type=float, default=None, help="Cutoff convergence threshold in Ry.")
    parser.add_argument("--ecut-energy-delta-ev", type=float, default=None, help="Cutoff convergence threshold in eV.")

    parser.add_argument("--kpoints", action="store_true", help="Enable k-point convergence.")
    parser.add_argument("--bands", action="store_true", help="Enable band-structure workflow.")
    parser.add_argument("--fat-bands", action="store_true", help="Enable fat-band workflow.")
    parser.add_argument("--elastic", action="store_true", help="Enable elastic-properties workflow.")
    parser.add_argument(
        "--elastic-what",
        choices=["scf_elastic_constants", "mur_lc_elastic_constants"],
        default="",
        help="Elastic thermo_pw 'what' setting.",
    )
    parser.add_argument(
        "--elastic-algorithm",
        choices=["standard", "advanced", "energy_std", "energy"],
        default="",
        help="Elastic thermo_pw elastic_algorithm setting.",
    )
    parser.add_argument("--elastic-ngeo-strain", type=int, default=None, help="Elastic thermo_pw ngeo_strain.")
    parser.add_argument(
        "--elastic-frozen-ions",
        action="store_true",
        default=None,
        help="Enable frozen_ions in thermo_control.",
    )
    parser.add_argument("--phonon", action="store_true", help="Enable phonon workflow.")
    parser.add_argument("--kpoint-start-spacing", type=float, default=None, help="Starting k-point spacing.")
    parser.add_argument("--kpoint-spacing-step", type=float, default=0.025, help="K-point spacing reduction step.")
    parser.add_argument("--kpoint-max-mesh-size", type=int, default=14000, help="Maximum total k-point mesh size.")
    parser.add_argument("--kpoint-window", type=int, default=5, help="Initial planned k-point job window.")
    parser.add_argument("--kpoint-expand-by", type=int, default=2, help="How many k-point jobs to append when extending.")
    parser.add_argument(
        "--kpoint-mesh",
        action="append",
        default=[],
        help="Optional planned mesh, e.g. 4x4x4. Can be repeated.",
    )
    parser.add_argument("--kpoint-energy-delta-ry", type=float, default=None, help="K-point threshold in Ry.")
    parser.add_argument("--kpoint-energy-delta-ev", type=float, default=None, help="K-point threshold in eV.")

    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest/state files.")
    parser.add_argument(
        "--no-copy-source",
        action="store_true",
        help="Do not copy the selected source into the workflow source directory.",
    )

    args = parser.parse_args()

    conversion_options = parse_json_object(args.conversion_json, "--conversion-json")
    relaxation_options = parse_json_object(args.relaxation_json, "--relaxation-json")
    convergence_options = parse_json_object(args.convergence_json, "--convergence-json")
    band_options = parse_json_object(args.band_json, "--band-json")
    phonon_options = parse_json_object(args.phonon_json, "--phonon-json")
    elastic_options = parse_json_object(args.elastic_json, "--elastic-json")
    convergence_options = merge_cli_convergence_options(args, convergence_options)
    elastic_options = merge_cli_elastic_options(args, elastic_options)

    selected = list(args.selected)
    if (args.cutoff or args.kpoints) and not any("convergence" in item for item in selected):
        selected.append("convergence")
    if args.bands:
        selected.append("bands")
    if args.fat_bands:
        selected.append("fat_bands")
    if args.elastic:
        selected.append("elastic")
    if args.phonon:
        selected.append("phonon")

    request = WorkflowBuildRequest(
        name=args.name,
        output_directory=Path(args.output_dir),
        source_structure=Path(args.source),
        selected_workflows=selected,
        relaxation_mode=args.relaxation,
        description=args.description,
        queue=args.queue,
        conversion_options=conversion_options,
        relaxation_options=relaxation_options,
        convergence_options=convergence_options,
        band_options=band_options,
        elastic_options=elastic_options,
        phonon_options=phonon_options,
    )

    result = create_workflow_files(
        request,
        copy_source=not args.no_copy_source,
        overwrite=bool(args.force),
    )
    print(f"Workflow root: {result.workflow_root}")
    print(f"Manifest: {result.manifest_path}")
    print(f"State: {result.state_path}")
    return 0


def parse_json_object(raw: str, flag_name: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag_name} must be a valid JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{flag_name} must be a JSON object.")
    return payload


def merge_cli_convergence_options(args: argparse.Namespace, base: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = dict(base or {})

    cutoff = dict(options.get("cutoff") or {})
    if args.cutoff:
        cutoff.update(
            {
                "enabled": True,
                "start": args.ecut_start,
                "step": args.ecut_step,
                "maximum": args.ecut_max,
                "window_size": args.ecut_window,
                "expand_by": args.ecut_expand_by,
            }
        )
        if args.ecut_energy_delta_ry is not None:
            cutoff["energy_delta_ry"] = args.ecut_energy_delta_ry
        if args.ecut_energy_delta_ev is not None:
            cutoff["energy_delta_ev"] = args.ecut_energy_delta_ev
    options["cutoff"] = cutoff

    kpoints = dict(options.get("kpoints") or {})
    if args.kpoints:
        kpoints.update(
            {
                "enabled": True,
                "start_spacing": args.kpoint_start_spacing,
                "spacing_step": args.kpoint_spacing_step,
                "max_mesh_size": args.kpoint_max_mesh_size,
                "window_size": args.kpoint_window,
                "expand_by": args.kpoint_expand_by,
                "planned_meshes": args.kpoint_mesh,
            }
        )
        if args.kpoint_energy_delta_ry is not None:
            kpoints["energy_delta_ry"] = args.kpoint_energy_delta_ry
        if args.kpoint_energy_delta_ev is not None:
            kpoints["energy_delta_ev"] = args.kpoint_energy_delta_ev
    options["kpoints"] = kpoints

    return options


def merge_cli_elastic_options(args: argparse.Namespace, base: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = dict(base or {})
    if args.elastic_what:
        options["what"] = args.elastic_what
    if args.elastic_algorithm:
        options["elastic_algorithm"] = args.elastic_algorithm
    if args.elastic_ngeo_strain is not None:
        options["ngeo_strain"] = args.elastic_ngeo_strain
    if args.elastic_frozen_ions is not None:
        options["frozen_ions"] = bool(args.elastic_frozen_ions)
    return options


if __name__ == "__main__":
    raise SystemExit(main())
