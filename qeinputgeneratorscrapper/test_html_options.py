#!/usr/bin/env python3
"""

Discover and test option values for the Materials Cloud QE input generator script.

"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


URL = "https://qeinputgenerator.materialscloud.io/"


FIELD_MAP = {
    "fileformat": "--file-format",
    "protocol": "--protocol",
    "xc_functional": "--xc",
    "magnetization": "--magnetization",
    "refine_cell": "--refine-cell",
    "pseudo_override": "--pseudo-override",
    "kmesh_override": "--kmesh-override",
    "threshold_override": "--threshold-override",
}


def default_baseline(fileformat: str) -> dict[str, str]:
    return {
        "fileformat": fileformat,
        "protocol": "balanced",
        "xc_functional": "PBEsol",
        "magnetization": "magnetic",
        "refine_cell": "yes",
        "pseudo_override": "precision",
        "kmesh_override": "fine",
        "threshold_override": "tight",
    }


def safe_name(text: str) -> str:
    text = str(text)
    if text == "":
        text = "EMPTY_VALUE"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def get_real_options() -> dict[str, list[dict[str, str]]]:
    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: dict[str, list[dict[str, str]]] = {}

    for field in FIELD_MAP:
        select = soup.find("select", {"name": field})
        if select is None:
            print(f"WARNING: could not find <select name='{field}'>")
            results[field] = []
            continue

        options = []
        for option in select.find_all("option"):
            if option.has_attr("disabled"):
                continue

            value = option.get("value", "")
            label = option.get_text(" ", strip=True)

            options.append(
                {
                    "value": value,
                    "label": label,
                }
            )

        results[field] = options

    return results


def print_options(options_by_field: dict[str, list[dict[str, str]]]) -> None:
    print("\nReal HTML option values found:\n")

    for field, options in options_by_field.items():
        print(f"{field}:")
        if not options:
            print("  WARNING: no options found")
        for opt in options:
            value = opt["value"]
            label = opt["label"]
            print(f"  value={value!r}    label={label!r}")
        print()


def build_command(generator_script: Path, structure: Path, outdir: Path, options: dict[str, str]) -> list[str]:
    cmd = [
        "python",
        str(generator_script),
        str(structure),
        "--out",
        str(outdir),
    ]

    for field, flag in FIELD_MAP.items():
        cmd.extend([flag, str(options[field])])

    return cmd


def run_one(
    generator_script: Path,
    structure: Path,
    outdir: Path,
    options: dict[str, str],
    timeout: int = 120,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(generator_script, structure, outdir, options)

    start = time.time()

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )

    elapsed = time.time() - start

    scf_file = outdir / "scf.in"
    html_file = outdir / "result_page.html"

    success = proc.returncode == 0 and scf_file.exists()

    if success:
        reason = "OK"
    else:
        combined = proc.stdout + "\n" + proc.stderr
        if "Could not find QE input" in combined:
            reason = "No QE input in HTML response"
        elif "HTTPError" in combined:
            reason = "HTTP error"
        elif "Could not guess file format" in combined:
            reason = "Could not guess file format"
        else:
            reason = "Script failed"

    return {
        "success": success,
        "reason": reason,
        "elapsed_sec": round(elapsed, 2),
        "returncode": proc.returncode,
        "scf_exists": scf_file.exists(),
        "html_exists": html_file.exists(),
        "cmd": " ".join(shlex.quote(x) for x in cmd),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("structure", nargs="?", help="Structure file to test")
    parser.add_argument(
        "--generator-script",
        default="mc_qeinputgenerator_web_automation_client.py",
        help="Your working Materials Cloud requests script",
    )
    parser.add_argument(
        "--out",
        default="mc_real_option_tests",
        help="Output folder",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between requests",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only print real website option values, do not run tests",
    )
    parser.add_argument(
        "--baseline-file-format",
        default="cif-ase",
        help=(
            "Fileformat value to use while testing non-fileformat options. "
            "For CIF files, use cif-ase or cif-pymatgen. For POSCAR, use vasp-ase, etc."
        ),
    )
    parser.add_argument(
        "--include-file-format-tests",
        action="store_true",
        help=(
            "Also test every fileformat value using the same uploaded structure. "
            "Many failures may simply mean the parser does not match the file type."
        ),
    )

    args = parser.parse_args()

    options_by_field = get_real_options()
    print_options(options_by_field)

    if args.discover_only:
        return

    if args.structure is None:
        raise SystemExit("Please provide a structure file, or use --discover-only.")

    structure = Path(args.structure).resolve()
    generator_script = Path(args.generator_script).resolve()
    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    if not structure.exists():
        raise FileNotFoundError(structure)

    if not generator_script.exists():
        raise FileNotFoundError(generator_script)

    baseline = default_baseline(args.baseline_file_format)

    tests = []

    for field, options in options_by_field.items():
        if field == "fileformat" and not args.include_file_format_tests:
            print(
                "Skipping fileformat tests by default. "
                "Use --include-file-format-tests to test them with this same structure file."
            )
            continue

        for opt in options:
            value = opt["value"]
            label = opt["label"]

            test_options = dict(baseline)
            test_options[field] = value

            outdir = (
                base_out
                / "individual"
                / safe_name(field)
                / safe_name(value)
            )

            tests.append((field, value, label, test_options, outdir))

    if not tests:
        raise SystemExit("No tests selected.")

    rows = []

    for i, (field, value, label, test_options, outdir) in enumerate(tests, start=1):
        print("=" * 80)
        print(f"Test {i}/{len(tests)}")
        print(f"{field}: value={value!r}, label={label!r}")
        print("Options:", test_options)

        result = run_one(
            generator_script=generator_script,
            structure=structure,
            outdir=outdir,
            options=test_options,
        )

        status = "PASS" if result["success"] else "FAIL"
        print(f"Result: {status} | {result['reason']} | {result['elapsed_sec']} s")

        rows.append(
            {
                "test_number": i,
                "field": field,
                "value": value,
                "label": label,
                **test_options,
                "success": result["success"],
                "reason": result["reason"],
                "elapsed_sec": result["elapsed_sec"],
                "returncode": result["returncode"],
                "outdir": str(outdir),
                "cmd": result["cmd"],
                "stdout_tail": result["stdout_tail"],
                "stderr_tail": result["stderr_tail"],
            }
        )

        time.sleep(args.sleep)

    csv_file = base_out / "real_option_test_results.csv"

    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["success"])
    failed = len(rows) - passed

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"CSV: {csv_file}")

    if failed:
        print("\nFailed tests:")
        for row in rows:
            if not row["success"]:
                print(
                    f"- {row['field']} value={row['value']!r}, "
                    f"label={row['label']!r} -> {row['reason']}"
                )


if __name__ == "__main__":
    main()
