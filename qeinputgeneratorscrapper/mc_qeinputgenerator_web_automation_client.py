#!/usr/bin/env python3
"""
Generate Quantum ESPRESSO input from a structure file using the Materials Cloud
QE input generator.

This version supports:
- Multiple structure file formats through --file-format
- ZIP download with pseudopotentials
- Clean capture of the warning modal shown in result_page.html

Dependencies:
    pip install requests beautifulsoup4

Examples:
    python mc_qeinputgenerator_web_automation_client.py ZrAsRu.cif --out qe_out --download-zip
    python mc_qeinputgenerator_web_automation_client.py ZrAsRu.cif --file-format cif-pymatgen --out qe_out
    python mc_qeinputgenerator_web_automation_client.py POSCAR --file-format vasp-ase --out qe_poscar
    python mc_qeinputgenerator_web_automation_client.py --list-form-options
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://qeinputgenerator.materialscloud.io"
HOME_URL = BASE_URL + "/"
PROCESS_URL = urljoin(BASE_URL, "/compute/process_structure/")
ZIP_URL = urljoin(BASE_URL, "/compute/request_zip")


SELECT_FIELDS = [
    "fileformat",
    "protocol",
    "xc_functional",
    "magnetization",
    "refine_cell",
    "pseudo_override",
    "kmesh_override",
    "threshold_override",
]


def get_form_options(timeout: int = 60) -> dict[str, list[dict[str, str]]]:
    response = requests.get(HOME_URL, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    options_by_field: dict[str, list[dict[str, str]]] = {}

    for field in SELECT_FIELDS:
        select = soup.find("select", {"name": field})
        if select is None:
            options_by_field[field] = []
            continue

        options: list[dict[str, str]] = []
        for option in select.find_all("option"):
            if option.has_attr("disabled"):
                continue
            value = option.get("value", "")
            label = option.get_text(" ", strip=True)
            options.append({"value": value, "label": label})

        options_by_field[field] = options

    return options_by_field


def print_form_options() -> None:
    options_by_field = get_form_options()

    print("Real HTML option values found:\n")
    for field, options in options_by_field.items():
        print(f"{field}:")
        if not options:
            print("  WARNING: no options found")
        for opt in options:
            print(f"  value={opt['value']!r}    label={opt['label']!r}")
        print()


def guess_fileformat(structure_path: Path) -> str:
    suffix = structure_path.suffix.lower()
    name_upper = structure_path.name.upper()

    if suffix == ".cif":
        return "cif-ase"
    if suffix == ".xyz":
        return "xyz-ase"
    if suffix == ".xsf":
        return "xsf-ase"
    if suffix == ".cell":
        return "castep-ase"
    if suffix == ".pdb":
        return "pdb-ase"
    if name_upper in {"POSCAR", "CONTCAR"} or suffix in {".poscar", ".vasp"}:
        return "vasp-ase"
    if suffix in {".in", ".pwi", ".pw", ".qe"}:
        return "qeinp-qetools"

    raise ValueError(
        f"Could not guess file format for {structure_path.name!r}. "
        "Run `python mc_qeinputgenerator_web_automation_client.py --list-form-options` "
        "and pass the correct value with `--file-format VALUE`."
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_warning_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[×x]\s*", "", text)
    text = re.sub(r"\s*Got it!\s*$", "", text, flags=re.I)
    text = normalize_text(text)
    return text


def extract_qe_input(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    node = soup.select_one("#qepwinput pre")
    if node is None:
        node = soup.find("pre")

    if node is None:
        raise RuntimeError("Could not find QE input in the response HTML.")

    return node.get_text()


def extract_warnings(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "pre", "code", "textarea"]):
        tag.decompose()

    warnings: list[str] = []
    seen: set[str] = set()

    def add_warning(text: str) -> None:
        text = clean_warning_text(text)
        if not text:
            return
        if "warning" not in text.lower():
            return
        if len(text) > 2000:
            return
        if text not in seen:
            seen.add(text)
            warnings.append(text)

    # Best target: the modal the website opens with $('#modal-warning').modal('show')
    modal_candidates = []

    modal = soup.select_one("#modal-warning")
    if modal is not None:
        modal_candidates.append(modal)

    # Future-proof fallback: any modal with warning in id/class.
    for candidate in soup.select(".modal"):
        candidate_id = candidate.get("id", "")
        candidate_class = " ".join(candidate.get("class", []))
        marker = f"{candidate_id} {candidate_class}".lower()
        if "warning" in marker and candidate not in modal_candidates:
            modal_candidates.append(candidate)

    for modal in modal_candidates:
        # Remove close/footer buttons so "×" and "Got it!" are not captured.
        for tag in modal.select("button, .close, .modal-footer"):
            tag.decompose()

        title = modal.select_one(".modal-title")
        body = modal.select_one(".modal-body")

        parts = []
        if title is not None:
            parts.append(title.get_text(" ", strip=True))
        if body is not None:
            parts.append(body.get_text(" ", strip=True))

        if parts:
            add_warning(" ".join(parts))
        else:
            add_warning(modal.get_text(" ", strip=True))

    # Fallback: alert-warning blocks, excluding notes that do not contain "warning".
    if not warnings:
        for node in soup.select(".alert-warning, [class*='alert-warning']"):
            add_warning(node.get_text(" ", strip=True))

    return warnings


def write_warnings(warnings: list[str], outdir: Path) -> Path | None:
    if not warnings:
        return None

    warning_path = outdir / "warnings.txt"
    lines = []
    for i, warning in enumerate(warnings, start=1):
        lines.append(f"WARNING {i}:")
        lines.append(warning)
        lines.append("")

    warning_path.write_text("\n".join(lines), encoding="utf-8")
    return warning_path


def extract_pseudo_info(html: str) -> tuple[str | None, list[str]]:
    family = None
    elements: list[str] = []

    family_match = re.search(
        r"""['"]pseudos['"]\s*:\s*\{\s*['"]family['"]\s*:\s*(['"])(.*?)\1""",
        html,
        flags=re.DOTALL,
    )
    if family_match:
        family = family_match.group(2)

    elements_match = re.search(
        r"""['"]elements['"]\s*:\s*\[([^\]]*)\]""",
        html,
        flags=re.DOTALL,
    )
    if elements_match:
        elements = re.findall(r"""['"]([^'"]+)['"]""", elements_match.group(1))

    return family, elements


def generate_qe_input(
    structure_path: Path,
    outdir: Path,
    fileformat: str,
    protocol: str = "balanced",
    xc: str = "PBEsol",
    magnetization: str = "non-magnetic-fractional",
    refine_cell: str = "no",
    tolerance: str = "-5",
    pseudo_override: str = "",
    kmesh_override: str = "",
    threshold_override: str = "",
    download_zip: bool = False,
    fail_on_warning: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if fileformat == "auto":
        fileformat = guess_fileformat(structure_path)

    data = {
        "file-select-toggle": "upload",
        "fileformat": fileformat,

        # These fields exist in the web form. They are relevant for XYZ uploads
        # when cell vectors are supplied manually. For most periodic structure
        # formats they can remain blank/default.
        "xyzCellVecAx": "",
        "xyzCellVecAy": "0",
        "xyzCellVecAz": "0",
        "xyzCellVecBx": "0",
        "xyzCellVecBy": "",
        "xyzCellVecBz": "0",
        "xyzCellVecCx": "0",
        "xyzCellVecCy": "0",
        "xyzCellVecCz": "",

        "examplestructure": "al.xsf",
        "protocol": protocol,
        "xc_functional": xc,
        "magnetization": magnetization,
        "refine_cell": refine_cell,
        "toleranceInput": tolerance,
        "pseudo_override": pseudo_override,
        "kmesh_override": kmesh_override,
        "threshold_override": threshold_override,
    }

    headers = {
        "Origin": BASE_URL,
        "Referer": HOME_URL,
        "User-Agent": "Mozilla/5.0",
    }

    with requests.Session() as session:
        with structure_path.open("rb") as handle:
            files = {
                "structurefile": (
                    structure_path.name,
                    handle,
                    "application/octet-stream",
                )
            }

            response = session.post(
                PROCESS_URL,
                data=data,
                files=files,
                headers=headers,
                timeout=120,
            )

        response.raise_for_status()

        html_path = outdir / "result_page.html"
        html_path.write_text(response.text, encoding="utf-8")

        warnings = extract_warnings(response.text)
        warning_path = write_warnings(warnings, outdir)

        if warning_path:
            print(f"Captured {len(warnings)} warning(s): {warning_path}")
            for i, warning in enumerate(warnings, start=1):
                print(f"WARNING {i}: {warning}")
        else:
            print("No modal warning found in result page.")

        if fail_on_warning and warnings:
            raise RuntimeError(
                f"Result page contains {len(warnings)} warning(s). "
                f"See {warning_path}"
            )

        qe_input = extract_qe_input(response.text)
        qe_path = outdir / "scf.in"
        qe_path.write_text(qe_input, encoding="utf-8")

        print(f"File format value: {fileformat!r}")
        print(f"Wrote QE input: {qe_path}")

        family, elements = extract_pseudo_info(response.text)
        if family:
            print(f"Pseudo family: {family}")
        if elements:
            print(f"Pseudo elements: {', '.join(elements)}")

        if download_zip:
            if not family or not elements:
                raise RuntimeError(
                    "Could not extract pseudo information, so the zip request cannot be built."
                )

            zip_inputs = {
                "input_content": qe_input,
                "pseudos": {
                    "family": family,
                    "elements": elements,
                },
            }

            zip_payload = json.dumps(zip_inputs, separators=(",", ":"))

            zip_response = session.post(
                ZIP_URL,
                data={"inputs": zip_payload, "create_zip": "1"},
                headers={
                    "Origin": BASE_URL,
                    "Referer": PROCESS_URL,
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=120,
            )

            if not zip_response.ok:
                debug_path = outdir / f"request_zip_error_{zip_response.status_code}.html"
                debug_path.write_bytes(zip_response.content)
                print(
                    f"ZIP endpoint failed with HTTP {zip_response.status_code}. "
                    f"Saved error page: {debug_path}"
                )
                print("The QE input file was still generated correctly.")
                return

            content_type = zip_response.headers.get("content-type", "")
            zip_path = outdir / "qe_input_and_pseudos.zip"

            if "zip" in content_type.lower() or zip_response.content.startswith(b"PK"):
                zip_path.write_bytes(zip_response.content)
                print(f"Wrote ZIP: {zip_path}")
            else:
                debug_path = outdir / "request_zip_response.html"
                debug_path.write_bytes(zip_response.content)
                print(
                    "The ZIP endpoint did not return a ZIP file. "
                    f"Saved response for debugging: {debug_path}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QE pw.x input from a structure file using the Materials Cloud input generator."
    )

    parser.add_argument(
        "structure",
        nargs="?",
        type=Path,
        help="Input structure file. Examples: .cif, .xsf, .xyz, POSCAR, .cell, .pdb, .in",
    )
    parser.add_argument("--out", type=Path, default=Path("qe_generated"))

    parser.add_argument(
        "--file-format",
        "--fileformat",
        dest="fileformat",
        default="auto",
        help=(
            "Backend value for 'Select the file format'. Use 'auto' for common extensions, "
            "or run --list-form-options and pass the exact value, e.g. qeinp-qetools, "
            "cif-ase, cif-pymatgen, xyz-ase, xsf-ase, vasp-ase, castep-ase, pdb-ase."
        ),
    )
    parser.add_argument(
        "--list-form-options",
        action="store_true",
        help="Print the real current <option value=...> entries from the Materials Cloud form and exit.",
    )

    parser.add_argument("--protocol", default="balanced", help="fast, balanced, or stringent")
    parser.add_argument("--xc", default="PBEsol", help="PBE or PBEsol")
    parser.add_argument(
        "--magnetization",
        "--magnetism",
        dest="magnetization",
        default="non-magnetic-fractional",
        help="non-magnetic-fractional, non-magnetic-fixed or magnetic",
    )
    parser.add_argument("--refine-cell", default="no", choices=["yes", "no"])
    parser.add_argument("--tolerance", default="-5")
    parser.add_argument("--pseudo-override", default="", help="efficiency, precision or """)
    parser.add_argument("--kmesh-override", default="", help="coarse, medium, fine or """)
    parser.add_argument("--threshold-override", default="", help="loose, intermediate, tight or """)
    parser.add_argument("--download-zip", action="store_true")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with an error if the warning modal is found in result_page.html.",
    )

    args = parser.parse_args()

    if args.list_form_options:
        print_form_options()
        return

    if args.structure is None:
        parser.error("the following arguments are required unless using --list-form-options: structure")

    if not args.structure.exists():
        raise FileNotFoundError(args.structure)

    try:
        generate_qe_input(
            structure_path=args.structure,
            outdir=args.out,
            fileformat=args.fileformat,
            protocol=args.protocol,
            xc=args.xc,
            magnetization=args.magnetization,
            refine_cell=args.refine_cell,
            tolerance=args.tolerance,
            pseudo_override=args.pseudo_override,
            kmesh_override=args.kmesh_override,
            threshold_override=args.threshold_override,
            download_zip=args.download_zip,
            fail_on_warning=args.fail_on_warning,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
