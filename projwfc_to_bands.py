#!/usr/bin/env python3
"""Extract band energies and selected projection weights from projwfc.x output.

This is a Python replacement for ``projwfc_to_bands.awk``.  It keeps the same
table format while offering a CLI that is easier to call from Python workflow
code.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
STATE_RE = re.compile(r"state\s+#\s*(\d+)\s*:", re.IGNORECASE)
KPOINT_RE = re.compile(r"\bk\s*=\s*(%s)\s+(%s)\s+(%s)" % (FLOAT_RE, FLOAT_RE, FLOAT_RE), re.IGNORECASE)
ENERGY_RE = re.compile(r"====\s*e\(\s*(\d+)\s*\)\s*=\s*(%s)\s*eV\s*====" % FLOAT_RE, re.IGNORECASE)
WEIGHT_RE = re.compile(r"(%s)\s*\*\s*\[#\s*(\d+)\s*\]" % FLOAT_RE)


@dataclass(frozen=True)
class BandRow:
    k_index: int
    k_length: float
    band_index: int
    energy_minus_ef: float
    total_weight: float
    weights: dict[int, float]


@dataclass(frozen=True)
class StateInfo:
    state: int
    atom_index: int | None
    element: str
    wfc_index: int | None
    l_value: int | None
    m_index: int | None
    orbital: str


@dataclass(frozen=True)
class ProjectionData:
    states: list[int]
    state_info: dict[int, StateInfo]
    state_comment_lines: list[str]
    rows: list[BandRow]
    ef: float


def fortran_float(value: str) -> float:
    """Parse regular or Fortran-style floating point text."""
    return float(str(value).replace("D", "E").replace("d", "E"))


def selected_states(firststate: int = 0, laststate: int = 0) -> list[int]:
    """Return the atomic wavefunction state numbers selected by AWK-style args."""
    if firststate and laststate:
        if laststate < firststate:
            raise ValueError("laststate must be greater than or equal to firststate")
        return list(range(firststate, laststate + 1))
    return []


def awk_state_label(state: int) -> str:
    """Format a state label as the original AWK script did."""
    return f"[#{state:4d}]"


def orbital_from_l_m(l_value: int | None, m_index: int | None) -> str:
    """Return the QE real-spherical-harmonic orbital name for an l/m index."""
    if l_value is None or m_index is None:
        return ""
    labels = {
        0: {1: "s"},
        1: {1: "pz", 2: "px", 3: "py"},
        2: {1: "dz2", 2: "dzx", 3: "dzy", 4: "dx2-y2", 5: "dxy"},
        3: {
            1: "fz3",
            2: "fxz2",
            3: "fyz2",
            4: "fzx2-zy2",
            5: "fxyz",
            6: "fx3-3xy2",
            7: "f3x2y-y3",
        },
    }
    return labels.get(l_value, {}).get(m_index, f"l{l_value}m{m_index}")


def parse_state_info(line: str) -> StateInfo | None:
    """Parse a projwfc.x atomic-state metadata line."""
    pattern = re.compile(
        r"state\s+#\s*(\d+)\s*:\s*atom\s+(\d+)\s+\(([^)]*)\)\s*,\s*"
        r"wfc\s+(\d+)\s+\(l=\s*(\d+)\s*m=\s*(\d+)\s*\)",
        re.IGNORECASE,
    )
    match = pattern.search(line)
    if not match:
        state_match = STATE_RE.search(line)
        if not state_match:
            return None
        state = int(state_match.group(1))
        return StateInfo(state, None, "", None, None, None, "")

    state = int(match.group(1))
    atom_index = int(match.group(2))
    element = match.group(3).strip()
    wfc_index = int(match.group(4))
    l_value = int(match.group(5))
    m_index = int(match.group(6))
    return StateInfo(
        state=state,
        atom_index=atom_index,
        element=element,
        wfc_index=wfc_index,
        l_value=l_value,
        m_index=m_index,
        orbital=orbital_from_l_m(l_value, m_index),
    )


def state_label(state: int, state_info: dict[int, StateInfo], label_style: str) -> str:
    """Format selected-state column labels."""
    if label_style == "number":
        return awk_state_label(state).replace(" ", "_")

    info = state_info.get(state)
    if info is None or not info.orbital:
        return awk_state_label(state).replace(" ", "_")

    if label_style == "orbital":
        return f"{info.orbital}"
    if label_style == "state-orbital":
        return f"#{state}:{info.orbital}"
    if label_style == "atom-orbital":
        atom = f"{info.element}{info.atom_index}" if info.element and info.atom_index is not None else f"#{state}"
        return f"{atom}:{info.orbital}"
    raise ValueError("label_style must be 'number', 'orbital', 'state-orbital', or 'atom-orbital'")


def header_lines(
    ef: float,
    states: Iterable[int],
    *,
    state_info: dict[int, StateInfo] | None = None,
    label_style: str = "number",
) -> list[str]:
    state_list = list(states)
    state_info = state_info or {}
    lines = [f"# energies were translated by aligning to the reference value EF= {ef:11.5f}"]
    header = f"{'#iK':>3s} {'K-length':>11s} {'E-EF(eV)':>11s}"
    if state_list:
        header += f" {'[TOTAL]':>7s}"
        for state in state_list:
            header += f" {state_label(state, state_info, label_style):>7s}"
    lines.append(header)
    return lines


def format_band_row(row: BandRow, states: Iterable[int]) -> str:
    state_list = list(states)
    text = f"{row.k_index:3d} {row.k_length:11.5f} {row.energy_minus_ef:11.5f}"
    if state_list:
        text += f" {row.total_weight:7.3f}"
        for state in state_list:
            text += f" {row.weights.get(state, 0.0):7.3f}"
    return text


def rows_by_band(rows: Iterable[BandRow]) -> dict[int, list[BandRow]]:
    grouped: dict[int, list[BandRow]] = {}
    for row in rows:
        grouped.setdefault(row.band_index, []).append(row)
    return grouped


def ordered_rows(rows: Iterable[BandRow], layout: str) -> list[BandRow]:
    row_list = list(rows)
    if layout == "kpoint":
        return row_list
    if layout == "band":
        return sorted(row_list, key=lambda row: (row.band_index, row.k_index))
    raise ValueError("layout must be 'kpoint' or 'band'")


def parse_projwfc_output(
    lines: Iterable[str],
    *,
    ef: float = 0.0,
    firststate: int = 0,
    laststate: int = 0,
) -> ProjectionData:
    """Parse projwfc.x output into reusable band/projection rows."""
    states = selected_states(firststate, laststate)
    state_set = set(states)

    state_comment_lines: list[str] = []
    state_info: dict[int, StateInfo] = {}
    rows: list[BandRow] = []

    k_length = 0.0
    ik = 0
    previous_k: tuple[float, float, float] | None = None
    current_energy: float | None = None
    current_band_index: int | None = None
    weights = {state: 0.0 for state in states}
    total_weight = 0.0

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        parsed_state = parse_state_info(line)
        if parsed_state is not None:
            state_info[parsed_state.state] = parsed_state
        if parsed_state is not None and parsed_state.state in state_set:
            state_comment_lines.append(f"# {line}")

        kpoint_match = KPOINT_RE.search(line)
        if kpoint_match:
            kx, ky, kz = (fortran_float(group) for group in kpoint_match.groups())
            current_k = (kx, ky, kz)
            if previous_k is not None:
                k_length += math.dist(current_k, previous_k)
            ik += 1
            previous_k = current_k

        energy_match = ENERGY_RE.search(line)
        if energy_match:
            current_band_index = int(energy_match.group(1))
            current_energy = fortran_float(energy_match.group(2))
            weights = {state: 0.0 for state in states}
            total_weight = 0.0

        if states:
            for weight_text, state_text in WEIGHT_RE.findall(line):
                state = int(state_text)
                if state in state_set:
                    weight = fortran_float(weight_text)
                    weights[state] = weight
                    total_weight += weight

        if "|psi|" in line and current_energy is not None and current_band_index is not None:
            rows.append(
                BandRow(
                    k_index=ik,
                    k_length=k_length,
                    band_index=current_band_index,
                    energy_minus_ef=current_energy - ef,
                    total_weight=total_weight,
                    weights=dict(weights),
                )
            )

    return ProjectionData(
        states=states,
        state_info=state_info,
        state_comment_lines=state_comment_lines,
        rows=rows,
        ef=ef,
    )


def convert_projection_data_to_text(
    data: ProjectionData,
    *,
    layout: str = "kpoint",
    label_style: str = "number",
    include_state_comments: bool = False,
) -> str:
    """Format parsed projection data as the original text table."""
    if layout not in {"kpoint", "band"}:
        raise ValueError("layout must be 'kpoint' or 'band'")
    if label_style not in {"number", "orbital", "state-orbital", "atom-orbital"}:
        raise ValueError("label_style must be 'number', 'orbital', 'state-orbital', or 'atom-orbital'")

    rows = data.rows
    if not rows:
        output = data.state_comment_lines if include_state_comments else []
        return "\n".join(output) + ("\n" if output else "")

    output = []
    if include_state_comments:
        output.extend(data.state_comment_lines)
    output.extend(
        header_lines(
            data.ef,
            data.states,
            state_info=data.state_info,
            label_style=label_style,
        )
    )
    if layout == "kpoint":
        output.extend(format_band_row(row, data.states) for row in rows)
        return "\n".join(output) + "\n"

    grouped_rows = rows_by_band(rows)
    for band_index in sorted(grouped_rows):
        if output and output[-1] != "":
            output.append("")
        output.append(f"# band {band_index}")
        output.extend(format_band_row(row, data.states) for row in grouped_rows[band_index])
    return "\n".join(output) + ("\n" if output else "")


def convert_projwfc_to_bands(
    lines: Iterable[str],
    *,
    ef: float = 0.0,
    firststate: int = 0,
    laststate: int = 0,
    layout: str = "kpoint",
    label_style: str = "number",
    include_state_comments: bool = False,
) -> str:
    """Convert projwfc.x output text into the bands/projection table."""
    data = parse_projwfc_output(
        lines,
        ef=ef,
        firststate=firststate,
        laststate=laststate,
    )
    return convert_projection_data_to_text(
        data,
        layout=layout,
        label_style=label_style,
        include_state_comments=include_state_comments,
    )


def unique_headers(headers: Iterable[str]) -> list[str]:
    """Return CSV-safe headers with duplicates disambiguated."""
    counts: dict[str, int] = {}
    unique: list[str] = []
    for header in headers:
        text = str(header or "value")
        counts[text] = counts.get(text, 0) + 1
        unique.append(text if counts[text] == 1 else f"{text}_{counts[text]}")
    return unique


def csv_number(value: float) -> str:
    return f"{value:.10g}"


def write_csv_table(path: Path, headers: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_csv_stdout(headers: list[str], rows: Iterable[list[object]]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    writer.writerows(rows)


def base_csv_values(row: BandRow) -> list[object]:
    return [
        row.band_index,
        row.k_index,
        csv_number(row.k_length),
        csv_number(row.energy_minus_ef),
    ]


def detailed_csv_table(
    data: ProjectionData,
    *,
    layout: str = "band",
    label_style: str = "atom-orbital",
) -> tuple[list[str], list[list[object]]]:
    """Build one row per band/k-point with individual selected-state weights."""
    state_headers = [
        state_label(state, data.state_info, label_style)
        for state in data.states
    ]
    headers = unique_headers(
        ["band", "k_index", "k_length", "energy_minus_ef", "selected_total", *state_headers]
    )
    rows = [
        [
            *base_csv_values(row),
            csv_number(row.total_weight),
            *(csv_number(row.weights.get(state, 0.0)) for state in data.states),
        ]
        for row in ordered_rows(data.rows, layout)
    ]
    return headers, rows


def orbital_sort_key(orbital: str) -> tuple[int, str]:
    order = {
        "s": 0,
        "pz": 10,
        "px": 11,
        "py": 12,
        "dz2": 20,
        "dzx": 21,
        "dzy": 22,
        "dx2-y2": 23,
        "dxy": 24,
    }
    return order.get(orbital, 1000), orbital


def atom_sort_key(info: StateInfo) -> tuple[int, str]:
    return (info.atom_index if info.atom_index is not None else 10**9, info.element)


def selected_state_infos(data: ProjectionData) -> list[StateInfo]:
    infos: list[StateInfo] = []
    for state in data.states:
        info = data.state_info.get(state)
        if info is not None:
            infos.append(info)
    return infos


def orbital_weight(row: BandRow, data: ProjectionData, *, atom_index: int | None, orbital: str) -> float:
    total = 0.0
    for state in data.states:
        info = data.state_info.get(state)
        if info is None or info.orbital != orbital:
            continue
        if atom_index is not None and info.atom_index != atom_index:
            continue
        total += row.weights.get(state, 0.0)
    return total


def element_csv_table(
    data: ProjectionData,
    *,
    element: str,
    layout: str = "band",
) -> tuple[list[str], list[list[object]]]:
    """Build an element-specific CSV with orbital contributions summed per orbital."""
    element_key = str(element or "").strip()
    orbitals = sorted(
        {
            info.orbital
            for info in selected_state_infos(data)
            if info.element == element_key and info.orbital
        },
        key=orbital_sort_key,
    )
    headers = ["band", "k_index", "k_length", "energy_minus_ef", "element_total", *orbitals]
    rows: list[list[object]] = []
    for row in ordered_rows(data.rows, layout):
        weights = [
            sum(
                row.weights.get(state, 0.0)
                for state in data.states
                if (
                    (info := data.state_info.get(state)) is not None
                    and info.element == element_key
                    and info.orbital == orbital
                )
            )
            for orbital in orbitals
        ]
        rows.append([*base_csv_values(row), csv_number(sum(weights)), *(csv_number(weight) for weight in weights)])
    return headers, rows


def total_orbital_csv_table(
    data: ProjectionData,
    *,
    layout: str = "band",
) -> tuple[list[str], list[list[object]]]:
    """Build a CSV with each orbital summed over all selected atoms."""
    orbitals = sorted(
        {info.orbital for info in selected_state_infos(data) if info.orbital},
        key=orbital_sort_key,
    )
    headers = ["band", "k_index", "k_length", "energy_minus_ef", "orbital_total", *orbitals]
    rows: list[list[object]] = []
    for row in ordered_rows(data.rows, layout):
        weights = [orbital_weight(row, data, atom_index=None, orbital=orbital) for orbital in orbitals]
        rows.append([*base_csv_values(row), csv_number(sum(weights)), *(csv_number(weight) for weight in weights)])
    return headers, rows


def safe_filename_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return cleaned.strip("._") or "value"


def write_split_csvs(
    data: ProjectionData,
    output_dir: Path,
    *,
    layout: str = "band",
    label_style: str = "atom-orbital",
) -> dict[str, Path]:
    """Write detailed, per-element, and all-element orbital-total CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    headers, rows = detailed_csv_table(data, layout=layout, label_style=label_style)
    all_states_path = output_dir / "all_states.csv"
    write_csv_table(all_states_path, headers, rows)
    written["all_states"] = all_states_path

    total_headers, total_rows = total_orbital_csv_table(data, layout=layout)
    total_path = output_dir / "orbital_totals.csv"
    write_csv_table(total_path, total_headers, total_rows)
    written["orbital_totals"] = total_path

    element_infos: dict[str, StateInfo] = {}
    for info in selected_state_infos(data):
        if info.element:
            element_infos.setdefault(info.element, info)

    for element, info in sorted(element_infos.items(), key=lambda item: atom_sort_key(item[1])):
        element_headers, element_rows = element_csv_table(data, element=element, layout=layout)
        element_path = output_dir / f"{safe_filename_part(element)}.csv"
        write_csv_table(element_path, element_headers, element_rows)
        written[element] = element_path

    return written


def convert_file(
    input_path: Path,
    output_path: Path | None = None,
    *,
    ef: float = 0.0,
    firststate: int = 0,
    laststate: int = 0,
    layout: str = "kpoint",
    label_style: str = "number",
    include_state_comments: bool = False,
) -> str:
    """Convert a projwfc.x output file and optionally write the result."""
    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        text = convert_projwfc_to_bands(
            handle,
            ef=ef,
            firststate=firststate,
            laststate=laststate,
            layout=layout,
            label_style=label_style,
            include_state_comments=include_state_comments,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract band energies and optional atomic-state weights from projwfc.x output.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="projwfc.x output file. Omit or use '-' to read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Main output path. Defaults to stdout unless --csv-dir is used alone.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "csv"),
        default="text",
        help="Main output format. CSV writes one row per band/k-point with selected-state weights.",
    )
    parser.add_argument(
        "--csv-dir",
        help=(
            "Directory for plot-ready CSV files: all_states.csv, orbital_totals.csv, "
            "and one ELEMENT.csv file per selected element."
        ),
    )
    parser.add_argument(
        "--ef",
        type=float,
        default=0.0,
        help="Reference Fermi energy in eV. The energy column is printed as E - EF.",
    )
    parser.add_argument(
        "--firststate",
        type=int,
        default=0,
        help="First atomic wavefunction state to include. Requires --laststate.",
    )
    parser.add_argument(
        "--laststate",
        type=int,
        default=0,
        help="Last atomic wavefunction state to include. Requires --firststate.",
    )
    parser.add_argument(
        "--layout",
        choices=("kpoint", "band"),
        default="kpoint",
        help="Output row order. 'kpoint' matches the AWK script; 'band' groups rows by band for plotting.",
    )
    parser.add_argument(
        "--group-by-band",
        action="store_true",
        help="Shortcut for --layout band.",
    )
    parser.add_argument(
        "--label-style",
        choices=("number", "orbital", "state-orbital", "atom-orbital"),
        default="atom-orbital",
        help=(
            "State-weight column labels. 'number' matches AWK; 'orbital' uses s/px/etc.; "
            "'state-orbital' keeps the state number; 'atom-orbital' includes the atom."
        ),
    )
    parser.add_argument(
        "--state-comments",
        action="store_true",
        help="Include the selected projwfc.x state metadata comments in text output.",
    )
    return parser


def read_stdin(stdin: TextIO) -> list[str]:
    return stdin.readlines()


def load_projection_data_from_cli(args: argparse.Namespace) -> ProjectionData:
    if args.input and args.input != "-":
        with Path(args.input).open("r", encoding="utf-8", errors="ignore") as handle:
            return parse_projwfc_output(
                handle,
                ef=args.ef,
                firststate=args.firststate,
                laststate=args.laststate,
            )
    return parse_projwfc_output(
        read_stdin(sys.stdin),
        ef=args.ef,
        firststate=args.firststate,
        laststate=args.laststate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.firststate) != bool(args.laststate):
        parser.error("--firststate and --laststate must be provided together")

    layout = "band" if args.group_by_band else args.layout

    try:
        data = load_projection_data_from_cli(args)

        written_split_files: dict[str, Path] = {}
        if args.csv_dir:
            written_split_files = write_split_csvs(
                data,
                Path(args.csv_dir),
                layout=layout,
                label_style=args.label_style,
            )

        if args.format == "csv":
            headers, rows = detailed_csv_table(data, layout=layout, label_style=args.label_style)
            if args.output:
                write_csv_table(Path(args.output), headers, rows)
            elif not args.csv_dir:
                write_csv_stdout(headers, rows)
        else:
            text = convert_projection_data_to_text(
                data,
                layout=layout,
                label_style=args.label_style,
                include_state_comments=args.state_comments,
            )
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
            elif not args.csv_dir:
                sys.stdout.write(text)

        if args.csv_dir and not args.output:
            for key in sorted(written_split_files):
                print(f"{key}: {written_split_files[key]}", file=sys.stderr)
        return 0
    except OSError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
