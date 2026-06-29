"""Text post-processing helpers for QE input decks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def remove_comments_lines(content: str, comment_char: str = "!") -> str:
    lines = [line for line in content.splitlines() if not line.lstrip().startswith(comment_char)]
    return "\n".join(lines)


def remove_lines_starting_with(content: str, start_strings: Iterable[str]) -> str:
    lines = [
        line
        for line in content.splitlines()
        if not any(line.lstrip().startswith(prefix) for prefix in start_strings)
    ]
    return "\n".join(lines)


def remove_lines_from_start_to_end_markers(content: str, start_marker: str, end_marker: str) -> str:
    lines = content.splitlines()
    result_lines = []
    skip = False

    for line in lines:
        if start_marker in line:
            skip = True
        if not skip:
            result_lines.append(line)
        if end_marker in line and skip:
            skip = False

    return "\n".join(result_lines)


def remove_ion_markers(content: str) -> str:
    start_markers = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS")
    lines = []
    inside_block = False

    for line in content.splitlines():
        stripped = line.lstrip()
        stripped_no_ws = line.strip()
        normalized = stripped_no_ws.upper()

        if any(normalized.startswith(marker) for marker in start_markers):
            inside_block = True
            lines.append(line)
            continue

        if inside_block and stripped_no_ws == "":
            inside_block = False
            lines.append(line)
            continue

        if inside_block:
            if not stripped:
                lines.append(line)
                continue
            parts = stripped.split(maxsplit=1)
            token = parts[0]
            remainder = parts[1] if len(parts) > 1 else ""
            cleaned_token = "".join(ch for ch in token if not ch.isdigit() and ch not in "+-")
            leading_ws = line[: len(line) - len(stripped)]
            lines.append(f"{leading_ws}{cleaned_token} {remainder}".rstrip())
            continue

        lines.append(line)
    return "\n".join(lines)


def trim_qe_input(content: str) -> str:
    trimmed = remove_comments_lines(content)
    trimmed = remove_lines_from_start_to_end_markers(trimmed, start_marker="&IONS", end_marker="/")
    start_strings = [
        "disk_io",
        "nstep",
        "vdw_corr",
        "xdm_a1",
        "xdm_a2",
        "diago_thr_init",
        "startingpot",
        "startingwfc",
        "mixing_mode",
        "mixing_ndim",
        "diagonalization ",
    ]
    trimmed = remove_lines_starting_with(trimmed, start_strings=start_strings)
    trimmed = remove_ion_markers(trimmed)
    return trimmed


def save_text_to_file(path: str, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
