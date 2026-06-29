#!/usr/bin/env python3
"""Estimate Wannier90 window values from a Quantum ESPRESSO bands.out.gnu file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Change this value to decide how close a band must be to the Fermi energy.
# Example: 2.0 means Ef - 2.0 eV <= band energy <= Ef + 2.0 eV.
DEFAULT_CLOSE_BAND_TOLERANCE = 2.0


@dataclass(frozen=True)
class Band:
    index: int
    points: tuple[tuple[float, float], ...]

    @property
    def energies(self) -> tuple[float, ...]:
        return tuple(energy for _, energy in self.points)

    @property
    def energy_min(self) -> float:
        return min(self.energies)

    @property
    def energy_max(self) -> float:
        return max(self.energies)

    def is_close_to_fermi(self, fermi_energy: float, tolerance: float) -> bool:
        return any(abs(energy - fermi_energy) <= tolerance for energy in self.energies)

    def overlaps_energy_window(self, energy_min: float, energy_max: float) -> bool:
        return self.energy_max >= energy_min and self.energy_min <= energy_max


@dataclass(frozen=True)
class WindowAnalysis:
    input_file: Path
    fermi_energy: float
    tolerance: float
    total_bands: int
    total_points: int
    close_bands: tuple[Band, ...]
    wannier_bands: tuple[Band, ...]
    dis_froz_min: float
    dis_froz_max: float
    dis_win_min: float
    dis_win_max: float

    @property
    def num_wann(self) -> int:
        return len(self.wannier_bands)


def parse_bands_gnu(path: Path) -> tuple[Band, ...]:
    bands: list[Band] = []
    current_points: list[tuple[float, float]] = []

    def flush_current_band() -> None:
        nonlocal current_points
        if current_points:
            bands.append(Band(index=len(bands) + 1, points=tuple(current_points)))
            current_points = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                flush_current_band()
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least 2 columns")

            try:
                k_point = float(parts[0])
                energy = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: could not parse k-point/energy values"
                ) from exc

            current_points.append((k_point, energy))

    flush_current_band()
    if not bands:
        raise ValueError(f"{path}: no band data found")

    return tuple(bands)


def enclosing_energy_window(bands: Iterable[Band]) -> tuple[float, float]:
    selected = tuple(bands)
    if not selected:
        raise ValueError("cannot determine an energy window without any bands")
    return (
        min(band.energy_min for band in selected),
        max(band.energy_max for band in selected),
    )


def analyze_windows(path: Path, fermi_energy: float, tolerance: float) -> WindowAnalysis:
    bands = parse_bands_gnu(path)
    close_bands = tuple(
        band for band in bands if band.is_close_to_fermi(fermi_energy, tolerance)
    )
    if not close_bands:
        low = fermi_energy - tolerance
        high = fermi_energy + tolerance
        raise ValueError(
            "no bands have sampled energies inside "
            f"[{low:.6f}, {high:.6f}]"
        )

    dis_froz_min, dis_froz_max = enclosing_energy_window(close_bands)
    wannier_bands = tuple(
        band
        for band in bands
        if band.overlaps_energy_window(dis_froz_min, dis_froz_max)
    )
    dis_win_min, dis_win_max = enclosing_energy_window(wannier_bands)

    return WindowAnalysis(
        input_file=path,
        fermi_energy=fermi_energy,
        tolerance=tolerance,
        total_bands=len(bands),
        total_points=sum(len(band.points) for band in bands),
        close_bands=close_bands,
        wannier_bands=wannier_bands,
        dis_froz_min=dis_froz_min,
        dis_froz_max=dis_froz_max,
        dis_win_min=dis_win_min,
        dis_win_max=dis_win_max,
    )


def format_band_indices(bands: Iterable[Band]) -> str:
    return ", ".join(str(band.index) for band in bands)


def format_analysis(analysis: WindowAnalysis) -> str:
    close_low = analysis.fermi_energy - analysis.tolerance
    close_high = analysis.fermi_energy + analysis.tolerance
    rows = [
        ("input_file", str(analysis.input_file)),
        ("fermi_energy", f"{analysis.fermi_energy:.6f}"),
        ("close_band_rule", f"any sampled energy in [{close_low:.6f}, {close_high:.6f}]"),
        ("total_bands", str(analysis.total_bands)),
        ("total_points", str(analysis.total_points)),
        ("close_bands", format_band_indices(analysis.close_bands)),
        ("wannier_bands", format_band_indices(analysis.wannier_bands)),
        ("num_wann", str(analysis.num_wann)),
        ("dis_froz_min", f"{analysis.dis_froz_min:.6f}"),
        ("dis_froz_max", f"{analysis.dis_froz_max:.6f}"),
        ("dis_win_min", f"{analysis.dis_win_min:.6f}"),
        ("dis_win_max", f"{analysis.dis_win_max:.6f}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{label_width}} : {value}" for label, value in rows)


def analysis_to_json(analysis: WindowAnalysis) -> str:
    payload = {
        "input_file": str(analysis.input_file),
        "fermi_energy": analysis.fermi_energy,
        "close_band_tolerance": analysis.tolerance,
        "total_bands": analysis.total_bands,
        "total_points": analysis.total_points,
        "close_bands": [band.index for band in analysis.close_bands],
        "wannier_bands": [band.index for band in analysis.wannier_bands],
        "num_wann": analysis.num_wann,
        "dis_froz_min": analysis.dis_froz_min,
        "dis_froz_max": analysis.dis_froz_max,
        "dis_win_min": analysis.dis_win_min,
        "dis_win_max": analysis.dis_win_max,
    }
    return json.dumps(payload, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a bands.out.gnu file and estimate num_wann, dis_froz_min/max, "
            "and dis_win_min/max."
        )
    )
    parser.add_argument("bands_file", type=Path, help="Path to bands.out.gnu")
    parser.add_argument("fermi_energy", type=float, help="Fermi energy in eV")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_CLOSE_BAND_TOLERANCE,
        help=(
            "Energy distance from the Fermi level used to find close bands "
            f"(default: {DEFAULT_CLOSE_BAND_TOLERANCE} eV)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the aligned text report",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    try:
        analysis = analyze_windows(args.bands_file, args.fermi_energy, args.tolerance)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(analysis_to_json(analysis) if args.json else format_analysis(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
