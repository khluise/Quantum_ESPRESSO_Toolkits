"""K-point generation helpers for Quantum ESPRESSO inputs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class HighSymmetryData:
    """Container for high-symmetry k-path data."""

    grouped_paths: List[List[str]]
    points_frac: Dict[str, Sequence[float]]


def generate_crystal_kmesh(n1: int, n2: int, n3: int) -> List[str]:
    """Generate a K_POINTS crystal block matching the legacy kmesh.pl script."""
    if min(n1, n2, n3) <= 0:
        raise ValueError("Grid divisions must be positive integers.")

    total = n1 * n2 * n3
    weight = 1.0 / float(total)
    lines: List[str] = ["K_POINTS crystal", str(total)]

    for ix in range(n1):
        fx = ix / n1
        for iy in range(n2):
            fy = iy / n2
            for iz in range(n3):
                fz = iz / n3
                lines.append(
                    f"{fx:12.8f}{fy:12.8f}{fz:12.8f}{weight:14.6e}"
                )

    return lines


def _vector_norm(vector: Sequence[float]) -> float:
    return sum(float(component) * float(component) for component in vector[:3]) ** 0.5


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _angle_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    import math

    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return 90.0
    cosine = max(-1.0, min(1.0, _dot(left, right) / (left_norm * right_norm)))
    return math.degrees(math.acos(cosine))


def _cell_rows(cell: Any) -> List[List[float]]:
    try:
        rows = cell.array.tolist()
    except Exception:
        rows = list(cell)
    return [[float(row[0]), float(row[1]), float(row[2])] for row in rows[:3]]


def _cellpar(cell: Any) -> Sequence[float]:
    try:
        values = cell.cellpar()
        return [float(value) for value in values[:6]]
    except Exception:
        rows = _cell_rows(cell)
        if len(rows) < 3:
            raise ValueError("Cell must contain three lattice vectors.")
        a_vec, b_vec, c_vec = rows[:3]
        return [
            _vector_norm(a_vec),
            _vector_norm(b_vec),
            _vector_norm(c_vec),
            _angle_degrees(b_vec, c_vec),
            _angle_degrees(a_vec, c_vec),
            _angle_degrees(a_vec, b_vec),
        ]


def _nearly_equal(left: float, right: float, *, relative_tolerance: float = 1.0e-3) -> bool:
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) <= relative_tolerance * scale


def _angle_near(value: float, target: float, *, tolerance: float = 2.0) -> bool:
    return abs(float(value) - float(target)) <= tolerance


def fallback_high_symmetry_path_from_cell(cell: Any) -> HighSymmetryData:
    """Return a conservative high-symmetry path when seekpath is unavailable."""

    a, b, c, alpha, beta, gamma = _cellpar(cell)
    right_angles = all(_angle_near(angle, 90.0) for angle in (alpha, beta, gamma))
    hexagonal = (
        _nearly_equal(a, b)
        and _angle_near(alpha, 90.0)
        and _angle_near(beta, 90.0)
        and (_angle_near(gamma, 120.0) or _angle_near(gamma, 60.0))
    )

    if hexagonal:
        return HighSymmetryData(
            grouped_paths=[
                ["GAMMA", "M", "K", "GAMMA", "A", "L", "H", "A"],
                ["L", "M"],
                ["K", "H"],
            ],
            points_frac={
                "GAMMA": (0.0, 0.0, 0.0),
                "M": (0.5, 0.0, 0.0),
                "K": (1.0 / 3.0, 1.0 / 3.0, 0.0),
                "A": (0.0, 0.0, 0.5),
                "L": (0.5, 0.0, 0.5),
                "H": (1.0 / 3.0, 1.0 / 3.0, 0.5),
            },
        )

    if right_angles and _nearly_equal(a, b) and _nearly_equal(b, c):
        return HighSymmetryData(
            grouped_paths=[["GAMMA", "X", "M", "GAMMA", "R", "X"], ["M", "R"]],
            points_frac={
                "GAMMA": (0.0, 0.0, 0.0),
                "X": (0.0, 0.5, 0.0),
                "M": (0.5, 0.5, 0.0),
                "R": (0.5, 0.5, 0.5),
            },
        )

    if right_angles and _nearly_equal(a, b):
        return HighSymmetryData(
            grouped_paths=[
                ["GAMMA", "X", "M", "GAMMA", "Z", "R", "A", "Z"],
                ["X", "R"],
                ["M", "A"],
            ],
            points_frac={
                "GAMMA": (0.0, 0.0, 0.0),
                "X": (0.0, 0.5, 0.0),
                "M": (0.5, 0.5, 0.0),
                "Z": (0.0, 0.0, 0.5),
                "R": (0.0, 0.5, 0.5),
                "A": (0.5, 0.5, 0.5),
            },
        )

    return HighSymmetryData(
        grouped_paths=[
            ["GAMMA", "X", "S", "Y", "GAMMA", "Z", "U", "R", "T", "Z"],
            ["Y", "T"],
            ["U", "X"],
            ["S", "R"],
        ],
        points_frac={
            "GAMMA": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.0),
            "Y": (0.0, 0.5, 0.0),
            "Z": (0.0, 0.0, 0.5),
            "S": (0.5, 0.5, 0.0),
            "R": (0.5, 0.5, 0.5),
            "T": (0.0, 0.5, 0.5),
            "U": (0.5, 0.0, 0.5),
        },
    )


def load_high_symmetry_path(cif_path: str) -> HighSymmetryData:
    """Return grouped high-symmetry paths and fractional coordinates using seekpath."""
    resolved = Path(cif_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CIF file not found: {resolved}")

    try:
        from ase.io import read  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError("The 'ase' package is required to analyse CIF files.") from exc

    structure = read(str(resolved))

    try:
        import seekpath  # type: ignore
    except ModuleNotFoundError:
        return fallback_high_symmetry_path_from_cell(structure.get_cell())

    cell = (
        structure.get_cell(),
        structure.get_scaled_positions(),
        structure.get_atomic_numbers(),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="seekpath")
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"spglib(\..*)?$",
            message=r".*dict interface \(SpglibDataset\['.*?'\]\) is deprecated.*",
        )
        result = seekpath.get_path(cell)
    grouped_paths: List[List[str]] = []
    segments = result.get("path") or []
    if not segments:
        return fallback_high_symmetry_path_from_cell(structure.get_cell())

    current_path: List[str] = [segments[0][0]]
    for index, (start, end) in enumerate(segments):
        if current_path[-1] != start:
            grouped_paths.append(current_path)
            current_path = [start]
        current_path.append(end)
        if index == len(segments) - 1:
            grouped_paths.append(current_path)

    points_frac_raw = result.get("point_coords") or {}
    points_frac: Dict[str, Sequence[float]] = {
        label: tuple(coords) for label, coords in points_frac_raw.items()
    }

    return HighSymmetryData(grouped_paths=grouped_paths, points_frac=points_frac)


def build_tpiba_b_block(
    selected_paths: Iterable[Sequence[str]],
    points_frac: Dict[str, Sequence[float]],
    weight: float,
) -> List[str]:
    """Build a K_POINTS crystal_b card from selected high-symmetry path labels."""
    if weight <= 0:
        raise ValueError("Weight must be positive.")

    ordered_labels: List[str] = []
    for path in selected_paths:
        for label in path:
            ordered_labels.append(label)

    if not ordered_labels:
        raise ValueError("At least one high-symmetry path must be selected.")

    lines: List[str] = ["K_POINTS crystal_b", str(len(ordered_labels))]

    for label in ordered_labels:
        coords = points_frac.get(label)
        if coords is None:
            raise ValueError(f"Missing fractional coordinates for label '{label}'.")
        kx, ky, kz = (float(coords[0]), float(coords[1]), float(coords[2]))
        lines.append(
            f"{kx:7.3f}  {ky:7.3f}  {kz:7.3f}  {weight:g} !{label.lower()}"
        )

    return lines


__all__ = [
    "HighSymmetryData",
    "generate_crystal_kmesh",
    "fallback_high_symmetry_path_from_cell",
    "load_high_symmetry_path",
    "build_tpiba_b_block",
]
