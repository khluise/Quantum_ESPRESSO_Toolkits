"""Workflow helpers for Quantum ESPRESSO Helper GUI."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CutoffConvergenceSettings",
    "KpointConvergenceSettings",
    "extract_total_energy",
    "save_cutoff_visualizations",
]


@dataclass
class CutoffConvergenceSettings:
    """Settings governing a cut-off convergence sweep."""

    start: int
    multiplier: int
    step: int = 10
    maximum: int = 120
    threshold: float = 0.0001
    required: int = 3

    def sanitized(self) -> "CutoffConvergenceSettings":
        start = max(1, int(self.start))
        multiplier = max(1, int(self.multiplier))
        step = max(1, int(self.step))
        maximum = max(start, int(self.maximum))
        threshold = max(0.0, float(self.threshold))
        required = max(1, int(self.required))
        return CutoffConvergenceSettings(
            start=start,
            multiplier=multiplier,
            step=step,
            maximum=maximum,
            threshold=threshold,
            required=required,
        )

    def ecut_values(self) -> list[int]:
        settings = self.sanitized()
        return list(range(settings.start, settings.maximum + settings.step, settings.step))


@dataclass
class KpointConvergenceSettings:
    """Settings governing a k-point convergence sweep."""

    start: int = 1
    step: int = 1
    maximum: int = 24
    threshold: float = 0.0001
    required: int = 3

    def sanitized(self) -> "KpointConvergenceSettings":
        start = max(1, int(self.start))
        step = max(1, int(self.step))
        maximum = max(start, int(self.maximum))
        threshold = max(0.0, float(self.threshold))
        required = max(1, int(self.required))
        return KpointConvergenceSettings(
            start=start,
            step=step,
            maximum=maximum,
            threshold=threshold,
            required=required,
        )

    def nk_values(self) -> list[int]:
        settings = self.sanitized()
        values: list[int] = []
        current = settings.start
        while current <= settings.maximum:
            values.append(current)
            current += settings.step
        return values


_TOTAL_ENERGY_PATTERN = re.compile(
    r"!\s*total energy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)


def extract_total_energy(output_path: str | Path) -> float | None:
    """Extract the last total energy value from a Quantum ESPRESSO output file."""

    path = Path(output_path).expanduser()
    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    energy: float | None = None
    for match in _TOTAL_ENERGY_PATTERN.finditer(text):
        raw = match.group(1)
        if not raw:
            continue
        candidate = raw.replace("D", "E").replace("d", "E")
        try:
            energy = float(candidate)
        except ValueError:
            continue
    return energy


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "cutoff_convergence"


def save_cutoff_visualizations(
    destination: str | Path,
    seed_label: str,
    ecut_values: Sequence[int],
    energies: Sequence[float],
    *,
    x_label: str = "Cutoff Energy (Ry)",
    title_metric: str = "Cutoff Energy",
    base_stem: str = "energy_vs_cutoff",
) -> dict[str, str]:
    """Save plot and XMGrace artifacts summarising a convergence sweep."""

    dest_dir = Path(destination).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    data = list(zip(ecut_values, energies))
    if not data:
        return {"plot": "", "agr": ""}

    data.sort(key=lambda pair: pair[0])
    xs = [pair[0] for pair in data]
    ys = [pair[1] for pair in data]

    label = _sanitize_label(seed_label)
    stem = _sanitize_label(base_stem)
    base_name = f"{label}_{stem}"

    plot_path = dest_dir / f"{base_name}.png"
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415 (local import to keep optional dependency)
    except Exception:
        plot_written = False
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            xs,
            ys,
            marker="o",
            color="red",
            linewidth=2,
            markersize=6,
            markerfacecolor="red",
            markeredgecolor="darkred",
        )
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Energy (Ry)", fontsize=12, fontweight="bold")
        ax.set_title(f"Energy Convergence vs {title_metric} for {label}", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        try:
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plot_written = True
        except Exception:
            plot_written = False
        finally:
            plt.close(fig)

    agr_path = dest_dir / f"{base_name}.agr"
    y_min = min(ys)
    y_max = max(ys)
    y_range = y_max - y_min
    if math.isclose(y_range, 0.0, abs_tol=1e-9):
        padding = 0.05
        y_lower = y_min - padding
        y_upper = y_max + padding
    else:
        padding = max(0.05, y_range * 0.05)
        y_lower = y_min - padding
        y_upper = y_max + padding

    try:
        with agr_path.open("w", encoding="ascii") as handle:
            handle.write("# XMGrace file generated by Quantum ESPRESSO Helper\n")
            handle.write("@with g0\n")
            handle.write(f"@title \"Energy Convergence vs {title_metric} for {label}\"\n")
            handle.write(f"@xaxis label \"{x_label}\"\n")
            handle.write("@yaxis label \"Total Energy (Ry)\"\n")
            x_min = min(xs)
            x_max = max(xs)
            handle.write(f"@world {x_min} {y_lower} {x_max} {y_upper}\n")
            handle.write("@type xy\n")
            for x_val, y_val in data:
                handle.write(f"{x_val} {y_val:.10f}\n")
        agr_written = True
    except Exception:
        agr_written = False

    return {
        "plot": str(plot_path) if plot_written else "",
        "agr": str(agr_path) if agr_written else "",
    }
