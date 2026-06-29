"""Quantum ESPRESSO input parsing and rendering helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import SystemConfig, load_system_config


@dataclass
class ScfSeedParameters:
    source: Path
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    atomic_species: list[tuple[str, str, str]] = field(default_factory=list)
    atomic_positions: list[dict[str, str]] = field(default_factory=list)
    cell_parameters: dict[str, Any] = field(default_factory=dict)
    k_points: dict[str, Any] = field(default_factory=dict)
    raw_cards: dict[str, list[str]] = field(default_factory=dict)


def _parse_namelist_block(lines: Sequence[str]) -> dict[str, str]:
    text = " ".join(line.split("!")[0].strip() for line in lines if line.strip())
    key_pat = r"[A-Za-z0-9_]+(?:\([^)]*\))?"
    pattern = re.compile(
        rf"({key_pat})\s*=\s*(.*?)(?:,(?![^'\"]*[\'\"])|(?=\s+{key_pat}\s*=)|$)",
        re.DOTALL,
    )
    parsed: dict[str, str] = {}
    for match in pattern.finditer(text):
        key = match.group(1).lower()
        value = match.group(2).strip().rstrip(",")
        parsed[key] = value
    return parsed


def _format_namelist(name: str, values: dict[str, str]) -> str:
    lines = [f"&{name.upper()}"]
    keys = list(values.keys())
    pad = max(len(k) for k in keys) if keys else 0
    for key in keys:
        val = values[key]
        lines.append(f"   {key.ljust(pad)} = {val}")
    lines.append("/")
    return "\n".join(lines)


def _format_card_block(lines: list[str]) -> str:
    if not lines:
        return ""
    header, *body = lines
    formatted = [header]
    formatted.extend(f"  {row}" for row in body)
    return "\n".join(formatted)


def parse_scf_seed(seed_path: str) -> ScfSeedParameters:
    path = Path(seed_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"The seed file '{seed_path}' does not exist.")

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    sections: dict[str, dict[str, str]] = {}
    raw_cards: dict[str, list[str]] = {}
    atomic_species: list[tuple[str, str, str]] = []
    atomic_positions: list[dict[str, str]] = []
    cell_parameters: dict[str, Any] = {}
    k_points: dict[str, Any] = {}

    card_headers = {
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "K_POINTS",
        "CELL_PARAMETERS",
    }

    current_section: str | None = None
    section_lines: list[str] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        stripped = line.strip()

        if not stripped or stripped.startswith("!"):
            idx += 1
            continue

        upper = stripped.upper()

        if upper.startswith("&"):
            if current_section and section_lines:
                sections[current_section] = _parse_namelist_block(section_lines)
            current_section = upper.lstrip("&").split()[0]
            section_lines = []
            idx += 1
            continue

        if stripped == "/" and current_section:
            sections[current_section] = _parse_namelist_block(section_lines)
            current_section = None
            section_lines = []
            idx += 1
            continue

        if current_section:
            section_lines.append(stripped)
            idx += 1
            continue

        header_key = upper.split()[0]
        if header_key in card_headers:
            header = stripped
            data: list[str] = []
            idx += 1
            while idx < total:
                probe = lines[idx].strip()
                if not probe:
                    break
                if probe.startswith("!"):
                    idx += 1
                    continue
                probe_upper = probe.upper().split()[0]
                if probe_upper in card_headers or probe.startswith("&"):
                    break
                data.append(probe)
                idx += 1

            raw_cards[header_key] = [header] + data

            if header_key == "ATOMIC_SPECIES":
                for row in data:
                    parts = row.split()
                    if len(parts) >= 3:
                        atomic_species.append((parts[0], parts[1], " ".join(parts[2:])))
            elif header_key == "ATOMIC_POSITIONS":
                mode = ""
                if "(" in header and ")" in header:
                    mode = header.split("(", 1)[1].split(")", 1)[0].strip()
                for row in data:
                    parts = row.split()
                    if len(parts) >= 4:
                        atomic_positions.append(
                            {
                                "label": parts[0],
                                "x": parts[1],
                                "y": parts[2],
                                "z": parts[3],
                                "mode": mode,
                                "extra": " ".join(parts[4:]) if len(parts) > 4 else "",
                            }
                        )
            elif header_key == "CELL_PARAMETERS":
                mode = ""
                if "(" in header and ")" in header:
                    mode = header.split("(", 1)[1].split(")", 1)[0].strip()
                cell_parameters = {"mode": mode, "rows": [row.split() for row in data]}
            elif header_key == "K_POINTS":
                raw_option = header.split(maxsplit=1)
                option = raw_option[1].strip() if len(raw_option) > 1 else ""
                if option.lower().startswith("automatic"):
                    values = data[0].split() if data else []
                    grid = [int(v) for v in values[:3]] if len(values) >= 3 else []
                    offset = [int(v) for v in values[3:6]] if len(values) >= 6 else []
                    k_points = {"type": "automatic", "grid": grid, "offset": offset}
                else:
                    points = []
                    for row in data:
                        parts = row.split()
                        if len(parts) >= 4:
                            try:
                                kx, ky, kz = (float(parts[0]), float(parts[1]), float(parts[2]))
                                weight = float(parts[3])
                            except ValueError:
                                continue
                            points.append({"k": [kx, ky, kz], "weight": weight})
                    k_points = {"type": option or "explicit", "points": points}

            if idx < total and not lines[idx].strip():
                idx += 1
            continue

        idx += 1

    if current_section and section_lines:
        sections[current_section] = _parse_namelist_block(section_lines)

    normalized_sections = {name.lower(): values for name, values in sections.items()}

    return ScfSeedParameters(
        source=path,
        sections=normalized_sections,
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        cell_parameters=cell_parameters,
        k_points=k_points,
        raw_cards=raw_cards,
    )


def render_scf_template(
    seed: ScfSeedParameters,
    *,
    config: SystemConfig | None = None,
    overrides: dict[str, Any] | None = None,
) -> str:
    cfg = config or load_system_config()
    section_overrides = (overrides or {}).get("sections", {}) if overrides else {}
    card_overrides = (overrides or {}).get("cards", {}) if overrides else {}

    final_sections: dict[str, dict[str, str]] = {
        name.lower(): dict(values) for name, values in seed.sections.items()
    }

    for name, updates in section_overrides.items():
        section_key = name.lower()
        dest = final_sections.setdefault(section_key, {})
        for key, value in updates.items():
            entry_key = key.lower()
            if value is None:
                dest.pop(entry_key, None)
            else:
                dest[entry_key] = str(value)
        if not dest:
            final_sections.pop(section_key, None)

    control_section = final_sections.setdefault("control", {})
    control_section["calculation"] = control_section.get("calculation", "'scf'") or "'scf'"
    control_section.setdefault("verbosity", "'high'")
    if cfg.pseudo_directory and not control_section.get("pseudo_dir"):
        control_section["pseudo_dir"] = f"'{cfg.pseudo_directory}'"

    parts: list[str] = []
    base_order = ["control", "system", "electrons", "ions"]
    for key in base_order:
        values = final_sections.pop(key, None)
        if values:
            parts.append(_format_namelist(key, values))

    for key in sorted(final_sections.keys()):
        parts.append(_format_namelist(key, final_sections[key]))

    def render_card(key: str, fallback: list[str] | None = None) -> None:
        if key in card_overrides:
            override = card_overrides[key]
            if override:
                parts.append(_format_card_block(list(override)))
            return
        if fallback:
            parts.append(_format_card_block(list(fallback)))

    render_card("ATOMIC_SPECIES", seed.raw_cards.get("ATOMIC_SPECIES"))
    render_card("ATOMIC_POSITIONS", seed.raw_cards.get("ATOMIC_POSITIONS"))
    render_card("CELL_PARAMETERS", seed.raw_cards.get("CELL_PARAMETERS"))
    render_card("K_POINTS", seed.raw_cards.get("K_POINTS"))

    if "ATOMIC_SPECIES" not in card_overrides and not seed.raw_cards.get("ATOMIC_SPECIES") and seed.atomic_species:
        species_lines = ["ATOMIC_SPECIES"]
        for symbol, mass, pseudo in seed.atomic_species:
            species_lines.append(f"{symbol} {mass} {pseudo}")
        parts.append(_format_card_block(species_lines))

    if "ATOMIC_POSITIONS" not in card_overrides and not seed.raw_cards.get("ATOMIC_POSITIONS") and seed.atomic_positions:
        mode = seed.atomic_positions[0].get("mode", "") if seed.atomic_positions else ""
        header = "ATOMIC_POSITIONS"
        if mode:
            header = f"{header} ({mode})"
        position_lines = [header]
        for atom in seed.atomic_positions:
            extras = f" {atom['extra']}" if atom.get("extra") else ""
            position_lines.append(f"{atom['label']} {atom['x']} {atom['y']} {atom['z']}{extras}")
        parts.append(_format_card_block(position_lines))

    if "CELL_PARAMETERS" not in card_overrides and not seed.raw_cards.get("CELL_PARAMETERS") and seed.cell_parameters:
        mode = seed.cell_parameters.get("mode") or ""
        header = "CELL_PARAMETERS"
        if mode:
            header = f"{header} {mode}"
        rows = seed.cell_parameters.get("rows") or []
        cell_lines = [header]
        for row in rows:
            cell_lines.append(" ".join(row))
        parts.append(_format_card_block(cell_lines))

    if "K_POINTS" not in card_overrides and not seed.raw_cards.get("K_POINTS") and seed.k_points:
        k_block: list[str] = []
        k_type = seed.k_points.get("type", "automatic")
        if k_type == "automatic":
            k_block.append("K_POINTS automatic")
            grid = seed.k_points.get("grid") or []
            offset = seed.k_points.get("offset") or []
            values = [str(v) for v in (grid + offset)]
            while len(values) < 6:
                values.append("0")
            k_block.append(" ".join(values[:6]))
        else:
            header = f"K_POINTS {k_type}".strip()
            k_block.append(header)
            for point in seed.k_points.get("points", []):
                coords = point.get("k", [0.0, 0.0, 0.0])
                weight = point.get("weight", 0.0)
                k_block.append(f"{coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f} {weight:.8f}")
        parts.append(_format_card_block(k_block))

    return "\n\n".join(part for part in parts if part).rstrip() + "\n"
