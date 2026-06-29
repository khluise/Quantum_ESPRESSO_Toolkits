"""Structure conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .structures import (
    AdvancedSettings,
    MAGNETISM_PRESETS,
    PseudoLibrary,
    build_qe_input,
    kpoints_from_kdist,
    read_structure,
    refine_structure_with_spglib,
    XC_FUNCTIONALS,
)

DEFAULT_PROTOCOL = "sssp_precision"
DEFAULT_XC = "PBE"
DEFAULT_KDIST = 0.15


def _determine_k_mesh(settings: AdvancedSettings, cell: np.ndarray, k_dist: float) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if settings.k_mode == "automatic":
        return kpoints_from_kdist(cell, k_dist), (0, 0, 0)
    return settings.k_points, settings.k_shift


def convert_cif_to_qe(
    cif_file: str,
    output_file: str,
    ibrav: Optional[bool] = None,
    sg: Optional[bool] = None,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    xc_label: str = DEFAULT_XC,
    k_distance: float = DEFAULT_KDIST,
    symprec: float = 0.01,
) -> str:
    """Convert a CIF structure to a Quantum ESPRESSO input deck."""

    cif_path = Path(cif_file).expanduser().resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(f"The CIF file '{cif_file}' does not exist.")

    if ibrav and sg:
        raise ValueError("Only one of 'ibrav' or 'sg' may be set. Choose either ibrav or sg, not both.")

    structure = read_structure(str(cif_path), fmt="cif")

    if sg:
        structure = refine_structure_with_spglib(structure, symprec=symprec)
    elif ibrav:
        # For historical compatibility, attempt a mild refinement as well.
        structure = refine_structure_with_spglib(structure, symprec=symprec)

    settings = AdvancedSettings()
    magnetism = MAGNETISM_PRESETS[0]
    pseudo_library = PseudoLibrary()

    xc_tag = XC_FUNCTIONALS.get(xc_label, xc_label.lower())
    pseudo_map, ecutwfc, ecutrho = pseudo_library.resolve(
        protocol,
        xc_tag,
        structure.get_chemical_symbols(),
    )

    cell = np.array(structure.get_cell())
    k_points, k_shift = _determine_k_mesh(settings, cell, k_distance)

    content = build_qe_input(
        structure,
        protocol,
        xc_label,
        magnetism,
        pseudo_map,
        ecutwfc,
        ecutrho,
        k_points,
        k_shift,
        settings,
    )

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content
