import json
import math
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from subprocess import CalledProcessError
import subprocess

from typing import Optional, Any, Dict, List, Tuple, Sequence, Mapping

# Suppress spglib dict-interface deprecation warnings emitted via seekpath/spglib.
# Keep this narrow so unrelated deprecations remain visible.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"spglib(\..*)?$",
    message=r".*dict interface \(SpglibDataset\['.*?'\]\) is deprecated.*",
)

# Robust file locking for cross-process coordination with the job daemon
try:
    from filelock import FileLock, Timeout
except Exception:  # pragma: no cover - handled at runtime
    FileLock = None
    Timeout = Exception

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSize, QTimer, QRect
from PyQt6.QtGui import QAction, QCloseEvent, QDesktopServices, QIcon, QIntValidator, QTextCursor, QColor, QFont, QMovie, QPalette, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QPlainTextEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QButtonGroup,
    QTabWidget,
    QTabBar,
    QStackedWidget,
    QSizePolicy,
    QSplitter,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QProgressBar,
    QTableWidgetItem,
    QStyledItemDelegate,
    QInputDialog,
    QMenu,
    QMenuBar,
    QFrame,
    QStyle,
)

from utilities import (
    JobConfig,
    apply_pseudo_settings_to_text,
    ensure_control_verbosity,
    build_job_commands,
    create_run_command,
    build_tpiba_b_block,
    generate_dummy_inputs,
    generate_job_script,
    generate_crystal_kmesh,
    parse_scf_seed,
    render_scf_template,
    load_system_config,
    save_system_config,
    SystemConfig,
    ScfSeedParameters,
    sanitize_job_name,
    load_high_symmetry_path,
    run_command_on_terminal,
    save_text_to_file,
    resolve_executable_bin_dir,
    resolve_qe_bin_dir,
    resolve_thermo_pw_bin_dir,
    trim_qe_input,
)

from qe_helper.converter import DEFAULT_PROTOCOL, DEFAULT_XC, DEFAULT_KDIST
from qe_helper.kpoints import fallback_high_symmetry_path_from_cell
from qe_helper.structures import read_structure
from workflow_builder import (
    WorkflowBuildRequest,
    _generate_workflow_qe_input,
    _run_generator_command,
    copy_selected_workflow_pseudos,
    create_workflow_files,
    normalize_qe_fortran_number,
    render_band_scf_input,
    set_namelist_values,
    pseudo_relativistic_type,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CIF_DEFAULT_SYMPREC = 0.01
RY_TO_EV = 13.605693
COMMAND_PREFS_PATH = PROJECT_ROOT / "command_preferences.json"
QE_INPUT_GENERATOR_DIR = PROJECT_ROOT / "qeinputgeneratorscrapper"
LOCAL_QE_INPUT_GENERATOR_SCRIPT = QE_INPUT_GENERATOR_DIR / "make_pw_input_from_cif.py"
MC_QE_INPUT_GENERATOR_SCRIPT = QE_INPUT_GENERATOR_DIR / "mc_qeinputgenerator_web_automation_client.py"

def _default_pwscf_output_path(structure_path: Path) -> Path:
    stem = structure_path.stem or structure_path.name or "scf"
    return structure_path.with_name(f"{stem}.scf.in")


class QeInputGeneratorThread(QThread):
    log_appended = pyqtSignal(str)
    finished_with_result = pyqtSignal(bool, str, str)

    def __init__(
        self,
        structure_path: str,
        output_path: str,
        use_materials_cloud: bool,
        options: Dict[str, str],
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.structure_path = Path(structure_path).expanduser().resolve()
        self.output_path = (
            Path(output_path).expanduser().resolve()
            if output_path
            else _default_pwscf_output_path(self.structure_path)
        )
        self.use_materials_cloud = use_materials_cloud
        self.options = dict(options)

    def run(self) -> None:  # noqa: D401 (Qt override)
        try:
            output_path = self._generate()
        except Exception as exc:  # noqa: BLE001
            self.finished_with_result.emit(False, str(exc), "")
            return

        self.finished_with_result.emit(True, f"Generated {output_path}", str(output_path))

    def _generate(self) -> Path:
        if not self.structure_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.structure_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_materials_cloud:
            return self._generate_with_materials_cloud()
        return self._generate_locally()

    def _generate_locally(self) -> Path:
        if self.structure_path.suffix.lower() != ".cif":
            raise ValueError("The local generator currently accepts CIF files only. Enable Materials Cloud for other formats.")
        if not LOCAL_QE_INPUT_GENERATOR_SCRIPT.is_file():
            raise FileNotFoundError(f"Local generator script not found: {LOCAL_QE_INPUT_GENERATOR_SCRIPT}")

        protocol = self.options.get("protocol", "balanced")
        magnetism = self.options.get("magnetization", "non-magnetic-fractional")
        command = [
            sys.executable,
            str(LOCAL_QE_INPUT_GENERATOR_SCRIPT),
            str(self.structure_path),
            str(self.output_path),
            "--protocol",
            protocol,
            "--magnetism",
            magnetism,
            "--kpoint-spacing",
            self.options.get("kpoint_spacing", "0.10"),
            "--refine",
            self.options.get("refine_cell", "yes"),
            "--symprec",
            self.options.get("symprec", "1e-5"),
        ]
        self._run_command(command, cwd=QE_INPUT_GENERATOR_DIR)
        if not self.output_path.is_file():
            raise RuntimeError(f"Local generator finished but did not create {self.output_path}")
        return self.output_path

    def _generate_with_materials_cloud(self) -> Path:
        if not MC_QE_INPUT_GENERATOR_SCRIPT.is_file():
            raise FileNotFoundError(f"Materials Cloud generator script not found: {MC_QE_INPUT_GENERATOR_SCRIPT}")

        with tempfile.TemporaryDirectory(prefix="qe_input_generator_", dir=str(self.output_path.parent)) as tmp_dir:
            staging_dir = Path(tmp_dir) / "materials_cloud"
            command = [
                sys.executable,
                str(MC_QE_INPUT_GENERATOR_SCRIPT),
                str(self.structure_path),
                "--out",
                str(staging_dir),
                "--download-zip",
                "--file-format",
                self.options.get("file_format", "auto"),
                "--protocol",
                self.options.get("protocol", "balanced"),
                "--xc",
                self.options.get("xc", "PBEsol"),
                "--magnetism",
                self.options.get("magnetization", "non-magnetic-fractional"),
                "--refine-cell",
                self.options.get("refine_cell", "yes"),
                "--tolerance",
                self.options.get("tolerance", "-5"),
            ]

            pseudo_override = self.options.get("pseudo_override", "")
            if pseudo_override:
                command.extend(["--pseudo-override", pseudo_override])
            kmesh_override = self.options.get("kmesh_override", "")
            if kmesh_override:
                command.extend(["--kmesh-override", kmesh_override])
            threshold_override = self.options.get("threshold_override", "")
            if threshold_override:
                command.extend(["--threshold-override", threshold_override])

            self._run_command(command, cwd=QE_INPUT_GENERATOR_DIR)

            zip_path = staging_dir / "qe_input_and_pseudos.zip"
            if not zip_path.is_file():
                fallback = staging_dir / "scf.in"
                if fallback.is_file():
                    shutil.copy2(fallback, self.output_path)
                    return self.output_path
                raise RuntimeError(f"Materials Cloud did not create the expected ZIP file: {zip_path}")

            extracted_dir = Path(tmp_dir) / "extracted"
            self._safe_extract_zip(zip_path, extracted_dir)

            pwscf_input = self._find_pwscf_input(extracted_dir)
            if pwscf_input is None:
                raise RuntimeError("The Materials Cloud ZIP did not contain PWscf/pwscf.in.")

            shutil.copy2(pwscf_input, self.output_path)

            pseudo_dir = pwscf_input.parent / "pseudo"
            if pseudo_dir.is_dir():
                shutil.copytree(pseudo_dir, self.output_path.parent / "pseudo", dirs_exist_ok=True)
            else:
                self.log_appended.emit("No pseudo directory found in the Materials Cloud ZIP.")

        return self.output_path

    def _run_command(self, command: Sequence[str], *, cwd: Path) -> None:
        _run_generator_command(command, cwd=cwd, log=self.log_appended.emit)

    @staticmethod
    def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise RuntimeError(f"Refusing to extract unsafe ZIP member: {member.filename}")
            archive.extractall(destination)

    @staticmethod
    def _find_pwscf_input(root: Path) -> Optional[Path]:
        preferred = root / "PWscf" / "pwscf.in"
        if preferred.is_file():
            return preferred
        for candidate in root.rglob("pwscf.in"):
            if candidate.is_file():
                return candidate
        return None


def _resolve_executable_candidate(candidate: str, executable_name: str) -> tuple[str, str]:
    bin_dir, resolved_executable = resolve_executable_bin_dir(candidate, executable_name)
    return bin_dir or "", resolved_executable or ""


def _resolve_pw_candidate(candidate: str) -> tuple[str, str]:
    bin_dir, resolved_pw = resolve_qe_bin_dir(candidate)
    return bin_dir or "", resolved_pw or ""


def _resolve_thermo_pw_candidate(candidate: str) -> tuple[str, str]:
    bin_dir, resolved_thermo = resolve_thermo_pw_bin_dir(candidate)
    return bin_dir or "", resolved_thermo or ""


ELEMENT_SYMBOLS: set[str] = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


def normalize_element_symbol(symbol: str) -> str:
    text = (symbol or "").strip()
    if not text:
        return ""
    if len(text) == 1:
        candidate = text.upper()
    else:
        candidate = text[0].upper() + text[1:].lower()
    return candidate if candidate in ELEMENT_SYMBOLS else text.strip()


@dataclass(frozen=True)
class AtomicSpeciesEntry:
    element: str
    mass: Optional[str] = None
    pseudo: Optional[str] = None


class PseudoBrowserDialog(QDialog):
    """Dialog that lists pseudopotentials for the detected atomic species."""

    ELEMENT_ROLE = Qt.ItemDataRole.UserRole + 1
    PATH_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(
        self,
        system_config: SystemConfig,
        species: List[AtomicSpeciesEntry],
        *,
        existing: Optional[Dict[str, str]] = None,
        require_full_relativistic: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Select full-relativistic pseudopotentials"
            if require_full_relativistic
            else "Select pseudopotentials"
        )
        self.resize(900, 560)

        self._system_config = system_config
        self._species = species
        self._require_full_relativistic = require_full_relativistic
        self._elements: List[str] = []
        self._species_defaults: Dict[str, str] = {}
        for entry in species:
            symbol = normalize_element_symbol(entry.element)
            if symbol and symbol not in self._elements:
                self._elements.append(symbol)
            if symbol and entry.pseudo and symbol not in self._species_defaults:
                self._species_defaults[symbol] = entry.pseudo
        self._expected_count = len(self._elements)

        self._assignments: Dict[str, str] = {}
        if existing:
            for symbol, path in existing.items():
                normalized = normalize_element_symbol(symbol)
                if normalized in self._elements and path:
                    self._assignments[normalized] = str(path)

        self._selected_mapping: Dict[str, str] = {}
        self._candidates: Dict[str, List[Path]] = {symbol: [] for symbol in self._elements}
        self._items_by_name: Dict[tuple[str, str], List[QListWidgetItem]] = {}
        self._items_by_path: Dict[tuple[str, str], QListWidgetItem] = {}
        self._suppress_selection_updates = False

        self._build_ui()
        self._gather_candidates()
        self._populate_files_list()
        self._apply_initial_selection()
        self._update_selection_info()
        self._update_missing_label()

    def selected_files(self) -> Dict[str, str]:
        if self._selected_mapping:
            return dict(self._selected_mapping)
        return dict(self._assignments)
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        species_text = ", ".join(self._elements) if self._elements else "<none>"
        pseudo_label = (
            "full-relativistic pseudopotential"
            if self._require_full_relativistic
            else "pseudopotential"
        )
        self.instructionsLabel = QLabel(
            f"Select {self._expected_count} {pseudo_label} file(s) matching: {species_text}"
        )
        self.instructionsLabel.setWordWrap(True)
        layout.addWidget(self.instructionsLabel)

        self.missingLabel = QLabel("", self)
        self.missingLabel.setWordWrap(True)
        self.missingLabel.setStyleSheet("color: #d9534f;")
        layout.addWidget(self.missingLabel)

        self.selectionInfoLabel = QLabel("", self)
        layout.addWidget(self.selectionInfoLabel)

        self.assignmentsSummaryLabel = QLabel("", self)
        self.assignmentsSummaryLabel.setWordWrap(True)
        self.assignmentsSummaryLabel.setStyleSheet("color: #555;")
        layout.addWidget(self.assignmentsSummaryLabel)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.filesList = QListWidget(splitter)
        self.filesList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.filesList.setAlternatingRowColors(True)
        self.filesList.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.preview = QPlainTextEdit(splitter)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        splitter_index = layout.indexOf(splitter)
        if splitter_index >= 0:
            layout.setStretch(splitter_index, 1)

        controls = QHBoxLayout()
        self.addFileBtn = QPushButton("Add file...", self)
        self.clearSelectionBtn = QPushButton("Clear selection", self)
        controls.addWidget(self.addFileBtn)
        controls.addWidget(self.clearSelectionBtn)
        controls.addStretch()
        layout.addLayout(controls)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addWidget(button_box)

        self.filesList.itemSelectionChanged.connect(self._on_selection_changed)
        self.filesList.currentItemChanged.connect(self._on_current_item_changed)
        self.addFileBtn.clicked.connect(self._on_add_file)
        self.clearSelectionBtn.clicked.connect(self._on_clear_selection)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

    def _gather_candidates(self) -> None:
        allowed = set(self._elements)
        if not allowed:
            return

        directories: List[Path] = []
        try:
            dirs = getattr(self._system_config, "pseudo_directories", None)
            if dirs:
                directories = [Path(p) for p in dirs if p]
            elif getattr(self._system_config, "pseudo_directory", None):
                directories = [Path(self._system_config.pseudo_directory)]
        except Exception:
            directories = []

        seen: set[str] = set()
        for base in directories:
            try:
                if not base.exists():
                    continue
                for path in base.rglob("*"):
                    if not path.is_file() or path.suffix.lower() != ".upf":
                        continue
                    element = self._identify_element(path, allowed)
                    if not element:
                        continue
                    if not self._candidate_matches_filter(path):
                        continue
                    key = self._normalise_path(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    resolved_path = Path(key)
                    self._candidates.setdefault(element, []).append(resolved_path)
            except Exception:
                continue

        for element in self._candidates:
            self._candidates[element].sort(key=lambda p: p.name.lower())

    def _populate_files_list(self) -> None:
        self.filesList.clear()
        self._items_by_name.clear()
        self._items_by_path.clear()

        ordered: List[tuple[str, Path]] = []
        for element in self._elements:
            for path in self._candidates.get(element, []):
                ordered.append((element, path))

        for element, path in ordered:
            self._add_candidate_item(element, path)

    def _add_candidate_item(self, element: str, path: Path, *, select: bool = False) -> QListWidgetItem:
        display = f"{element} — {path.name} ({path.parent})"
        item = QListWidgetItem(display)
        item.setToolTip(str(path))
        item.setData(self.ELEMENT_ROLE, element)
        item.setData(self.PATH_ROLE, str(path))
        self.filesList.addItem(item)
        key_name = (element, path.name.lower())
        self._items_by_name.setdefault(key_name, []).append(item)
        resolved = self._normalise_path(path)
        self._items_by_path[(element, resolved)] = item
        if select:
            item.setSelected(True)
        return item

    def _apply_initial_selection(self) -> None:
        selected_elements: set[str] = set()
        self._suppress_selection_updates = True
        try:
            for element, stored_path in list(self._assignments.items()):
                item = self._find_item_by_path(element, stored_path)
                if item:
                    item.setSelected(True)
                    selected_elements.add(element)
                else:
                    self._assignments.pop(element, None)

            for element, pseudo_name in self._species_defaults.items():
                if element in selected_elements:
                    continue
                item = self._find_item_by_name(element, pseudo_name)
                if item:
                    item.setSelected(True)
                    selected_elements.add(element)

            self._auto_assign_missing(selected_elements)
        finally:
            self._suppress_selection_updates = False

    def _auto_assign_missing(self, selected_elements: set[str]) -> None:
        for element in self._elements:
            if element in selected_elements:
                continue
            candidates = self._candidates.get(element) or []
            preferred = self._pick_preferred_candidate(element, candidates)
            if not preferred:
                continue
            item = self._find_item_by_path(element, self._normalise_path(preferred))
            if item:
                item.setSelected(True)
                selected_elements.add(element)

    def _pick_preferred_candidate(self, element: str, candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None

        pattern = re.compile(rf"^{re.escape(element)}[._-]", re.IGNORECASE)
        matches = [path for path in candidates if pattern.match(path.name)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return sorted(matches, key=lambda p: (len(p.name), p.name.lower()))[0]
        return candidates[0]

    def _find_item_by_path(self, element: str, stored_path: str) -> Optional[QListWidgetItem]:
        if not stored_path:
            return None
        normalized = normalize_element_symbol(element)
        resolved = self._normalise_path(stored_path)
        item = self._items_by_path.get((normalized, resolved))
        if item:
            return item
        name = Path(stored_path).name.lower()
        matches = self._items_by_name.get((normalized, name))
        return matches[0] if matches else None

    def _find_item_by_name(self, element: str, pseudo_name: str) -> Optional[QListWidgetItem]:
        if not pseudo_name:
            return None
        normalized = normalize_element_symbol(element)
        key = (normalized, Path(pseudo_name).name.lower())
        matches = self._items_by_name.get(key)
        return matches[0] if matches else None

    def _on_selection_changed(self) -> None:
        if self._suppress_selection_updates:
            return

        selected = self.filesList.selectedItems()
        duplicates: Dict[str, List[QListWidgetItem]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            duplicates.setdefault(element, []).append(item)

        needs_cleanup = any(len(items) > 1 for items in duplicates.values())
        if needs_cleanup:
            current = self.filesList.currentItem()
            self._suppress_selection_updates = True
            try:
                for element, items in duplicates.items():
                    if len(items) <= 1:
                        continue
                    ordered = sorted(items, key=lambda it: self.filesList.row(it))
                    if current and current in ordered and current.data(self.ELEMENT_ROLE) == element:
                        keep = current
                    else:
                        keep = ordered[-1]
                    for item in ordered:
                        if item is keep:
                            continue
                        item.setSelected(False)
            finally:
                self._suppress_selection_updates = False

        self._update_selection_info()
        item = self._current_preview_item()
        self._update_preview(item)

    def _on_current_item_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        self._update_preview(current)

    def _current_preview_item(self) -> Optional[QListWidgetItem]:
        selected = self.filesList.selectedItems()
        if selected:
            return selected[-1]
        return self.filesList.currentItem()

    def _update_preview(self, item: Optional[QListWidgetItem]) -> None:
        if not item:
            self.preview.clear()
            return
        path = item.data(self.PATH_ROLE)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(8000)
        except Exception as exc:  # noqa: BLE001
            snippet = f"Could not read file: {exc}"
        self.preview.setPlainText(snippet)

    def _update_selection_info(self) -> None:
        selected = self.filesList.selectedItems()
        count = len(selected)
        info = f"Selected {count}/{self._expected_count} pseudopotential file(s)."
        mapping: Dict[str, str] = {}
        duplicates: List[str] = []
        path_usage: Dict[str, List[str]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            raw_path = item.data(self.PATH_ROLE) or ""
            normalized_path = self._normalise_path(raw_path)
            path = Path(normalized_path)
            if element in mapping:
                duplicates.append(element)
            else:
                mapping[element] = path.name
            path_usage.setdefault(normalized_path, []).append(element)
        if duplicates:
            dup_text = ", ".join(sorted(set(duplicates)))
            info += f" Duplicate selections for: {dup_text}."

        path_duplicates = []
        for normalized_path, elements in path_usage.items():
            if len(elements) > 1:
                label = f"{Path(normalized_path).name} → {', '.join(sorted(set(elements)))}"
                path_duplicates.append(label)
        if path_duplicates:
            info += f" File reused by multiple elements: {'; '.join(path_duplicates)}."
        self.selectionInfoLabel.setText(info)

        if mapping:
            ordered_elements = [el for el in self._elements if el in mapping]
            remaining = [el for el in mapping if el not in ordered_elements]
            for el in remaining:
                ordered_elements.append(el)
            summary = "; ".join(f"{element} ← {mapping[element]}" for element in ordered_elements)
        else:
            summary = ""
        self.assignmentsSummaryLabel.setText(summary)

    def _update_missing_label(self) -> None:
        missing = [element for element in self._elements if not self._candidates.get(element)]
        if missing:
            names = ", ".join(missing)
            label = "full-relativistic pseudopotentials" if self._require_full_relativistic else "pseudopotentials"
            self.missingLabel.setText(
                f"No {label} were found for: {names}. Use 'Add file...' to include them manually."
            )
            self.missingLabel.show()
        else:
            self.missingLabel.hide()

    def _on_add_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pseudopotential file",
            str(PROJECT_ROOT),
            "Pseudopotentials (*.upf *.UPF);;All files (*)",
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".upf":
            self._show_warning("Invalid file", "Select a file ending with .upf or .UPF.")
            return

        element = self._identify_element(path_obj, set(self._elements))
        if not element:
            self._show_warning(
                "Cannot determine element",
                "The file name does not start with a recognised element symbol (e.g. Mg_, Mg., Mg-).",
            )
            return
        if not self._candidate_matches_filter(path_obj):
            self._show_warning(
                "Pseudopotential does not match",
                "Select a full-relativistic pseudopotential for spin-orbit coupling.",
            )
            return

        resolved = self._normalise_path(path_obj)

        existing_paths = {self._normalise_path(p) for p in self._candidates.get(element, [])}
        if resolved in existing_paths:
            item = self._find_item_by_path(element, resolved)
            if item:
                item.setSelected(True)
                self._update_selection_info()
            return

        resolved_path = Path(resolved)
        self._candidates.setdefault(element, []).append(resolved_path)
        self._candidates[element].sort(key=lambda p: p.name.lower())
        self._populate_files_list()
        item = self._find_item_by_path(element, resolved)
        if item:
            item.setSelected(True)
            self.filesList.scrollToItem(item)
        self._update_missing_label()
        self._update_selection_info()

    def _on_clear_selection(self) -> None:
        self.filesList.clearSelection()
        self._update_selection_info()
        self.preview.clear()

    def _on_accept(self) -> None:
        if self._expected_count == 0:
            self._selected_mapping = {}
            super().accept()
            return

        selected = self.filesList.selectedItems()
        if len(selected) != self._expected_count:
            self._show_warning(
                "Incomplete selection",
                f"Please select exactly {self._expected_count} pseudopotential file(s).",
            )
            return

        mapping: Dict[str, str] = {}
        duplicates: List[str] = []
        path_usage: Dict[str, List[str]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            raw_path = item.data(self.PATH_ROLE)
            normalized_path = self._normalise_path(raw_path)
            if element in mapping:
                duplicates.append(element)
            else:
                mapping[element] = normalized_path
            path_usage.setdefault(normalized_path, []).append(element)

        if duplicates:
            dup_text = ", ".join(sorted(set(duplicates)))
            self._show_warning("Duplicate selection", f"Multiple files selected for: {dup_text}. Please pick one per element.")
            return

        missing = [element for element in self._elements if element not in mapping]
        if missing:
            names = ", ".join(missing)
            self._show_warning("Missing selection", f"Select a pseudopotential for: {names}.")
            return

        reuse_conflicts = []
        for normalized_path, elements in path_usage.items():
            if len(elements) > 1:
                label = f"{Path(normalized_path).name}: {', '.join(sorted(set(elements)))}"
                reuse_conflicts.append(label)
        if reuse_conflicts:
            detail = "; ".join(reuse_conflicts)
            self._show_warning(
                "Duplicate file usage",
                f"The same pseudopotential file is assigned to multiple elements ({detail}). Choose unique files per element.",
            )
            return

        ordered_mapping: Dict[str, str] = {}
        for element in self._elements:
            if element in mapping:
                ordered_mapping[element] = mapping[element]
        for element, path in mapping.items():
            if element not in ordered_mapping:
                ordered_mapping[element] = path

        self._selected_mapping = ordered_mapping
        super().accept()

    @staticmethod
    def _normalise_path(path: Any) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(path)

    @staticmethod
    def _identify_element(path: Path, allowed: set[str]) -> Optional[str]:
        name = path.name
        token_match = re.match(r"([A-Za-z]{1,2})(?=[^A-Za-z])", name)
        if not token_match:
            token = re.split(r"[._-]", name, maxsplit=1)[0]
        else:
            token = token_match.group(1)
        candidate = normalize_element_symbol(token)
        return candidate if candidate in allowed else None

    def _candidate_matches_filter(self, path: Path) -> bool:
        if not self._require_full_relativistic:
            return True
        return pseudo_relativistic_type(path).lower() == "full"

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


class WorkflowPseudoSelectorDialog(QDialog):
    """Assign the pseudopotential roles required by a workflow."""

    ORDINARY_ROLE = "ordinary"
    FULL_ROLE = "full"

    def __init__(
        self,
        system_config: SystemConfig,
        species: List[AtomicSpeciesEntry],
        *,
        require_pseudopotentials: bool,
        require_full_relativistic: bool,
        existing_pseudopotentials: Optional[Dict[str, str]] = None,
        existing_full_relativistic: Optional[Dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select workflow pseudopotentials")
        self.resize(1040, 620)

        self._system_config = system_config
        self._require_pseudopotentials = require_pseudopotentials
        self._require_full_relativistic = require_full_relativistic
        self._elements: List[str] = []
        for entry in species:
            symbol = normalize_element_symbol(entry.element)
            if symbol and symbol not in self._elements:
                self._elements.append(symbol)

        self._existing = {
            self.ORDINARY_ROLE: self._normalize_mapping(existing_pseudopotentials),
            self.FULL_ROLE: self._normalize_mapping(existing_full_relativistic),
        }
        self._candidates: Dict[str, Dict[str, List[Path]]] = {
            self.ORDINARY_ROLE: {element: [] for element in self._elements},
            self.FULL_ROLE: {element: [] for element in self._elements},
        }
        self._candidate_keys: Dict[str, Dict[str, set[str]]] = {
            self.ORDINARY_ROLE: {element: set() for element in self._elements},
            self.FULL_ROLE: {element: set() for element in self._elements},
        }
        self._combos: Dict[Tuple[str, str], QComboBox] = {}
        self._selected_pseudopotentials: Dict[str, str] = {}
        self._selected_full_relativistic: Dict[str, str] = {}

        self._gather_candidates()
        self._include_existing_candidates()
        self._build_ui()
        self._populate_assignment_table()
        self._update_status_labels()

    def selected_pseudopotentials(self) -> Dict[str, str]:
        if self._selected_pseudopotentials:
            return dict(self._selected_pseudopotentials)
        return self._selected_mapping(self.ORDINARY_ROLE)

    def selected_full_relativistic_pseudopotentials(self) -> Dict[str, str]:
        if self._selected_full_relativistic:
            return dict(self._selected_full_relativistic)
        return self._selected_mapping(self.FULL_ROLE)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        species_text = ", ".join(self._elements) if self._elements else "<none>"
        roles: List[str] = []
        if self._require_pseudopotentials:
            roles.append("pseudopotential")
        if self._require_full_relativistic:
            roles.append("full-relativistic pseudopotential")
        role_text = " and ".join(roles) if roles else "pseudopotential"
        self.instructionsLabel = QLabel(
            f"Assign the required {role_text} file(s) for: {species_text}",
            self,
        )
        self.instructionsLabel.setWordWrap(True)
        layout.addWidget(self.instructionsLabel)

        self.missingCandidatesLabel = QLabel("", self)
        self.missingCandidatesLabel.setWordWrap(True)
        self.missingCandidatesLabel.setStyleSheet("color: #d9534f;")
        layout.addWidget(self.missingCandidatesLabel)

        self.selectionStatusLabel = QLabel("", self)
        self.selectionStatusLabel.setWordWrap(True)
        layout.addWidget(self.selectionStatusLabel)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.assignmentsTable = QTableWidget(len(self._elements), 3, splitter)
        self.assignmentsTable.setHorizontalHeaderLabels(
            ["Species", "Pseudopotential", "Full-relativistic pseudopotential"]
        )
        self.assignmentsTable.verticalHeader().setVisible(False)
        self.assignmentsTable.setAlternatingRowColors(True)
        self.assignmentsTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.assignmentsTable.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.assignmentsTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        if not self._require_pseudopotentials:
            self.assignmentsTable.setColumnHidden(1, True)
        if not self._require_full_relativistic:
            self.assignmentsTable.setColumnHidden(2, True)

        self.preview = QPlainTextEdit(splitter)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview.setPlaceholderText("Select an assignment to preview its pseudopotential file.")
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.addFileButton = QPushButton("Add file...", self)
        self.clearSelectionsButton = QPushButton("Clear selections", self)
        actions.addWidget(self.addFileButton)
        actions.addWidget(self.clearSelectionsButton)
        actions.addStretch()
        layout.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        layout.addWidget(buttons)

        self.assignmentsTable.currentCellChanged.connect(self._preview_current_assignment)
        self.addFileButton.clicked.connect(self._add_file)
        self.clearSelectionsButton.clicked.connect(self._clear_selections)
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)

    def _populate_assignment_table(self) -> None:
        for row, element in enumerate(self._elements):
            element_item = QTableWidgetItem(element)
            element_item.setFlags(element_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.assignmentsTable.setItem(row, 0, element_item)
            if self._require_pseudopotentials:
                self._set_combo(row, 1, element, self.ORDINARY_ROLE)
            if self._require_full_relativistic:
                self._set_combo(row, 2, element, self.FULL_ROLE)

    def _set_combo(self, row: int, column: int, element: str, role: str) -> None:
        combo = QComboBox(self.assignmentsTable)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._populate_combo(combo, element, role)
        combo.currentIndexChanged.connect(
            lambda _index, watched=combo: self._handle_combo_changed(watched),
        )
        self.assignmentsTable.setCellWidget(row, column, combo)
        self._combos[(role, element)] = combo

    def _populate_combo(self, combo: QComboBox, element: str, role: str) -> None:
        previous = str(combo.currentData() or "") if combo.count() else ""
        existing = self._existing.get(role, {}).get(element, "")
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select file...", "")
        for path in self._candidates[role].get(element, []):
            combo.addItem(f"{path.name} ({path.parent})", str(path))

        selection = previous or existing
        index = self._combo_index_for_path(combo, selection)
        if index < 0:
            preferred = self._preferred_candidate(element, role)
            index = self._combo_index_for_path(combo, str(preferred or ""))
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._set_combo_tooltip(combo)

    def _handle_combo_changed(self, combo: QComboBox) -> None:
        self._set_combo_tooltip(combo)
        self._preview_path(str(combo.currentData() or ""))
        self._update_status_labels()

    def _set_combo_tooltip(self, combo: QComboBox) -> None:
        combo.setToolTip(str(combo.currentData() or ""))

    def _preview_current_assignment(self, row: int, column: int, _old_row: int, _old_column: int) -> None:
        widget = self.assignmentsTable.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            self._preview_path(str(widget.currentData() or ""))
            return
        self.preview.clear()

    def _preview_path(self, raw_path: str) -> None:
        path = Path(raw_path).expanduser() if raw_path else None
        if path is None or not path.is_file():
            self.preview.clear()
            return
        try:
            self.preview.setPlainText(path.read_text(encoding="utf-8", errors="ignore")[:8000])
        except Exception as exc:  # noqa: BLE001
            self.preview.setPlainText(f"Could not read file: {exc}")

    def _gather_candidates(self) -> None:
        allowed = set(self._elements)
        if not allowed:
            return
        for base in self._pseudo_directories():
            try:
                if not base.exists():
                    continue
                for path in base.rglob("*"):
                    if not path.is_file() or path.suffix.lower() != ".upf":
                        continue
                    element = PseudoBrowserDialog._identify_element(path, allowed)
                    if element:
                        self._add_candidate(element, path)
            except Exception:
                continue

    def _pseudo_directories(self) -> List[Path]:
        directories = getattr(self._system_config, "pseudo_directories", None)
        if directories:
            return [Path(path) for path in directories if path]
        legacy = getattr(self._system_config, "pseudo_directory", None)
        return [Path(legacy)] if legacy else []

    def _include_existing_candidates(self) -> None:
        for mapping in self._existing.values():
            for element, raw_path in mapping.items():
                self._add_candidate(element, Path(raw_path).expanduser())

    def _add_candidate(self, element: str, path: Path) -> None:
        if element not in self._elements or not path.is_file():
            return
        role = self.FULL_ROLE if pseudo_relativistic_type(path).lower() == "full" else self.ORDINARY_ROLE
        resolved = PseudoBrowserDialog._normalise_path(path)
        if resolved in self._candidate_keys[role][element]:
            return
        self._candidate_keys[role][element].add(resolved)
        self._candidates[role][element].append(Path(resolved))
        self._candidates[role][element].sort(key=lambda item: item.name.lower())

    def _add_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pseudopotential file",
            str(PROJECT_ROOT),
            "Pseudopotentials (*.upf *.UPF);;All files (*)",
        )
        if not file_path:
            return

        path = Path(file_path).expanduser()
        if path.suffix.lower() != ".upf":
            self._show_warning("Invalid file", "Select a file ending with .upf or .UPF.")
            return
        element = PseudoBrowserDialog._identify_element(path, set(self._elements))
        if not element:
            self._show_warning(
                "Cannot determine element",
                "The file name must start with one of the workflow species.",
            )
            return

        is_full = pseudo_relativistic_type(path).lower() == "full"
        if is_full and not self._require_full_relativistic:
            self._show_warning(
                "Full-relativistic file not required",
                "This workflow selection only needs ordinary pseudopotentials.",
            )
            return
        if not is_full and not self._require_pseudopotentials:
            self._show_warning(
                "Ordinary file not required",
                "This workflow selection only needs full-relativistic pseudopotentials.",
            )
            return

        self._add_candidate(element, path)
        role = self.FULL_ROLE if is_full else self.ORDINARY_ROLE
        combo = self._combos.get((role, element))
        if combo is not None:
            self._populate_combo(combo, element, role)
            index = self._combo_index_for_path(combo, str(path))
            if index >= 0:
                combo.setCurrentIndex(index)
        self._update_status_labels()

    def _clear_selections(self) -> None:
        for combo in self._combos.values():
            combo.setCurrentIndex(0)
        self.preview.clear()
        self._update_status_labels()

    def _accept_selection(self) -> None:
        pseudopotentials = self._selected_mapping(self.ORDINARY_ROLE)
        full_relativistic = self._selected_mapping(self.FULL_ROLE)
        error = self._selection_error(pseudopotentials, full_relativistic)
        if error:
            self._show_warning("Incomplete selection", error)
            return
        self._selected_pseudopotentials = pseudopotentials
        self._selected_full_relativistic = full_relativistic
        self.accept()

    def _selection_error(
        self,
        pseudopotentials: Dict[str, str],
        full_relativistic: Dict[str, str],
    ) -> str:
        if self._require_pseudopotentials:
            missing = [element for element in self._elements if not pseudopotentials.get(element)]
            if missing:
                return f"Select pseudopotentials for: {', '.join(missing)}."
        if self._require_full_relativistic:
            missing = [element for element in self._elements if not full_relativistic.get(element)]
            if missing:
                return f"Select full-relativistic pseudopotentials for: {', '.join(missing)}."
            invalid = [
                element
                for element, raw_path in full_relativistic.items()
                if pseudo_relativistic_type(Path(raw_path)).lower() != "full"
            ]
            if invalid:
                return f"Full-relativistic files are required for: {', '.join(invalid)}."
        return ""

    def _selected_mapping(self, role: str) -> Dict[str, str]:
        if role == self.ORDINARY_ROLE and not self._require_pseudopotentials:
            return {}
        if role == self.FULL_ROLE and not self._require_full_relativistic:
            return {}
        mapping: Dict[str, str] = {}
        for element in self._elements:
            combo = self._combos.get((role, element))
            raw_path = str(combo.currentData() or "") if combo is not None else ""
            if raw_path:
                mapping[element] = raw_path
        return mapping

    def _update_status_labels(self) -> None:
        status: List[str] = []
        if self._require_pseudopotentials:
            selected = len(self._selected_mapping(self.ORDINARY_ROLE))
            status.append(f"Pseudopotentials: {selected}/{len(self._elements)}")
        if self._require_full_relativistic:
            selected = len(self._selected_mapping(self.FULL_ROLE))
            status.append(f"Full-relativistic: {selected}/{len(self._elements)}")
        self.selectionStatusLabel.setText(" | ".join(status))

        missing: List[str] = []
        if self._require_pseudopotentials:
            elements = [element for element in self._elements if not self._candidates[self.ORDINARY_ROLE][element]]
            if elements:
                missing.append(f"pseudopotentials for {', '.join(elements)}")
        if self._require_full_relativistic:
            elements = [element for element in self._elements if not self._candidates[self.FULL_ROLE][element]]
            if elements:
                missing.append(f"full-relativistic pseudopotentials for {', '.join(elements)}")
        if missing:
            self.missingCandidatesLabel.setText(
                "No " + "; no ".join(missing) + ". Use Add file to include them manually."
            )
            self.missingCandidatesLabel.show()
        else:
            self.missingCandidatesLabel.hide()

    def _normalize_mapping(self, mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for element, raw_path in (mapping or {}).items():
            symbol = normalize_element_symbol(element)
            if symbol in self._elements and raw_path:
                normalized[symbol] = str(raw_path)
        return normalized

    def _preferred_candidate(self, element: str, role: str) -> Optional[Path]:
        candidates = self._candidates[role].get(element) or []
        if not candidates:
            return None
        pattern = re.compile(rf"^{re.escape(element)}[._-]", re.IGNORECASE)
        matches = [path for path in candidates if pattern.match(path.name)]
        return sorted(matches or candidates, key=lambda path: (len(path.name), path.name.lower()))[0]

    @staticmethod
    def _combo_index_for_path(combo: QComboBox, raw_path: str) -> int:
        if not raw_path:
            return -1
        expected = PseudoBrowserDialog._normalise_path(raw_path)
        for index in range(combo.count()):
            value = str(combo.itemData(index) or "")
            if value and PseudoBrowserDialog._normalise_path(value) == expected:
                return index
        return -1

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


class WorkflowComboPopupDelegate(QStyledItemDelegate):
    """Paint workflow combo popup rows so they stay readable in all states."""

    def paint(self, painter: QPainter, option: Any, index: Any) -> None:  # noqa: D401 (Qt override)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        if selected or hovered:
            painter.fillRect(option.rect, QColor(PRIMARY_COLOR))
            painter.setPen(QColor(TEXT_LIGHT))
        else:
            painter.fillRect(option.rect, QColor("#FFFFFF"))
            painter.setPen(QColor(TEXT_DARK))

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        painter.drawText(
            option.rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )
        painter.restore()

    def sizeHint(self, option: Any, index: Any) -> QSize:  # noqa: D401 (Qt override)
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height(), 28))


class TrimmedDoubleSpinBox(QDoubleSpinBox):
    """Display double values without unnecessary trailing zeroes."""

    def textFromValue(self, value: float) -> str:  # noqa: D401 (Qt override)
        text = f"{value:.{self.decimals()}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text


class ReorderablePathTableWidget(QTableWidget):
    """Small table helper that lets selected path rows be reordered by dragging."""

    rowDropped = pyqtSignal(int, int)

    def __init__(self, rows: int, columns: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(rows, columns, parent)
        self._drag_current_row = -1

    def _event_row(self, event: Any) -> int:
        try:
            point = event.position().toPoint()
        except Exception:
            point = event.pos()
        row = self.rowAt(int(point.y()))
        if row < 0 and self.rowCount() > 0:
            row = 0 if int(point.y()) < 0 else self.rowCount() - 1
        return row

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_current_row = self._event_row(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        if self._drag_current_row >= 0 and event.buttons() & Qt.MouseButton.LeftButton:
            target_row = self._event_row(event)
            if target_row >= 0 and target_row != self._drag_current_row:
                source_row = self._drag_current_row
                self._drag_current_row = target_row
                self.rowDropped.emit(source_row, target_row)
                self.selectRow(target_row)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        self._drag_current_row = -1
        super().mouseReleaseEvent(event)


class NoWheelTabBar(QTabBar):
    """Keep mouse-wheel scrolling from changing the selected tab."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        event.ignore()


class NoWheelTabWidget(QTabWidget):
    """Tab widget whose tab bar ignores mouse-wheel tab changes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTabBar(NoWheelTabBar(self))


class JobRunner(QThread):
    """Execute a single job record sequentially in a background thread."""

    progress_updated = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    log_appended = pyqtSignal(str)
    finished_with_success = pyqtSignal(bool)

    def __init__(self, commands: List[str], working_directory: Path, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._commands = list(commands)
        self._working_directory = working_directory
        self._should_stop = False
        self._current_process: Optional[subprocess.Popen] = None
        self._current_pgid: Optional[int] = None

    def request_stop(self) -> None:
        self._should_stop = True
        self._send_interrupt()

    def _register_process(self, process: subprocess.Popen) -> None:
        self._current_process = process
        try:
            self._current_pgid = os.getpgid(process.pid)
        except Exception:
            self._current_pgid = None

    def _clear_process_reference(self) -> None:
        self._current_process = None
        self._current_pgid = None

    def _send_interrupt(self) -> None:
        process = self._current_process
        if process is None or process.poll() is not None:
            return
        try:
            if self._current_pgid is not None and os.name != "nt":
                os.killpg(self._current_pgid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

    def run(self) -> None:  # noqa: D401 (Qt override)
        if not self._commands:
            self.status_changed.emit("Completed")
            self.progress_updated.emit(100)
            self.finished_with_success.emit(True)
            return

        total = len(self._commands)
        for idx, command in enumerate(self._commands, start=1):
            if self._should_stop:
                self.status_changed.emit("Stopped")
                self.finished_with_success.emit(False)
                return

            step_label = f"Running ({idx}/{total})"
            self.status_changed.emit(step_label)

            try:
                result = run_command_on_terminal(
                    command,
                    cwd=self._working_directory,
                    show_output=True,
                    check=False,
                    stop_event=lambda: self._should_stop,
                    on_start=self._register_process,
                )
            except Exception as exc:  # noqa: BLE001
                self.log_appended.emit(str(exc))
                self.status_changed.emit("Failed")
                self.finished_with_success.emit(False)
                return
            finally:
                self._clear_process_reference()

            stdout_text = (result.stdout or "").strip()
            stderr_text = (result.stderr or "").strip()
            if stdout_text:
                self.log_appended.emit(stdout_text)

            if self._should_stop:
                self.status_changed.emit("Stopped")
                self.finished_with_success.emit(False)
                return

            if result.returncode != 0:
                if stderr_text:
                    self.log_appended.emit(stderr_text)
                self.status_changed.emit("Failed")
                self.finished_with_success.emit(False)
                return

            if stderr_text:
                self.log_appended.emit(stderr_text)

            progress = int(idx / total * 100)
            self.progress_updated.emit(progress)

        self.status_changed.emit("Completed")
        self.progress_updated.emit(100)
        self.finished_with_success.emit(True)


class PspBuilderThread(QThread):
    """Run build_psp.prepare_all(...) in a background thread and emit logs.

    This imports the local `build_psp` module inside the thread to avoid
    performing network I/O at import time.
    """

    log_appended = pyqtSignal(str)
    finished_with_success = pyqtSignal(bool)

    def __init__(self, qe_path: str, dest: Path, run_flag: bool, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.qe_path = qe_path
        self.dest = dest
        self.run_flag = run_flag
        self._proc: subprocess.Popen | None = None
        self._should_stop = False

    def request_stop(self) -> None:
        """Request the running subprocess to terminate."""
        self._should_stop = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self) -> None:
        # Build the command to run the bundled script so we can capture realtime output
        script_path = PROJECT_ROOT / "build_psp.py"
        if not script_path.exists():
            try:
                self.log_appended.emit(f"build_psp.py not found at {script_path}")
            except Exception:
                pass
            self.finished_with_success.emit(False)
            return

        cmd = [sys.executable, str(script_path), "--qe-path", str(self.qe_path or ""), "--dest", str(self.dest)]
        if self.run_flag:
            cmd.append("--run")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                self.log_appended.emit(f"Failed to start process: {exc}")
            except Exception:
                pass
            self.finished_with_success.emit(False)
            return

        try:
            assert self._proc is not None
            # Read stdout line by line and emit logs
            for raw in self._proc.stdout or []:
                if raw is None:
                    break
                line = raw.rstrip("\n")
                try:
                    self.log_appended.emit(line)
                except Exception:
                    pass
                if self._should_stop:
                    break

            # Wait for process to exit (if not already)
            if self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=1)
                except Exception:
                    # If still running after stop request, kill it
                    if self._should_stop:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass

            returncode = self._proc.returncode
            success = returncode == 0 and not self._should_stop
            self.finished_with_success.emit(bool(success))
        except Exception as exc:  # noqa: BLE001
            try:
                self.log_appended.emit(f"Builder error: {exc}")
            except Exception:
                pass
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.kill()
            except Exception:
                pass
            self.finished_with_success.emit(False)


class QePathSearchThread(QThread):
    log_appended = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    finished_with_results = pyqtSignal(list)

    def __init__(
        self,
        parent: Optional[Any] = None,
        *,
        executable_name: str = "pw.x",
        display_name: str = "Quantum ESPRESSO",
    ) -> None:
        super().__init__(parent)
        self.executable_name = executable_name
        self.display_name = display_name

    def _run_find(self, root: Path) -> list[str]:
        self.status_changed.emit(f"Searching {root} for {self.executable_name}...")
        self.log_appended.emit(f"Searching for {self.executable_name} in {root}...")
        try:
            result = subprocess.run(
                [
                    "find",
                    str(root),
                    "-name",
                    self.executable_name,
                    "(",
                    "-type",
                    "f",
                    "-o",
                    "-type",
                    "l",
                    ")",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_appended.emit(f"Search failed: {exc}")
            return []

        seen: set[str] = set()
        matches: list[str] = []
        for line in (result.stdout or "").splitlines():
            candidate = line.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)
        return matches

    def run(self) -> None:  # noqa: D401 (Qt override)
        home_results = self._run_find(Path.home())
        if home_results:
            self.finished_with_results.emit(home_results)
            return

        self.status_changed.emit(f"Searching system-wide for {self.executable_name}...")
        self.log_appended.emit(
            f"No {self.executable_name} found in the home directory. "
            "Searching system-wide (this may take a while)..."
        )
        system_results = self._run_find(Path("/"))
        self.finished_with_results.emit(system_results)


class QeInstallThread(QThread):
    log_appended = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    finished_with_result = pyqtSignal(bool, str, list, list)

    def __init__(
        self,
        install_dir: Path,
        sudo_password: str,
        parent: Optional[Any] = None,
        *,
        executable_name: str = "pw.x",
        display_name: str = "Quantum ESPRESSO",
        install_args: Optional[list[str]] = None,
        extra_executable_name: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.install_dir = install_dir
        self._sudo_password = sudo_password
        self.executable_name = executable_name
        self.display_name = display_name
        self.install_args = list(install_args or ["--with-thermo"])
        self.extra_executable_name = extra_executable_name
        self._proc: subprocess.Popen | None = None
        self._should_stop = False
        self._stage: Optional[str] = None

    def request_stop(self) -> None:
        self._should_stop = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def _set_stage(self, stage: str) -> None:
        if stage == self._stage:
            return
        self._stage = stage
        labels = {
            "download": f"Downloading {self.display_name}...",
            "extract": "Extracting source files...",
            "compile": f"Compiling {self.display_name}...",
            "validate": "Validating installation...",
        }
        self.status_changed.emit(labels.get(stage, "Working..."))

    def _filter_valid_candidates(self, paths: list[str], executable_name: str) -> list[str]:
        valid: list[str] = []
        for candidate in paths:
            bin_dir, _ = _resolve_executable_candidate(candidate, executable_name)
            if bin_dir:
                valid.append(candidate)
        return valid

    def _find_binaries(self, executable_name: str) -> list[str]:
        try:
            result = subprocess.run(
                [
                    "find",
                    str(self.install_dir),
                    "-name",
                    executable_name,
                    "(",
                    "-type",
                    "f",
                    "-o",
                    "-type",
                    "l",
                    ")",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_appended.emit(f"Validation failed: {exc}")
            return []

        seen: set[str] = set()
        matches: list[str] = []
        for line in (result.stdout or "").splitlines():
            candidate = line.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)
        return matches

    def run(self) -> None:  # noqa: D401 (Qt override)
        tool_path = PROJECT_ROOT / "qe_helper" / "QE_download_tool"
        if not tool_path.exists():
            self.finished_with_result.emit(
                False,
                f"QE_download_tool not found at {tool_path}.",
                [],
                [],
            )
            return

        self._set_stage("download")
        command = ["sudo", "-S", "-p", "", str(tool_path), *self.install_args]
        self.log_appended.emit(f"Running with sudo: {tool_path} {' '.join(self.install_args)}")

        try:
            self._proc = subprocess.Popen(
                command,
                cwd=str(self.install_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            if self._proc.stdin is not None:
                self._proc.stdin.write(f"{self._sudo_password}\n")
                self._proc.stdin.flush()
                self._proc.stdin.close()
            self._sudo_password = ""
        except Exception as exc:  # noqa: BLE001
            self.finished_with_result.emit(False, f"Failed to start QE_download_tool: {exc}", [], [])
            return

        output_lines: list[str] = []
        try:
            for raw in self._proc.stdout or []:
                if raw is None:
                    break
                line = raw.rstrip("\n")
                if line:
                    output_lines.append(line)
                    self.log_appended.emit(line)

                if self._should_stop:
                    break

                lowered = line.lower()
                if "download" in lowered:
                    self._set_stage("download")
                elif "extract" in lowered or "unpack" in lowered or "tar" in lowered:
                    self._set_stage("extract")
                elif "compile" in lowered or "building" in lowered or "make" in lowered:
                    self._set_stage("compile")

            if self._should_stop:
                try:
                    if self._proc and self._proc.poll() is None:
                        self._proc.terminate()
                except Exception:
                    pass
                self.finished_with_result.emit(False, "Installation canceled by user.", [], [])
                return

            assert self._proc is not None
            return_code = self._proc.wait()
        except Exception as exc:  # noqa: BLE001
            self.finished_with_result.emit(False, f"Installation failed: {exc}", [], [])
            return

        if return_code != 0:
            tail = "\n".join(output_lines[-20:]) if output_lines else ""
            message = f"QE_download_tool exited with code {return_code}."
            if tail:
                message = f"{message}\n\n{tail}"
            self.finished_with_result.emit(False, message, [], [])
            return

        self._set_stage("validate")
        preferred = str(self.install_dir / "bin" / self.executable_name)
        preferred_valid = self._filter_valid_candidates([preferred], self.executable_name)

        paths = preferred_valid or self._filter_valid_candidates(
            self._find_binaries(self.executable_name),
            self.executable_name,
        )
        if not paths:
            self.finished_with_result.emit(
                False,
                f"QE_download_tool completed, but no {self.executable_name} binary was found "
                "in the installation directory.",
                [],
                [],
            )
            return

        extra_paths: list[str] = []
        if self.extra_executable_name:
            extra_paths = self._filter_valid_candidates(
                self._find_binaries(self.extra_executable_name),
                self.extra_executable_name,
            )

        self.finished_with_result.emit(True, "", paths, extra_paths)


class QeConfigureProgressDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None,
        *,
        allow_cancel: bool = False,
        initial_status: str = "Working...",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(720, 420)

        self._canceled = False
        self._finished = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.statusLabel = QLabel(initial_status, self)
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 0)
        layout.addWidget(self.progressBar)

        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self.logView.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.logView.setPlaceholderText("Waiting for output...")
        layout.addWidget(self.logView, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.cancelButton = QPushButton("Cancel", self)
        self.closeButton = QPushButton("Close", self)
        self.closeButton.setEnabled(False)
        buttons_layout.addWidget(self.cancelButton)
        buttons_layout.addWidget(self.closeButton)
        layout.addLayout(buttons_layout)

        if allow_cancel:
            self.cancelButton.clicked.connect(self._handle_cancel)
        else:
            self.cancelButton.hide()

        self.closeButton.clicked.connect(self.accept)

    def append_log(self, text: str) -> None:
        if not text:
            return
        cursor = self.logView.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text if text.endswith("\n") else f"{text}\n")
        self.logView.setTextCursor(cursor)
        self.logView.ensureCursorVisible()

    def set_status(self, text: str) -> None:
        if text:
            self.statusLabel.setText(text)

    def set_finished(self, success: bool) -> None:
        if self._finished:
            return
        self._finished = True
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(1)
        self.cancelButton.setEnabled(False)
        self.closeButton.setEnabled(True)
        state = "completed" if success else "finished with errors"
        self.statusLabel.setText(f"Operation {state}. You may close this dialog.")

    def was_canceled(self) -> bool:
        return self._canceled

    def _handle_cancel(self) -> None:
        if self._finished:
            self.reject()
            return
        self._canceled = True
        self.cancelButton.setEnabled(False)
        self.statusLabel.setText("Cancel requested. Waiting for the installer to stop...")
        self.append_log("\n[User requested cancellation]\n")
        self.cancel_requested.emit()

    def reject(self) -> None:  # noqa: D401 (Qt override)
        if not self._finished:
            self._canceled = True
        super().reject()


class DownloadProgressDialog(QDialog):
    """Modal dialog that shows progress and logs for the pseudo builder."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preparing pseudopotentials")
        self.setModal(True)
        self.resize(720, 420)

        self._canceled = False
        self._finished = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.statusLabel = QLabel("Starting pseudopotential preparation…", self)
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 0)  # Indeterminate by default
        layout.addWidget(self.progressBar)

        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self.logView.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.logView.setPlaceholderText("Waiting for output…")
        layout.addWidget(self.logView, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.cancelButton = QPushButton("Cancel", self)
        self.closeButton = QPushButton("Close", self)
        self.closeButton.setEnabled(False)
        buttons_layout.addWidget(self.cancelButton)
        buttons_layout.addWidget(self.closeButton)
        layout.addLayout(buttons_layout)

        self.cancelButton.clicked.connect(self._handle_cancel)
        self.closeButton.clicked.connect(self.accept)

    def append_log(self, text: str) -> None:
        if not text:
            return
        cursor = self.logView.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text if text.endswith("\n") else f"{text}\n")
        self.logView.setTextCursor(cursor)
        self.logView.ensureCursorVisible()

    def set_finished(self, success: bool) -> None:
        if self._finished:
            return
        self._finished = True
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(1)
        self.cancelButton.setEnabled(False)
        self.closeButton.setEnabled(True)
        state = "completed" if success else "finished with errors"
        self.statusLabel.setText(f"Preparation {state}. You may close this dialog.")

    def was_canceled(self) -> bool:
        return self._canceled

    def _handle_cancel(self) -> None:
        if self._finished:
            self.reject()
            return
        self._canceled = True
        self.cancelButton.setEnabled(False)
        self.statusLabel.setText("Cancel requested. Waiting for builder to stop…")
        self.append_log("\n[User requested cancellation]\n")

    def reject(self) -> None:  # noqa: D401 (Qt override)
        if not self._finished:
            self._canceled = True
        super().reject()

    def _add_candidate_item(self, element: str, path: Path, *, select: bool = False) -> QListWidgetItem:
        display = f"{element} — {path.name} ({path.parent})"
        item = QListWidgetItem(display)
        item.setToolTip(str(path))
        item.setData(self.ELEMENT_ROLE, element)
        item.setData(self.PATH_ROLE, str(path))
        self.filesList.addItem(item)
        key_name = (element, path.name.lower())
        self._items_by_name.setdefault(key_name, []).append(item)
        resolved = self._normalise_path(path)
        self._items_by_path[(element, resolved)] = item
        if select:
            item.setSelected(True)
        return item

    def _apply_initial_selection(self) -> None:
        selected_elements: set[str] = set()
        self._suppress_selection_updates = True
        try:
            for element, stored_path in list(self._assignments.items()):
                item = self._find_item_by_path(element, stored_path)
                if item:
                    item.setSelected(True)
                    selected_elements.add(element)
                else:
                    self._assignments.pop(element, None)

            for element, pseudo_name in self._species_defaults.items():
                if element in selected_elements:
                    continue
                item = self._find_item_by_name(element, pseudo_name)
                if item:
                    item.setSelected(True)
                    selected_elements.add(element)

            self._auto_assign_missing(selected_elements)
        finally:
            self._suppress_selection_updates = False

    def _find_item_by_path(self, element: str, stored_path: str) -> Optional[QListWidgetItem]:
        if not stored_path:
            return None
        normalized = normalize_element_symbol(element)
        resolved = self._normalise_path(stored_path)
        item = self._items_by_path.get((normalized, resolved))
        if item:
            return item
        name = Path(stored_path).name.lower()
        matches = self._items_by_name.get((normalized, name))
        return matches[0] if matches else None

    def _find_item_by_name(self, element: str, pseudo_name: str) -> Optional[QListWidgetItem]:
        if not pseudo_name:
            return None
        normalized = normalize_element_symbol(element)
        key = (normalized, Path(pseudo_name).name.lower())
        matches = self._items_by_name.get(key)
        return matches[0] if matches else None

    def _auto_assign_missing(self, selected_elements: set[str]) -> None:
        for element in self._elements:
            if element in selected_elements:
                continue
            candidates = self._candidates.get(element) or []
            preferred = self._pick_preferred_candidate(element, candidates)
            if not preferred:
                continue
            item = self._find_item_by_path(element, self._normalise_path(preferred))
            if item:
                item.setSelected(True)
                selected_elements.add(element)

    def _pick_preferred_candidate(self, element: str, candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None

        pattern = re.compile(rf"^{re.escape(element)}[._-]", re.IGNORECASE)
        matches = [path for path in candidates if pattern.match(path.name)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return sorted(matches, key=lambda p: (len(p.name), p.name.lower()))[0]
        return candidates[0]

    def _on_selection_changed(self) -> None:
        if self._suppress_selection_updates:
            return

        selected = self.filesList.selectedItems()
        duplicates: Dict[str, List[QListWidgetItem]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            duplicates.setdefault(element, []).append(item)

        needs_cleanup = any(len(items) > 1 for items in duplicates.values())
        if needs_cleanup:
            current = self.filesList.currentItem()
            self._suppress_selection_updates = True
            try:
                for element, items in duplicates.items():
                    if len(items) <= 1:
                        continue
                    ordered = sorted(items, key=lambda it: self.filesList.row(it))
                    if current and current in ordered and current.data(self.ELEMENT_ROLE) == element:
                        keep = current
                    else:
                        keep = ordered[-1]
                    for item in ordered:
                        if item is keep:
                            continue
                        item.setSelected(False)
            finally:
                self._suppress_selection_updates = False

        self._update_selection_info()
        item = self._current_preview_item()
        self._update_preview(item)

    def _on_current_item_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        self._update_preview(current)

    def _current_preview_item(self) -> Optional[QListWidgetItem]:
        selected = self.filesList.selectedItems()
        if selected:
            return selected[-1]
        return self.filesList.currentItem()

    def _update_preview(self, item: Optional[QListWidgetItem]) -> None:
        if not item:
            self.preview.clear()
            return
        path = item.data(self.PATH_ROLE)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                snippet = handle.read(8000)
        except Exception as exc:  # noqa: BLE001
            snippet = f"Could not read file: {exc}"
        self.preview.setPlainText(snippet)

    def _update_selection_info(self) -> None:
        selected = self.filesList.selectedItems()
        count = len(selected)
        info = f"Selected {count}/{self._expected_count} pseudopotential file(s)."
        mapping: Dict[str, str] = {}
        duplicates: List[str] = []
        path_usage: Dict[str, List[str]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            raw_path = item.data(self.PATH_ROLE) or ""
            normalized_path = self._normalise_path(raw_path)
            path = Path(normalized_path)
            if element in mapping:
                duplicates.append(element)
            else:
                mapping[element] = path.name
            path_usage.setdefault(normalized_path, []).append(element)
        if duplicates:
            dup_text = ", ".join(sorted(set(duplicates)))
            info += f" Duplicate selections for: {dup_text}."

        path_duplicates = []
        for normalized_path, elements in path_usage.items():
            if len(elements) > 1:
                label = f"{Path(normalized_path).name} → {', '.join(sorted(set(elements)))}"
                path_duplicates.append(label)
        if path_duplicates:
            info += f" File reused by multiple elements: {'; '.join(path_duplicates)}."
        self.selectionInfoLabel.setText(info)

        if mapping:
            ordered_elements = [el for el in self._elements if el in mapping]
            remaining = [el for el in mapping if el not in ordered_elements]
            for el in remaining:
                ordered_elements.append(el)
            summary = "; ".join(f"{element} ← {mapping[element]}" for element in ordered_elements)
        else:
            summary = ""
        self.assignmentsSummaryLabel.setText(summary)

    def _update_missing_label(self) -> None:
        missing = [element for element in self._elements if not self._candidates.get(element)]
        if missing:
            names = ", ".join(missing)
            self.missingLabel.setText(
                f"No pseudopotentials were found for: {names}. Use 'Add file...' to include them manually."
            )
            self.missingLabel.show()
        else:
            self.missingLabel.hide()

    def _on_add_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pseudopotential file",
            str(PROJECT_ROOT),
            "Pseudopotentials (*.upf *.UPF);;All files (*)",
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if path_obj.suffix.lower() != ".upf":
            self._show_warning("Invalid file", "Select a file ending with .upf or .UPF.")
            return

        element = self._identify_element(path_obj, set(self._elements))
        if not element:
            self._show_warning(
                "Cannot determine element",
                "The file name does not start with a recognised element symbol (e.g. Mg_, Mg., Mg-).",
            )
            return

        resolved = self._normalise_path(path_obj)

        existing_paths = {self._normalise_path(p) for p in self._candidates.get(element, [])}
        if resolved in existing_paths:
            # Already present; locate existing item and select it.
            item = self._find_item_by_path(element, resolved)
            if item:
                item.setSelected(True)
                self._update_selection_info()
            return

        resolved_path = Path(resolved)
        self._candidates.setdefault(element, []).append(resolved_path)
        self._candidates[element].sort(key=lambda p: p.name.lower())
        self._populate_files_list()
        item = self._find_item_by_path(element, resolved)
        if item:
            item.setSelected(True)
            self.filesList.scrollToItem(item)
        self._update_missing_label()
        self._update_selection_info()

    def _on_clear_selection(self) -> None:
        self.filesList.clearSelection()
        self._update_selection_info()
        self.preview.clear()

    def _on_accept(self) -> None:
        if self._expected_count == 0:
            self._selected_mapping = {}
            super().accept()
            return

        selected = self.filesList.selectedItems()
        if len(selected) != self._expected_count:
            self._show_warning(
                "Incomplete selection",
                f"Please select exactly {self._expected_count} pseudopotential file(s).",
            )
            return

        mapping: Dict[str, str] = {}
        duplicates: List[str] = []
        path_usage: Dict[str, List[str]] = {}
        for item in selected:
            element = item.data(self.ELEMENT_ROLE)
            raw_path = item.data(self.PATH_ROLE)
            normalized_path = self._normalise_path(raw_path)
            if element in mapping:
                duplicates.append(element)
            else:
                mapping[element] = normalized_path
            path_usage.setdefault(normalized_path, []).append(element)

        if duplicates:
            dup_text = ", ".join(sorted(set(duplicates)))
            self._show_warning("Duplicate selection", f"Multiple files selected for: {dup_text}. Please pick one per element.")
            return

        missing = [element for element in self._elements if element not in mapping]
        if missing:
            names = ", ".join(missing)
            self._show_warning("Missing selection", f"Select a pseudopotential for: {names}.")
            return

        reuse_conflicts = []
        for normalized_path, elements in path_usage.items():
            if len(elements) > 1:
                label = f"{Path(normalized_path).name}: {', '.join(sorted(set(elements)))}"
                reuse_conflicts.append(label)
        if reuse_conflicts:
            detail = "; ".join(reuse_conflicts)
            self._show_warning(
                "Duplicate file usage",
                f"The same pseudopotential file is assigned to multiple elements ({detail}). Choose unique files per element.",
            )
            return

        ordered_mapping: Dict[str, str] = {}
        for element in self._elements:
            if element in mapping:
                ordered_mapping[element] = mapping[element]
        for element, path in mapping.items():
            if element not in ordered_mapping:
                ordered_mapping[element] = path

        self._selected_mapping = ordered_mapping
        super().accept()

    @staticmethod
    def _normalise_path(path: Any) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(path)

    @staticmethod
    def _identify_element(path: Path, allowed: set[str]) -> Optional[str]:
        name = path.name
        token_match = re.match(r"([A-Za-z]{1,2})(?=[^A-Za-z])", name)
        if not token_match:
            token = re.split(r"[._-]", name, maxsplit=1)[0]
        else:
            token = token_match.group(1)
        candidate = normalize_element_symbol(token)
        return candidate if candidate in allowed else None

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

class SystemConfigDialog(QDialog):
    def __init__(self, config: SystemConfig, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("System configuration")
        self.setModal(True)

        self._result_config = config
        self._base_mpi_procs = max(1, int(getattr(config, "mpi_procs", 1) or 1))
        self._qe_detect_thread: Optional[QThread] = None
        self._qe_install_thread: Optional[QThread] = None
        self._qe_progress_dialog: Optional[QDialog] = None
        self._thermo_detect_thread: Optional[QThread] = None
        self._thermo_install_thread: Optional[QThread] = None
        self._thermo_progress_dialog: Optional[QDialog] = None

        layout = QVBoxLayout(self)

        form = QFormLayout()
        layout.addLayout(form)

        qe_path_layout = QHBoxLayout()
        self.qePathLineEdit = QLineEdit(self)
        if config.qe_path:
            self.qePathLineEdit.setText(config.qe_path)
        browse_qe_button = QPushButton("Browse...", self)
        detect_button = QPushButton("Configure", self)
        qe_path_layout.addWidget(self.qePathLineEdit)
        qe_path_layout.addWidget(browse_qe_button)
        qe_path_layout.addWidget(detect_button)
        form.addRow("Quantum ESPRESSO bin directory:", qe_path_layout)

        thermo_path_layout = QHBoxLayout()
        self.thermoPwPathLineEdit = QLineEdit(self)
        if getattr(config, "thermo_pw_path", None):
            self.thermoPwPathLineEdit.setText(str(config.thermo_pw_path))
        browse_thermo_button = QPushButton("Browse...", self)
        detect_thermo_button = QPushButton("Configure", self)
        thermo_path_layout.addWidget(self.thermoPwPathLineEdit)
        thermo_path_layout.addWidget(browse_thermo_button)
        thermo_path_layout.addWidget(detect_thermo_button)
        form.addRow("Thermo_pw bin directory:", thermo_path_layout)

        # Pseudopotential directories: allow multiple entries and downloading
        pseudo_layout = QVBoxLayout()
        self.pseudoListWidget = QListWidget(self)
        self.pseudoListWidget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Populate from config: support both legacy single value and new list
        try:
            dirs = list(getattr(config, "pseudo_directories", [])) or ([] if config.pseudo_directory is None else [config.pseudo_directory])
        except Exception:
            dirs = [config.pseudo_directory] if getattr(config, "pseudo_directory", None) else []
        for d in dirs:
            if d:
                self.pseudoListWidget.addItem(d)

        btn_row = QHBoxLayout()
        self.pseudoAddButton = QPushButton("Add...", self)
        self.pseudoRemoveButton = QPushButton("Remove", self)
        self.pseudoDownloadButton = QPushButton("Download / Prepare", self)
        btn_row.addWidget(self.pseudoAddButton)
        btn_row.addWidget(self.pseudoRemoveButton)
        btn_row.addStretch()
        btn_row.addWidget(self.pseudoDownloadButton)

        pseudo_layout.addWidget(self.pseudoListWidget)
        pseudo_layout.addLayout(btn_row)
        form.addRow("Pseudopotential directories:", pseudo_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addWidget(button_box)

        browse_qe_button.clicked.connect(self._browse_qe_path)
        detect_button.clicked.connect(self._configure_qe_path)
        browse_thermo_button.clicked.connect(self._browse_thermo_pw_path)
        detect_thermo_button.clicked.connect(self._configure_thermo_pw_path)
        self.pseudoAddButton.clicked.connect(self._add_pseudo_dir)
        self.pseudoRemoveButton.clicked.connect(self._remove_selected_pseudo_dirs)
        self.pseudoDownloadButton.clicked.connect(self._download_pseudos)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def _browse_qe_path(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Quantum ESPRESSO bin directory",
            self.qePathLineEdit.text().strip() or str(PROJECT_ROOT),
        )
        if not directory:
            return
        bin_dir = self._normalize_qe_bin_dir(directory)
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Selected directory does not contain a valid pw.x executable.",
            )
            return
        self.qePathLineEdit.setText(bin_dir)

    def _browse_thermo_pw_path(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select thermo_pw bin directory",
            self.thermoPwPathLineEdit.text().strip() or str(PROJECT_ROOT),
        )
        if not directory:
            return
        bin_dir = self._normalize_thermo_pw_bin_dir(directory)
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Selected directory does not contain a valid thermo_pw.x executable.",
            )
            return
        self.thermoPwPathLineEdit.setText(bin_dir)

    def _configure_qe_path(self) -> None:
        if self._qe_detect_thread is not None and self._qe_detect_thread.isRunning():
            QMessageBox.information(
                self,
                "Configuration in progress",
                "A Quantum ESPRESSO configuration task is already running.",
            )
            return

        dialog = QeConfigureProgressDialog(
            "Configuring Quantum ESPRESSO",
            self,
            allow_cancel=False,
            initial_status="Searching for pw.x...",
        )
        self._qe_progress_dialog = dialog
        thread = QePathSearchThread(
            self,
            executable_name="pw.x",
            display_name="Quantum ESPRESSO",
        )
        self._qe_detect_thread = thread

        thread.log_appended.connect(dialog.append_log)
        thread.status_changed.connect(dialog.set_status)
        thread.finished_with_results.connect(self._on_qe_search_finished)

        dialog.show()
        thread.start()

    def _configure_thermo_pw_path(self) -> None:
        if self._thermo_detect_thread is not None and self._thermo_detect_thread.isRunning():
            QMessageBox.information(
                self,
                "Configuration in progress",
                "A thermo_pw configuration task is already running.",
            )
            return

        dialog = QeConfigureProgressDialog(
            "Configuring thermo_pw",
            self,
            allow_cancel=False,
            initial_status="Searching for thermo_pw.x...",
        )
        self._thermo_progress_dialog = dialog
        thread = QePathSearchThread(
            self,
            executable_name="thermo_pw.x",
            display_name="thermo_pw",
        )
        self._thermo_detect_thread = thread

        thread.log_appended.connect(dialog.append_log)
        thread.status_changed.connect(dialog.set_status)
        thread.finished_with_results.connect(self._on_thermo_pw_search_finished)

        dialog.show()
        thread.start()

    def _on_qe_search_finished(self, paths: list[str]) -> None:
        self._finalize_qe_progress(True)
        self._qe_detect_thread = None
        self._handle_qe_search_results(paths)

    def _on_thermo_pw_search_finished(self, paths: list[str]) -> None:
        self._finalize_thermo_progress(True)
        self._thermo_detect_thread = None
        self._handle_thermo_pw_search_results(paths)

    def _on_qe_install_finished(
        self,
        success: bool,
        message: str,
        paths: list[str],
        thermo_paths: list[str],
    ) -> None:
        self._finalize_qe_progress(success)
        self._qe_install_thread = None
        if not success:
            detail = message or "Automatic installation failed."
            QMessageBox.warning(
                self,
                "Automatic installation failed",
                f"{detail}\n\nPlease install Quantum ESPRESSO manually and then configure the bin directory.",
            )
            return

        qe_bin = self._set_first_qe_result(paths)
        thermo_bin = self._set_first_thermo_result(thermo_paths)
        message_parts: list[str] = []
        if qe_bin:
            message_parts.append(f"Quantum ESPRESSO bin directory:\n{qe_bin}")
        if thermo_bin:
            message_parts.append(f"thermo_pw bin directory:\n{thermo_bin}")
        if not message_parts:
            self._handle_qe_search_results(paths)
            return
        QMessageBox.information(
            self,
            "Installation configured",
            "\n\n".join(message_parts),
        )

    def _on_thermo_pw_install_finished(
        self,
        success: bool,
        message: str,
        paths: list[str],
        _extra_paths: list[str],
    ) -> None:
        self._finalize_thermo_progress(success)
        self._thermo_install_thread = None
        if not success:
            detail = message or "Automatic installation failed."
            QMessageBox.warning(
                self,
                "Automatic installation failed",
                f"{detail}\n\nPlease install thermo_pw manually and then configure the bin directory.",
            )
            return

        thermo_bin = self._set_first_thermo_result(paths)
        if thermo_bin:
            QMessageBox.information(
                self,
                "thermo_pw configured",
                f"Using bin directory:\n{thermo_bin}",
            )
        else:
            self._handle_thermo_pw_search_results(paths)

    def _finalize_qe_progress(self, success: bool) -> None:
        dialog = self._qe_progress_dialog
        if dialog is None:
            return
        dialog.set_finished(success)
        dialog.accept()
        self._qe_progress_dialog = None

    def _finalize_thermo_progress(self, success: bool) -> None:
        dialog = self._thermo_progress_dialog
        if dialog is None:
            return
        dialog.set_finished(success)
        dialog.accept()
        self._thermo_progress_dialog = None

    def _handle_qe_search_results(self, paths: list[str]) -> None:
        bin_dirs, details = self._collect_qe_bin_dirs(paths)
        options: list[str] = []
        mapping: dict[str, tuple[str, str]] = {}
        for bin_dir in bin_dirs:
            label = f"Detected installation: {bin_dir}"
            options.append(label)
            mapping[label] = ("detected", bin_dir)

        browse_label = "Browse for existing Quantum ESPRESSO bin directory"
        install_label = "Create a new Quantum ESPRESSO installation"
        options.append(browse_label)
        options.append(install_label)
        mapping[browse_label] = ("browse", "")
        mapping[install_label] = ("install", "")

        prompt = (
            "Select a Quantum ESPRESSO installation to use:"
            if bin_dirs
            else "No installations were detected. Select how you want to proceed:"
        )
        selection, ok = QInputDialog.getItem(
            self,
            "Quantum ESPRESSO setup",
            prompt,
            options,
            0,
            False,
        )
        if not ok or not selection:
            return

        action, value = mapping.get(selection, ("", ""))
        if action == "detected" and value:
            self._confirm_qe_bin_dir(value, details.get(value, {}))
        elif action == "browse":
            self._browse_existing_qe_bin_dir()
        elif action == "install":
            self._run_qe_install_flow()

    def _handle_thermo_pw_search_results(self, paths: list[str]) -> None:
        bin_dirs, details = self._collect_thermo_pw_bin_dirs(paths)
        options: list[str] = []
        mapping: dict[str, tuple[str, str]] = {}
        for bin_dir in bin_dirs:
            label = f"Detected installation: {bin_dir}"
            options.append(label)
            mapping[label] = ("detected", bin_dir)

        browse_label = "Browse for existing thermo_pw bin directory"
        install_label = "Create a new thermo_pw installation"
        options.append(browse_label)
        options.append(install_label)
        mapping[browse_label] = ("browse", "")
        mapping[install_label] = ("install", "")

        prompt = (
            "Select a thermo_pw installation to use:"
            if bin_dirs
            else "No thermo_pw installations were detected. Select how you want to proceed:"
        )
        selection, ok = QInputDialog.getItem(
            self,
            "thermo_pw setup",
            prompt,
            options,
            0,
            False,
        )
        if not ok or not selection:
            return

        action, value = mapping.get(selection, ("", ""))
        if action == "detected" and value:
            self._confirm_thermo_pw_bin_dir(value, details.get(value, {}))
        elif action == "browse":
            self._browse_existing_thermo_pw_bin_dir()
        elif action == "install":
            self._run_thermo_pw_install_flow()

    def _set_first_qe_result(self, paths: list[str]) -> str:
        bin_dirs, _details = self._collect_qe_bin_dirs(paths)
        if not bin_dirs:
            return ""
        bin_dir = bin_dirs[0]
        self.qePathLineEdit.setText(bin_dir)
        return bin_dir

    def _set_first_thermo_result(self, paths: list[str]) -> str:
        bin_dirs, _details = self._collect_thermo_pw_bin_dirs(paths)
        if not bin_dirs:
            return ""
        bin_dir = bin_dirs[0]
        self.thermoPwPathLineEdit.setText(bin_dir)
        return bin_dir

    def _confirm_qe_bin_dir(self, bin_dir: str, info: dict[str, str]) -> None:
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Could not determine the Quantum ESPRESSO bin directory from the selected path.",
            )
            return

        source_path = info.get("candidate")
        resolved_pw = info.get("resolved_pw")
        message_parts: list[str] = []
        if source_path and resolved_pw and source_path != resolved_pw:
            message_parts.append(f"Detected pw.x via:\n{source_path}")
            message_parts.append(f"Resolves to:\n{resolved_pw}")
        elif resolved_pw:
            message_parts.append(f"Detected pw.x at:\n{resolved_pw}")
        elif source_path:
            message_parts.append(f"Detected pw.x at:\n{source_path}")
        message_parts.append(f"Use this Quantum ESPRESSO bin directory?\n{bin_dir}")
        message = "\n\n".join(message_parts)
        answer = QMessageBox.question(
            self,
            "Confirm Quantum ESPRESSO location",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.qePathLineEdit.setText(bin_dir)
            QMessageBox.information(
                self,
                "Quantum ESPRESSO configured",
                f"Using bin directory:\n{bin_dir}",
            )

    def _confirm_thermo_pw_bin_dir(self, bin_dir: str, info: dict[str, str]) -> None:
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Could not determine the thermo_pw bin directory from the selected path.",
            )
            return

        source_path = info.get("candidate")
        resolved_thermo = info.get("resolved_thermo")
        message_parts: list[str] = []
        if source_path and resolved_thermo and source_path != resolved_thermo:
            message_parts.append(f"Detected thermo_pw.x via:\n{source_path}")
            message_parts.append(f"Resolves to:\n{resolved_thermo}")
        elif resolved_thermo:
            message_parts.append(f"Detected thermo_pw.x at:\n{resolved_thermo}")
        elif source_path:
            message_parts.append(f"Detected thermo_pw.x at:\n{source_path}")
        message_parts.append(f"Use this thermo_pw bin directory?\n{bin_dir}")
        message = "\n\n".join(message_parts)
        answer = QMessageBox.question(
            self,
            "Confirm thermo_pw location",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.thermoPwPathLineEdit.setText(bin_dir)
            QMessageBox.information(
                self,
                "thermo_pw configured",
                f"Using bin directory:\n{bin_dir}",
            )

    def _browse_existing_qe_bin_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Quantum ESPRESSO bin directory",
            self.qePathLineEdit.text().strip() or str(PROJECT_ROOT),
        )
        if not directory:
            return
        bin_dir, resolved_pw = _resolve_pw_candidate(directory)
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Selected directory does not contain a valid pw.x executable.",
            )
            return
        info = {"candidate": directory, "resolved_pw": resolved_pw} if resolved_pw else {"candidate": directory}
        self._confirm_qe_bin_dir(bin_dir, info)

    def _browse_existing_thermo_pw_bin_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select thermo_pw bin directory",
            self.thermoPwPathLineEdit.text().strip() or str(PROJECT_ROOT),
        )
        if not directory:
            return
        bin_dir, resolved_thermo = _resolve_thermo_pw_candidate(directory)
        if not bin_dir:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Selected directory does not contain a valid thermo_pw.x executable.",
            )
            return
        info = (
            {"candidate": directory, "resolved_thermo": resolved_thermo}
            if resolved_thermo
            else {"candidate": directory}
        )
        self._confirm_thermo_pw_bin_dir(bin_dir, info)

    def _run_qe_install_flow(self) -> None:
        install_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose installation directory",
            str(PROJECT_ROOT),
        )
        if not install_dir:
            return

        password, ok = QInputDialog.getText(
            self,
            "Administrator privileges required",
            "Enter your sudo password to install Quantum ESPRESSO:\n"
            "(Your password is used only for this installation and is not saved.)",
            QLineEdit.EchoMode.Password,
            "",
        )
        if not ok:
            return
        if not password:
            QMessageBox.warning(
                self,
                "Password required",
                "A sudo password is required to continue the installation.",
            )
            return

        dialog = QeConfigureProgressDialog(
            "Installing Quantum ESPRESSO",
            self,
            allow_cancel=True,
            initial_status="Downloading Quantum ESPRESSO...",
        )
        self._qe_progress_dialog = dialog
        thread = QeInstallThread(
            Path(install_dir),
            password,
            self,
            executable_name="pw.x",
            display_name="Quantum ESPRESSO",
            install_args=["--with-thermo"],
            extra_executable_name="thermo_pw.x",
        )
        self._qe_install_thread = thread

        dialog.cancel_requested.connect(thread.request_stop)
        thread.log_appended.connect(dialog.append_log)
        thread.status_changed.connect(dialog.set_status)
        thread.finished_with_result.connect(self._on_qe_install_finished)

        dialog.show()
        thread.start()

    def _run_thermo_pw_install_flow(self) -> None:
        install_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose installation directory",
            str(PROJECT_ROOT),
        )
        if not install_dir:
            return

        password, ok = QInputDialog.getText(
            self,
            "Administrator privileges required",
            "Enter your sudo password to install thermo_pw:\n"
            "(Your password is used only for this installation and is not saved.)",
            QLineEdit.EchoMode.Password,
            "",
        )
        if not ok:
            return
        if not password:
            QMessageBox.warning(
                self,
                "Password required",
                "A sudo password is required to continue the installation.",
            )
            return

        dialog = QeConfigureProgressDialog(
            "Installing thermo_pw",
            self,
            allow_cancel=True,
            initial_status="Downloading thermo_pw...",
        )
        self._thermo_progress_dialog = dialog
        thread = QeInstallThread(
            Path(install_dir),
            password,
            self,
            executable_name="thermo_pw.x",
            display_name="thermo_pw",
            install_args=["--thermo-only"],
        )
        self._thermo_install_thread = thread

        dialog.cancel_requested.connect(thread.request_stop)
        thread.log_appended.connect(dialog.append_log)
        thread.status_changed.connect(dialog.set_status)
        thread.finished_with_result.connect(self._on_thermo_pw_install_finished)

        dialog.show()
        thread.start()

    def _collect_qe_bin_dirs(self, paths: list[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
        details: dict[str, dict[str, str]] = {}
        for candidate in paths:
            bin_dir, resolved_pw = _resolve_pw_candidate(candidate)
            if not bin_dir:
                continue
            if bin_dir not in details:
                details[bin_dir] = {"candidate": candidate, "resolved_pw": resolved_pw}
        return sorted(details.keys()), details

    def _normalize_qe_bin_dir(self, raw: str) -> str:
        bin_dir, _ = _resolve_pw_candidate(raw)
        return bin_dir

    def _collect_thermo_pw_bin_dirs(self, paths: list[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
        details: dict[str, dict[str, str]] = {}
        for candidate in paths:
            bin_dir, resolved_thermo = _resolve_thermo_pw_candidate(candidate)
            if not bin_dir:
                continue
            if bin_dir not in details:
                details[bin_dir] = {"candidate": candidate, "resolved_thermo": resolved_thermo}
        return sorted(details.keys()), details

    def _normalize_thermo_pw_bin_dir(self, raw: str) -> str:
        bin_dir, _ = _resolve_thermo_pw_candidate(raw)
        return bin_dir

    def accept(self) -> None:  # noqa: D401 (Qt override)
        # Build a SystemConfig with possibly multiple pseudo directories.
        pseudo_dirs: list[str] = []
        for i in range(self.pseudoListWidget.count()):
            item = self.pseudoListWidget.item(i)
            if item and item.text().strip():
                pseudo_dirs.append(item.text().strip())

        # Keep legacy field for compatibility (first entry or None)
        first = pseudo_dirs[0] if pseudo_dirs else None
        qe_bin = self._normalize_qe_bin_dir(self.qePathLineEdit.text().strip())
        if self.qePathLineEdit.text().strip() and not qe_bin:
            QMessageBox.warning(
                self,
                "Invalid Quantum ESPRESSO path",
                "Enter a valid bin directory containing pw.x before saving.",
            )
            return
        if qe_bin:
            self.qePathLineEdit.setText(qe_bin)
        thermo_bin = self._normalize_thermo_pw_bin_dir(self.thermoPwPathLineEdit.text().strip())
        if self.thermoPwPathLineEdit.text().strip() and not thermo_bin:
            QMessageBox.warning(
                self,
                "Invalid thermo_pw path",
                "Enter a valid bin directory containing thermo_pw.x before saving.",
            )
            return
        if thermo_bin:
            self.thermoPwPathLineEdit.setText(thermo_bin)
        self._result_config = SystemConfig(
            qe_path=qe_bin or None,
            thermo_pw_path=thermo_bin or None,
            mpi_procs=self._base_mpi_procs,
            pseudo_directory=first,
            pseudo_directories=pseudo_dirs,
        )
        super().accept()

    # ---- Pseudopotential list helpers ----
    def _add_pseudo_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select pseudopotential directory",
            str(PROJECT_ROOT),
        )
        if not directory:
            return
        # Avoid duplicates
        for i in range(self.pseudoListWidget.count()):
            if self.pseudoListWidget.item(i).text() == directory:
                return
        self.pseudoListWidget.addItem(directory)

    def _remove_selected_pseudo_dirs(self) -> None:
        rows = sorted({idx.row() for idx in self.pseudoListWidget.selectedIndexes()}, reverse=True)
        if not rows and self.pseudoListWidget.count():
            rows = [self.pseudoListWidget.count() - 1]
        for r in rows:
            if 0 <= r < self.pseudoListWidget.count():
                self.pseudoListWidget.takeItem(r)

    def _download_pseudos(self) -> None:
        # Ask for destination directory
        dest = QFileDialog.getExistingDirectory(self, "Choose destination for prepared pseudopotentials", str(PROJECT_ROOT))
        if not dest:
            return

        # Ask whether to run the make_all_ps step (may be long)
        run_make = QMessageBox.question(
            self,
            "Run make_all_ps?",
            "Do you want to run the patched make_all_ps after preparing pslibrary?\n(This can be long and may compile many files)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        run_flag = run_make == QMessageBox.StandardButton.Yes

        qe_path = self._normalize_qe_bin_dir(self.qePathLineEdit.text().strip())
        if qe_path.endswith("/bin"):
            qe_path = qe_path[:-4]
        builder = PspBuilderThread(qe_path, Path(dest), run_flag, parent=self)

        dlg = DownloadProgressDialog(self)

        # Append initial note
        dlg.append_log(f"Starting prepare in: {dest}")

        # Wire builder signals to the dialog
        builder.log_appended.connect(dlg.append_log)

        def _on_finished(ok: bool) -> None:
            dlg.append_log("\nCompleted successfully." if ok else "\nFailed or cancelled.")
            dlg.set_finished(ok)

        builder.finished_with_success.connect(_on_finished)

        # Cancel from dialog should request stop
        def _on_cancel_request() -> None:
            if dlg.was_canceled():
                builder.request_stop()

        dlg.cancelButton.clicked.connect(_on_cancel_request)

        # Start builder and show modal dialog
        builder.start()
        dlg.exec()

    def result_config(self) -> SystemConfig:
        return self._result_config


class CommandPreferencesDialog(QDialog):
    _NO_MPI_DEFAULT_COMMANDS = {"alpha2f.x", "projwfc.x", "fermi_velocity.x", "lambda.x"}

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Command preferences")
        self.setModal(True)
        # Always load current system config when constructing
        system_config = load_system_config()
        self._qe_bin_dir = (system_config.qe_path or "").strip()
        self._thermo_pw_bin_dir = (getattr(system_config, "thermo_pw_path", None) or "").strip()
        prefs = self._load_preferences()
        self._use_mpirun = bool(prefs.get("use_mpirun", True))
        self._mpirun_suffix = str(prefs.get("mpirun_suffix", "mpirun" if self._use_mpirun else "")).strip()
        self._auto_defaults = self._build_default_commands(self._use_mpirun, self._mpirun_suffix)
        self._custom_commands: set[str] = set()  # Track which commands have been customized by user

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.useMpirunCheckBox = QCheckBox("use mpirun", self)
        self.useMpirunCheckBox.setChecked(self._use_mpirun)
        layout.addWidget(self.useMpirunCheckBox)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        layout.addLayout(form)

        self.mpirunSuffixLineEdit = QLineEdit(self)
        self.mpirunSuffixLineEdit.setText(self._mpirun_suffix)
        form.addRow("mpirun suffix", self.mpirunSuffixLineEdit)

        self._command_edits: dict[str, QLineEdit] = {}
        for label, key in self._command_fields():
            edit = QLineEdit(self)
            value = str((prefs.get("commands") or {}).get(key, "")).strip()
            if not value:
                value = self._auto_defaults.get(key, "")
            edit.setText(value)
            form.addRow(label, edit)
            self._command_edits[key] = edit
            edit.textEdited.connect(lambda text, k=key: self._mark_as_custom(k))

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.updateButton = QPushButton("Update", self)
        self.saveButton = QPushButton("Save", self)
        self.closeButton = QPushButton("Close", self)
        buttons_layout.addWidget(self.updateButton)
        buttons_layout.addWidget(self.saveButton)
        buttons_layout.addWidget(self.closeButton)
        layout.addLayout(buttons_layout)

        self.useMpirunCheckBox.toggled.connect(self._on_use_mpirun_toggled)
        self.mpirunSuffixLineEdit.textEdited.connect(self._on_mpirun_suffix_changed)
        self.saveButton.clicked.connect(self._save_preferences)
        self.closeButton.clicked.connect(self.reject)
        self.updateButton.clicked.connect(self._on_update_clicked)

    def _command_fields(self) -> list[tuple[str, str]]:
        return [
            ("pw.x command", "pw.x"),
            ("ph.x command", "ph.x"),
            ("q2r.x command", "q2r.x"),
            ("matdyn.x command", "matdyn.x"),
            ("lambda.x command", "lambda.x"),
            ("alpha2f.x command", "alpha2f.x"),
            ("bands.x command", "bands.x"),
            ("dos.x command", "dos.x"),
            ("projwfc.x command", "projwfc.x"),
            ("fermi_velocity.x command", "fermi_velocity.x"),
            ("pp.x command", "pp.x"),
            ("thermo_pw.x command", "thermo_pw.x"),
        ]

    def _build_default_commands(self, use_mpirun: bool, mpirun_suffix: str) -> dict[str, str]:
        prefix = mpirun_suffix.strip() if use_mpirun else ""
        bin_dir = Path(self._qe_bin_dir) if self._qe_bin_dir else None
        thermo_bin_dir = Path(self._thermo_pw_bin_dir) if self._thermo_pw_bin_dir else None

        def full_path(binary: str) -> str:
            if binary == "thermo_pw.x":
                if thermo_bin_dir is None:
                    return binary
                return str(thermo_bin_dir / binary)
            if bin_dir is None:
                return binary
            return str(bin_dir / binary)

        def with_prefix(binary: str) -> str:
            command = full_path(binary)
            if binary in self._NO_MPI_DEFAULT_COMMANDS:
                return command
            return f"{prefix} {command}".strip() if prefix else command

        return {
            "pw.x": with_prefix("pw.x"),
            "ph.x": with_prefix("ph.x"),
            "q2r.x": with_prefix("q2r.x"),
            "matdyn.x": with_prefix("matdyn.x"),
            "lambda.x": with_prefix("lambda.x"),
            "alpha2f.x": with_prefix("alpha2f.x"),
            "bands.x": with_prefix("bands.x"),
            "dos.x": with_prefix("dos.x"),
            "projwfc.x": with_prefix("projwfc.x"),
            "fermi_velocity.x": with_prefix("fermi_velocity.x"),
            "pp.x": with_prefix("pp.x"),
            "thermo_pw.x": with_prefix("thermo_pw.x"),
        }

    def _mark_as_custom(self, key: str) -> None:
        """Mark a command as customized by the user."""
        self._custom_commands.add(key)

    def _on_use_mpirun_toggled(self, checked: bool) -> None:
        old_use_mpirun = self._use_mpirun
        old_suffix = self._mpirun_suffix

        self._use_mpirun = checked
        suffix_text = self.mpirunSuffixLineEdit.text().strip()
        if checked and not suffix_text:
            suffix_text = "mpirun"
            self.mpirunSuffixLineEdit.blockSignals(True)
            self.mpirunSuffixLineEdit.setText(suffix_text)
            self.mpirunSuffixLineEdit.blockSignals(False)
        elif not checked:
            suffix_text = ""
            self.mpirunSuffixLineEdit.blockSignals(True)
            self.mpirunSuffixLineEdit.setText(suffix_text)
            self.mpirunSuffixLineEdit.blockSignals(False)

        self._mpirun_suffix = suffix_text
        self._auto_defaults = self._build_default_commands(self._use_mpirun, self._mpirun_suffix)
        self._update_auto_commands()

    def _on_mpirun_suffix_changed(self, text: str) -> None:
        suffix_text = text.strip()
        self._mpirun_suffix = suffix_text
        self._auto_defaults = self._build_default_commands(self._use_mpirun, self._mpirun_suffix)
        self._update_auto_commands()

    def _update_auto_commands(self) -> None:
        """Update command fields that haven't been customized."""
        for key, edit in self._command_edits.items():
            if key in self._custom_commands:
                continue
            edit.setText(self._auto_defaults.get(key, ""))

    def _on_update_clicked(self) -> None:
        """Manual update: reload system config QE path and refresh defaults."""
        try:
            system_config = load_system_config()
            self._qe_bin_dir = (system_config.qe_path or "").strip()
            self._thermo_pw_bin_dir = (getattr(system_config, "thermo_pw_path", None) or "").strip()
            self._auto_defaults = self._build_default_commands(self._use_mpirun, self._mpirun_suffix)
            self._update_auto_commands()
            QMessageBox.information(
                self,
                "Updated",
                "Command defaults refreshed from current Quantum ESPRESSO and thermo_pw paths.",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Update failed", f"Could not refresh command defaults: {exc}")

    def _load_preferences(self) -> dict[str, Any]:
        if not COMMAND_PREFS_PATH.exists():
            return {}
        try:
            with COMMAND_PREFS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_preferences(self) -> None:
        prefs = {
            "use_mpirun": self.useMpirunCheckBox.isChecked(),
            "mpirun_suffix": self.mpirunSuffixLineEdit.text().strip(),
            "commands": {key: edit.text().strip() for key, edit in self._command_edits.items()},
        }
        try:
            with COMMAND_PREFS_PATH.open("w", encoding="utf-8") as handle:
                json.dump(prefs, handle, indent=2)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", f"Could not save preferences: {exc}")
            return

        QMessageBox.information(self, "Preferences saved", "Command preferences have been saved.")


class CalculationOptionsDialog(QDialog):
    """Modal dialog to tweak calculation-specific options before generating inputs."""

    def __init__(
        self,
        calculation: str,
        seed: Optional[ScfSeedParameters],
        job_config: JobConfig,
        system_config: SystemConfig,
        parent: Optional[QWidget] = None,
        *,
        cif_path: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{(calculation or '').upper()} options")
        self.setModal(True)

        self.calculation = (calculation or "").lower()
        self.seed = seed
        self.job_config = job_config
        self.system_config = system_config
        self.cif_path = (cif_path or "").strip() or None
        self._overrides: dict[str, Any] = {}
        # Calculation mode flags
        self._is_nscf = self.calculation in {"nscf", "bands"}
        self._is_vc_relax = self.calculation in {"vc-relax", "relax"}
        # Advanced flags to expose in UI (backends for others are kept for future)
        self._advanced_flag_keys = [
            "tstress",
            "tprnfor",
            "nosym",
            "noinv",
            "noncolin",
            "lspinorb",
            "wf_collect",
            "restart_mode",
        ]
        self._occupations_choices: list[str] = [
            "smearing",
            "tetrahedra",
            "tetrahedra_lin",
            "tetrahedra_opt",
            "fixed",
            "from_input",
        ]

        self._flag_defaults: dict[str, bool] = {
            "tstress": False,
            "tprnfor": False,
            "nosym": False,
            "nosym_evc": False,
            "noinv": False,
            "force_symmorphic": False,
            "wf_collect": False,
            # Doc-backed defaults that are true by default
            "ace": True,
            "x_gamma_extrapolation": True,
            "restart_mode": False,
        }
        self._flag_labels: dict[str, str] = {
            "tstress": "Write stress tensor (tstress)",
            "tprnfor": "Write forces (tprnfor)",
            "nosym": "Disable symmetry (nosym)",
            "nosym_evc": "Disable symmetry of eigenvectors (nosym_evc)",
            "noinv": "Disable inversion symmetry (noinv)",
            "force_symmorphic": "Force symmorphic group (force_symmorphic)",
            "wf_collect": "Collect wavefunctions (wf_collect)",
            "ace": "Adaptively Compressed Exchange (ace)",
            "x_gamma_extrapolation": "EXX gamma extrapolation (x_gamma_extrapolation)",
            "restart_mode": "Restart calculation (restart_mode='restart')",
        }
        self._numeric_defaults = {
            ("control", "etot_conv_thr"): "1.0D-4",
            ("control", "forc_conv_thr"): "1.0D-3",
            ("control", "outdir"): None,
            ("control", "pseudo_dir"): None,
            ("system", "ecutrho"): "4*ecutwfc",
            ("electrons", "conv_thr"): "1.0D-6",
            ("electrons", "mixing_beta"): "0.7D0",
            ("electrons", "electron_maxstep"): "100",
            ("cell", "press"): "0.D0",
            ("cell", "press_conv_thr"): "0.D0",
        }
        self._flag_initial = {}
        self._flag_values = {}
        self._initial_k_type = "automatic"
        self._initial_k_grid = None
        self._initial_k_offset = None
        self._initial_k_card: list[str] = []
        self._tpiba_loaded = False
        self._tpiba_error: Optional[str] = None
        self._tpiba_paths: List[List[str]] = []
        self._tpiba_points: Dict[str, Sequence[float]] = {}
        self.ionDynamicsCombo: Optional[QComboBox] = None
        self.cellDynamicsCombo: Optional[QComboBox] = None
        self.pressLineEdit: Optional[QLineEdit] = None
        self.pressConvThrLineEdit: Optional[QLineEdit] = None
        self.nbndLineEdit: Optional[QLineEdit] = None
        self.nbndCheck: Optional[QCheckBox] = None
        self.nbndHintLabel: Optional[QLabel] = None
        self._build_ui()
        self._populate_defaults()
        # Wire dynamic updates after defaults are populated
        try:
            self.ecutwfcLineEdit.textChanged.connect(self._on_ecutwfc_changed)
            self.ecutrhoLineEdit.textEdited.connect(self._on_ecutrho_edited)
        except Exception:
            pass

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        numeric_group = QGroupBox("Common parameters", self)
        numeric_form = QFormLayout(numeric_group)
        numeric_form.setContentsMargins(12, 12, 12, 12)
        self.ecutwfcLineEdit = QLineEdit(numeric_group)
        self.ecutrhoLineEdit = QLineEdit(numeric_group)
        self.etotConvLineEdit = QLineEdit(numeric_group)
        self.forcConvLineEdit = QLineEdit(numeric_group)
        self.convThrLineEdit = QLineEdit(numeric_group)
        self.mixingBetaLineEdit = QLineEdit(numeric_group)
        self.electronMaxStepLineEdit = QLineEdit(numeric_group)
        self.electronMaxStepLineEdit.setValidator(QIntValidator(1, 100000, self))
        # Integer-only fields
        self.ecutwfcLineEdit.setValidator(QIntValidator(1, 1000000, self))
        self.ecutrhoLineEdit.setValidator(QIntValidator(1, 10000000, self))
        self.ecutwfcLineEdit.setPlaceholderText("Required (integer Ry)")
        self.ecutrhoLineEdit.setPlaceholderText("Default: 4 * ecutwfc or value from source")
        self.etotConvLineEdit.setPlaceholderText("Default: 1.0D-4")
        self.forcConvLineEdit.setPlaceholderText("Default: 1.0D-3")
        self.convThrLineEdit.setPlaceholderText("Default: 1.0D-6")
        self.mixingBetaLineEdit.setPlaceholderText("Default: 0.7D0")
        self.electronMaxStepLineEdit.setPlaceholderText("Default: 100")
        # New common fields: occupations dropdown plus smearing controls
        self.occupationsCombo = QComboBox(numeric_group)
        self.occupationsCombo.addItems(self._occupations_choices)
        self.smearingCombo = QComboBox(numeric_group)
        self.smearingCombo.addItems([
            "(none)",
            "gaussian",
            "mp",
            "methfessel-paxton",
            "marzari-vanderbilt",
            "fermi-dirac",
            "cold",
        ])
        self.degaussLineEdit = QLineEdit(numeric_group)
        self.degaussLineEdit.setPlaceholderText("Default: 0.D0 (Ry)")
        numeric_form.addRow("ecutwfc:", self.ecutwfcLineEdit)
        numeric_form.addRow("ecutrho:", self.ecutrhoLineEdit)
        numeric_form.addRow("etot_conv_thr:", self.etotConvLineEdit)
        numeric_form.addRow("forc_conv_thr:", self.forcConvLineEdit)
        numeric_form.addRow("conv_thr:", self.convThrLineEdit)
        numeric_form.addRow("mixing_beta:", self.mixingBetaLineEdit)
        numeric_form.addRow("electron_maxstep:", self.electronMaxStepLineEdit)
        numeric_form.addRow("occupations:", self.occupationsCombo)
        numeric_form.addRow("smearing:", self.smearingCombo)
        numeric_form.addRow("degauss:", self.degaussLineEdit)
        if self._is_vc_relax:
            self.ionDynamicsCombo = QComboBox(numeric_group)
            self.ionDynamicsCombo.addItems(["bfgs", "damp"])
            self.cellDynamicsCombo = QComboBox(numeric_group)
            self.cellDynamicsCombo.addItems(["bfgs", "none", "sd", "damp-pr", "damp-w"])
            self.ionDynamicsCombo.setToolTip("&IONS: ion_dynamics - ionic relaxation algorithm")
            self.cellDynamicsCombo.setToolTip("&CELL: cell_dynamics - cell relaxation algorithm")
            self.ionDynamicsCombo.currentTextChanged.connect(self._on_ion_dynamics_changed)
            self.pressLineEdit = QLineEdit(numeric_group)
            self.pressLineEdit.setPlaceholderText("Default: 0.D0")
            self.pressLineEdit.setToolTip("&CELL: press (Ry/bohr^3). Default 0.D0")
            self.pressConvThrLineEdit = QLineEdit(numeric_group)
            self.pressConvThrLineEdit.setPlaceholderText("Default: 0.D0")
            self.pressConvThrLineEdit.setToolTip("&CELL: press_conv_thr. Default 0.D0")
            numeric_form.addRow("ion_dynamics:", self.ionDynamicsCombo)
            numeric_form.addRow("cell_dynamics:", self.cellDynamicsCombo)
            numeric_form.addRow("press:", self.pressLineEdit)
            numeric_form.addRow("press_conv_thr:", self.pressConvThrLineEdit)
        layout.addWidget(numeric_group)
        self.occupationsCombo.currentTextChanged.connect(self._on_occupations_changed)

        k_group = QGroupBox("K-point sampling", self)
        k_layout = QVBoxLayout(k_group)
        k_layout.setContentsMargins(12, 12, 12, 12)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:", k_group))
        self.kTypeCombo = QComboBox(k_group)
        self.kTypeCombo.addItems([
            "automatic",
            "crystal",
            "crystal_b",
        ])
        self.kTypeCombo.currentTextChanged.connect(self._on_k_type_changed)
        type_row.addWidget(self.kTypeCombo)
        type_row.addStretch()
        k_layout.addLayout(type_row)
        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)
        self.kGridSpinBoxes = []
        for label in ("nk1", "nk2", "nk3"):
            spin = QSpinBox(k_group)
            spin.setRange(1, 240)
            spin.setValue(6)
            spin.setToolTip(f"Set {label} for K_POINTS automatic grid")
            self.kGridSpinBoxes.append(spin)
            grid_row.addWidget(spin)
        offset_row = QHBoxLayout()
        offset_row.setSpacing(6)
        self.kOffsetSpinBoxes = []
        for label in ("s1", "s2", "s3"):
            spin = QSpinBox(k_group)
            spin.setRange(0, 1)
            spin.setToolTip(f"Set {label} shift for automatic grid")
            self.kOffsetSpinBoxes.append(spin)
            offset_row.addWidget(spin)
        # Container for automatic grid controls
        self.kAutoContainer = QWidget(k_group)
        auto_layout = QVBoxLayout(self.kAutoContainer)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.addLayout(grid_row)
        auto_layout.addLayout(offset_row)
        k_layout.addWidget(self.kAutoContainer)
        # Container for explicit points list
        self.kExplicitContainer = QWidget(k_group)
        explicit_layout = QVBoxLayout(self.kExplicitContainer)
        explicit_layout.setContentsMargins(0, 0, 0, 0)
        self.kPointsTable = QTableWidget(0, 4, self.kExplicitContainer)
        self.kPointsTable.setHorizontalHeaderLabels(["kx", "ky", "kz", "weight"])
        self.kPointsTable.horizontalHeader().setStretchLastSection(True)
        explicit_layout.addWidget(self.kPointsTable)
        row_btns = QHBoxLayout()
        self.kAddRowBtn = QPushButton("Add row", self.kExplicitContainer)
        self.kRemoveRowBtn = QPushButton("Remove row", self.kExplicitContainer)
        self.kAddRowBtn.clicked.connect(lambda: self.kPointsTable.insertRow(self.kPointsTable.rowCount()))
        self.kRemoveRowBtn.clicked.connect(self._remove_selected_kpoint_rows)
        row_btns.addWidget(self.kAddRowBtn)
        row_btns.addWidget(self.kRemoveRowBtn)
        row_btns.addStretch()
        explicit_layout.addLayout(row_btns)
        k_layout.addWidget(self.kExplicitContainer)
        # Container for crystal_b path selection
        self.kPathContainer = QWidget(k_group)
        path_layout = QVBoxLayout(self.kPathContainer)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.tpibaPathList = QListWidget(self.kPathContainer)
        self.tpibaPathList.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.tpibaPathList.setAlternatingRowColors(True)
        self.tpibaPathList.setEnabled(False)
        path_layout.addWidget(self.tpibaPathList)
        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Weight:", self.kPathContainer))
        self.tpibaWeightSpinBox = QDoubleSpinBox(self.kPathContainer)
        self.tpibaWeightSpinBox.setRange(0.001, 1000.0)
        self.tpibaWeightSpinBox.setDecimals(4)
        self.tpibaWeightSpinBox.setSingleStep(0.5)
        self.tpibaWeightSpinBox.setValue(1.0)
        weight_row.addWidget(self.tpibaWeightSpinBox)
        weight_row.addStretch()
        path_layout.addLayout(weight_row)
        self.tpibaInfoLabel = QLabel("", self.kPathContainer)
        self.tpibaInfoLabel.setWordWrap(True)
        path_layout.addWidget(self.tpibaInfoLabel)
        k_layout.addWidget(self.kPathContainer)
        self.kPointInfoLabel = QLabel("", k_group)
        self.kPointInfoLabel.setWordWrap(True)
        k_layout.addWidget(self.kPointInfoLabel)
        # Default visibility
        self._on_k_type_changed(self.kTypeCombo.currentText())
        layout.addWidget(k_group)

        # Advanced flags are shown via a separate dialog
        bool_group = QGroupBox("Advanced", self)
        bool_layout = QVBoxLayout(bool_group)
        bool_layout.setContentsMargins(12, 12, 12, 12)
        self.nbndCheck = QCheckBox("Set nbnd", bool_group)
        self.nbndCheck.setChecked(False)
        self.nbndCheck.toggled.connect(self._on_nbnd_toggled)
        bool_layout.addWidget(self.nbndCheck)
        nbnd_form = QFormLayout()
        nbnd_form.setContentsMargins(0, 0, 0, 0)
        self.nbndLineEdit = QLineEdit(bool_group)
        self.nbndLineEdit.setValidator(QIntValidator(1, 10000000, self))
        self.nbndLineEdit.setPlaceholderText("Suggested: auto")
        self.nbndLineEdit.setEnabled(False)
        nbnd_form.addRow("nbnd:", self.nbndLineEdit)
        bool_layout.addLayout(nbnd_form)
        self.nbndHintLabel = QLabel("", bool_group)
        self.nbndHintLabel.setWordWrap(True)
        self.nbndHintLabel.setStyleSheet("color: #555;")
        bool_layout.addWidget(self.nbndHintLabel)
        # Extend flags set with commonly used logical options across CONTROL and SYSTEM (backend-only set kept)
        self._flag_sections = {
            "tstress": "control",
            "tprnfor": "control",
            "wf_collect": "control",
            "restart_mode": "control",
            "nosym": "system",
            "noinv": "system",
            "noncolin": "system",
            "lspinorb": "system",
            "ace": "system",
            "x_gamma_extrapolation": "system",
        }
        # Advanced button opens a modal dialog to edit flags
        advanced_button = QPushButton("Advanced…", self)
        advanced_button.clicked.connect(self._open_advanced_flags)
        bool_layout.addWidget(QLabel("Click to configure optional flags"))
        bool_layout.addWidget(advanced_button)
        layout.addWidget(bool_group)

        # Provide hover help for K_POINTS
        k_group.setToolTip(
            "K_POINTS types: automatic | crystal | crystal_b\n"
            "automatic: nk1 nk2 nk3 s1 s2 s3 (shift 0 or 1)\n"
            "crystal: uniform mesh with 1/total weights\n"
            "crystal_b: select high-symmetry paths derived from the CIF."
        )

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        apply_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        apply_button.setText("Apply")
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self._on_accept)
        layout.addWidget(button_box)

    def _populate_defaults(self) -> None:
        self._seed_int_or_default(self.ecutwfcLineEdit, "system", "ecutwfc")
        seed_ecutrho = self._strip_numeric(self._seed_value("system", "ecutrho"))
        self._ecutrho_auto = False
        if seed_ecutrho:
            self._seed_int_or_default(self.ecutrhoLineEdit, "system", "ecutrho")
        else:
            self._ecutrho_auto = True
            self._recompute_ecutrho_from_ecutwfc()
        self._seed_or_default(self.etotConvLineEdit, "control", "etot_conv_thr")
        self._seed_or_default(self.forcConvLineEdit, "control", "forc_conv_thr")
        self._seed_or_default(self.convThrLineEdit, "electrons", "conv_thr")
        self._seed_or_default(self.mixingBetaLineEdit, "electrons", "mixing_beta")
        self._seed_int_or_default(self.electronMaxStepLineEdit, "electrons", "electron_maxstep")
        # Seed occupations, smearing, and degauss
        self._seed_occupations_defaults()
        self._seed_or_default(self.degaussLineEdit, "system", "degauss")
        self._seed_nbnd_default()

        # Initialize flag values from seed/defaults for advanced set only
        control_section = self.seed.sections.get("control", {}) if self.seed else {}
        system_section = self.seed.sections.get("system", {}) if self.seed else {}
        for key in self._advanced_flag_keys:
            section = self._flag_sections.get(key, "control")
            source = control_section if section == "control" else system_section
            if key == "restart_mode":
                raw_restart = self._strip_quotes(source.get("restart_mode", ""))
                if raw_restart:
                    value = raw_restart.lower() == "restart"
                    self._flag_initial[key] = value
                    self._flag_values[key] = value
                else:
                    self._flag_initial[key] = None
                    self._flag_values[key] = self._flag_defaults.get(key, False)
                continue

            value = self._parse_bool(source.get(key))
            self._flag_initial[key] = value
            if value is None:
                self._flag_values[key] = self._flag_defaults.get(key, False)
            else:
                self._flag_values[key] = value

        # Add contextual tooltips from INPUT_PW documentation for common parameters
        self.ecutwfcLineEdit.setToolTip("&SYSTEM: ecutwfc (Ry) - REQUIRED. Kinetic energy cutoff for wavefunctions.")
        self.ecutrhoLineEdit.setToolTip("&SYSTEM: ecutrho (Ry) - Default 4*ecutwfc. Kinetic energy cutoff for charge density.")
        self.etotConvLineEdit.setToolTip("&CONTROL: etot_conv_thr - Default 1.0D-4. Convergence threshold on total energy.")
        self.forcConvLineEdit.setToolTip("&CONTROL: forc_conv_thr - Default 1.0D-3. Convergence threshold on forces.")
        self.convThrLineEdit.setToolTip("&ELECTRONS: conv_thr - Default 1.0D-6. SCF convergence threshold.")
        self.mixingBetaLineEdit.setToolTip("&ELECTRONS: mixing_beta - Default 0.7D0. Mixing factor for self-consistency.")
        self.electronMaxStepLineEdit.setToolTip("&ELECTRONS: electron_maxstep (INTEGER) - Default 100. Max SCF iterations per step.")
        self.occupationsCombo.setToolTip("&SYSTEM: occupations - Select the electron occupation scheme. Default 'smearing'.")
        self.smearingCombo.setToolTip("&SYSTEM: smearing - Use with occupations='smearing'. Common: gaussian, mp, marzari-vanderbilt, fermi-dirac, cold.")
        self.degaussLineEdit.setToolTip("&SYSTEM: degauss (Ry) - Default 0.D0. Gaussian spreading for metallic occupations.")
        if self.nbndLineEdit is not None:
            self.nbndLineEdit.setToolTip(
                "&SYSTEM: nbnd (INTEGER) - Number of bands. Suggested from valence electrons."
            )

        default_grid = [6, 6, 6]
        default_shift = [0, 0, 0]
        grid_values = default_grid[:]
        shift_values = default_shift[:]
        displayed_kind = "automatic"
        actual_kind = "automatic"

        if self.seed:
            self._initial_k_card = list(self.seed.raw_cards.get("K_POINTS", []))
        else:
            self._initial_k_card = []

        if self.seed and isinstance(self.seed.k_points, dict):
            kind_raw = str(self.seed.k_points.get("type", "automatic")).lower()
            kind_map = {
                "tpiba": "crystal_b",
                "tpiba_b": "crystal_b",
                "tpiba_c": "crystal_b",
                "crystal_b": "crystal_b",
                "crystal_c": "crystal",
            }
            kind = kind_map.get(kind_raw, kind_raw)
            actual_kind = kind

            if kind == "automatic":
                grid = self.seed.k_points.get("grid") or default_grid
                shift = self.seed.k_points.get("offset") or default_shift
                grid_values = [
                    self._safe_int(grid[idx] if idx < len(grid) else fallback, fallback, minimum=1, maximum=240)
                    for idx, fallback in enumerate(default_grid)
                ]
                shift_values = [
                    self._safe_int(shift[idx] if idx < len(shift) else fallback, fallback, minimum=0, maximum=1)
                    for idx, fallback in enumerate(default_shift)
                ]
                self.kPointInfoLabel.setText("Values copied from the seed K_POINTS automatic grid.")
            elif kind == "crystal":
                displayed_kind = "crystal"
                self.kPointInfoLabel.setText(
                    "Existing crystal mesh detected. Adjust nk divisions to regenerate a new mesh."
                )
            elif kind == "crystal_b":
                displayed_kind = "crystal_b"
                self.kPointInfoLabel.setText(
                    "Existing crystal_b block detected. Select new paths to regenerate the card."
                )
            elif kind == "gamma":
                self.kPointInfoLabel.setText("Gamma-only sampling detected. Switch type to override if desired.")
            else:
                self.kPointInfoLabel.setText(
                    "Unsupported K_POINTS card detected. Existing block will be left untouched."
                )
        else:
            actual_kind = "automatic"
            self.kPointInfoLabel.setText("Adjust the automatic grid as needed.")

        idx = self.kTypeCombo.findText(displayed_kind, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.kTypeCombo.setCurrentIndex(idx)
        else:
            self.kTypeCombo.setCurrentIndex(0)

        for spin, value in zip(self.kGridSpinBoxes, grid_values):
            spin.setValue(value)
        for spin, value in zip(self.kOffsetSpinBoxes, shift_values):
            spin.setValue(value)

        self._initial_k_type = actual_kind
        if actual_kind == "automatic" or displayed_kind == "automatic":
            self._initial_k_grid = grid_values[:]
            self._initial_k_offset = shift_values[:]
        else:
            self._initial_k_grid = None
            self._initial_k_offset = None

        if self._is_vc_relax:
            ion_default = "bfgs"
            ion_value = self._strip_quotes(self._seed_value("ions", "ion_dynamics")).lower()
            if ion_value not in {"bfgs", "damp"}:
                ion_value = ion_default
            if self.ionDynamicsCombo is not None:
                idx = self.ionDynamicsCombo.findText(ion_value, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.ionDynamicsCombo.setCurrentIndex(idx)

            cell_default = "bfgs" if ion_value == "bfgs" else "none"
            cell_value = self._strip_quotes(self._seed_value("cell", "cell_dynamics")).lower()
            if ion_value == "bfgs":
                cell_value = "bfgs"
            if cell_value not in {"bfgs", "none", "sd", "damp-pr", "damp-w"}:
                cell_value = cell_default
            if self.cellDynamicsCombo is not None:
                idx = self.cellDynamicsCombo.findText(cell_value, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.cellDynamicsCombo.setCurrentIndex(idx)

            if self.pressLineEdit is not None:
                self._seed_or_default(self.pressLineEdit, "cell", "press")
            if self.pressConvThrLineEdit is not None:
                self._seed_or_default(self.pressConvThrLineEdit, "cell", "press_conv_thr")

            if self.ionDynamicsCombo is not None:
                self._on_ion_dynamics_changed(self.ionDynamicsCombo.currentText())

    def _safe_int(self, value: Any, fallback: int, *, minimum: int, maximum: Optional[int] = None) -> int:
        try:
            number = int(float(value))
        except Exception:
            number = fallback
        number = max(minimum, number)
        if maximum is not None:
            number = min(maximum, number)
        return number

    def _seed_or_default(self, widget: QLineEdit, section: str, key: str) -> None:
        value = self._strip_numeric(self._seed_value(section, key))
        if not value:
            fallback = self._numeric_defaults.get((section, key))
            if fallback:
                value = fallback
        widget.setText(value or "")

    def _seed_int_or_default(self, widget: QLineEdit, section: str, key: str) -> None:
        raw = self._strip_numeric(self._seed_value(section, key))
        text = ""
        if raw:
            # Accept floats like 60.0 and cast to int
            try:
                ival = int(float(raw.replace("D", "e").replace("d", "e")))
                text = str(ival)
            except Exception:
                text = raw  # fallback to raw if unexpected, validator will prevent bad edits
        else:
            fallback = self._numeric_defaults.get((section, key))
            if fallback:
                try:
                    text = str(int(float(str(fallback))))
                except Exception:
                    text = str(fallback)
        widget.setText(text)

    def _strip_numeric(self, value: str) -> str:
        cleaned = self._strip_quotes(value)
        if not cleaned:
            return ""
        return cleaned.replace(",", "").strip()

    def _seed_value(self, section: str, key: str) -> str:
        if not self.seed:
            return ""
        section_dict = self.seed.sections.get(section.lower()) or {}
        return section_dict.get(key.lower(), "")

    def _seed_nbnd_default(self) -> None:
        if self.nbndLineEdit is None:
            return
        seed_nbnd = self._strip_numeric(self._seed_value("system", "nbnd"))
        if seed_nbnd:
            self.nbndLineEdit.setText(seed_nbnd)
            if self.nbndHintLabel is not None:
                self.nbndHintLabel.setVisible(False)
            if self.nbndCheck is not None:
                self.nbndCheck.setChecked(False)
                self._on_nbnd_toggled(self.nbndCheck.isChecked())
            return

        suggested, notes = self._suggest_nbnd_value()
        if suggested is not None:
            self.nbndLineEdit.setText(str(suggested))
            self.nbndLineEdit.setPlaceholderText(f"Suggested: {suggested}")
            if self.nbndHintLabel is not None and notes:
                self.nbndHintLabel.setText(
                    "NBND suggestion may be low; missing pseudo data for: " + "; ".join(notes)
                )
        else:
            self.nbndLineEdit.setPlaceholderText("Suggested: unavailable")
            if self.nbndHintLabel is not None and notes:
                self.nbndHintLabel.setText("NBND suggestion unavailable: " + "; ".join(notes))

        if self.nbndHintLabel is not None:
            self.nbndHintLabel.setVisible(bool(self.nbndHintLabel.text()))
        if self.nbndCheck is not None:
            self.nbndCheck.setChecked(False)
            self._on_nbnd_toggled(self.nbndCheck.isChecked())

    def _on_nbnd_toggled(self, checked: bool) -> None:
        if self.nbndLineEdit is not None:
            self.nbndLineEdit.setEnabled(bool(checked))

    def _suggest_nbnd_value(self) -> tuple[Optional[int], list[str]]:
        if not self.seed:
            return None, ["seed input not loaded"]

        atom_counts: Counter[str] = Counter()
        for position in self.seed.atomic_positions:
            label = normalize_element_symbol(position.get("label", ""))
            if label:
                atom_counts[label] += 1
        if not atom_counts:
            return None, ["no atomic positions found"]

        seed_pseudos: dict[str, str] = {}
        for element, _mass, pseudo in self.seed.atomic_species:
            symbol = normalize_element_symbol(element)
            if symbol and pseudo:
                seed_pseudos[symbol] = pseudo
        if not seed_pseudos:
            return None, ["no ATOMIC_SPECIES entries found"]

        pseudo_overrides = {
            normalize_element_symbol(key): value
            for key, value in (self.job_config.pseudo_files or {}).items()
            if key and value
        }

        total_valence_electrons = 0.0
        notes: list[str] = []
        for element, pseudo_name in seed_pseudos.items():
            pseudo_name = pseudo_overrides.get(element, pseudo_name)
            pseudo_path = self._resolve_pseudo_path(pseudo_name)
            if pseudo_path is None:
                notes.append(f"{element}: {pseudo_name}")
                continue
            valence_e, issue = self._read_valence_electrons(pseudo_path)
            if valence_e is None:
                reason = f"{element}: {pseudo_path.name}"
                if issue:
                    reason += f" ({issue})"
                notes.append(reason)
                continue
            total_valence_electrons += valence_e * atom_counts.get(element, 0)

        if total_valence_electrons <= 0:
            if not notes:
                notes.append("valence electrons not found in pseudopotentials")
            return None, notes

        suggested_nbnd = int(2 * total_valence_electrons + 0.5)
        return suggested_nbnd, notes

    def _resolve_pseudo_path(self, pseudo_name: str) -> Optional[Path]:
        if not pseudo_name:
            return None
        candidate = Path(pseudo_name).expanduser()
        if candidate.is_file():
            return candidate
        for base in self._pseudo_search_dirs():
            resolved = (base / pseudo_name).expanduser()
            if resolved.is_file():
                return resolved
        return None

    def _pseudo_search_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        if self.seed:
            candidates.append(self.seed.source.parent)
        if self.job_config.working_directory:
            candidates.append(Path(self.job_config.working_directory).expanduser())
        if self.job_config.pseudo_directory:
            candidates.append(Path(self.job_config.pseudo_directory).expanduser())
        for path in self.system_config.pseudo_directories or []:
            if path:
                candidates.append(Path(path).expanduser())
        if self.system_config.pseudo_directory:
            candidates.append(Path(self.system_config.pseudo_directory).expanduser())

        seen: set[str] = set()
        ordered: list[Path] = []
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(path)
        return ordered

    def _read_valence_electrons(self, pseudo_path: Path) -> tuple[Optional[float], Optional[str]]:
        try:
            with pseudo_path.open("r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError as exc:
            return None, f"could not read file: {exc}"

        in_valence = False
        found_valence = False
        valence_e = 0.0
        for line in lines:
            if "Valence configuration" in line:
                in_valence = True
                found_valence = True
                continue
            if "Generation configuration" in line:
                break
            if in_valence:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        occ = float(parts[3])
                    except ValueError:
                        continue
                    valence_e += occ
        if found_valence and valence_e > 0:
            return valence_e, None

        # Fallback: read z_valence from PP_HEADER if valence config is missing or empty.
        joined = " ".join(lines)
        match = re.search(r"z_valence\s*=\s*['\"]?\s*([0-9.+\-EeDd]+)", joined, re.IGNORECASE)
        if match:
            raw = match.group(1)
            try:
                return float(raw.replace("D", "e").replace("d", "e")), None
            except ValueError:
                pass
        if not found_valence:
            return None, "valence configuration not found"
        return None, "valence configuration empty; z_valence not found"

    def _strip_quotes(self, value: str) -> str:
        cleaned = (value or "").strip()
        if len(cleaned) >= 2 and cleaned[0] in "'\"" and cleaned[-1] in "'\"":
            return cleaned[1:-1]
        return cleaned

    def _quote(self, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return cleaned
        if cleaned.startswith("'") and cleaned.endswith("'"):
            return cleaned
        return f"'{cleaned}'"

    def _parse_bool(self, value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        cleaned = str(value).strip().lower()
        if cleaned in {".true.", "true", "1"}:
            return True
        if cleaned in {".false.", "false", "0"}:
            return False
        return None

    def _normalize_float(self, text: str, label: str) -> Optional[str]:
        cleaned = text.strip()
        if not cleaned:
            return None
        # Allow expressions like 4*ecutwfc for ecutrho specifically
        if label.lower() == "ecutrho" and ("*" in cleaned or cleaned.lower().endswith("ecutwfc")):
            return cleaned
        # Numeric with D or E exponents
        try:
            float(cleaned.replace("d", "e").replace("D", "e"))
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        return cleaned

    def _normalize_int(self, text: str, label: str) -> Optional[str]:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            value = int(float(cleaned))
        except ValueError:
            raise ValueError(f"{label} must be an integer.")
        return str(value)

    def _on_accept(self) -> None:
        try:
            self._overrides = self._collect_overrides()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid option", str(exc))
            return
        self.accept()

    def _collect_overrides(self) -> dict[str, Any]:
        if self.calculation not in {"scf", "nscf", "bands", "vc-relax", "relax"}:
            return {}

        sections: dict[str, dict[str, Optional[str]]] = {}
        cards: dict[str, list[str]] = {}

        control = sections.setdefault("control", {})
        control["calculation"] = f"'{self.calculation}'"

        job_name = (self.job_config.name or "").strip() or self.job_config.sanitized_name()
        control["title"] = self._quote(job_name)

        if self.job_config.working_directory:
            control["outdir"] = self._quote(self.job_config.working_directory)

        pseudo_text = (self.job_config.pseudo_directory or "").strip()
        if not pseudo_text and getattr(self.job_config, "pseudo_files", None):
            try:
                parent_dirs = {str(Path(path).expanduser().parent) for path in self.job_config.pseudo_files.values() if path}
                if len(parent_dirs) == 1:
                    pseudo_text = parent_dirs.pop()
            except Exception:
                pseudo_text = ""
        if not pseudo_text:
            pseudo_text = (self.system_config.pseudo_directory or "").strip()
        if pseudo_text:
            control["pseudo_dir"] = self._quote(pseudo_text)

        # Use integer normalization for ecutwfc and ecutrho
        mapping = (
            (self.ecutwfcLineEdit, "system", "ecutwfc", "ecutwfc", True),
            (self.ecutrhoLineEdit, "system", "ecutrho", "ecutrho", True),
            (self.etotConvLineEdit, "control", "etot_conv_thr", "etot_conv_thr", False),
            (self.forcConvLineEdit, "control", "forc_conv_thr", "forc_conv_thr", False),
            (self.convThrLineEdit, "electrons", "conv_thr", "conv_thr", False),
            (self.mixingBetaLineEdit, "electrons", "mixing_beta", "mixing_beta", False),
        )

        mapping_iterable = mapping

        for widget, section_name, key_name, label, is_int in mapping_iterable:
            if is_int:
                value = self._normalize_int(widget.text(), label)
            else:
                value = self._normalize_float(widget.text(), label)
            if value is not None:
                sections.setdefault(section_name, {})[key_name] = value

        max_step = self._normalize_int(self.electronMaxStepLineEdit.text(), "electron_maxstep")
        if max_step is not None:
            sections.setdefault("electrons", {})["electron_maxstep"] = max_step
        if self.nbndLineEdit is not None and self.nbndCheck is not None and self.nbndCheck.isChecked():
            nbnd_value = self._normalize_int(self.nbndLineEdit.text(), "nbnd")
            if nbnd_value is not None:
                sections.setdefault("system", {})["nbnd"] = nbnd_value

        seed_system = self.seed.sections.get("system", {}) if self.seed else {}
        system_section = sections.setdefault("system", {})

        occ_value = (self.occupationsCombo.currentText() or "").strip().lower()
        if occ_value:
            system_section["occupations"] = f"'{occ_value}'"

        if occ_value == "smearing":
            smearing = (self.smearingCombo.currentText() or "").strip().lower()
            if smearing and smearing != "(none)":
                system_section["smearing"] = f"'{smearing}'"
            elif "smearing" in seed_system:
                system_section["smearing"] = None

            degauss_val = self._normalize_float(self.degaussLineEdit.text(), "degauss")
            if degauss_val is not None:
                system_section["degauss"] = degauss_val
            elif "degauss" in seed_system:
                system_section["degauss"] = None
        else:
            if "smearing" in seed_system:
                system_section["smearing"] = None
            if "degauss" in seed_system:
                system_section["degauss"] = None

        for key in self._advanced_flag_keys:
            default = self._flag_defaults.get(key, False)
            initial = self._flag_initial.get(key)
            current_seed_value = default if initial is None else initial
            new_value = self._flag_values.get(key, default)
            if new_value == current_seed_value:
                continue
            if key == "restart_mode":
                control_section = sections.setdefault("control", {})
                control_section["restart_mode"] = "'restart'" if new_value else "'from_scratch'"
                continue
            target_section = getattr(self, "_flag_sections", {}).get(key, "control")
            if new_value == default:
                # Revert to default: for default True flags, write explicit .false. when toggled off.
                if default is True and not new_value:
                    sections.setdefault(target_section, {})[key] = ".false."
                else:
                    sections.setdefault(target_section, {})[key] = None
            else:
                # Changing away from default: always write explicit value
                sections.setdefault(target_section, {})[key] = ".true." if new_value else ".false."

        if self._is_vc_relax:
            ion_value = "bfgs"
            if self.ionDynamicsCombo is not None:
                ion_value = (self.ionDynamicsCombo.currentText() or "bfgs").strip().lower() or "bfgs"
            ion_value = ion_value if ion_value in {"bfgs", "damp"} else "bfgs"

            ions_section = sections.setdefault("ions", {})
            ions_section["ion_dynamics"] = f"'{ion_value}'"

            cell_section = sections.setdefault("cell", {})
            cell_section["cell_dofree"] = "'ibrav'"

            if ion_value == "bfgs":
                cell_value = "bfgs"
            else:
                cell_value = "none"
                if self.cellDynamicsCombo is not None:
                    selected = (self.cellDynamicsCombo.currentText() or "none").strip().lower()
                    if selected in {"none", "sd", "damp-pr", "damp-w"}:
                        cell_value = selected
            cell_section["cell_dynamics"] = f"'{cell_value}'"

            press_val = None
            if self.pressLineEdit is not None:
                press_val = self._normalize_float(self.pressLineEdit.text(), "press")
            if press_val is None:
                press_val = "0.D0"
            cell_section["press"] = press_val

            press_thr_val = None
            if self.pressConvThrLineEdit is not None:
                press_thr_val = self._normalize_float(self.pressConvThrLineEdit.text(), "press_conv_thr")
            if press_thr_val is None:
                press_thr_val = "0.D0"
            cell_section["press_conv_thr"] = press_thr_val

        # K_POINTS override
        ktype = "automatic"
        if hasattr(self, "kTypeCombo") and self.kTypeCombo is not None:
            ktype = (self.kTypeCombo.currentText() or "automatic").strip().lower()

        k_card: Optional[list[str]] = None
        changed = False
        if self.calculation in {"scf", "nscf", "bands", "vc-relax", "relax"}:
            k_card, changed = self._build_kpoints_override(ktype)
        if changed and k_card:
            cards["K_POINTS"] = k_card

        result: dict[str, Any] = {}
        if sections:
            result["sections"] = sections
        if cards:
            result["cards"] = cards

        return result

    def _build_kpoints_override(self, ktype: str) -> tuple[Optional[list[str]], bool]:
        kind = (ktype or "automatic").strip().lower()
        if kind == "automatic":
            grid = [spin.value() for spin in self.kGridSpinBoxes]
            offset = [spin.value() for spin in self.kOffsetSpinBoxes]
            values = [str(val) for val in grid + offset]
            card = ["K_POINTS automatic", " ".join(values)]
            changed = True
            if self._initial_k_grid is not None and self._initial_k_offset is not None:
                if grid == self._initial_k_grid and offset == self._initial_k_offset:
                    changed = False
            elif self._initial_k_type == "automatic":
                changed = self._cards_changed("automatic", card)
            return card, changed

        if kind == "crystal":
            n_values = [spin.value() for spin in self.kGridSpinBoxes]
            try:
                card = generate_crystal_kmesh(n_values[0], n_values[1], n_values[2])
            except ValueError as exc:  # noqa: PERF203 - user-facing validation
                raise ValueError(f"Crystal mesh: {exc}") from exc
            return card, self._cards_changed("crystal", card)

        if kind == "crystal_b":
            if self._tpiba_error:
                raise ValueError(self._tpiba_error)
            if not self._tpiba_paths:
                raise ValueError("No crystal_b paths are available for this structure.")
            selected_paths: list[Sequence[str]] = []
            for index in range(self.tpibaPathList.count()):
                item = self.tpibaPathList.item(index)
                if item.isSelected():
                    path = item.data(Qt.ItemDataRole.UserRole) or []
                    if path:
                        selected_paths.append(path)
            if not selected_paths:
                raise ValueError("Select at least one high-symmetry path for crystal_b.")
            weight = float(self.tpibaWeightSpinBox.value())
            try:
                card = build_tpiba_b_block(selected_paths, self._tpiba_points, weight)
            except ValueError as exc:  # noqa: PERF203 - user-facing validation
                raise ValueError(f"crystal_b: {exc}") from exc
            return card, self._cards_changed("crystal_b", card)

        raise ValueError(f"Unsupported K_POINTS type '{kind}'.")

    def _cards_changed(self, target_kind: str, candidate: list[str]) -> bool:
        if self._initial_k_type != target_kind:
            return True
        if not self._initial_k_card:
            return True
        initial_norm = self._normalize_card(self._initial_k_card)
        candidate_norm = self._normalize_card(candidate)
        return initial_norm != candidate_norm

    @staticmethod
    def _normalize_card(lines: Sequence[str]) -> str:
        normalized: list[str] = []
        for raw in lines:
            text = str(raw).strip()
            if not text:
                continue
            normalized.append(" ".join(text.split()))
        return "\n".join(normalized)

    def result_overrides(self) -> dict[str, Any]:
        return self._overrides

    def _seed_occupations_defaults(self) -> None:
        seed_occ = self._strip_quotes(self._seed_value("system", "occupations")).lower()
        target_occ = seed_occ if seed_occ in self._occupations_choices else "smearing"

        idx = self.occupationsCombo.findText(target_occ, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.occupationsCombo.setCurrentIndex(idx)
        else:
            self.occupationsCombo.insertItem(0, target_occ)
            self.occupationsCombo.setCurrentIndex(0)

        if target_occ == "smearing":
            self._apply_seed_smearing_value()
        else:
            idx_none = self.smearingCombo.findText("(none)", Qt.MatchFlag.MatchFixedString)
            if idx_none >= 0:
                self.smearingCombo.setCurrentIndex(idx_none)

        self._on_occupations_changed(self.occupationsCombo.currentText())

    def _apply_seed_smearing_value(self) -> None:
        seed_smearing = self._strip_quotes(self._seed_value("system", "smearing")).lower()
        value = seed_smearing or "gaussian"
        idx = self.smearingCombo.findText(value, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.smearingCombo.setCurrentIndex(idx)
        else:
            self.smearingCombo.insertItem(0, value)
            self.smearingCombo.setCurrentIndex(0)

    def _on_occupations_changed(self, text: str) -> None:
        is_smearing = (text or "").strip().lower() == "smearing"
        self.smearingCombo.setEnabled(is_smearing)
        self.degaussLineEdit.setEnabled(is_smearing)

    def _open_advanced_flags(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Advanced flags")
        v = QVBoxLayout(dlg)
        checkboxes: dict[str, QCheckBox] = {}
        for key in self._advanced_flag_keys:
            label = self._flag_labels.get(key, key)
            cb = QCheckBox(label, dlg)
            cb.setChecked(self._flag_values.get(key, self._flag_defaults.get(key, False)))
            checkboxes[key] = cb
            v.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        v.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for key, cb in checkboxes.items():
                self._flag_values[key] = cb.isChecked()

    # ---- NSCF / vc-relax helpers ----
    def _on_ion_dynamics_changed(self, text: str) -> None:
        if not self._is_vc_relax or self.cellDynamicsCombo is None:
            return

        value = (text or "").strip().lower()
        if value == "bfgs":
            idx = self.cellDynamicsCombo.findText("bfgs", Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.cellDynamicsCombo.setCurrentIndex(idx)
            self.cellDynamicsCombo.setEnabled(False)
        else:
            if not self.cellDynamicsCombo.isEnabled():
                self.cellDynamicsCombo.setEnabled(True)
            current = (self.cellDynamicsCombo.currentText() or "").strip().lower()
            if current in {"", "bfgs"}:
                idx = self.cellDynamicsCombo.findText("none", Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.cellDynamicsCombo.setCurrentIndex(idx)

    # ---- NSCF K_POINTS helpers ----
    def _on_k_type_changed(self, text: str) -> None:
        ktype = (text or "").strip().lower()
        if ktype == "automatic":
            self.kAutoContainer.setVisible(True)
            self.kExplicitContainer.setVisible(False)
            self.kPathContainer.setVisible(False)
            self.kPointInfoLabel.setText("In automatic: nk1 nk2 nk3 s1 s2 s3 (shift 0 or 1).")
            for spin in self.kOffsetSpinBoxes:
                spin.setEnabled(True)
        elif ktype == "crystal":
            self.kAutoContainer.setVisible(True)
            self.kExplicitContainer.setVisible(False)
            self.kPathContainer.setVisible(False)
            for spin in self.kOffsetSpinBoxes:
                spin.setValue(0)
                spin.setEnabled(False)
            self.kPointInfoLabel.setText("Crystal mesh uses nk1/nk2/nk3. Offsets are disabled and treated as zero.")
        elif ktype == "crystal_b":
            self.kAutoContainer.setVisible(False)
            self.kExplicitContainer.setVisible(False)
            self.kPathContainer.setVisible(True)
            self.kPointInfoLabel.setText("Select one or more high-symmetry paths and set a weight.")
            for spin in self.kOffsetSpinBoxes:
                spin.setEnabled(False)
            self._ensure_tpiba_data_loaded()
        else:
            self.kAutoContainer.setVisible(False)
            self.kExplicitContainer.setVisible(True)
            self.kPathContainer.setVisible(False)
            self.kPointInfoLabel.setText("Enter kx ky kz and weight per row.")
            for spin in self.kOffsetSpinBoxes:
                spin.setEnabled(True)

    def _ensure_tpiba_data_loaded(self) -> None:
        if self._tpiba_loaded:
            return

        if not self.cif_path:
            self._tpiba_error = "Select a CIF file to enable crystal_b sampling."
            self.tpibaPathList.clear()
            self.tpibaPathList.setEnabled(False)
            self.tpibaInfoLabel.setText(self._tpiba_error)
            return

        try:
            data = load_high_symmetry_path(self.cif_path)
        except Exception as exc:  # noqa: BLE001
            self._tpiba_error = str(exc)
            self.tpibaInfoLabel.setText(f"Failed to load high-symmetry paths: {exc}")
            self.tpibaPathList.clear()
            self.tpibaPathList.setEnabled(False)
            return

        self._tpiba_loaded = True
        self._tpiba_error = None
        self._tpiba_paths = data.grouped_paths
        self._tpiba_points = data.points_frac

        self.tpibaPathList.clear()
        for group in self._tpiba_paths:
            label = " → ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, group)
            self.tpibaPathList.addItem(item)

        self.tpibaPathList.setEnabled(bool(self._tpiba_paths))
        if not self._tpiba_paths:
            self.tpibaInfoLabel.setText("No high-symmetry paths available for this structure.")
        else:
            self.tpibaInfoLabel.setText(
                "Hold Ctrl/Cmd to select multiple paths. Weight applies to each listed point."
            )

    def _remove_selected_kpoint_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.kPointsTable.selectedIndexes()}, reverse=True)
        if not rows and self.kPointsTable.rowCount():
            rows = [self.kPointsTable.rowCount() - 1]
        for r in rows:
            if 0 <= r < self.kPointsTable.rowCount():
                self.kPointsTable.removeRow(r)

    # ---------------------- Helpers: dynamic defaults ------------------
    def _on_ecutwfc_changed(self, _text: str) -> None:
        if getattr(self, "_ecutrho_auto", False):
            self._recompute_ecutrho_from_ecutwfc()

    def _on_ecutrho_edited(self, _text: str) -> None:
        # User touched ecutrho -> disable auto mode
        self._ecutrho_auto = False

    def _recompute_ecutrho_from_ecutwfc(self) -> None:
        try:
            ew = int(float((self.ecutwfcLineEdit.text() or "0").strip()))
            if ew > 0:
                self.ecutrhoLineEdit.setText(str(ew * 4))
        except Exception:
            # leave as-is if invalid
            pass


class BandxOptionsDialog(QDialog):
    """Modal dialog to adjust bandx (bands.x) overrides before generating inputs."""

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BANDS options")
        self.setModal(True)

        self._defaults = defaults
        self._optional_meta: Dict[str, Dict[str, Any]] = defaults.get("optional", {})
        self._optional_order: List[str] = list(defaults.get("optional_order", self._optional_meta.keys()))
        self._result: Dict[str, Any] = {}
        self._optional_fields: Dict[str, Dict[str, Any]] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        prefix_default = str(self._defaults.get("prefix", ""))
        outdir_default = str(self._defaults.get("outdir", ""))
        filband_default = str(self._defaults.get("filband", ""))

        self.prefixLineEdit = QLineEdit(prefix_default, self)
        self.prefixLineEdit.setPlaceholderText("Required (e.g. pwscf)")

        self.outdirLineEdit = QLineEdit(outdir_default, self)
        self.outdirLineEdit.setPlaceholderText("Default: ./")

        self.filbandLineEdit = QLineEdit(filband_default, self)
        self.filbandLineEdit.setPlaceholderText("Required (e.g. bands.dat)")

        form.addRow("prefix:", self.prefixLineEdit)
        form.addRow("outdir:", self.outdirLineEdit)
        form.addRow("filband:", self.filbandLineEdit)

        layout.addLayout(form)

        if self._optional_meta:
            optional_group = QGroupBox("Optional overrides", self)
            optional_form = QFormLayout(optional_group)
            optional_form.setContentsMargins(12, 12, 12, 12)
            optional_form.setSpacing(8)

            for key in self._optional_order:
                meta = self._optional_meta.get(key)
                if not meta:
                    continue
                row_widget = self._build_optional_row(optional_group, key, meta)
                label = meta.get("label", key)
                optional_form.addRow(f"{label}:", row_widget)

            layout.addWidget(optional_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_optional_row(self, parent: QWidget, key: str, meta: Dict[str, Any]) -> QWidget:
        include_box = QCheckBox("Set", parent)
        include_box.setToolTip("Enable to include this parameter in the generated BANDS input.")
        widget: QWidget

        field_type = (meta.get("type") or "string").lower()
        if field_type == "bool":
            combo = QComboBox(parent)
            combo.addItem(".false.", False)
            combo.addItem(".true.", True)
            combo.setCurrentIndex(1 if bool(meta.get("default")) else 0)
            widget = combo
        elif field_type == "int":
            spin = QSpinBox(parent)
            minimum = int(meta.get("minimum", -99999))
            maximum = int(meta.get("maximum", 99999))
            spin.setRange(minimum, maximum)
            try:
                spin.setValue(int(meta.get("default", 0)))
            except Exception:
                spin.setValue(0)
            widget = spin
        else:
            line_edit = QLineEdit(parent)
            default_text = str(meta.get("default", ""))
            line_edit.setText(default_text)
            if default_text:
                line_edit.setPlaceholderText(default_text)
            widget = line_edit

        widget.setEnabled(False)
        include_box.toggled.connect(partial(self._handle_optional_toggle, widget, meta))

        container = QWidget(parent)
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(include_box)
        row_layout.addWidget(widget, 1)

        self._optional_fields[key] = {
            "include": include_box,
            "widget": widget,
            "meta": meta,
        }

        return container

    @staticmethod
    def _handle_optional_toggle(widget: QWidget, meta: Dict[str, Any], checked: bool) -> None:
        widget.setEnabled(checked)
        if not checked:
            return
        if isinstance(widget, QLineEdit):
            default_text = str(meta.get("default", ""))
            if default_text and not widget.text().strip():
                widget.setText(default_text)

    def _on_accept(self) -> None:
        prefix = self.prefixLineEdit.text().strip()
        if not prefix:
            QMessageBox.warning(self, "Missing value", "Provide a prefix before continuing.")
            return

        outdir = self.outdirLineEdit.text().strip() or "./"

        filband = self.filbandLineEdit.text().strip()
        if not filband:
            QMessageBox.warning(self, "Missing value", "Provide a filband value before continuing.")
            return

        try:
            extra = self._collect_optional_overrides()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid option", str(exc))
            return

        self._result = {
            "prefix": prefix,
            "outdir": outdir,
            "filband": filband,
            "extra": extra,
        }
        self.accept()

    def _collect_optional_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for key, data in self._optional_fields.items():
            include_box: QCheckBox = data["include"]
            if not include_box.isChecked():
                continue
            widget = data["widget"]
            meta = data["meta"]
            value = self._extract_optional_value(key, widget, meta)
            overrides[key] = value
        return overrides

    def _extract_optional_value(self, key: str, widget: QWidget, meta: Dict[str, Any]) -> Any:
        label = meta.get("label", key)
        field_type = (meta.get("type") or "string").lower()

        if field_type == "bool" and isinstance(widget, QComboBox):
            data = widget.currentData()
            return bool(data)

        if field_type == "int":
            if isinstance(widget, QSpinBox):
                return int(widget.value())
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    raise ValueError(f"{label} cannot be empty when enabled.")
                try:
                    return int(text)
                except ValueError:  # noqa: PERF203 - user-facing validation
                    raise ValueError(f"{label} must be an integer.")
            raise ValueError(f"Unsupported widget for '{label}'.")

        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if not text:
                raise ValueError(f"{label} cannot be empty when enabled.")
            return text

        if isinstance(widget, QComboBox):
            return str(widget.currentText()).strip()

        raise ValueError(f"Unsupported control for '{label}'.")

    def result_options(self) -> Dict[str, Any]:
        return dict(self._result)


class MatdynOptionsDialog(QDialog):
    """Modal dialog to configure matdyn.x parameters."""

    _ASR_CHOICES: tuple[str, ...] = ("no", "simple", "crystal", "all", "one-dim", "two-dim")

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("matdyn.x options")
        self.setModal(True)

        self._defaults = defaults
        self._result: Dict[str, Any] = {}
        self._hs_loaded = False
        self._hs_paths: List[List[str]] = []
        self._hs_points: Dict[str, Sequence[float]] = {}
        self._hs_error: Optional[str] = None
        self._flfrc_hint = str(self._defaults.get("flfrc_hint", "")).strip()

        self._build_ui()
        self._apply_defaults()
        self._update_q_controls()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        input_group = QGroupBox("&INPUT parameters", self)
        input_form = QFormLayout(input_group)
        input_form.setContentsMargins(12, 12, 12, 12)
        input_form.setSpacing(8)

        self.flfrcLineEdit = QLineEdit(input_group)
        placeholder = "Required force-constant file from q2r.x"
        if self._flfrc_hint:
            placeholder = f"{placeholder} (e.g. {self._flfrc_hint})"
        self.flfrcLineEdit.setPlaceholderText(placeholder)
        tooltip = "flfrc: Force-constant file produced by q2r.x (mandatory)."
        if self._flfrc_hint:
            tooltip = f"{tooltip} Common choice: {self._flfrc_hint}."
        self.flfrcLineEdit.setToolTip(tooltip)

        self.flfrqLineEdit = QLineEdit(input_group)
        self.flfrqLineEdit.setPlaceholderText("Default: matdyn.freq")
        self.flfrqLineEdit.setToolTip("flfrq: Frequency output file. Default matdyn.freq.")

        self.fldosLineEdit = QLineEdit(input_group)
        self.fldosLineEdit.setPlaceholderText("Default: matdyn.dos")
        self.fldosLineEdit.setToolTip("fldos: Density of states output file (used when dos=.true.).")

        self.fldynLineEdit = QLineEdit(input_group)
        self.fldynLineEdit.setPlaceholderText("Optional (leave blank to omit)")
        self.fldynLineEdit.setToolTip("fldyn: Optional dynamical matrices output file. If empty it is skipped.")

        self.asrCombo = QComboBox(input_group)
        self.asrCombo.addItems(list(self._ASR_CHOICES))
        self.asrCombo.setToolTip("asr: Acoustic sum rule enforcement (default 'no').")

        self.deltaELineEdit = QLineEdit(input_group)
        self.deltaELineEdit.setPlaceholderText("Optional (energy step, e.g. 0.5)")
        self.deltaELineEdit.setToolTip("deltaE: Energy spacing (Ry) for DOS. Leave blank for internal default.")

        self.ndosSpin = QSpinBox(input_group)
        self.ndosSpin.setRange(0, 50000)
        self.ndosSpin.setSpecialValueText("auto")
        self.ndosSpin.setToolTip("ndos: Number of energy points for DOS integration. Set 0 for default.")

        nk_container = QWidget(input_group)
        nk_layout = QHBoxLayout(nk_container)
        nk_layout.setContentsMargins(0, 0, 0, 0)
        nk_layout.setSpacing(6)
        self.nkSpinBoxes: List[QSpinBox] = []
        for label in ("nk1", "nk2", "nk3"):
            spin = QSpinBox(nk_container)
            spin.setRange(0, 400)
            spin.setToolTip(f"{label}: Monkhorst-Pack grid divisions for phonon DOS.")
            self.nkSpinBoxes.append(spin)
            nk_layout.addWidget(spin)
        nk_layout.addStretch()

        input_form.addRow("flfrc:", self.flfrcLineEdit)
        input_form.addRow("flfrq:", self.flfrqLineEdit)
        input_form.addRow("fldos:", self.fldosLineEdit)
        input_form.addRow("fldyn:", self.fldynLineEdit)
        input_form.addRow("asr:", self.asrCombo)
        input_form.addRow("deltaE:", self.deltaELineEdit)
        input_form.addRow("ndos:", self.ndosSpin)
        input_form.addRow("nk1/nk2/nk3:", nk_container)

        layout.addWidget(input_group)

        switches_group = QGroupBox("Switches", self)
        switches_layout = QVBoxLayout(switches_group)
        switches_layout.setContentsMargins(12, 12, 12, 12)
        switches_layout.setSpacing(6)

        self.dosCheck = QCheckBox("Compute phonon DOS (dos)", switches_group)
        self.dosCheck.setToolTip(
            "dos: When enabled, matdyn.x integrates the phonon DOS on the nk mesh (default .false.)."
        )

        self.la2FCheck = QCheckBox("Write Eliashberg function (la2F)", switches_group)
        self.la2FCheck.setToolTip(
            "la2F: Emit the electron-phonon spectral function alpha^2F(omega) (only written when enabled)."
        )

        self.qBandFormCheck = QCheckBox("Interpret q list as band path (q_in_band_form)", switches_group)
        self.qBandFormCheck.setToolTip(
            "q_in_band_form: Treat q-points as a dispersive path; supply explicit points when DOS remains disabled."
        )

        switches_layout.addWidget(self.dosCheck)
        switches_layout.addWidget(self.la2FCheck)
        switches_layout.addWidget(self.qBandFormCheck)
        switches_layout.addStretch()

        layout.addWidget(switches_group)

        self.qGroupBox = QGroupBox("Band-path specification", self)
        self.qGroupBox.setEnabled(False)
        q_layout = QVBoxLayout(self.qGroupBox)
        q_layout.setContentsMargins(12, 12, 12, 12)
        q_layout.setSpacing(8)

        self.qInfoLabel = QLabel("Enable a CIF-backed path to select segments.", self.qGroupBox)
        self.qInfoLabel.setWordWrap(True)
        q_layout.addWidget(self.qInfoLabel)

        self.pathList = QListWidget(self.qGroupBox)
        self.pathList.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.pathList.setAlternatingRowColors(True)
        q_layout.addWidget(self.pathList)

        control_row = QHBoxLayout()
        control_row.setSpacing(6)
        control_row.addWidget(QLabel("Default nq:", self.qGroupBox))
        self.weightSpin = QSpinBox(self.qGroupBox)
        self.weightSpin.setRange(1, 1000)
        control_row.addWidget(self.weightSpin)
        self.applyWeightButton = QPushButton("Apply to all", self.qGroupBox)
        control_row.addWidget(self.applyWeightButton)
        control_row.addStretch()
        q_layout.addLayout(control_row)

        self.qPointsTable = QTableWidget(0, 4, self.qGroupBox)
        self.qPointsTable.setHorizontalHeaderLabels(["xq1", "xq2", "xq3", "nq"])
        self.qPointsTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        q_layout.addWidget(self.qPointsTable)

        self.nqsLabel = QLabel("nqs = 0", self.qGroupBox)
        q_layout.addWidget(self.nqsLabel)

        layout.addWidget(self.qGroupBox)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Apply")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        outer_layout.addWidget(button_box)

        self.dosCheck.toggled.connect(self._update_q_controls)
        self.qBandFormCheck.toggled.connect(self._update_q_controls)
        self.pathList.itemSelectionChanged.connect(self._populate_q_point_table)
        self.applyWeightButton.clicked.connect(self._apply_weight_to_all)

    def _apply_defaults(self) -> None:
        self.flfrcLineEdit.setText(str(self._defaults.get("flfrc", "")))
        self.flfrqLineEdit.setText(str(self._defaults.get("flfrq", "matdyn.freq")))
        self.fldosLineEdit.setText(str(self._defaults.get("fldos", "matdyn.dos")))
        self.fldynLineEdit.setText(str(self._defaults.get("fldyn", "")))

        asr_default = str(self._defaults.get("asr", "no")).lower()
        idx = self.asrCombo.findText(asr_default, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.asrCombo.setCurrentIndex(idx)

        delta_default = str(self._defaults.get("deltaE", "") or "")
        self.deltaELineEdit.setText(delta_default)

        ndos_default = int(self._defaults.get("ndos", 0) or 0)
        self.ndosSpin.setValue(ndos_default)

        nk_defaults = tuple(self._defaults.get("nk", (0, 0, 0)) or (0, 0, 0))
        for spin, value in zip(self.nkSpinBoxes, nk_defaults):
            spin.setValue(int(value))

        weight_default = int(self._defaults.get("weight", 50) or 50)
        self.weightSpin.setValue(weight_default)

        self.dosCheck.setChecked(bool(self._defaults.get("dos", False)))
        self.la2FCheck.setChecked(bool(self._defaults.get("la2f", False)))
        self.qBandFormCheck.setChecked(bool(self._defaults.get("q_in_band_form", False)))

    def _on_accept(self) -> None:
        try:
            self._result = self._collect_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid option", str(exc))
            return
        self.accept()

    def _collect_values(self) -> Dict[str, Any]:
        flfrc = self.flfrcLineEdit.text().strip()
        if not flfrc:
            raise ValueError("Provide a value for flfrc before continuing.")

        fldos = self.fldosLineEdit.text().strip() or "matdyn.dos"
        flfrq = self.flfrqLineEdit.text().strip() or "matdyn.freq"
        fldyn = self.fldynLineEdit.text().strip()

        asr_value = (self.asrCombo.currentText() or "no").strip().lower() or "no"
        if asr_value not in self._ASR_CHOICES:
            raise ValueError("Select a valid acoustic sum rule (asr).")

        deltaE_text = self.deltaELineEdit.text().strip()
        if deltaE_text:
            self._parse_float(deltaE_text, "deltaE")

        ndos_value = int(self.ndosSpin.value())

        nk_values = tuple(int(spin.value()) for spin in self.nkSpinBoxes)

        dos_flag = self.dosCheck.isChecked()
        la2f_flag = self.la2FCheck.isChecked()
        q_band_flag = self.qBandFormCheck.isChecked()

        q_points: List[Dict[str, Any]] = []
        if q_band_flag and not dos_flag:
            if self.qPointsTable.rowCount() == 0:
                raise ValueError("Select at least one high-symmetry segment when q_in_band_form is enabled.")
            for row in range(self.qPointsTable.rowCount()):
                header_item = self.qPointsTable.verticalHeaderItem(row)
                label = header_item.text() if header_item else f"q{row + 1}"
                coords: List[float] = []
                for col in range(3):
                    item = self.qPointsTable.item(row, col)
                    if item is None:
                        raise ValueError("Incomplete q-point table entry.")
                    try:
                        coords.append(float(item.text()))
                    except ValueError as exc:  # noqa: PERF203 - user validation
                        raise ValueError("Q-point coordinates must be numeric.") from exc
                widget = self.qPointsTable.cellWidget(row, 3)
                weight = int(widget.value()) if isinstance(widget, QSpinBox) else int(self.weightSpin.value())
                q_points.append({"label": label, "coords": tuple(coords), "weight": weight})

        return {
            "flfrc": flfrc,
            "fldos": fldos,
            "flfrq": flfrq,
            "fldyn": fldyn,
            "asr": asr_value,
            "deltaE": deltaE_text,
            "ndos": ndos_value,
            "nk": nk_values,
            "dos": dos_flag,
            "la2f": la2f_flag,
            "q_in_band_form": q_band_flag,
            "q_points": q_points,
        }

    def result_options(self) -> Dict[str, Any]:
        return dict(self._result)

    def _update_q_controls(self) -> None:
        enabled = bool(self.qBandFormCheck.isChecked()) and not self.dosCheck.isChecked()
        if enabled and not self._ensure_high_symmetry_data():
            enabled = False
            self.qBandFormCheck.blockSignals(True)
            self.qBandFormCheck.setChecked(False)
            self.qBandFormCheck.blockSignals(False)

        self.qGroupBox.setEnabled(enabled)
        if not enabled:
            if self.dosCheck.isChecked() and self.qBandFormCheck.isChecked():
                self.qInfoLabel.setText("Disable DOS to configure a dispersion path.")
            elif not self._defaults.get("cif_path"):
                self.qInfoLabel.setText("Select a workflow CIF source to enable band-path sampling.")
            elif self._hs_error:
                self.qInfoLabel.setText(self._hs_error)
            else:
                self.qInfoLabel.setText("Enable q_in_band_form to configure a dispersion path.")
            self.qPointsTable.setRowCount(0)
            self.nqsLabel.setText("nqs = 0")
            return

        if self.pathList.count() and not any(self.pathList.item(i).isSelected() for i in range(self.pathList.count())):
            self.pathList.item(0).setSelected(True)
        self.qInfoLabel.setText("Select high-symmetry segments to build the path. Hold Ctrl to choose multiple.")
        self._populate_q_point_table()

    def _ensure_high_symmetry_data(self) -> bool:
        if self._hs_loaded:
            return bool(self._hs_paths)

        cif_path = str(self._defaults.get("cif_path", "")).strip()
        if not cif_path:
            self._hs_error = "Select a workflow CIF source to enable band-path sampling."
            self.pathList.clear()
            return False

        try:
            data = load_high_symmetry_path(cif_path)
        except Exception as exc:  # noqa: BLE001 - surface to user
            self._hs_error = str(exc)
            self.pathList.clear()
            return False

        self._hs_loaded = True
        self._hs_error = None
        self._hs_paths = data.grouped_paths
        self._hs_points = data.points_frac

        self.pathList.clear()
        for group in self._hs_paths:
            label = " → ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, group)
            self.pathList.addItem(item)

        if not self._hs_paths:
            self._hs_error = "No high-symmetry segments available for the loaded structure."
            return False

        return True

    def _populate_q_point_table(self) -> None:
        if not self.qGroupBox.isEnabled():
            return

        selected_paths: List[List[str]] = []
        for idx in range(self.pathList.count()):
            item = self.pathList.item(idx)
            if item and item.isSelected():
                path = item.data(Qt.ItemDataRole.UserRole) or []
                if path:
                    selected_paths.append(list(path))

        if not selected_paths:
            self.qPointsTable.setRowCount(0)
            self.nqsLabel.setText("nqs = 0")
            return

        labels: List[str] = []
        for path in selected_paths:
            for pos, label in enumerate(path):
                if labels and labels[-1] == label and pos == 0:
                    continue
                labels.append(label)

        row_count = len(labels)
        self.qPointsTable.setRowCount(row_count)
        self.qPointsTable.setColumnCount(4)
        self.qPointsTable.setVerticalHeaderLabels(labels)

        for row, label in enumerate(labels):
            coords = self._hs_points.get(label)
            if coords is None:
                coords = (0.0, 0.0, 0.0)
            for col, value in enumerate(coords[:3]):
                item = QTableWidgetItem(f"{float(value):.10f}")
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.qPointsTable.setItem(row, col, item)
            spin = QSpinBox(self.qPointsTable)
            spin.setRange(1, 1000)
            spin.setValue(self.weightSpin.value())
            self.qPointsTable.setCellWidget(row, 3, spin)

        self.nqsLabel.setText(f"nqs = {row_count}")

    def _apply_weight_to_all(self) -> None:
        value = self.weightSpin.value()
        for row in range(self.qPointsTable.rowCount()):
            widget = self.qPointsTable.cellWidget(row, 3)
            if isinstance(widget, QSpinBox):
                widget.setValue(value)

    @staticmethod
    def _parse_float(text: str, label: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        try:
            float(cleaned.replace("d", "e").replace("D", "e"))
        except ValueError as exc:  # noqa: PERF203 - user validation
            raise ValueError(f"{label} must be a number.") from exc
        return cleaned

class PdosOptionsDialog(QDialog):
    """Modal dialog to configure projwfc.x parameters for PDOS generation."""

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PDOS options")
        self.setModal(True)

        self._defaults = defaults
        self._result: Dict[str, Any] = {}
        self._current_auto_filpdos = str(defaults.get("filpdos", ""))
        self._filpdos_user_overridden = False

        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        common_group = QGroupBox("Common parameters", self)
        common_form = QFormLayout(common_group)
        common_form.setContentsMargins(12, 12, 12, 12)
        common_form.setSpacing(8)

        prefix_default = str(self._defaults.get("prefix", ""))
        outdir_default = str(self._defaults.get("outdir", "")) or "./"
        filpdos_default = str(self._defaults.get("filpdos", ""))

        self.prefixLineEdit = QLineEdit(prefix_default, self)
        self.prefixLineEdit.setPlaceholderText("Required")
        self.prefixLineEdit.setToolTip("prefix: Identifier of the pw.x run providing the wavefunctions.")

        self.outdirLineEdit = QLineEdit(outdir_default, self)
        self.outdirLineEdit.setPlaceholderText("Default: ./")
        self.outdirLineEdit.setToolTip("outdir: Directory containing pw.x scratch data (same as pw.x).")

        self.filpdosLineEdit = QLineEdit(filpdos_default, self)
        self.filpdosLineEdit.setPlaceholderText("Default: {prefix}.pdos")
        self.filpdosLineEdit.setToolTip("filpdos: Output file collecting the projected DOS.")

        common_form.addRow("prefix:", self.prefixLineEdit)
        common_form.addRow("outdir:", self.outdirLineEdit)
        common_form.addRow("filpdos:", self.filpdosLineEdit)

        optional_frame = QWidget(common_group)
        optional_layout = QGridLayout(optional_frame)
        optional_layout.setContentsMargins(0, 0, 0, 0)
        optional_layout.setHorizontalSpacing(8)
        optional_layout.setVerticalSpacing(6)

        self.ngaussLineEdit = QLineEdit(optional_frame)
        self.ngaussLineEdit.setPlaceholderText("Optional")
        self.ngaussLineEdit.setToolTip("ngauss: Smearing type index (see INPUT_PROJWFC).")

        self.degaussLineEdit = QLineEdit(optional_frame)
        self.degaussLineEdit.setPlaceholderText("Optional")
        self.degaussLineEdit.setToolTip("degauss (Ry): Width of the smearing function.")

        self.eminLineEdit = QLineEdit(optional_frame)
        self.eminLineEdit.setPlaceholderText("Optional")
        self.eminLineEdit.setToolTip("Emin (eV): Lower bound of the energy window.")

        self.emaxLineEdit = QLineEdit(optional_frame)
        self.emaxLineEdit.setPlaceholderText("Optional")
        self.emaxLineEdit.setToolTip("Emax (eV): Upper bound of the energy window.")

        self.deltaELineEdit = QLineEdit(optional_frame)
        self.deltaELineEdit.setPlaceholderText("Optional")
        self.deltaELineEdit.setToolTip("DeltaE (eV): Energy step for DOS sampling.")

        self.filprojLineEdit = QLineEdit(optional_frame)
        self.filprojLineEdit.setPlaceholderText("Optional")
        self.filprojLineEdit.setToolTip("filproj: File name for atomic projection weights.")

        optional_layout.addWidget(QLabel("ngauss:", optional_frame), 0, 0)
        optional_layout.addWidget(self.ngaussLineEdit, 0, 1)
        optional_layout.addWidget(QLabel("degauss:", optional_frame), 1, 0)
        optional_layout.addWidget(self.degaussLineEdit, 1, 1)
        optional_layout.addWidget(QLabel("Emin:", optional_frame), 2, 0)
        optional_layout.addWidget(self.eminLineEdit, 2, 1)
        optional_layout.addWidget(QLabel("Emax:", optional_frame), 3, 0)
        optional_layout.addWidget(self.emaxLineEdit, 3, 1)
        optional_layout.addWidget(QLabel("DeltaE:", optional_frame), 4, 0)
        optional_layout.addWidget(self.deltaELineEdit, 4, 1)
        optional_layout.addWidget(QLabel("filproj:", optional_frame), 5, 0)
        optional_layout.addWidget(self.filprojLineEdit, 5, 1)

        common_form.addRow(optional_frame)
        layout.addWidget(common_group)

        advanced_group = QGroupBox("Advanced flags", self)
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setContentsMargins(12, 12, 12, 12)
        advanced_layout.setSpacing(6)

        self.lsymCheck = QCheckBox("Preserve symmetry (lsym)", advanced_group)
        self.lsymCheck.setChecked(True)
        self.lsymCheck.setToolTip("lsym: Keep crystal symmetries in projections (default .true.).")

        self.diagBasisCheck = QCheckBox("Diagonalize overlap basis (diag_basis)", advanced_group)
        self.diagBasisCheck.setToolTip("diag_basis: Report projections in the diagonalized overlap basis.")

        self.pawprojCheck = QCheckBox("PAW projections (pawproj)", advanced_group)
        self.pawprojCheck.setToolTip("pawproj: Enable PAW-specific projection output when available.")

        self.lbinaryCheck = QCheckBox("Binary data output (lbinary_data)", advanced_group)
        self.lbinaryCheck.setToolTip("lbinary_data: Write PDOS data using binary format.")

        self.kresolvedCheck = QCheckBox("k-resolved DOS (kresolveddos)", advanced_group)
        self.kresolvedCheck.setToolTip("kresolveddos: Produce k-resolved projected DOS.")

        self.plotboxesCheck = QCheckBox("Plot bounding boxes (plotboxes)", advanced_group)
        self.plotboxesCheck.setToolTip("plotboxes: Draw boxes around selected atoms in plot files.")

        for widget in (
            self.lsymCheck,
            self.diagBasisCheck,
            self.pawprojCheck,
            self.lbinaryCheck,
            self.kresolvedCheck,
            self.plotboxesCheck,
        ):
            advanced_layout.addWidget(widget)

        layout.addWidget(advanced_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Apply")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        outer_layout.addWidget(button_box)

        self._optional_fields: Dict[str, Dict[str, Any]] = {
            "ngauss": {"widget": self.ngaussLineEdit, "type": "int", "label": "ngauss"},
            "degauss": {"widget": self.degaussLineEdit, "type": "float", "label": "degauss"},
            "Emin": {"widget": self.eminLineEdit, "type": "float", "label": "Emin"},
            "Emax": {"widget": self.emaxLineEdit, "type": "float", "label": "Emax"},
            "DeltaE": {"widget": self.deltaELineEdit, "type": "float", "label": "DeltaE"},
            "filproj": {"widget": self.filprojLineEdit, "type": "string", "label": "filproj"},
        }

        self.prefixLineEdit.textChanged.connect(self._on_prefix_changed)
        self.filpdosLineEdit.textEdited.connect(self._on_filpdos_edited)

    def _on_prefix_changed(self, text: str) -> None:
        cleaned = text.strip()
        fallback = cleaned or "projwfc"
        auto_value = f"{fallback}.pdos"
        self._current_auto_filpdos = auto_value
        if not self._filpdos_user_overridden:
            self.filpdosLineEdit.blockSignals(True)
            try:
                self.filpdosLineEdit.setText(auto_value)
            finally:
                self.filpdosLineEdit.blockSignals(False)

    def _on_filpdos_edited(self, _text: str) -> None:
        self._filpdos_user_overridden = True

    @staticmethod
    def _parse_int(text: str, label: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        try:
            value = int(float(cleaned.replace("d", "e").replace("D", "e")))
        except ValueError:
            raise ValueError(f"{label} must be an integer.")
        return str(value)

    @staticmethod
    def _parse_float(text: str, label: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        try:
            float(cleaned.replace("d", "e").replace("D", "e"))
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        return cleaned

    def _collect_values(self) -> Dict[str, Any]:
        prefix = self.prefixLineEdit.text().strip()
        if not prefix:
            raise ValueError("Provide a prefix before continuing.")

        outdir = self.outdirLineEdit.text().strip() or "./"

        filpdos_text = self.filpdosLineEdit.text().strip()
        if not filpdos_text:
            filpdos_text = f"{prefix}.pdos"

        optional: Dict[str, str] = {}
        for key, meta in self._optional_fields.items():
            widget: QLineEdit = meta["widget"]  # type: ignore[assignment]
            raw = widget.text().strip()
            if not raw:
                continue
            kind = (meta.get("type") or "string").lower()
            if kind == "int":
                parsed = self._parse_int(raw, meta.get("label", key))
            elif kind == "float":
                parsed = self._parse_float(raw, meta.get("label", key))
            else:
                parsed = raw
            optional[key] = parsed

        advanced: Dict[str, bool] = {}
        if not self.lsymCheck.isChecked():
            advanced["lsym"] = False
        if self.diagBasisCheck.isChecked():
            advanced["diag_basis"] = True
        if self.pawprojCheck.isChecked():
            advanced["pawproj"] = True
        if self.lbinaryCheck.isChecked():
            advanced["lbinary_data"] = True
        if self.kresolvedCheck.isChecked():
            advanced["kresolveddos"] = True
        if self.plotboxesCheck.isChecked():
            advanced["plotboxes"] = True

        return {
            "prefix": prefix,
            "outdir": outdir,
            "filpdos": filpdos_text,
            "optional": optional,
            "advanced": advanced,
        }

    def _on_accept(self) -> None:
        try:
            self._result = self._collect_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid option", str(exc))
            return
        self.accept()

    def result_options(self) -> Dict[str, Any]:
        return dict(self._result)


class EpsilonOptionsDialog(QDialog):
    """Modal dialog to configure epsilon.x (optics) parameters."""

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optics options")
        self.setModal(True)

        self._defaults = defaults
        self._result: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        common_group = QGroupBox("Common parameters", self)
        common_form = QFormLayout(common_group)
        common_form.setContentsMargins(12, 12, 12, 12)
        common_form.setSpacing(8)

        prefix_default = str(self._defaults.get("prefix", ""))
        outdir_default = str(self._defaults.get("outdir", "")) or "./"

        self.prefixLineEdit = QLineEdit(prefix_default, self)
        self.prefixLineEdit.setPlaceholderText("Required")
        self.prefixLineEdit.setToolTip("prefix: Identifier matching the SCF/NSCF runs.")

        self.outdirLineEdit = QLineEdit(outdir_default, self)
        self.outdirLineEdit.setPlaceholderText("Default: ./")
        self.outdirLineEdit.setToolTip("outdir: Scratch directory hosting the prior pw.x wavefunctions.")

        common_form.addRow("prefix:", self.prefixLineEdit)
        common_form.addRow("outdir:", self.outdirLineEdit)

        layout.addWidget(common_group)

        grid_group = QGroupBox("Energy grid", self)
        grid_form = QFormLayout(grid_group)
        grid_form.setContentsMargins(12, 12, 12, 12)
        grid_form.setSpacing(8)

        smeartype_default = str(self._defaults.get("smeartype", "lorentzian"))
        self.smeartypeCombo = QComboBox(self)
        self.smeartypeCombo.addItems(["lorentzian", "gauss"])
        idx = self.smeartypeCombo.findText(smeartype_default, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.smeartypeCombo.setCurrentIndex(idx)

        self.interSmearLineEdit = QLineEdit(str(self._defaults.get("intersmear", "0.1d0")), self)
        self.interSmearLineEdit.setPlaceholderText("0.1d0")
        self.interSmearLineEdit.setToolTip("intersmear (Ry): Lifetime broadening for interband transitions.")

        self.intraSmearLineEdit = QLineEdit(str(self._defaults.get("intrasmear", "0.0d0")), self)
        self.intraSmearLineEdit.setPlaceholderText("0.0d0")
        self.intraSmearLineEdit.setToolTip("intrasmear (Ry): Intraband smearing width.")

        self.wmaxLineEdit = QLineEdit(str(self._defaults.get("wmax", "30.0d0")), self)
        self.wmaxLineEdit.setPlaceholderText("30.0d0")
        self.wmaxLineEdit.setToolTip("wmax (Ry): Maximum frequency for the dielectric spectrum.")

        self.wminLineEdit = QLineEdit(str(self._defaults.get("wmin", "0.02d0")), self)
        self.wminLineEdit.setPlaceholderText("0.02d0")
        self.wminLineEdit.setToolTip("wmin (Ry): Minimum frequency for the dielectric spectrum.")

        self.nwLineEdit = QLineEdit(str(self._defaults.get("nw", "500")), self)
        self.nwLineEdit.setPlaceholderText("500")
        self.nwLineEdit.setToolTip("nw: Number of frequency points (integer).")

        self.shiftLineEdit = QLineEdit(str(self._defaults.get("shift", "0.0d0")), self)
        self.shiftLineEdit.setPlaceholderText("0.0d0")
        self.shiftLineEdit.setToolTip("shift (Ry): Apply an energy shift to the spectrum.")

        grid_form.addRow("smeartype:", self.smeartypeCombo)
        grid_form.addRow("intersmear:", self.interSmearLineEdit)
        grid_form.addRow("intrasmear:", self.intraSmearLineEdit)
        grid_form.addRow("wmax:", self.wmaxLineEdit)
        grid_form.addRow("wmin:", self.wminLineEdit)
        grid_form.addRow("nw:", self.nwLineEdit)
        grid_form.addRow("shift:", self.shiftLineEdit)

        layout.addWidget(grid_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_accept(self) -> None:
        prefix = self.prefixLineEdit.text().strip()
        if not prefix:
            QMessageBox.warning(self, "Missing value", "Provide a prefix before continuing.")
            return

        outdir = self.outdirLineEdit.text().strip() or "./"

        smeartype = (self.smeartypeCombo.currentText() or "lorentzian").strip()
        if smeartype.lower() not in {"lorentzian", "gauss"}:
            QMessageBox.warning(self, "Invalid smearing", "Select a valid smeartype (lorentzian or gauss).")
            return

        intersmear = self._require_text(self.interSmearLineEdit, "intersmear")
        if intersmear is None:
            return
        intrasmear = self._require_text(self.intraSmearLineEdit, "intrasmear")
        if intrasmear is None:
            return
        wmax = self._require_text(self.wmaxLineEdit, "wmax")
        if wmax is None:
            return
        wmin = self._require_text(self.wminLineEdit, "wmin")
        if wmin is None:
            return
        shift = self._require_text(self.shiftLineEdit, "shift")
        if shift is None:
            return

        nw_text = self.nwLineEdit.text().strip()
        if not nw_text:
            QMessageBox.warning(self, "Missing value", "Provide a value for nw before continuing.")
            return
        try:
            nw_value = int(float(nw_text))
        except ValueError:
            QMessageBox.warning(self, "Invalid value", "nw must be an integer.")
            return
        if nw_value <= 0:
            QMessageBox.warning(self, "Invalid value", "nw must be positive.")
            return

        self._result = {
            "prefix": prefix,
            "outdir": outdir,
            "smeartype": smeartype.lower(),
            "intersmear": intersmear,
            "intrasmear": intrasmear,
            "wmax": wmax,
            "wmin": wmin,
            "nw": str(nw_value),
            "shift": shift,
        }
        self.accept()

    @staticmethod
    def _require_text(widget: QLineEdit, label: str) -> Optional[str]:
        text = widget.text().strip()
        if not text:
            QMessageBox.warning(widget.window(), "Missing value", f"Provide a value for {label} before continuing.")
            return None
        return text

    def result_options(self) -> Dict[str, Any]:
        return dict(self._result)


class PhononOptionsDialog(QDialog):
    """Modal dialog to configure ph.x (phonon) parameters."""

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phonon options")
        self.setModal(True)

        self._defaults = defaults
        self._result: Dict[str, Any] = {}
        self._default_values: Dict[str, Any] = {}
        self._flag_checks: Dict[str, QCheckBox] = {}
        self._flag_initial: Dict[str, bool] = {}
        self._initial_state: Dict[str, Any] = {}
        self._tpiba_loaded = False
        self._tpiba_paths: List[List[str]] = []
        self._tpiba_points: Dict[str, Sequence[float]] = {}
        self._tpiba_error: Optional[str] = None

        self._build_ui()
        self._apply_defaults()
        self._update_q_controls()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(12)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget(scroll_area)
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        common_group = QGroupBox("Common parameters", self)
        common_form = QFormLayout(common_group)
        common_form.setContentsMargins(12, 12, 12, 12)
        common_form.setSpacing(8)

        self.prefixLineEdit = QLineEdit(common_group)
        self.prefixLineEdit.setPlaceholderText("Required (matches pw.x prefix)")

        self.outdirLineEdit = QLineEdit(common_group)
        self.outdirLineEdit.setPlaceholderText("Default: ./")

        self.tr2PhLineEdit = QLineEdit(common_group)
        self.tr2PhLineEdit.setPlaceholderText("Default: 1.0d-12")

        self.nmixSpinBox = QSpinBox(common_group)
        self.nmixSpinBox.setRange(1, 200)

        self.verbosityCombo = QComboBox(common_group)
        self.verbosityCombo.addItems(["default", "low", "medium", "high", "minimal", "debug"])

        self.fildynLineEdit = QLineEdit(common_group)
        self.fildynLineEdit.setPlaceholderText("Default: matdyn")

        self.fildvscfLineEdit = QLineEdit(common_group)
        self.fildvscfLineEdit.setPlaceholderText("Optional (e.g. prefix.dvscf)")

        common_form.addRow("prefix:", self.prefixLineEdit)
        common_form.addRow("outdir:", self.outdirLineEdit)
        common_form.addRow("tr2_ph:", self.tr2PhLineEdit)
        common_form.addRow("nmix_ph:", self.nmixSpinBox)
        common_form.addRow("verbosity:", self.verbosityCombo)
        common_form.addRow("fildyn:", self.fildynLineEdit)
        common_form.addRow("fildvscf:", self.fildvscfLineEdit)
        layout.addWidget(common_group)

        flags_group = QGroupBox("Logical flags", self)
        flags_layout = QGridLayout(flags_group)
        flags_layout.setContentsMargins(12, 12, 12, 12)
        flags_layout.setHorizontalSpacing(12)
        flags_layout.setVerticalSpacing(6)

        flag_specs = [
            ("epsil", "Compute dielectric constant (epsil)"),
            ("trans", "Compute phonons (trans)"),
            ("lmultipole", "Include multipole corrections (lmultipole)"),
            ("recover", "Resume from previous run (recover)"),
            ("low_directory_check", "Strict phsave lookup (low_directory_check)"),
            ("ldisp", "Use q-point mesh (ldisp)"),
            ("qplot", "Read q list (qplot)"),
            ("q2d", "Interpret q list as 2D grid (q2d)"),
            ("q_in_band_form", "Interpret q list as band path (q_in_band_form)"),
            ("lqdir", "Create per-q directories (lqdir)"),
            ("search_sym", "Search mode symmetry (search_sym)"),
        ]

        for index, (name, tooltip) in enumerate(flag_specs):
            checkbox = QCheckBox(name, flags_group)
            checkbox.setToolTip(tooltip)
            row = index // 3
            col = index % 3
            flags_layout.addWidget(checkbox, row, col)
            self._flag_checks[name] = checkbox

        layout.addWidget(flags_group)

        elph_group = QGroupBox("Electron-phonon options", self)
        elph_form = QFormLayout(elph_group)
        elph_form.setContentsMargins(12, 12, 12, 12)
        elph_form.setSpacing(8)

        self.electronPhononCombo = QComboBox(elph_group)
        self.electronPhononCombo.addItem("Disabled (default)", "")
        for option in ["simple", "interpolated", "lambda_tetra", "gamma_tetra", "epa", "ahc"]:
            self.electronPhononCombo.addItem(option, option)

        self.elPhNsigmaSpin = QSpinBox(elph_group)
        self.elPhNsigmaSpin.setRange(1, 9999)

        self.elPhSigmaLineEdit = QLineEdit(elph_group)
        self.elPhSigmaLineEdit.setPlaceholderText("Default: 0.02")

        self.diagonalizationCombo = QComboBox(elph_group)
        self.diagonalizationCombo.addItems(["david", "cg"])

        elph_form.addRow("electron_phonon:", self.electronPhononCombo)
        elph_form.addRow("el_ph_nsigma:", self.elPhNsigmaSpin)
        elph_form.addRow("el_ph_sigma:", self.elPhSigmaLineEdit)
        elph_form.addRow("diagonalization:", self.diagonalizationCombo)
        layout.addWidget(elph_group)

        grid_group = QGroupBox("Grid parameters", self)
        grid_layout = QVBoxLayout(grid_group)
        grid_layout.setContentsMargins(12, 12, 12, 12)
        grid_layout.setSpacing(6)

        nq_row = QHBoxLayout()
        nq_row.setSpacing(6)
        nq_row.addWidget(QLabel("nq1/nq2/nq3:", grid_group))
        self.nqSpinBoxes: List[QSpinBox] = []
        for name in ("nq1", "nq2", "nq3"):
            spin = QSpinBox(grid_group)
            spin.setRange(0, 400)
            spin.setObjectName(name)
            self.nqSpinBoxes.append(spin)
            nq_row.addWidget(spin)
        nq_row.addStretch()
        grid_layout.addLayout(nq_row)

        nk_row = QHBoxLayout()
        nk_row.setSpacing(6)
        nk_row.addWidget(QLabel("nk1/nk2/nk3:", grid_group))
        self.nkSpinBoxes: List[QSpinBox] = []
        for name in ("nk1", "nk2", "nk3"):
            spin = QSpinBox(grid_group)
            spin.setRange(0, 400)
            spin.setObjectName(name)
            self.nkSpinBoxes.append(spin)
            nk_row.addWidget(spin)
        nk_row.addStretch()
        grid_layout.addLayout(nk_row)

        kshift_row = QHBoxLayout()
        kshift_row.setSpacing(6)
        kshift_row.addWidget(QLabel("k1/k2/k3:", grid_group))
        self.kShiftSpinBoxes: List[QSpinBox] = []
        for name in ("k1", "k2", "k3"):
            spin = QSpinBox(grid_group)
            spin.setRange(-1, 1)
            spin.setObjectName(name)
            self.kShiftSpinBoxes.append(spin)
            kshift_row.addWidget(spin)
        kshift_row.addStretch()
        grid_layout.addLayout(kshift_row)

        layout.addWidget(grid_group)

        q_group = QGroupBox("Q-point specification", self)
        q_layout = QVBoxLayout(q_group)
        q_layout.setContentsMargins(12, 12, 12, 12)
        q_layout.setSpacing(6)

        self.includeSingleQCheck = QCheckBox("Include single q-point", q_group)
        q_layout.addWidget(self.includeSingleQCheck)

        self.singleQContainer = QWidget(q_group)
        single_layout = QHBoxLayout(self.singleQContainer)
        single_layout.setContentsMargins(24, 0, 0, 0)
        single_layout.setSpacing(6)
        single_layout.addWidget(QLabel("xq(1)/xq(2)/xq(3):", self.singleQContainer))
        self.singleQEdits: List[QLineEdit] = []
        for _ in range(3):
            edit = QLineEdit(self.singleQContainer)
            edit.setPlaceholderText("0.0")
            edit.setFixedWidth(70)
            self.singleQEdits.append(edit)
            single_layout.addWidget(edit)
        single_layout.addStretch()
        q_layout.addWidget(self.singleQContainer)

        self.includeQPathCheck = QCheckBox("Include high-symmetry path", q_group)
        q_layout.addWidget(self.includeQPathCheck)

        self.qPathContainer = QWidget(q_group)
        path_layout = QVBoxLayout(self.qPathContainer)
        path_layout.setContentsMargins(24, 0, 0, 0)
        path_layout.setSpacing(6)

        self.qPathList = QListWidget(self.qPathContainer)
        self.qPathList.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.qPathList.setAlternatingRowColors(True)
        self.qPathList.setEnabled(False)
        path_layout.addWidget(self.qPathList)

        weight_row = QHBoxLayout()
        weight_row.setSpacing(6)
        weight_row.addWidget(QLabel("Weight (nq):", self.qPathContainer))
        self.qPathWeightSpin = QSpinBox(self.qPathContainer)
        self.qPathWeightSpin.setRange(1, 9999)
        self.qPathWeightSpin.setValue(1)
        weight_row.addWidget(self.qPathWeightSpin)
        weight_row.addStretch()
        path_layout.addLayout(weight_row)

        self.qPathInfoLabel = QLabel("Enable ldisp and qplot to select a path.", self.qPathContainer)
        self.qPathInfoLabel.setWordWrap(True)
        path_layout.addWidget(self.qPathInfoLabel)

        q_layout.addWidget(self.qPathContainer)
        layout.addWidget(q_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Apply")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        outer_layout.addWidget(button_box)

        self.includeSingleQCheck.toggled.connect(self._update_q_controls)
        self.includeQPathCheck.toggled.connect(self._update_q_controls)
        self._flag_checks["ldisp"].toggled.connect(self._update_q_controls)
        self._flag_checks["qplot"].toggled.connect(self._update_q_controls)

    def _apply_defaults(self) -> None:
        prefix_default = str(self._defaults.get("prefix", "pwscf"))
        outdir_default = str(self._defaults.get("outdir", "./")) or "./"
        tr2_default = str(self._defaults.get("tr2_ph", "1.0d-12"))
        nmix_default = int(self._defaults.get("nmix_ph", 4))
        verbosity_default = str(self._defaults.get("verbosity", "default"))
        fildyn_default = str(self._defaults.get("fildyn", "matdyn"))
        fildvscf_default = str(self._defaults.get("fildvscf", "")).strip()

        self.prefixLineEdit.setText(prefix_default)
        self.outdirLineEdit.setText(outdir_default)
        self.tr2PhLineEdit.setText(tr2_default)
        self.nmixSpinBox.setValue(nmix_default)
        idx = self.verbosityCombo.findText(verbosity_default, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.verbosityCombo.setCurrentIndex(idx)
        self.fildynLineEdit.setText(fildyn_default)
        self.fildvscfLineEdit.setText(fildvscf_default)

        flag_defaults: Dict[str, bool] = dict(self._defaults.get("flags", {}))
        for name, checkbox in self._flag_checks.items():
            value = flag_defaults.get(name, False)
            checkbox.setChecked(bool(value))

        elph_default = str(self._defaults.get("electron_phonon", "")).strip()
        idx = self.electronPhononCombo.findData(elph_default, flags=Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.electronPhononCombo.setCurrentIndex(idx)
        else:
            self.electronPhononCombo.setCurrentIndex(0)

        self.elPhNsigmaSpin.setValue(int(self._defaults.get("el_ph_nsigma", 10)))
        self.elPhSigmaLineEdit.setText(str(self._defaults.get("el_ph_sigma", "0.02")))

        diag_default = str(self._defaults.get("diagonalization", "david"))
        idx = self.diagonalizationCombo.findText(diag_default, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.diagonalizationCombo.setCurrentIndex(idx)

        for spin, value in zip(self.nqSpinBoxes, self._sequence_or_default(self._defaults.get("nq"), 3, 0)):
            spin.setValue(value)
        for spin, value in zip(self.nkSpinBoxes, self._sequence_or_default(self._defaults.get("nk"), 3, 0)):
            spin.setValue(value)
        for spin, value in zip(self.kShiftSpinBoxes, self._sequence_or_default(self._defaults.get("kshift"), 3, 0)):
            spin.setValue(value)

        self.qPathWeightSpin.setValue(int(self._defaults.get("q_weight", 1)))

        self._default_values = {
            "prefix": prefix_default.strip(),
            "outdir": outdir_default.strip() or "./",
            "tr2_ph": tr2_default.strip(),
            "nmix_ph": self.nmixSpinBox.value(),
            "verbosity": verbosity_default.strip(),
            "fildyn": fildyn_default.strip(),
            "fildvscf": fildvscf_default,
            "electron_phonon": str(self.electronPhononCombo.currentData() or ""),
            "el_ph_nsigma": self.elPhNsigmaSpin.value(),
            "el_ph_sigma": self.elPhSigmaLineEdit.text().strip(),
            "diagonalization": self.diagonalizationCombo.currentText().strip(),
            "nq": tuple(spin.value() for spin in self.nqSpinBoxes),
            "nk": tuple(spin.value() for spin in self.nkSpinBoxes),
            "kshift": tuple(spin.value() for spin in self.kShiftSpinBoxes),
        }
        self._initial_state = dict(self._default_values)
        self._flag_initial = {name: checkbox.isChecked() for name, checkbox in self._flag_checks.items()}

    @staticmethod
    def _sequence_or_default(value: Any, expected: int, fallback: int) -> Tuple[int, ...]:
        if isinstance(value, (list, tuple)) and len(value) == expected:
            result: List[int] = []
            for entry in value:
                try:
                    result.append(int(float(entry)))
                except Exception:
                    result.append(fallback)
            return tuple(result)
        return tuple(fallback for _ in range(expected))

    def _update_q_controls(self) -> None:
        ldisp_on = self._flag_checks["ldisp"].isChecked()
        qplot_on = self._flag_checks["qplot"].isChecked()

        allow_single = not ldisp_on and not qplot_on
        if not allow_single and self.includeSingleQCheck.isChecked():
            self.includeSingleQCheck.blockSignals(True)
            self.includeSingleQCheck.setChecked(False)
            self.includeSingleQCheck.blockSignals(False)
        self.includeSingleQCheck.setEnabled(allow_single)
        self.singleQContainer.setEnabled(allow_single and self.includeSingleQCheck.isChecked())

        allow_path = ldisp_on and qplot_on
        if not allow_path and self.includeQPathCheck.isChecked():
            self.includeQPathCheck.blockSignals(True)
            self.includeQPathCheck.setChecked(False)
            self.includeQPathCheck.blockSignals(False)
        self.includeQPathCheck.setEnabled(allow_path)
        if allow_path:
            self._ensure_tpiba_loaded()
        enabled = allow_path and self.includeQPathCheck.isChecked()
        self.qPathContainer.setEnabled(enabled)
        if not allow_path:
            self.qPathInfoLabel.setText("Enable ldisp and qplot to select a path.")

    def _ensure_tpiba_loaded(self) -> None:
        if self._tpiba_loaded or self._tpiba_error:
            return

        cif_path = str(self._defaults.get("cif_path") or "").strip()
        if not cif_path:
            self._tpiba_error = "Select a workflow CIF source to enable high-symmetry paths."
            self.qPathList.setEnabled(False)
            self.qPathInfoLabel.setText(self._tpiba_error)
            return

        try:
            data = load_high_symmetry_path(cif_path)
        except Exception as exc:  # noqa: BLE001
            self._tpiba_error = str(exc)
            self.qPathList.setEnabled(False)
            self.qPathInfoLabel.setText(self._tpiba_error)
            return

        self._tpiba_loaded = True
        self._tpiba_paths = data.grouped_paths
        self._tpiba_points = data.points_frac
        self.qPathList.clear()
        for group in self._tpiba_paths:
            label = " -> ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, group)
            self.qPathList.addItem(item)
        if self._tpiba_paths:
            self.qPathList.setEnabled(True)
            self.qPathInfoLabel.setText("Select one or more segments; weight applies to each point.")
        else:
            self.qPathList.setEnabled(False)
            self.qPathInfoLabel.setText("No high-symmetry paths available for this structure.")

    def _on_accept(self) -> None:
        try:
            self._result = self._collect_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid option", str(exc))
            return
        self.accept()

    def _collect_values(self) -> Dict[str, Any]:
        prefix = self.prefixLineEdit.text().strip()
        if not prefix:
            raise ValueError("Provide a prefix before continuing.")
        outdir = self.outdirLineEdit.text().strip() or "./"

        namelist: Dict[str, str] = {
            "prefix": self._quote(prefix),
            "outdir": self._quote(outdir),
        }

        tr2_value = self.tr2PhLineEdit.text().strip()
        if tr2_value and tr2_value != self._initial_state.get("tr2_ph"):
            self._validate_float(tr2_value, "tr2_ph")
            namelist["tr2_ph"] = tr2_value

        if self.nmixSpinBox.value() != self._initial_state.get("nmix_ph"):
            namelist["nmix_ph"] = str(self.nmixSpinBox.value())

        verbosity_value = self.verbosityCombo.currentText().strip()
        if verbosity_value != self._initial_state.get("verbosity"):
            namelist["verbosity"] = self._quote(verbosity_value)

        fildyn_value = self.fildynLineEdit.text().strip()
        if fildyn_value and fildyn_value != self._initial_state.get("fildyn"):
            namelist["fildyn"] = self._quote(fildyn_value)

        fildvscf_value = self.fildvscfLineEdit.text().strip()
        if fildvscf_value and fildvscf_value != self._initial_state.get("fildvscf"):
            namelist["fildvscf"] = self._quote(fildvscf_value)

        for name, checkbox in self._flag_checks.items():
            initial = self._flag_initial.get(name, False)
            current = checkbox.isChecked()
            if current == initial:
                continue
            namelist[name] = ".true." if current else ".false."

        elph_choice = str(self.electronPhononCombo.currentData() or "").strip()
        if elph_choice != self._initial_state.get("electron_phonon"):
            namelist["electron_phonon"] = self._quote(elph_choice)

        if self.elPhNsigmaSpin.value() != self._initial_state.get("el_ph_nsigma"):
            namelist["el_ph_nsigma"] = str(self.elPhNsigmaSpin.value())

        sigma_text = self.elPhSigmaLineEdit.text().strip()
        if sigma_text and sigma_text != self._initial_state.get("el_ph_sigma"):
            self._validate_float(sigma_text, "el_ph_sigma")
            namelist["el_ph_sigma"] = sigma_text

        diagonalization_choice = self.diagonalizationCombo.currentText().strip()
        if diagonalization_choice != self._initial_state.get("diagonalization"):
            namelist["diagonalization"] = self._quote(diagonalization_choice)

        nq_values = tuple(spin.value() for spin in self.nqSpinBoxes)
        if nq_values != tuple(self._initial_state.get("nq", (0, 0, 0))):
            for key, value in zip(("nq1", "nq2", "nq3"), nq_values):
                namelist[key] = str(value)

        nk_values = tuple(spin.value() for spin in self.nkSpinBoxes)
        if nk_values != tuple(self._initial_state.get("nk", (0, 0, 0))):
            for key, value in zip(("nk1", "nk2", "nk3"), nk_values):
                namelist[key] = str(value)

        kshift_values = tuple(spin.value() for spin in self.kShiftSpinBoxes)
        if kshift_values != tuple(self._initial_state.get("kshift", (0, 0, 0))):
            for key, value in zip(("k1", "k2", "k3"), kshift_values):
                namelist[key] = str(value)

        q_lines: List[str] = []
        ldisp_on = self._flag_checks["ldisp"].isChecked()
        qplot_on = self._flag_checks["qplot"].isChecked()

        if self.includeSingleQCheck.isChecked():
            if ldisp_on or qplot_on:
                raise ValueError("Single q-point is available only when ldisp and qplot are disabled.")
            coords: List[float] = []
            for edit in self.singleQEdits:
                text = edit.text().strip()
                if not text:
                    raise ValueError("Provide all components of the q-point.")
                coords.append(self._parse_float(text, "q-point component"))
            formatted = " ".join(self._format_float(value) for value in coords)
            q_lines.append(formatted)

        if self.includeQPathCheck.isChecked():
            if not (ldisp_on and qplot_on):
                raise ValueError("High-symmetry paths require both ldisp and qplot enabled.")
            if self._tpiba_error:
                raise ValueError(self._tpiba_error)
            if not self._tpiba_loaded:
                self._ensure_tpiba_loaded()
            selected: List[List[str]] = []
            for row in range(self.qPathList.count()):
                item = self.qPathList.item(row)
                if item and item.isSelected():
                    group = item.data(Qt.ItemDataRole.UserRole)
                    if group:
                        selected.append(list(group))
            if not selected:
                raise ValueError("Select at least one path segment to include.")
            weight = self.qPathWeightSpin.value()
            gathered: List[str] = []
            for group in selected:
                for label in group:
                    coords = self._tpiba_points.get(label)
                    if coords is None:
                        raise ValueError(f"Missing coordinates for high-symmetry point '{label}'.")
                    base = " ".join(self._format_float(value) for value in coords[:3])
                    gathered.append(f"{base} {weight}    ! {label}")
            if not gathered:
                raise ValueError("Could not assemble q-point list from the selected path.")
            q_lines.append(str(len(gathered)))
            q_lines.extend(gathered)

        if namelist.get("ldisp") == ".true." and namelist.get("qplot") != ".true.":
            namelist.setdefault("qplot", ".false.")

        return {"namelist": namelist, "q_lines": q_lines}

    @staticmethod
    def _quote(value: str) -> str:
        text = value.strip()
        if not text:
            return "''"
        if text.startswith("'") and text.endswith("'"):
            return text
        return f"'{text}'"

    @staticmethod
    def _validate_float(text: str, label: str) -> None:
        try:
            float(text.replace("D", "e").replace("d", "e"))
        except ValueError as exc:  # noqa: PERF203 - user input validation
            raise ValueError(f"{label} must be numeric.") from exc

    @staticmethod
    def _parse_float(text: str, label: str) -> float:
        try:
            return float(text.replace("D", "e").replace("d", "e"))
        except ValueError as exc:  # noqa: PERF203 - user input validation
            raise ValueError(f"{label} must be numeric.") from exc

    @staticmethod
    def _format_float(value: float) -> str:
        formatted = f"{value:.8f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        if formatted == "-0":
            formatted = "0"
        return formatted

    def result_options(self) -> Dict[str, Any]:
        return dict(self._result)

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()

        self.job_queue: List[Dict[str, Any]] = []
        self._updating_generated_preview = False
        self.active_runner: Optional[JobRunner] = None
        self.job_queue_file = PROJECT_ROOT / "job_queue.json"
        self.job_daemon_script = PROJECT_ROOT / "job_daemon.py"
        self.job_daemon_pid_file = PROJECT_ROOT / ".job_daemon.pid"
        self.system_config: SystemConfig = load_system_config()
        self._pseudo_files: Dict[str, str] = {}
        self._current_species_order: List[str] = []
        self._workflow_local_pseudo_files: Dict[str, str] = {}
        self._workflow_full_relativistic_pseudo_files: Dict[str, str] = {}
        self._workflow_species_order: List[str] = []
        self._warned_missing_pseudo_dir = False
        self._generated_preview_paths: List[Optional[str]] = []
        self._updating_command_preview = False
        self._custom_commands_override = False
        self._custom_job_active = False
        self._custom_preview_cache: Optional[tuple[str, str, Optional[str]]] = None
        self._custom_command_cache = ""
        self._qe_input_generator_thread: Optional[QeInputGeneratorThread] = None
        self._job_mpi_procs = self._system_mpi_procs()

        self._job_preview_dirty = True
        self._active_job_index: Optional[int] = None

        self._build_ui()
        self._configure_status_bar()
        self._connect_signals()
        self._apply_system_config_defaults(force=True)
        self._load_saved_job_queue()
        self._sync_job_status_table()
        # Start periodic polling of the job queue file so GUI reflects daemon updates.
        try:
            interval_sec = int(os.environ.get("QE_JOB_POLL_INTERVAL", "1"))
        except Exception:
            interval_sec = 1
        self._job_poll_interval_ms = max(100, interval_sec * 1000)
        self._job_poll_timer = QTimer(self)
        self._job_poll_timer.timeout.connect(self._poll_job_queue_file)
        self._job_poll_timer.start(self._job_poll_interval_ms)
        if hasattr(self, "jobCalcTypeComboBox"):
            self._apply_calculation_type_side_effects(self.jobCalcTypeComboBox.currentText())

    def _build_ui(self) -> None:
        self.setWindowTitle("Quantum ESPRESSO Helper by Luise")
        self.setMinimumSize(640, 480)
        self.resize(900, 700)

        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.setSpacing(12)

        self.tabWidget = NoWheelTabWidget(central_widget)
        self.tabWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        central_layout.addWidget(self.tabWidget)
        self.setCentralWidget(central_widget)

        self._build_qe_input_generator_tab()
        self._build_job_creator_tab()
        self._build_workflow_tab()
        self._build_job_status_tab()

        self._build_menu_bar()

    def _build_qe_input_generator_tab(self) -> None:
        generator_tab = QWidget()
        generator_layout = QVBoxLayout(generator_tab)
        generator_layout.setContentsMargins(0, 0, 0, 0)
        generator_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal, generator_tab)

        config_scroll = QScrollArea(splitter)
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.Shape.NoFrame)
        config_panel = QWidget(config_scroll)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(12)

        file_group = QGroupBox("Files", config_panel)
        file_layout = QGridLayout(file_group)
        self.qeGeneratorInputLineEdit = QLineEdit(file_group)
        self.qeGeneratorInputLineEdit.setPlaceholderText("Select a structure file")
        self.qeGeneratorInputBrowseButton = QPushButton("Browse...", file_group)
        self.qeGeneratorOutputLineEdit = QLineEdit(file_group)
        self.qeGeneratorOutputLineEdit.setPlaceholderText("Output PWscf input file")
        self.qeGeneratorOutputBrowseButton = QPushButton("Browse...", file_group)
        file_layout.addWidget(QLabel("Upload file:", file_group), 0, 0)
        file_layout.addWidget(self.qeGeneratorInputLineEdit, 0, 1)
        file_layout.addWidget(self.qeGeneratorInputBrowseButton, 0, 2)
        file_layout.addWidget(QLabel("Output file:", file_group), 1, 0)
        file_layout.addWidget(self.qeGeneratorOutputLineEdit, 1, 1)
        file_layout.addWidget(self.qeGeneratorOutputBrowseButton, 1, 2)
        config_layout.addWidget(file_group)

        common_group = QGroupBox("Input settings", config_panel)
        common_form = QFormLayout(common_group)
        self.qeGeneratorKpointsCombo = QComboBox(common_group)
        self.qeGeneratorKpointsCombo.addItem(
            "fine (0.10 1/Angstrom, 0.0125 Ry)",
            ("fine", 0.10),
        )
        self.qeGeneratorKpointsCombo.addItem(
            "medium (0.15 1/Angstrom, 0.0200 Ry)",
            ("medium", 0.15),
        )
        self.qeGeneratorKpointsCombo.addItem(
            "coarse (0.30 1/Angstrom, 0.0275 Ry)",
            ("coarse", 0.30),
        )
        self.qeGeneratorKpointsCombo.addItem(
            "Protocol default (local: 0.10 1/Angstrom)",
            ("", 0.10),
        )
        self.qeGeneratorRefineCellCombo = QComboBox(common_group)
        self._set_combo_items(
            self.qeGeneratorRefineCellCombo,
            [
                ("Yes", "yes"),
                ("No", "no"),
            ],
        )
        self.qeGeneratorSymprecCombo = QComboBox(common_group)
        for exponent in range(-2, -9, -1):
            self.qeGeneratorSymprecCombo.addItem(f"10^{exponent} Angstrom", str(exponent))
        self._set_combo_current_data(self.qeGeneratorSymprecCombo, "-5")
        common_form.addRow("K-points distance:", self.qeGeneratorKpointsCombo)
        common_form.addRow("Refine cell:", self.qeGeneratorRefineCellCombo)
        common_form.addRow("Symmetry tolerance (symprec):", self.qeGeneratorSymprecCombo)
        config_layout.addWidget(common_group)

        service_group = QGroupBox("Generator", config_panel)
        service_layout = QVBoxLayout(service_group)
        self.materialsCloudCheckBox = QCheckBox("Use Materials Cloud service", service_group)
        self.materialsCloudCheckBox.setToolTip("Requires an internet connection.")
        service_layout.addWidget(self.materialsCloudCheckBox)
        internet_note = QLabel("Materials Cloud generation requires an internet connection.", service_group)
        internet_note.setWordWrap(True)
        service_layout.addWidget(internet_note)
        config_layout.addWidget(service_group)

        self.materialsCloudOptionsGroup = QGroupBox("Materials Cloud options", config_panel)
        mc_layout = QVBoxLayout(self.materialsCloudOptionsGroup)
        mc_layout.setSpacing(10)

        mc_form = QFormLayout()
        self.mcFileFormatCombo = QComboBox(self.materialsCloudOptionsGroup)
        self._set_combo_items(
            self.mcFileFormatCombo,
            [
                ("CIF File (.cif) [parser: ase]", "cif-ase"),
                ("CIF File (.cif) [parser: pymatgen]", "cif-pymatgen"),
                ("Quantum ESPRESSO input [parser: qetools]", "qeinp-qetools"),
                ("XCrySDen (.xsf) [parser: ase]", "xsf-ase"),
                ("VASP POSCAR [parser: ase]", "vasp-ase"),
                ("CASTEP (.cell) [parser: ase]", "castep-ase"),
                ("Protein Data Bank format (.pdb)", "pdb-ase"),
                ("Auto-detect", "auto"),
            ],
        )
        self.mcProtocolCombo = QComboBox(self.materialsCloudOptionsGroup)
        self._set_combo_items(
            self.mcProtocolCombo,
            [("fast", "fast"), ("balanced", "balanced"), ("stringent", "stringent")],
        )
        self._set_combo_current_data(self.mcProtocolCombo, "balanced")
        self.mcXcCombo = QComboBox(self.materialsCloudOptionsGroup)
        self._set_combo_items(self.mcXcCombo, [("PBEsol", "PBEsol"), ("PBE", "PBE")])
        self.mcMagnetizationCombo = QComboBox(self.materialsCloudOptionsGroup)
        self._set_combo_items(
            self.mcMagnetizationCombo,
            [
                ("non-magnetic metal (fractional occupations)", "non-magnetic-fractional"),
                ("non-magnetic insulator (fixed occupations)", "non-magnetic-fixed"),
                ("magnetic (fractional occupations)", "magnetic"),
            ],
        )
        mc_form.addRow("Select the file format:", self.mcFileFormatCombo)
        mc_form.addRow("Select the protocol:", self.mcProtocolCombo)
        mc_form.addRow("Select the XC functional:", self.mcXcCombo)
        mc_form.addRow("Select the magnetism/smearing:", self.mcMagnetizationCombo)
        mc_layout.addLayout(mc_form)

        advanced_group = QGroupBox("Advanced settings", self.materialsCloudOptionsGroup)
        advanced_form = QFormLayout(advanced_group)
        self.mcPseudoOverrideCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.mcPseudoOverrideCombo,
            [
                ("Protocol default", ""),
                ("SSSP 1.3: Efficiency", "efficiency"),
                ("SSSP 1.3: Precision", "precision"),
            ],
        )
        self.mcThresholdOverrideCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.mcThresholdOverrideCombo,
            [
                ("Protocol default", ""),
                ("loose", "loose"),
                ("intermediate", "intermediate"),
                ("tight", "tight"),
            ],
        )
        advanced_form.addRow("Override the pseudopotential:", self.mcPseudoOverrideCombo)
        advanced_form.addRow("Override the energy/forces thresholds:", self.mcThresholdOverrideCombo)
        mc_layout.addWidget(advanced_group)

        self.materialsCloudOptionsGroup.setVisible(False)
        config_layout.addWidget(self.materialsCloudOptionsGroup)

        action_group = QGroupBox("Generation", config_panel)
        action_layout = QVBoxLayout(action_group)
        self.generatePWscfButton = QPushButton("Generate PWscf file", action_group)
        self.qeInputGeneratorStatusLabel = QLabel("Ready", action_group)
        self.qeInputGeneratorStatusLabel.setWordWrap(True)
        action_layout.addWidget(self.generatePWscfButton)
        action_layout.addWidget(self.qeInputGeneratorStatusLabel)
        config_layout.addWidget(action_group)
        config_layout.addStretch()

        preview_panel = QWidget(splitter)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        preview_toolbar = QHBoxLayout()
        preview_toolbar.addWidget(QLabel("Preview", preview_panel))
        preview_toolbar.addStretch()
        self.qeInputGeneratorSavePreviewButton = QPushButton("Save", preview_panel)
        preview_toolbar.addWidget(self.qeInputGeneratorSavePreviewButton)
        preview_layout.addLayout(preview_toolbar)

        self.qeInputGeneratorPreviewEdit = QPlainTextEdit(preview_panel)
        self.qeInputGeneratorPreviewEdit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.qeInputGeneratorPreviewEdit.setPlaceholderText("Generated PWscf input will appear here.")
        preview_layout.addWidget(self.qeInputGeneratorPreviewEdit)

        config_scroll.setWidget(config_panel)
        splitter.addWidget(config_scroll)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        generator_layout.addWidget(splitter)
        self.tabWidget.addTab(generator_tab, "QE input generator")

    @staticmethod
    def _set_combo_items(combo: QComboBox, items: Sequence[Tuple[str, str]]) -> None:
        combo.clear()
        for label, value in items:
            combo.addItem(label, value)

    def _build_job_creator_tab(self) -> None:
        job_tab = QWidget()
        job_layout = QVBoxLayout(job_tab)
        job_layout.setContentsMargins(0, 0, 0, 0)
        job_layout.setSpacing(12)

        job_splitter = QSplitter(Qt.Orientation.Horizontal, job_tab)

        config_panel = QWidget(job_splitter)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(12)

        job_info_group = QGroupBox("Job details", config_panel)
        job_info_layout = QFormLayout(job_info_group)
        self.jobNameLineEdit = QLineEdit(job_info_group)
        self.jobNameLineEdit.setPlaceholderText("e.g. self-consistent run")
        job_info_layout.addRow("Job name:", self.jobNameLineEdit)

        self.calculation_label = QLabel("Calculation:", job_info_group)
        self.jobCalcTypeComboBox = QComboBox(job_info_group)
        for option in [
            "scf",
            "nscf",
            "vc-relax",
            "band",  # alias to "bands"
            "bandx",
            "phonon",
            "matdyn.x",
            "pdos",
            "optics",  # alias to epsilon.x
            "custom",
        ]:
            self.jobCalcTypeComboBox.addItem(option)
        self.comboBox = self.jobCalcTypeComboBox
        job_info_layout.addRow(self.calculation_label, self.jobCalcTypeComboBox)

        self.jobQueueLabel = QLabel("Queue / profile:", job_info_group)
        self.jobQueueComboBox = QComboBox(job_info_group)
        self.jobQueueComboBox.addItems(["default", "short", "long"])
        job_info_layout.addRow(self.jobQueueLabel, self.jobQueueComboBox)

        config_layout.addWidget(job_info_group)

        paths_group = QGroupBox("Paths", config_panel)
        paths_layout = QGridLayout(paths_group)

        self.working_directory_job_creator_label = QLabel("Working directory:", paths_group)
        working_dir_layout = QHBoxLayout()
        self.outputDirectoryLineEdit = QLineEdit(paths_group)
        self.outputDirectoryLineEdit.setPlaceholderText("Where to write generated inputs")
        self.outputDirectoryLineEdit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.outputDirectoryBrowseButton = QPushButton("Browse...", paths_group)
        working_dir_layout.addWidget(self.outputDirectoryLineEdit)
        working_dir_layout.addWidget(self.outputDirectoryBrowseButton)
        paths_layout.addWidget(self.working_directory_job_creator_label, 0, 0)
        paths_layout.addLayout(working_dir_layout, 0, 1)

        config_layout.addWidget(paths_group)

        pseudo_group = QGroupBox("Pseudopotentials", config_panel)
        self.pseudoGroupBox = pseudo_group
        pseudo_layout = QVBoxLayout(pseudo_group)
        pseudo_layout.setSpacing(6)

        pseudo_button_row = QHBoxLayout()
        self.pseudoSelectButton = QPushButton("Select pseudopotentials...", pseudo_group)
        self.pseudoClearButton = QPushButton("Clear", pseudo_group)
        pseudo_button_row.addWidget(self.pseudoSelectButton)
        pseudo_button_row.addWidget(self.pseudoClearButton)
        pseudo_button_row.addStretch()
        pseudo_layout.addLayout(pseudo_button_row)

        self.pseudoFilesListWidget = QListWidget(pseudo_group)
        self.pseudoFilesListWidget.setAlternatingRowColors(True)
        self.pseudoFilesListWidget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pseudoFilesListWidget.setMinimumHeight(80)
        pseudo_layout.addWidget(self.pseudoFilesListWidget)

        self.pseudoSummaryLabel = QLabel("No pseudopotentials selected.", pseudo_group)
        self.pseudoSummaryLabel.setWordWrap(True)
        pseudo_layout.addWidget(self.pseudoSummaryLabel)

        self._pseudo_group_tooltip = pseudo_group.toolTip()
        config_layout.addWidget(pseudo_group)
        self._refresh_pseudo_selection_view()

        input_preview_group = QGroupBox("Generated files and commands", config_panel)
        input_preview_layout = QVBoxLayout(input_preview_group)

        self.generatedFilesListWidget = QListWidget(input_preview_group)
        self.generatedFilesListWidget.setAlternatingRowColors(True)
        self.generatedFilesListWidget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.generatedFilesListWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self.commandPreviewEdit = QPlainTextEdit(input_preview_group)
        self.commandPreviewEdit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.commandPreviewEdit.setPlaceholderText("Required commands will appear here once generated.")
        self.commandPreviewEdit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.commandPreviewEdit.textChanged.connect(self._handle_command_preview_text_changed)

        input_preview_splitter = QSplitter(Qt.Orientation.Vertical, input_preview_group)
        input_preview_splitter.addWidget(self.generatedFilesListWidget)
        input_preview_splitter.addWidget(self.commandPreviewEdit)
        input_preview_splitter.setStretchFactor(0, 3)
        input_preview_splitter.setStretchFactor(1, 2)
        input_preview_splitter.setSizes([220, 160])

        input_preview_layout.addWidget(input_preview_splitter)

        config_layout.addWidget(input_preview_group)

        config_layout.addStretch()

        right_panel = QWidget(job_splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        source_group = QGroupBox("Source input file", right_panel)
        source_layout = QHBoxLayout(source_group)
        self.jobInputPathLineEdit = QLineEdit(source_group)
        self.jobInputPathLineEdit.setPlaceholderText("Select a QE input file to base the job on")
        self.jobInputBrowseButton = QPushButton("Browse...", source_group)
        source_layout.addWidget(self.jobInputPathLineEdit)
        source_layout.addWidget(self.jobInputBrowseButton)
        right_layout.addWidget(source_group)

        generated_preview_group = QGroupBox("Generated input preview", right_panel)
        generated_preview_layout = QVBoxLayout(generated_preview_group)

        self.generatedFilePreviewEdit = QPlainTextEdit(generated_preview_group)
        self.generatedFilePreviewEdit.setReadOnly(False)
        self.generatedFilePreviewEdit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.generatedFilePreviewEdit.setPlaceholderText(
            "Generated input files for the calculation will appear here. You can edit this text before saving."
        )
        self.generatedFilePreviewEdit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        generated_preview_layout.addWidget(self.generatedFilePreviewEdit)

        generated_preview_buttons = QHBoxLayout()
        generated_preview_buttons.addStretch()
        self.saveGeneratedPreviewButton = QPushButton("Save preview...", generated_preview_group)
        generated_preview_buttons.addWidget(self.saveGeneratedPreviewButton)
        generated_preview_layout.addLayout(generated_preview_buttons)
        right_layout.addWidget(generated_preview_group)

        job_action_layout = QHBoxLayout()
        self.jobsInputFilesGenerateButton = QPushButton("Generate input", job_tab)
        self.jobsInputFilesSaveButton = QPushButton("Save", job_tab)
        self.addJobsToQueueButton = QPushButton("Add to queue", job_tab)
        job_action_layout.addWidget(self.jobsInputFilesGenerateButton)
        job_action_layout.addWidget(self.jobsInputFilesSaveButton)
        job_action_layout.addStretch()
        job_action_layout.addWidget(self.addJobsToQueueButton)
        right_layout.addLayout(job_action_layout)

        self.jobStatusLabel = QLabel("Awaiting configuration", right_panel)
        self.jobStatusLabel.setWordWrap(True)
        right_layout.addWidget(self.jobStatusLabel)

        job_splitter.addWidget(config_panel)
        job_splitter.addWidget(right_panel)
        job_splitter.setStretchFactor(0, 2)
        job_splitter.setStretchFactor(1, 3)

        job_layout.addWidget(job_splitter)

        self.tabWidget.addTab(job_tab, "Job Creator")

    def _build_workflow_tab(self) -> None:
        workflow_tab = QWidget()
        workflow_layout = QVBoxLayout(workflow_tab)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(12)

        scroll_area = QScrollArea(workflow_tab)
        scroll_area.setWidgetResizable(True)
        workflow_layout.addWidget(scroll_area)

        content = QWidget(scroll_area)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        workflow_template_row = QHBoxLayout()
        self.workflowLoadButton = QPushButton("Load workflow", content)
        self.workflowSaveButton = QPushButton("Save workflow", content)
        workflow_template_row.addStretch()
        workflow_template_row.addWidget(self.workflowLoadButton)
        workflow_template_row.addWidget(self.workflowSaveButton)
        content_layout.addLayout(workflow_template_row)

        structure_group = QGroupBox("Step 1: Select structure", content)
        structure_layout = QGridLayout(structure_group)
        self.workflowStructurePathLineEdit = QLineEdit(structure_group)
        self.workflowStructurePathLineEdit.setPlaceholderText("Select a CIF or pw.x input file")
        self.workflowStructureBrowseButton = QPushButton("Browse...", structure_group)
        self.workflowSourceInputPath = ""
        self.workflowStructureStatusLabel = QLabel("No structure selected.", structure_group)
        self.workflowStructureStatusLabel.setWordWrap(True)
        structure_layout.addWidget(QLabel("Structure file:", structure_group), 0, 0)
        structure_layout.addWidget(self.workflowStructurePathLineEdit, 0, 1)
        structure_layout.addWidget(self.workflowStructureBrowseButton, 0, 2)
        structure_layout.addWidget(self.workflowStructureStatusLabel, 1, 0, 1, 3)
        content_layout.addWidget(structure_group)

        relaxation_group = QGroupBox("Step 2: Structure relaxation", content)
        relaxation_layout = QHBoxLayout(relaxation_group)
        self.workflowRelaxationButtonGroup = QButtonGroup(relaxation_group)
        self.workflowRelaxationButtonGroup.setExclusive(True)
        self.workflowRelaxationButtons: Dict[str, QPushButton] = {}
        for index, (key, label) in enumerate(
            [
                ("as_is", "Structure as is"),
                ("atomic_positions", "Atomic positions"),
                ("full_geometry", "Full geometry"),
            ]
        ):
            button = QPushButton(label, relaxation_group)
            button.setObjectName("workflowSegmentButton")
            button.setCheckable(True)
            if index == 0:
                button.setChecked(True)
            button.setMinimumHeight(36)
            self.workflowRelaxationButtonGroup.addButton(button, index)
            self.workflowRelaxationButtons[key] = button
            relaxation_layout.addWidget(button)
        content_layout.addWidget(relaxation_group)

        selection_group = QGroupBox("Step 3: Configure workflow", content)
        selection_layout = QVBoxLayout(selection_group)
        self.workflowSelectionCheckboxes: Dict[str, QCheckBox] = {}
        workflow_choices = [
            ("convergence", "Convergence (k-points and ecut)"),
            ("bands", "Band structure"),
            ("pdos", "Projected density of states (PDOS)"),
            ("fermi_surface", "Fermi surface"),
            ("charge_density", "Charge density"),
            ("elastic", "Elastic properties"),
            ("phonon", "Phonon"),
            ("wannier", "Wannier90 + SCDM"),
        ]
        for key, label in workflow_choices:
            checkbox = QCheckBox(label, selection_group)
            checkbox.setMinimumHeight(30)
            self.workflowSelectionCheckboxes[key] = checkbox
            selection_layout.addWidget(checkbox)
        content_layout.addWidget(selection_group)

        options_group = QGroupBox("Step 4: Customize calculation parameters", content)
        options_layout = QVBoxLayout(options_group)
        self.workflowOptionsTabWidget = NoWheelTabWidget(options_group)
        self.workflowOptionsTabWidget.setDocumentMode(True)
        self.workflowOptionsTabWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        self.workflowOptionsTabWidget.currentChanged.connect(
            lambda _index: self._workflow_fit_options_tab_height(),
        )
        self.workflowConversionOptionsTab = self._build_workflow_conversion_tab()
        self.workflowPseudosOptionsTab = self._build_workflow_pseudos_tab()
        self.workflowConvergenceOptionsTab = self._build_workflow_convergence_tab()
        self.workflowBandsOptionsTab = self._build_workflow_bands_tab()
        self.workflowDosOptionsTab = self._build_workflow_dos_tab()
        self.workflowFermiSurfaceOptionsTab = self._build_workflow_fermi_surface_tab()
        self.workflowChargeDensityOptionsTab = self._build_workflow_charge_density_tab()
        self.workflowElasticOptionsTab = self._build_workflow_elastic_tab()
        self.workflowPhononOptionsTab = self._build_workflow_phonon_tab()
        self.workflowWannierOptionsTab = self._build_workflow_wannier_tab()
        options_layout.addWidget(self.workflowOptionsTabWidget)
        content_layout.addWidget(options_group)
        self._workflow_refresh_option_tabs()

        submit_group = QGroupBox("Step 5: Submit workflow", content)
        submit_layout = QVBoxLayout(submit_group)
        submit_form = QFormLayout()
        self.workflowJobNameLineEdit = QLineEdit(submit_group)
        self.workflowJobNameLineEdit.setPlaceholderText("Workflow job name")
        self.workflowOutputDirectoryLineEdit = QLineEdit(submit_group)
        self.workflowOutputDirectoryLineEdit.setPlaceholderText("Where to write workflow outputs")
        self.workflowOutputDirectoryBrowseButton = QPushButton("Browse...", submit_group)
        workflow_output_directory_row = QWidget(submit_group)
        workflow_output_directory_layout = QHBoxLayout(workflow_output_directory_row)
        workflow_output_directory_layout.setContentsMargins(0, 0, 0, 0)
        workflow_output_directory_layout.addWidget(self.workflowOutputDirectoryLineEdit)
        workflow_output_directory_layout.addWidget(self.workflowOutputDirectoryBrowseButton)
        self.workflowDescriptionEdit = QPlainTextEdit(submit_group)
        self.workflowDescriptionEdit.setPlaceholderText("Optional description")
        self.workflowDescriptionEdit.setMaximumHeight(90)
        submit_form.addRow("Name:", self.workflowJobNameLineEdit)
        submit_form.addRow("Output directory:", workflow_output_directory_row)
        submit_form.addRow("Description:", self.workflowDescriptionEdit)
        submit_layout.addLayout(submit_form)
        self.workflowJobNameLineEdit.textEdited.connect(
            lambda _text: self._workflow_update_charge_density_fileout(),
        )

        submit_row = QHBoxLayout()
        self.workflowSubmitStatusLabel = QLabel("Workflow will be written and queued for daemon execution.", submit_group)
        self.workflowSubmitStatusLabel.setWordWrap(True)
        self.workflowSubmitButton = QPushButton("Submit to daemon", submit_group)
        submit_row.addWidget(self.workflowSubmitStatusLabel)
        submit_row.addStretch()
        submit_row.addWidget(self.workflowSubmitButton)
        submit_layout.addLayout(submit_row)
        content_layout.addWidget(submit_group)

        content_layout.addStretch()
        scroll_area.setWidget(content)

        self.tabWidget.addTab(workflow_tab, "Workflow Creator")

    def _workflow_refresh_option_tabs(self) -> None:
        if not hasattr(self, "workflowOptionsTabWidget"):
            return

        self._workflow_refresh_dos_controls()
        self._workflow_refresh_elastic_controls()
        self._workflow_refresh_phonon_controls()
        self._workflow_refresh_pseudos_tab()

        selected = [
            key
            for key, checkbox in getattr(self, "workflowSelectionCheckboxes", {}).items()
            if checkbox.isChecked()
        ]

        self.workflowOptionsTabWidget.clear()
        has_structure = bool(str(getattr(self, "workflowSourceInputPath", "")).strip())
        if has_structure:
            self.workflowOptionsTabWidget.addTab(self.workflowConversionOptionsTab, "Conversion")
            if self._workflow_pseudos_tab_needed():
                self.workflowOptionsTabWidget.addTab(self.workflowPseudosOptionsTab, "Pseudos")
        if not selected:
            if not has_structure:
                self.workflowOptionsTabWidget.addTab(
                    self._build_workflow_options_empty_panel(),
                    "Options",
                )
            self._workflow_fit_options_tab_height()
            return

        labels = {
            "convergence": "Convergence",
            "bands": "Band",
            "pdos": "PDOS",
            "fermi_surface": "Fermi Surface",
            "charge_density": "Charge Density",
            "elastic": "Elastic",
            "phonon": "Phonon",
            "wannier": "Wannier90 + SCDM",
        }
        for key in selected:
            if key == "convergence":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowConvergenceOptionsTab,
                    labels[key],
                )
                self._workflow_update_kpoint_mesh_preview()
                continue
            if key == "bands":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowBandsOptionsTab,
                    labels[key],
                )
                self._workflow_refresh_bands_tab()
                continue
            if key == "pdos":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowDosOptionsTab,
                    labels[key],
                )
                continue
            if key == "fermi_surface":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowFermiSurfaceOptionsTab,
                    labels[key],
                )
                self._workflow_update_fermi_surface_kmesh_preview()
                continue
            if key == "charge_density":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowChargeDensityOptionsTab,
                    labels[key],
                )
                self._workflow_update_charge_density_fileout()
                self._workflow_update_charge_density_kmesh_preview()
                continue
            if key == "elastic":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowElasticOptionsTab,
                    labels[key],
                )
                self._workflow_refresh_elastic_controls()
                continue
            if key == "phonon":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowPhononOptionsTab,
                    labels[key],
                )
                self._workflow_refresh_phonon_path_options()
                self._workflow_refresh_phonon_controls()
                continue
            if key == "wannier":
                self.workflowOptionsTabWidget.addTab(
                    self.workflowWannierOptionsTab,
                    labels[key],
                )
                self._workflow_refresh_wannier_tab()
                continue
            self.workflowOptionsTabWidget.addTab(
                self._build_workflow_option_placeholder(key),
                labels.get(key, key.title()),
            )
        self._workflow_fit_options_tab_height()

    def _workflow_fit_options_tab_height(self) -> None:
        if not hasattr(self, "workflowOptionsTabWidget"):
            return
        current = self.workflowOptionsTabWidget.currentWidget()
        if current is None:
            return
        tab_bar = self.workflowOptionsTabWidget.tabBar()
        tab_bar_height = tab_bar.sizeHint().height() if tab_bar is not None else 0
        page_height = current.sizeHint().height()
        target_height = page_height + tab_bar_height + 12
        self.workflowOptionsTabWidget.setMinimumHeight(target_height)
        self.workflowOptionsTabWidget.setMaximumHeight(target_height)
        self.workflowOptionsTabWidget.updateGeometry()

    def _build_workflow_options_empty_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        label = QLabel("Select a structure file to configure conversion options.", panel)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
        return panel

    def _build_workflow_option_placeholder(self, key: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        label_map = {
            "convergence": "K-point and cutoff convergence settings will be added here.",
            "bands": "Band structure settings will be added here.",
            "pdos": "Projected density of states settings will be added here.",
            "phonon": "Phonon settings will be added here.",
            "wannier": "Wannier90 + SCDM settings will be added here.",
        }
        label = QLabel(label_map.get(key, "Workflow settings will be added here."), panel)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
        return panel

    def _build_workflow_convergence_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        routine_group = QGroupBox("Convergence routines", panel)
        routine_layout = QVBoxLayout(routine_group)
        self.workflowCutoffConvergenceCheckbox = QCheckBox("Cutoff energy convergence", routine_group)
        self.workflowKpointConvergenceCheckbox = QCheckBox("K-points convergence", routine_group)
        self.workflowCutoffConvergenceCheckbox.setChecked(True)
        routine_layout.addWidget(self.workflowCutoffConvergenceCheckbox)
        routine_layout.addWidget(self.workflowKpointConvergenceCheckbox)
        layout.addWidget(routine_group)

        cutoff_group = QGroupBox("Cutoff energy convergence", panel)
        self.workflowCutoffConvergenceGroup = cutoff_group
        cutoff_form = QFormLayout(cutoff_group)

        self.workflowEcutStepSpin = QSpinBox(cutoff_group)
        self.workflowEcutStepSpin.setRange(1, 1000)
        self.workflowEcutStepSpin.setValue(10)
        self.workflowEcutStepSpin.setSuffix(" Ry")

        self.workflowEcutMinSpin = QSpinBox(cutoff_group)
        self.workflowEcutMinSpin.setRange(1, 10000)
        self.workflowEcutMinSpin.setValue(35)
        self.workflowEcutMinSpin.setSuffix(" Ry")

        self.workflowEcutMaxSpin = QSpinBox(cutoff_group)
        self.workflowEcutMaxSpin.setRange(1, 10000)
        self.workflowEcutMaxSpin.setValue(120)
        self.workflowEcutMaxSpin.setSuffix(" Ry")

        self.workflowEcutEnergyDeltaSpin = TrimmedDoubleSpinBox(cutoff_group)
        self.workflowEcutEnergyDeltaSpin.setRange(0.0, 100.0 * RY_TO_EV)
        self.workflowEcutEnergyDeltaSpin.setDecimals(8)
        self.workflowEcutEnergyDeltaSpin.setSingleStep(0.001)
        self.workflowEcutEnergyDeltaSpin.setValue(0.001)
        self.workflowEcutEnergyDeltaSpin.setSuffix(" eV")
        cutoff_delta_row = QWidget(cutoff_group)
        cutoff_delta_layout = QHBoxLayout(cutoff_delta_row)
        cutoff_delta_layout.setContentsMargins(0, 0, 0, 0)
        cutoff_delta_layout.setSpacing(10)
        self.workflowEcutEnergyDeltaRyLabel = QLabel(cutoff_delta_row)
        cutoff_delta_layout.addWidget(self.workflowEcutEnergyDeltaSpin)
        cutoff_delta_layout.addWidget(self.workflowEcutEnergyDeltaRyLabel)
        cutoff_delta_layout.addStretch()

        cutoff_form.addRow("ecutwfc step size:", self.workflowEcutStepSpin)
        cutoff_form.addRow("Minimum ecutwfc:", self.workflowEcutMinSpin)
        cutoff_form.addRow("Maximum ecutwfc:", self.workflowEcutMaxSpin)
        cutoff_form.addRow("Total energy delta:", cutoff_delta_row)
        layout.addWidget(cutoff_group)

        kpoint_group = QGroupBox("K-points convergence", panel)
        self.workflowKpointConvergenceGroup = kpoint_group
        kpoint_form = QFormLayout(kpoint_group)

        self.workflowKpointStartSpacingSpin = TrimmedDoubleSpinBox(kpoint_group)
        self.workflowKpointStartSpacingSpin.setRange(0.001, 10.0)
        self.workflowKpointStartSpacingSpin.setDecimals(4)
        self.workflowKpointStartSpacingSpin.setSingleStep(0.01)
        self.workflowKpointStartSpacingSpin.setValue(0.30)
        self.workflowKpointStartSpacingSpin.setSuffix(" 1/Angstrom")

        spacing_row = QWidget(kpoint_group)
        spacing_layout = QHBoxLayout(spacing_row)
        spacing_layout.setContentsMargins(0, 0, 0, 0)
        spacing_layout.setSpacing(10)
        self.workflowKpointMeshPreviewLabel = QLabel("Mesh: select a structure file", spacing_row)
        self.workflowKpointMeshPreviewLabel.setWordWrap(True)
        self.workflowKpointMeshPreviewLabel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        spacing_layout.addWidget(self.workflowKpointStartSpacingSpin)
        spacing_layout.addWidget(self.workflowKpointMeshPreviewLabel, 1)

        self.workflowKpointSpacingStepSpin = TrimmedDoubleSpinBox(kpoint_group)
        self.workflowKpointSpacingStepSpin.setRange(0.001, 10.0)
        self.workflowKpointSpacingStepSpin.setDecimals(4)
        self.workflowKpointSpacingStepSpin.setSingleStep(0.005)
        self.workflowKpointSpacingStepSpin.setValue(0.025)
        self.workflowKpointSpacingStepSpin.setSuffix(" 1/Angstrom")

        self.workflowKpointMaxMeshSpin = QSpinBox(kpoint_group)
        self.workflowKpointMaxMeshSpin.setRange(1, 1000000000)
        self.workflowKpointMaxMeshSpin.setValue(14000)

        self.workflowKpointEnergyDeltaSpin = TrimmedDoubleSpinBox(kpoint_group)
        self.workflowKpointEnergyDeltaSpin.setRange(0.0, 100.0 * RY_TO_EV)
        self.workflowKpointEnergyDeltaSpin.setDecimals(8)
        self.workflowKpointEnergyDeltaSpin.setSingleStep(0.001)
        self.workflowKpointEnergyDeltaSpin.setValue(0.001)
        self.workflowKpointEnergyDeltaSpin.setSuffix(" eV")
        kpoint_delta_row = QWidget(kpoint_group)
        kpoint_delta_layout = QHBoxLayout(kpoint_delta_row)
        kpoint_delta_layout.setContentsMargins(0, 0, 0, 0)
        kpoint_delta_layout.setSpacing(10)
        self.workflowKpointEnergyDeltaRyLabel = QLabel(kpoint_delta_row)
        kpoint_delta_layout.addWidget(self.workflowKpointEnergyDeltaSpin)
        kpoint_delta_layout.addWidget(self.workflowKpointEnergyDeltaRyLabel)
        kpoint_delta_layout.addStretch()

        kpoint_form.addRow("Starting/max k-point spacing:", spacing_row)
        kpoint_form.addRow("Spacing reduction step:", self.workflowKpointSpacingStepSpin)
        kpoint_form.addRow("Max k-point mesh size:", self.workflowKpointMaxMeshSpin)
        kpoint_form.addRow("Total energy delta:", kpoint_delta_row)
        layout.addWidget(kpoint_group)

        self.workflowCutoffConvergenceCheckbox.toggled.connect(
            self._workflow_refresh_convergence_sections,
        )
        self.workflowKpointConvergenceCheckbox.toggled.connect(
            self._workflow_refresh_convergence_sections,
        )
        self.workflowKpointStartSpacingSpin.valueChanged.connect(
            lambda _value: self._workflow_update_kpoint_mesh_preview(),
        )
        self.workflowKpointMaxMeshSpin.valueChanged.connect(
            lambda _value: self._workflow_update_kpoint_mesh_preview(),
        )
        self.workflowEcutEnergyDeltaSpin.valueChanged.connect(
            lambda _value: self._workflow_update_energy_delta_labels(),
        )
        self.workflowKpointEnergyDeltaSpin.valueChanged.connect(
            lambda _value: self._workflow_update_energy_delta_labels(),
        )

        self._workflow_update_energy_delta_labels()
        self._workflow_refresh_convergence_sections()
        layout.addStretch()
        return panel

    def _build_workflow_bands_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF for band workflow", panel)
        scf_form = QFormLayout(scf_group)
        self.workflowBandScfKpointSpacingSpin = TrimmedDoubleSpinBox(scf_group)
        self.workflowBandScfKpointSpacingSpin.setRange(0.001, 10.0)
        self.workflowBandScfKpointSpacingSpin.setDecimals(4)
        self.workflowBandScfKpointSpacingSpin.setSingleStep(0.01)
        self.workflowBandScfKpointSpacingSpin.setValue(0.10)
        self.workflowBandScfKpointSpacingSpin.setSuffix(" 1/Angstrom")
        self.workflowBandScfKpointSpacingSpin.setToolTip("Dense Monkhorst-Pack spacing used by the SCF stage.")

        nbnd_row = QWidget(scf_group)
        nbnd_row_layout = QHBoxLayout(nbnd_row)
        nbnd_row_layout.setContentsMargins(0, 0, 0, 0)
        nbnd_row_layout.setSpacing(8)
        self.workflowBandsNbndLineEdit = QLineEdit("64", nbnd_row)
        self.workflowBandsNbndLineEdit.setValidator(QIntValidator(1, 10000000, self))
        self.workflowBandsNbndLineEdit.setMaximumWidth(120)
        self.workflowBandsSuggestNbndButton = QPushButton("Suggest via dry run", nbnd_row)
        nbnd_row_layout.addWidget(self.workflowBandsNbndLineEdit)
        nbnd_row_layout.addWidget(self.workflowBandsSuggestNbndButton)
        nbnd_row_layout.addStretch()
        self.workflowBandsNbndHintLabel = QLabel("", scf_group)
        self.workflowBandsNbndHintLabel.setWordWrap(True)
        self.workflowBandsNbndHintLabel.setStyleSheet("color: #555;")

        scf_form.addRow("SCF k-point spacing:", self.workflowBandScfKpointSpacingSpin)
        scf_form.addRow("nbnd:", nbnd_row)
        scf_form.addRow("", self.workflowBandsNbndHintLabel)
        layout.addWidget(scf_group)

        path_group = QGroupBox("Band path (K_POINTS crystal_b)", panel)
        path_layout = QVBoxLayout(path_group)
        path_layout.setSpacing(10)

        path_splitter = QSplitter(Qt.Orientation.Horizontal, path_group)
        path_splitter.setChildrenCollapsible(False)

        suggestions_panel = QWidget(path_splitter)
        suggestions_layout = QVBoxLayout(suggestions_panel)
        suggestions_layout.setContentsMargins(0, 0, 0, 0)
        suggestions_layout.setSpacing(6)
        suggestions_title = QLabel("Suggested paths", suggestions_panel)
        suggestions_title.setStyleSheet("font-weight: 600;")
        self.workflowBandsPathInfoLabel = QLabel("Select a structure file to suggest high-symmetry paths.", suggestions_panel)
        self.workflowBandsPathInfoLabel.setWordWrap(True)
        self.workflowBandsPathInfoLabel.setStyleSheet("color: #555;")
        self.workflowBandsPathList = QListWidget(suggestions_panel)
        self.workflowBandsPathList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.workflowBandsPathList.setMinimumHeight(150)
        suggestions_toolbar = QHBoxLayout()
        self.workflowBandsRefreshPathsButton = self._workflow_path_tool_button(
            suggestions_panel,
            "Refresh suggested paths",
            "Refresh",
            QStyle.StandardPixmap.SP_BrowserReload,
        )
        self.workflowBandsLoadSuggestedPathButton = self._workflow_path_tool_button(
            suggestions_panel,
            "Load selected suggested path",
            "Load",
            QStyle.StandardPixmap.SP_ArrowRight,
        )
        suggestions_toolbar.addWidget(self.workflowBandsRefreshPathsButton)
        suggestions_toolbar.addWidget(self.workflowBandsLoadSuggestedPathButton)
        suggestions_toolbar.addStretch()
        suggestions_layout.addWidget(suggestions_title)
        suggestions_layout.addWidget(self.workflowBandsPathInfoLabel)
        suggestions_layout.addWidget(self.workflowBandsPathList, 1)
        suggestions_layout.addLayout(suggestions_toolbar)

        selected_panel = QWidget(path_splitter)
        selected_layout = QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.setSpacing(6)
        selected_header = QHBoxLayout()
        selected_title = QLabel("Selected path", selected_panel)
        selected_title.setStyleSheet("font-weight: 600;")
        selected_header.addWidget(selected_title)
        selected_header.addStretch()
        self.workflowBandsAddPathPointButton = self._workflow_text_path_tool_button(selected_panel, "+ Add", "Add custom point")
        self.workflowBandsRemovePathPointButton = self._workflow_path_tool_button(
            selected_panel,
            "Remove selected point",
            "Delete",
            QStyle.StandardPixmap.SP_DialogDiscardButton,
        )
        self.workflowBandsMovePathPointUpButton = self._workflow_path_tool_button(
            selected_panel,
            "Move selected point up",
            "Up",
            QStyle.StandardPixmap.SP_ArrowUp,
        )
        self.workflowBandsMovePathPointDownButton = self._workflow_path_tool_button(
            selected_panel,
            "Move selected point down",
            "Down",
            QStyle.StandardPixmap.SP_ArrowDown,
        )
        for button in (
            self.workflowBandsAddPathPointButton,
            self.workflowBandsRemovePathPointButton,
            self.workflowBandsMovePathPointUpButton,
            self.workflowBandsMovePathPointDownButton,
        ):
            selected_header.addWidget(button)
        self.workflowBandsSelectedPathLabel = QLabel("Selected path: -", selected_panel)
        self.workflowBandsSelectedPathLabel.setWordWrap(True)
        self.workflowBandsSelectedPathLabel.setStyleSheet("color: #555;")
        self.workflowBandsPathTable = ReorderablePathTableWidget(0, 4, selected_panel)
        self._workflow_configure_path_table(self.workflowBandsPathTable)
        self.workflowBandsPathTable.setMinimumHeight(150)
        selected_layout.addLayout(selected_header)
        selected_layout.addWidget(self.workflowBandsSelectedPathLabel)
        selected_layout.addWidget(self.workflowBandsPathTable, 1)

        path_splitter.addWidget(suggestions_panel)
        path_splitter.addWidget(selected_panel)
        path_splitter.setStretchFactor(0, 1)
        path_splitter.setStretchFactor(1, 2)
        path_layout.addWidget(path_splitter)

        preview_row = QHBoxLayout()
        preview_title = QLabel("Rendered preview", path_group)
        preview_title.setStyleSheet("font-weight: 600;")
        preview_row.addWidget(preview_title)
        preview_row.addStretch()
        preview_row.addWidget(QLabel("Weight:", path_group))
        self.workflowBandsPathWeightLineEdit = QLineEdit("50", path_group)
        self.workflowBandsPathWeightLineEdit.setValidator(QIntValidator(1, 1000000, self))
        self.workflowBandsPathWeightLineEdit.setMaximumWidth(120)
        preview_row.addWidget(self.workflowBandsPathWeightLineEdit)
        path_layout.addLayout(preview_row)

        self.workflowBandsCrystalBPreview = QPlainTextEdit(path_group)
        self.workflowBandsCrystalBPreview.setReadOnly(True)
        self.workflowBandsCrystalBPreview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.workflowBandsCrystalBPreview.setPlaceholderText("Selected high-symmetry points will appear here.")
        self.workflowBandsCrystalBPreview.setMaximumHeight(155)
        path_layout.addWidget(self.workflowBandsCrystalBPreview)
        layout.addWidget(path_group)

        post_group = QGroupBox("Post-processing", panel)
        post_form = QFormLayout(post_group)
        self.workflowCalculateFatBandCheckBox = QCheckBox(
            "Calculate fat band (run projwfc.x)",
            post_group,
        )
        self.workflowCalculateFatBandCheckBox.setChecked(False)
        post_form.addRow(self.workflowCalculateFatBandCheckBox)
        self.workflowBandsFilbandLineEdit = QLineEdit("bands.out", post_group)
        post_form.addRow("bands.x filband:", self.workflowBandsFilbandLineEdit)
        self.workflowFatBandColorLabel = QLabel("Fat-band coloring:", post_group)
        self.workflowFatBandColorCombo = QComboBox(post_group)
        self.workflowFatBandColorCombo.addItem("Dominant orbital", "dominant")
        self.workflowFatBandColorCombo.addItem("Blended colors", "blended")
        post_form.addRow(self.workflowFatBandColorLabel, self.workflowFatBandColorCombo)
        layout.addWidget(post_group)

        self.workflowBandsSuggestNbndButton.clicked.connect(
            lambda: self._workflow_suggest_nbnd_dry_run("bands"),
        )
        self.workflowBandsRefreshPathsButton.clicked.connect(
            lambda: self._workflow_refresh_band_path_options(force_reload=True),
        )
        self.workflowBandsPathList.itemSelectionChanged.connect(self._workflow_update_band_points_preview)
        self.workflowBandsLoadSuggestedPathButton.clicked.connect(
            lambda: self._workflow_load_selected_path_suggestion("bands"),
        )
        self.workflowBandsPathTable.itemChanged.connect(
            lambda _item: self._workflow_update_band_points_preview(),
        )
        self.workflowBandsPathTable.rowDropped.connect(
            lambda source, target: self._workflow_drag_move_path_table_point("bands", source, target),
        )
        self.workflowBandsAddPathPointButton.clicked.connect(
            lambda: self._workflow_add_path_table_point("bands"),
        )
        self.workflowBandsRemovePathPointButton.clicked.connect(
            lambda: self._workflow_remove_path_table_point("bands"),
        )
        self.workflowBandsMovePathPointUpButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("bands", -1),
        )
        self.workflowBandsMovePathPointDownButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("bands", 1),
        )
        self.workflowBandsPathWeightLineEdit.textChanged.connect(self._workflow_update_band_points_preview)
        self.workflowCalculateFatBandCheckBox.toggled.connect(
            lambda _checked: self._workflow_refresh_bands_mode_controls(),
        )

        self._workflow_refresh_bands_tab()
        layout.addStretch()
        return panel

    def _build_workflow_dos_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF step", panel)
        self.workflowPhononScfGroup = scf_group
        scf_form = QFormLayout(scf_group)
        self.workflowDosScfKpointSpacingSpin = TrimmedDoubleSpinBox(scf_group)
        self.workflowDosScfKpointSpacingSpin.setRange(0.001, 10.0)
        self.workflowDosScfKpointSpacingSpin.setDecimals(4)
        self.workflowDosScfKpointSpacingSpin.setSingleStep(0.01)
        self.workflowDosScfKpointSpacingSpin.setValue(0.30)
        self.workflowDosScfKpointSpacingSpin.setSuffix(" 1/Angstrom")
        self.workflowDosScfKpointSpacingSpin.setToolTip(
            "K-point spacing used by the dedicated DOS SCF stage."
        )
        self.workflowDosUseConvergedKmeshCheckBox = QCheckBox(
            "Use converged k-mesh from convergence workflow when available",
            scf_group,
        )
        self.workflowDosUseConvergedKmeshCheckBox.setChecked(True)
        scf_form.addRow("SCF k-point spacing:", self.workflowDosScfKpointSpacingSpin)
        scf_form.addRow("", self.workflowDosUseConvergedKmeshCheckBox)
        layout.addWidget(scf_group)

        nscf_group = QGroupBox("NSCF step", panel)
        nscf_form = QFormLayout(nscf_group)
        self.workflowDosNscfKpointSpacingSpin = TrimmedDoubleSpinBox(nscf_group)
        self.workflowDosNscfKpointSpacingSpin.setRange(0.001, 10.0)
        self.workflowDosNscfKpointSpacingSpin.setDecimals(4)
        self.workflowDosNscfKpointSpacingSpin.setSingleStep(0.01)
        self.workflowDosNscfKpointSpacingSpin.setValue(0.10)
        self.workflowDosNscfKpointSpacingSpin.setSuffix(" 1/Angstrom")
        self.workflowDosNscfKpointSpacingSpin.setToolTip(
            "Dense K-point spacing used by DOS NSCF stage."
        )
        nscf_form.addRow("NSCF k-point spacing:", self.workflowDosNscfKpointSpacingSpin)
        layout.addWidget(nscf_group)

        note = QLabel(
            "PDOS always runs its own SCF + NSCF sequence with occupations='tetrahedra_opt' (no smearing).",
            panel,
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        self._workflow_refresh_dos_controls()
        layout.addStretch()
        return panel

    def _build_workflow_wannier_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF and NSCF mesh", panel)
        scf_form = QFormLayout(scf_group)

        kmesh_row = QWidget(scf_group)
        kmesh_layout = QHBoxLayout(kmesh_row)
        kmesh_layout.setContentsMargins(0, 0, 0, 0)
        kmesh_layout.setSpacing(8)
        self.workflowWannierKmeshSpinBoxes: List[QSpinBox] = []
        for label, value in (("nk1", 8), ("nk2", 8), ("nk3", 6)):
            spin = QSpinBox(kmesh_row)
            spin.setRange(1, 1000000)
            spin.setValue(value)
            spin.setPrefix(f"{label} ")
            spin.setMaximumWidth(120)
            self.workflowWannierKmeshSpinBoxes.append(spin)
            kmesh_layout.addWidget(spin)
        self.workflowWannierKmeshPreviewLabel = QLabel("NSCF crystal k-points: 384", kmesh_row)
        self.workflowWannierKmeshPreviewLabel.setWordWrap(True)
        kmesh_layout.addWidget(self.workflowWannierKmeshPreviewLabel, 1)

        nbnd_row = QWidget(scf_group)
        nbnd_row_layout = QHBoxLayout(nbnd_row)
        nbnd_row_layout.setContentsMargins(0, 0, 0, 0)
        nbnd_row_layout.setSpacing(8)
        self.workflowWannierNbndLineEdit = QLineEdit("64", nbnd_row)
        self.workflowWannierNbndLineEdit.setValidator(QIntValidator(1, 10000000, self))
        self.workflowWannierNbndLineEdit.setMaximumWidth(120)
        self.workflowWannierSuggestNbndButton = QPushButton("Suggest via dry run", nbnd_row)
        nbnd_row_layout.addWidget(self.workflowWannierNbndLineEdit)
        nbnd_row_layout.addWidget(self.workflowWannierSuggestNbndButton)
        nbnd_row_layout.addStretch()
        self.workflowWannierNbndHintLabel = QLabel("", scf_group)
        self.workflowWannierNbndHintLabel.setWordWrap(True)
        self.workflowWannierNbndHintLabel.setStyleSheet("color: #555;")
        scf_form.addRow("K_POINTS automatic mesh:", kmesh_row)
        scf_form.addRow("nbnd:", nbnd_row)
        scf_form.addRow("", self.workflowWannierNbndHintLabel)
        layout.addWidget(scf_group)

        path_group = QGroupBox("Band path for pwbands.in", panel)
        path_layout = QVBoxLayout(path_group)
        path_layout.setSpacing(10)

        path_splitter = QSplitter(Qt.Orientation.Horizontal, path_group)
        path_splitter.setChildrenCollapsible(False)

        suggestions_panel = QWidget(path_splitter)
        suggestions_layout = QVBoxLayout(suggestions_panel)
        suggestions_layout.setContentsMargins(0, 0, 0, 0)
        suggestions_layout.setSpacing(6)
        suggestions_title = QLabel("Suggested paths", suggestions_panel)
        suggestions_title.setStyleSheet("font-weight: 600;")
        self.workflowWannierPathInfoLabel = QLabel("Select a structure file to suggest high-symmetry paths.", suggestions_panel)
        self.workflowWannierPathInfoLabel.setWordWrap(True)
        self.workflowWannierPathInfoLabel.setStyleSheet("color: #555;")
        self.workflowWannierPathList = QListWidget(suggestions_panel)
        self.workflowWannierPathList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.workflowWannierPathList.setMinimumHeight(130)
        suggestions_toolbar = QHBoxLayout()
        self.workflowWannierRefreshPathsButton = self._workflow_path_tool_button(
            suggestions_panel,
            "Refresh suggested paths",
            "Refresh",
            QStyle.StandardPixmap.SP_BrowserReload,
        )
        self.workflowWannierLoadSuggestedPathButton = self._workflow_path_tool_button(
            suggestions_panel,
            "Load selected suggested path",
            "Load",
            QStyle.StandardPixmap.SP_ArrowRight,
        )
        suggestions_toolbar.addWidget(self.workflowWannierRefreshPathsButton)
        suggestions_toolbar.addWidget(self.workflowWannierLoadSuggestedPathButton)
        suggestions_toolbar.addStretch()
        suggestions_layout.addWidget(suggestions_title)
        suggestions_layout.addWidget(self.workflowWannierPathInfoLabel)
        suggestions_layout.addWidget(self.workflowWannierPathList, 1)
        suggestions_layout.addLayout(suggestions_toolbar)

        selected_panel = QWidget(path_splitter)
        selected_layout = QVBoxLayout(selected_panel)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.setSpacing(6)
        selected_header = QHBoxLayout()
        selected_title = QLabel("Selected path", selected_panel)
        selected_title.setStyleSheet("font-weight: 600;")
        selected_header.addWidget(selected_title)
        selected_header.addStretch()
        self.workflowWannierAddPathPointButton = self._workflow_text_path_tool_button(selected_panel, "+ Add", "Add custom point")
        self.workflowWannierRemovePathPointButton = self._workflow_path_tool_button(
            selected_panel,
            "Remove selected point",
            "Delete",
            QStyle.StandardPixmap.SP_DialogDiscardButton,
        )
        self.workflowWannierMovePathPointUpButton = self._workflow_path_tool_button(
            selected_panel,
            "Move selected point up",
            "Up",
            QStyle.StandardPixmap.SP_ArrowUp,
        )
        self.workflowWannierMovePathPointDownButton = self._workflow_path_tool_button(
            selected_panel,
            "Move selected point down",
            "Down",
            QStyle.StandardPixmap.SP_ArrowDown,
        )
        for button in (
            self.workflowWannierAddPathPointButton,
            self.workflowWannierRemovePathPointButton,
            self.workflowWannierMovePathPointUpButton,
            self.workflowWannierMovePathPointDownButton,
        ):
            selected_header.addWidget(button)
        self.workflowWannierSelectedPathLabel = QLabel("Selected path: -", selected_panel)
        self.workflowWannierSelectedPathLabel.setWordWrap(True)
        self.workflowWannierSelectedPathLabel.setStyleSheet("color: #555;")
        self.workflowWannierPathTable = ReorderablePathTableWidget(0, 4, selected_panel)
        self._workflow_configure_path_table(self.workflowWannierPathTable)
        self.workflowWannierPathTable.setMinimumHeight(130)
        selected_layout.addLayout(selected_header)
        selected_layout.addWidget(self.workflowWannierSelectedPathLabel)
        selected_layout.addWidget(self.workflowWannierPathTable, 1)

        path_splitter.addWidget(suggestions_panel)
        path_splitter.addWidget(selected_panel)
        path_splitter.setStretchFactor(0, 1)
        path_splitter.setStretchFactor(1, 2)
        path_layout.addWidget(path_splitter)

        preview_row = QHBoxLayout()
        preview_title = QLabel("K_POINTS crystal_b preview", path_group)
        preview_title.setStyleSheet("font-weight: 600;")
        preview_row.addWidget(preview_title)
        preview_row.addStretch()
        preview_row.addWidget(QLabel("Weight:", path_group))
        self.workflowWannierPathWeightLineEdit = QLineEdit("50", path_group)
        self.workflowWannierPathWeightLineEdit.setValidator(QIntValidator(1, 1000000, self))
        self.workflowWannierPathWeightLineEdit.setMaximumWidth(120)
        preview_row.addWidget(self.workflowWannierPathWeightLineEdit)
        path_layout.addLayout(preview_row)

        self.workflowWannierCrystalBPreview = QPlainTextEdit(path_group)
        self.workflowWannierCrystalBPreview.setReadOnly(True)
        self.workflowWannierCrystalBPreview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.workflowWannierCrystalBPreview.setPlaceholderText("Selected high-symmetry points will appear here.")
        self.workflowWannierCrystalBPreview.setMaximumHeight(135)
        path_layout.addWidget(self.workflowWannierCrystalBPreview)
        layout.addWidget(path_group)

        control_group = QGroupBox("Wannier control", panel)
        control_form = QFormLayout(control_group)
        self.workflowWannierWriteUnkCheckBox = QCheckBox("write_unk = .true.", control_group)
        self.workflowWannierWriteUnkCheckBox.setChecked(False)
        self.workflowWannierNumIterSpin = QSpinBox(control_group)
        self.workflowWannierNumIterSpin.setRange(1, 1000000)
        self.workflowWannierNumIterSpin.setValue(500)
        self.workflowWannierNumIterSpin.setMaximumWidth(120)
        self.workflowWannierDisNumIterSpin = QSpinBox(control_group)
        self.workflowWannierDisNumIterSpin.setRange(1, 1000000)
        self.workflowWannierDisNumIterSpin.setValue(500)
        self.workflowWannierDisNumIterSpin.setMaximumWidth(120)
        control_form.addRow("", self.workflowWannierWriteUnkCheckBox)
        control_form.addRow("num_iter:", self.workflowWannierNumIterSpin)
        control_form.addRow("dis_num_iter:", self.workflowWannierDisNumIterSpin)
        layout.addWidget(control_group)

        for spin in self.workflowWannierKmeshSpinBoxes:
            spin.valueChanged.connect(lambda _value: self._workflow_update_wannier_kmesh_preview())
        self.workflowWannierSuggestNbndButton.clicked.connect(
            lambda: self._workflow_suggest_nbnd_dry_run("wannier"),
        )
        self.workflowWannierRefreshPathsButton.clicked.connect(
            lambda: self._workflow_refresh_wannier_path_options(force_reload=True),
        )
        self.workflowWannierPathList.itemSelectionChanged.connect(self._workflow_update_wannier_points_preview)
        self.workflowWannierLoadSuggestedPathButton.clicked.connect(
            lambda: self._workflow_load_selected_path_suggestion("wannier"),
        )
        self.workflowWannierPathTable.itemChanged.connect(
            lambda _item: self._workflow_update_wannier_points_preview(),
        )
        self.workflowWannierPathTable.rowDropped.connect(
            lambda source, target: self._workflow_drag_move_path_table_point("wannier", source, target),
        )
        self.workflowWannierAddPathPointButton.clicked.connect(
            lambda: self._workflow_add_path_table_point("wannier"),
        )
        self.workflowWannierRemovePathPointButton.clicked.connect(
            lambda: self._workflow_remove_path_table_point("wannier"),
        )
        self.workflowWannierMovePathPointUpButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("wannier", -1),
        )
        self.workflowWannierMovePathPointDownButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("wannier", 1),
        )
        self.workflowWannierPathWeightLineEdit.textChanged.connect(self._workflow_update_wannier_points_preview)

        self._workflow_refresh_wannier_tab()
        layout.addStretch()
        return panel

    def _build_workflow_fermi_surface_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF step", panel)
        scf_form = QFormLayout(scf_group)

        spacing_row = QWidget(scf_group)
        spacing_layout = QHBoxLayout(spacing_row)
        spacing_layout.setContentsMargins(0, 0, 0, 0)
        spacing_layout.setSpacing(10)

        self.workflowFermiSurfaceKpointSpacingSpin = TrimmedDoubleSpinBox(spacing_row)
        self.workflowFermiSurfaceKpointSpacingSpin.setRange(0.001, 10.0)
        self.workflowFermiSurfaceKpointSpacingSpin.setDecimals(4)
        self.workflowFermiSurfaceKpointSpacingSpin.setSingleStep(0.01)
        self.workflowFermiSurfaceKpointSpacingSpin.setValue(0.10)
        self.workflowFermiSurfaceKpointSpacingSpin.setSuffix(" 1/Angstrom")
        self.workflowFermiSurfaceKpointSpacingSpin.setToolTip(
            "Monkhorst-Pack spacing used to generate the SCF K_POINTS automatic mesh."
        )
        self.workflowFermiSurfaceKmeshPreviewLabel = QLabel("Mesh: select a structure file", spacing_row)
        self.workflowFermiSurfaceKmeshPreviewLabel.setWordWrap(True)
        self.workflowFermiSurfaceKmeshPreviewLabel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        spacing_layout.addWidget(self.workflowFermiSurfaceKpointSpacingSpin)
        spacing_layout.addWidget(self.workflowFermiSurfaceKmeshPreviewLabel, 1)
        scf_form.addRow("K-point spacing:", spacing_row)
        layout.addWidget(scf_group)

        self.workflowFermiSurfaceKpointSpacingSpin.valueChanged.connect(
            lambda _value: self._workflow_update_fermi_surface_kmesh_preview(),
        )
        self._workflow_update_fermi_surface_kmesh_preview()
        layout.addStretch()
        return panel

    def _build_workflow_charge_density_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF step", panel)
        scf_form = QFormLayout(scf_group)

        spacing_row = QWidget(scf_group)
        spacing_layout = QHBoxLayout(spacing_row)
        spacing_layout.setContentsMargins(0, 0, 0, 0)
        spacing_layout.setSpacing(10)

        self.workflowChargeDensityKpointSpacingSpin = TrimmedDoubleSpinBox(spacing_row)
        self.workflowChargeDensityKpointSpacingSpin.setRange(0.001, 10.0)
        self.workflowChargeDensityKpointSpacingSpin.setDecimals(4)
        self.workflowChargeDensityKpointSpacingSpin.setSingleStep(0.01)
        self.workflowChargeDensityKpointSpacingSpin.setValue(0.10)
        self.workflowChargeDensityKpointSpacingSpin.setSuffix(" 1/Angstrom")
        self.workflowChargeDensityKpointSpacingSpin.setToolTip(
            "Monkhorst-Pack spacing used to generate the SCF K_POINTS automatic mesh."
        )
        self.workflowChargeDensityKmeshPreviewLabel = QLabel("Mesh: select a structure file", spacing_row)
        self.workflowChargeDensityKmeshPreviewLabel.setWordWrap(True)
        self.workflowChargeDensityKmeshPreviewLabel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        spacing_layout.addWidget(self.workflowChargeDensityKpointSpacingSpin)
        spacing_layout.addWidget(self.workflowChargeDensityKmeshPreviewLabel, 1)
        scf_form.addRow("K-point spacing:", spacing_row)
        layout.addWidget(scf_group)

        pp_group = QGroupBox("pp.x output", panel)
        pp_form = QFormLayout(pp_group)
        self._workflowChargeDensityFileoutDirty = False
        self.workflowChargeDensityFileoutLineEdit = QLineEdit(self._workflow_charge_density_default_fileout(), pp_group)
        self.workflowChargeDensityFileoutLineEdit.setToolTip(
            "Cube file written by pp.x."
        )
        grid_row = QWidget(pp_group)
        grid_layout = QHBoxLayout(grid_row)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(8)
        self.workflowChargeDensityNxSpin = QSpinBox(grid_row)
        self.workflowChargeDensityNySpin = QSpinBox(grid_row)
        self.workflowChargeDensityNzSpin = QSpinBox(grid_row)
        for label, spin in (
            ("nx", self.workflowChargeDensityNxSpin),
            ("ny", self.workflowChargeDensityNySpin),
            ("nz", self.workflowChargeDensityNzSpin),
        ):
            spin.setRange(1, 1000000)
            spin.setValue(64)
            spin.setPrefix(f"{label} ")
            spin.setMaximumWidth(120)
            grid_layout.addWidget(spin)
        grid_layout.addStretch()
        pp_form.addRow("fileout:", self.workflowChargeDensityFileoutLineEdit)
        pp_form.addRow("Grid:", grid_row)
        layout.addWidget(pp_group)

        self.workflowChargeDensityKpointSpacingSpin.valueChanged.connect(
            lambda _value: self._workflow_update_charge_density_kmesh_preview(),
        )
        self.workflowChargeDensityFileoutLineEdit.textEdited.connect(
            lambda _text: setattr(self, "_workflowChargeDensityFileoutDirty", True),
        )
        self._workflow_update_charge_density_fileout()
        self._workflow_update_charge_density_kmesh_preview()
        layout.addStretch()
        return panel

    def _build_workflow_elastic_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        control_group = QGroupBox("thermo_control", panel)
        control_form = QFormLayout(control_group)

        self.workflowElasticWhatCombo = QComboBox(control_group)
        self.workflowElasticWhatCombo.addItem("scf_elastic_constants", "scf_elastic_constants")
        self.workflowElasticWhatCombo.addItem("mur_lc_elastic_constants", "mur_lc_elastic_constants")

        self.workflowElasticAlgorithmCombo = QComboBox(control_group)
        self.workflowElasticAlgorithmCombo.addItem("standard", "standard")
        self.workflowElasticAlgorithmCombo.addItem("advanced", "advanced")
        self.workflowElasticAlgorithmCombo.addItem("energy_std", "energy_std")
        self.workflowElasticAlgorithmCombo.addItem("energy", "energy")

        self._workflowElasticNgeoTouched = False
        self.workflowElasticNgeoStrainSpin = QSpinBox(control_group)
        self.workflowElasticNgeoStrainSpin.setRange(1, 1000)
        self.workflowElasticNgeoStrainSpin.setValue(4)
        self.workflowElasticNgeoStrainSpin.setMaximumWidth(120)

        self.workflowElasticFrozenIonsCheckBox = QCheckBox("frozen_ions = .true.", control_group)
        self.workflowElasticFrozenIonsCheckBox.setChecked(False)

        control_form.addRow("what:", self.workflowElasticWhatCombo)
        control_form.addRow("elastic_algorithm:", self.workflowElasticAlgorithmCombo)
        control_form.addRow("ngeo_strain:", self.workflowElasticNgeoStrainSpin)
        control_form.addRow("", self.workflowElasticFrozenIonsCheckBox)
        layout.addWidget(control_group)

        self.workflowElasticAlgorithmCombo.currentIndexChanged.connect(
            lambda _index: self._workflow_refresh_elastic_controls(),
        )
        self.workflowElasticNgeoStrainSpin.valueChanged.connect(
            lambda _value: setattr(self, "_workflowElasticNgeoTouched", True),
        )

        self._workflow_refresh_elastic_controls()
        layout.addStretch()
        return panel

    def _build_workflow_phonon_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        scf_group = QGroupBox("SCF step", panel)
        scf_layout = QVBoxLayout(scf_group)
        scf_form = QFormLayout()
        scf_mesh_row = QWidget(scf_group)
        scf_mesh_layout = QHBoxLayout(scf_mesh_row)
        scf_mesh_layout.setContentsMargins(0, 0, 0, 0)
        scf_mesh_layout.setSpacing(8)
        self.workflowPhononScfKmeshSpinBoxes: List[QSpinBox] = []
        for value in (16, 16, 16):
            spin = QSpinBox(scf_mesh_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononScfKmeshSpinBoxes.append(spin)
            scf_mesh_layout.addWidget(spin)
        scf_mesh_layout.addStretch()
        scf_form.addRow("Uniform SCF k-mesh:", scf_mesh_row)
        scf_layout.addLayout(scf_form)
        scf_note = QLabel(
            "The phonon SCF step writes K_POINTS automatic from this mesh instead of keeping the generated K_POINTS block.",
            scf_group,
        )
        scf_note.setWordWrap(True)
        scf_note.setStyleSheet("color: #555;")
        scf_layout.addWidget(scf_note)
        layout.addWidget(scf_group)

        ph_group = QGroupBox("ph.x step", panel)
        ph_form = QFormLayout(ph_group)
        self.workflowPhononModeCombo = QComboBox(ph_group)
        self.workflowPhononModeCombo.addItem("Dispersion grid (SCF -> ph.x -> q2r.x -> matdyn.x)", "dispersion")
        self.workflowPhononModeCombo.addItem("Single q-point (SCF -> ph.x)", "single_q")
        self.workflowPhononModeCombo.addItem("Electron-phonon coupling", "epc")
        self.workflowPhononEpcModeCombo = QComboBox(ph_group)
        self.workflowPhononEpcModeCombo.addItem("EPC: interpolated", "epc_interpolated")

        self.workflowPhononTr2PhLineEdit = QLineEdit("1.0d-12", ph_group)
        self.workflowPhononAlphaMixSpin = TrimmedDoubleSpinBox(ph_group)
        self.workflowPhononAlphaMixSpin.setRange(0.0, 1.0)
        self.workflowPhononAlphaMixSpin.setDecimals(4)
        self.workflowPhononAlphaMixSpin.setSingleStep(0.05)
        self.workflowPhononAlphaMixSpin.setValue(0.7)
        self.workflowPhononAlphaMixSpin.setToolTip("ph.x alpha_mix(1) mixing parameter.")
        self.workflowPhononFildynLineEdit = QLineEdit("phonon.dyn", ph_group)
        self.workflowPhononTransCheckBox = QCheckBox("Compute dynamical matrix (trans)", ph_group)
        self.workflowPhononTransCheckBox.setChecked(True)
        self.workflowPhononEpsilCheckBox = QCheckBox("Calculate dielectric tensor / LO-TO terms (epsil)", ph_group)
        self.workflowPhononEpsilCheckBox.setChecked(False)
        self.workflowPhononNmixSpin = QSpinBox(ph_group)
        self.workflowPhononNmixSpin.setRange(1, 1000)
        self.workflowPhononNmixSpin.setValue(4)
        self.workflowPhononVerbosityCombo = QComboBox(ph_group)
        self.workflowPhononVerbosityCombo.addItem("default", "default")
        self.workflowPhononVerbosityCombo.addItem("high", "high")
        self.workflowPhononVerbosityCombo.addItem("debug", "debug")

        qgrid_row = QWidget(ph_group)
        qgrid_layout = QHBoxLayout(qgrid_row)
        qgrid_layout.setContentsMargins(0, 0, 0, 0)
        qgrid_layout.setSpacing(8)
        self.workflowPhononQGridSpinBoxes: List[QSpinBox] = []
        for value in (3, 3, 3):
            spin = QSpinBox(qgrid_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononQGridSpinBoxes.append(spin)
            qgrid_layout.addWidget(spin)
        qgrid_layout.addStretch()
        self.workflowPhononQGridRow = qgrid_row

        self.workflowPhononQPointLineEdit = QLineEdit("0.0 0.0 0.0", ph_group)
        self.workflowPhononQPointLineEdit.setToolTip("Single-q ph.x vector, for example: 0.0 0.0 0.0")

        ph_form.addRow("Mode:", self.workflowPhononModeCombo)
        ph_form.addRow("EPC mode:", self.workflowPhononEpcModeCombo)
        self.workflowPhononEpcModeLabel = ph_form.labelForField(self.workflowPhononEpcModeCombo)
        ph_form.addRow("tr2_ph:", self.workflowPhononTr2PhLineEdit)
        ph_form.addRow("alpha_mix(1):", self.workflowPhononAlphaMixSpin)
        ph_form.addRow("fildyn:", self.workflowPhononFildynLineEdit)
        ph_form.addRow("Uniform q-grid nq1/nq2/nq3:", qgrid_row)
        self.workflowPhononQGridLabel = ph_form.labelForField(qgrid_row)
        ph_form.addRow("Single q-vector:", self.workflowPhononQPointLineEdit)
        self.workflowPhononQPointLabel = ph_form.labelForField(self.workflowPhononQPointLineEdit)
        ph_form.addRow("nmix_ph:", self.workflowPhononNmixSpin)
        ph_form.addRow("verbosity:", self.workflowPhononVerbosityCombo)
        ph_form.addRow("", self.workflowPhononTransCheckBox)
        ph_form.addRow("", self.workflowPhononEpsilCheckBox)
        layout.addWidget(ph_group)

        interpolation_group = QGroupBox("q2r.x and matdyn.x", panel)
        self.workflowPhononInterpolationGroup = interpolation_group
        interpolation_form = QFormLayout(interpolation_group)
        self.workflowPhononFlfrcLineEdit = QLineEdit("phonon.fc", interpolation_group)
        self.workflowPhononFlfrqLineEdit = QLineEdit("phonon.freq.band", interpolation_group)
        self.workflowPhononMatdynFlvecLineEdit = QLineEdit("phonon_band.modes", interpolation_group)
        self.workflowPhononQ2rZasrCombo = QComboBox(interpolation_group)
        self.workflowPhononMatdynAsrCombo = QComboBox(interpolation_group)
        for combo in (self.workflowPhononQ2rZasrCombo, self.workflowPhononMatdynAsrCombo):
            combo.addItem("crystal", "crystal")
            combo.addItem("simple", "simple")
            combo.addItem("one-dim", "one-dim")
            combo.addItem("zero-dim", "zero-dim")
            combo.addItem("no", "no")
        self.workflowPhononBandCheckBox = QCheckBox("Compute phonon band structure", interpolation_group)
        self.workflowPhononBandCheckBox.setChecked(True)
        self.workflowPhononMatdynQInBandFormCheckBox = QCheckBox(
            "Use q_in_band_form for band-path interpolation",
            interpolation_group,
        )
        self.workflowPhononMatdynQInBandFormCheckBox.setChecked(True)
        self.workflowPhononPathEditorGroup = QGroupBox("High-symmetry path editor", interpolation_group)
        phonon_path_layout = QVBoxLayout(self.workflowPhononPathEditorGroup)
        phonon_path_layout.setSpacing(10)

        phonon_path_splitter = QSplitter(Qt.Orientation.Horizontal, self.workflowPhononPathEditorGroup)
        phonon_path_splitter.setChildrenCollapsible(False)

        phonon_suggestions_panel = QWidget(phonon_path_splitter)
        phonon_suggestions_layout = QVBoxLayout(phonon_suggestions_panel)
        phonon_suggestions_layout.setContentsMargins(0, 0, 0, 0)
        phonon_suggestions_layout.setSpacing(6)
        phonon_suggestions_title = QLabel("Suggested paths", phonon_suggestions_panel)
        phonon_suggestions_title.setStyleSheet("font-weight: 600;")
        self.workflowPhononPathInfoLabel = QLabel("Select a structure file to suggest high-symmetry paths.", phonon_suggestions_panel)
        self.workflowPhononPathInfoLabel.setWordWrap(True)
        self.workflowPhononPathInfoLabel.setStyleSheet("color: #555;")
        self.workflowPhononPathList = QListWidget(phonon_suggestions_panel)
        self.workflowPhononPathList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.workflowPhononPathList.setMinimumHeight(130)
        phonon_suggestions_toolbar = QHBoxLayout()
        self.workflowPhononPathRefreshButton = self._workflow_path_tool_button(
            phonon_suggestions_panel,
            "Refresh suggested paths",
            "Refresh",
            QStyle.StandardPixmap.SP_BrowserReload,
        )
        self.workflowPhononLoadSuggestedPathButton = self._workflow_path_tool_button(
            phonon_suggestions_panel,
            "Load selected suggested path",
            "Load",
            QStyle.StandardPixmap.SP_ArrowRight,
        )
        phonon_suggestions_toolbar.addWidget(self.workflowPhononPathRefreshButton)
        phonon_suggestions_toolbar.addWidget(self.workflowPhononLoadSuggestedPathButton)
        phonon_suggestions_toolbar.addStretch()
        phonon_suggestions_layout.addWidget(phonon_suggestions_title)
        phonon_suggestions_layout.addWidget(self.workflowPhononPathInfoLabel)
        phonon_suggestions_layout.addWidget(self.workflowPhononPathList, 1)
        phonon_suggestions_layout.addLayout(phonon_suggestions_toolbar)

        phonon_selected_panel = QWidget(phonon_path_splitter)
        phonon_selected_layout = QVBoxLayout(phonon_selected_panel)
        phonon_selected_layout.setContentsMargins(0, 0, 0, 0)
        phonon_selected_layout.setSpacing(6)
        phonon_selected_header = QHBoxLayout()
        phonon_selected_title = QLabel("Selected path", phonon_selected_panel)
        phonon_selected_title.setStyleSheet("font-weight: 600;")
        phonon_selected_header.addWidget(phonon_selected_title)
        phonon_selected_header.addStretch()
        self.workflowPhononAddPathPointButton = self._workflow_text_path_tool_button(
            phonon_selected_panel,
            "+ Add",
            "Add custom point",
        )
        self.workflowPhononRemovePathPointButton = self._workflow_path_tool_button(
            phonon_selected_panel,
            "Remove selected point",
            "Delete",
            QStyle.StandardPixmap.SP_DialogDiscardButton,
        )
        self.workflowPhononMovePathPointUpButton = self._workflow_path_tool_button(
            phonon_selected_panel,
            "Move selected point up",
            "Up",
            QStyle.StandardPixmap.SP_ArrowUp,
        )
        self.workflowPhononMovePathPointDownButton = self._workflow_path_tool_button(
            phonon_selected_panel,
            "Move selected point down",
            "Down",
            QStyle.StandardPixmap.SP_ArrowDown,
        )
        for button in (
            self.workflowPhononAddPathPointButton,
            self.workflowPhononRemovePathPointButton,
            self.workflowPhononMovePathPointUpButton,
            self.workflowPhononMovePathPointDownButton,
        ):
            phonon_selected_header.addWidget(button)
        self.workflowPhononSelectedPathLabel = QLabel("Selected path: -", phonon_selected_panel)
        self.workflowPhononSelectedPathLabel.setWordWrap(True)
        self.workflowPhononSelectedPathLabel.setStyleSheet("color: #555;")
        self.workflowPhononPathTable = ReorderablePathTableWidget(0, 4, phonon_selected_panel)
        self._workflow_configure_path_table(self.workflowPhononPathTable)
        self.workflowPhononPathTable.setMinimumHeight(130)
        phonon_selected_layout.addLayout(phonon_selected_header)
        phonon_selected_layout.addWidget(self.workflowPhononSelectedPathLabel)
        phonon_selected_layout.addWidget(self.workflowPhononPathTable, 1)

        phonon_path_splitter.addWidget(phonon_suggestions_panel)
        phonon_path_splitter.addWidget(phonon_selected_panel)
        phonon_path_splitter.setStretchFactor(0, 1)
        phonon_path_splitter.setStretchFactor(1, 2)
        phonon_path_layout.addWidget(phonon_path_splitter)

        phonon_preview_row = QHBoxLayout()
        phonon_preview_title = QLabel("matdyn q-path preview", self.workflowPhononPathEditorGroup)
        phonon_preview_title.setStyleSheet("font-weight: 600;")
        phonon_preview_row.addWidget(phonon_preview_title)
        phonon_preview_row.addStretch()
        phonon_preview_row.addWidget(QLabel("Weight:", self.workflowPhononPathEditorGroup))
        self.workflowPhononPathWeightLineEdit = QLineEdit("50", self.workflowPhononPathEditorGroup)
        self.workflowPhononPathWeightLineEdit.setValidator(QIntValidator(1, 1000000, self))
        self.workflowPhononPathWeightLineEdit.setMaximumWidth(120)
        phonon_preview_row.addWidget(self.workflowPhononPathWeightLineEdit)
        self.workflowPhononPathWeightRow = QWidget(self.workflowPhononPathEditorGroup)
        self.workflowPhononPathWeightRow.setLayout(phonon_preview_row)
        phonon_path_layout.addWidget(self.workflowPhononPathWeightRow)

        self.workflowPhononMatdynPathEdit = QPlainTextEdit(self.workflowPhononPathEditorGroup)
        self.workflowPhononMatdynPathEdit.setPlainText(
            "\n".join(
                [
                    "5",
                    "0.0 0.0 0.0 40",
                    "0.5 0.0 0.0 40",
                    "0.3333333333 0.3333333333 0.0 40",
                    "0.0 0.0 0.0 40",
                    "0.0 0.0 0.5 1",
                ]
            )
        )
        self.workflowPhononMatdynPathEdit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.workflowPhononMatdynPathEdit.setMaximumHeight(130)
        self.workflowPhononMatdynPathEdit.setReadOnly(True)
        self.workflowPhononMatdynPathEdit.setToolTip(
            "matdyn.x q-path preview generated from the selected high-symmetry point path."
        )
        phonon_path_layout.addWidget(self.workflowPhononMatdynPathEdit)

        interpolation_form.addRow("q2r.x flfrc:", self.workflowPhononFlfrcLineEdit)
        interpolation_form.addRow("", self.workflowPhononBandCheckBox)
        interpolation_form.addRow("matdyn.x band flfrq:", self.workflowPhononFlfrqLineEdit)
        self.workflowPhononFlfrqLabel = interpolation_form.labelForField(self.workflowPhononFlfrqLineEdit)
        interpolation_form.addRow("matdyn.x band flvec:", self.workflowPhononMatdynFlvecLineEdit)
        self.workflowPhononMatdynFlvecLabel = interpolation_form.labelForField(self.workflowPhononMatdynFlvecLineEdit)
        interpolation_form.addRow("q2r.x zasr:", self.workflowPhononQ2rZasrCombo)
        interpolation_form.addRow("matdyn.x asr:", self.workflowPhononMatdynAsrCombo)
        interpolation_form.addRow("", self.workflowPhononMatdynQInBandFormCheckBox)
        self.workflowPhononPathQInBandWidget = self.workflowPhononMatdynQInBandFormCheckBox
        self.workflowPhononPathHeaderWidget = self.workflowPhononPathEditorGroup
        self.workflowPhononPathHeaderLabel = None
        self.workflowPhononPathButtonsRow = self.workflowPhononPathEditorGroup
        self.workflowPhononMatdynPathLabel = None
        interpolation_form.addRow(self.workflowPhononPathEditorGroup)
        layout.addWidget(interpolation_group)

        dos_group = QGroupBox("Phonon DOS", panel)
        self.workflowPhononDosGroup = dos_group
        dos_form = QFormLayout(dos_group)
        self.workflowPhononDosCheckBox = QCheckBox("Compute phonon DOS", dos_group)
        self.workflowPhononDosCheckBox.setChecked(True)
        self.workflowPhononFldosLineEdit = QLineEdit("phonon.phdos", dos_group)
        self.workflowPhononPhdosFlvecLineEdit = QLineEdit("phonon_dos.modes", dos_group)
        phdos_nk_row = QWidget(dos_group)
        phdos_nk_layout = QHBoxLayout(phdos_nk_row)
        phdos_nk_layout.setContentsMargins(0, 0, 0, 0)
        phdos_nk_layout.setSpacing(8)
        self.workflowPhononDosNkSpinBoxes: List[QSpinBox] = []
        for value in (20, 20, 20):
            spin = QSpinBox(phdos_nk_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononDosNkSpinBoxes.append(spin)
            phdos_nk_layout.addWidget(spin)
        phdos_nk_layout.addStretch()
        self.workflowPhononDosNkRow = phdos_nk_row
        dos_form.addRow("", self.workflowPhononDosCheckBox)
        dos_form.addRow("fldos:", self.workflowPhononFldosLineEdit)
        dos_form.addRow("DOS flvec:", self.workflowPhononPhdosFlvecLineEdit)
        dos_form.addRow("DOS mesh nk1/nk2/nk3:", phdos_nk_row)
        self.workflowPhononDosFldosLabel = dos_form.labelForField(self.workflowPhononFldosLineEdit)
        self.workflowPhononDosFlvecLabel = dos_form.labelForField(self.workflowPhononPhdosFlvecLineEdit)
        self.workflowPhononDosNkLabel = dos_form.labelForField(phdos_nk_row)
        layout.addWidget(dos_group)

        epc_group = QGroupBox("EPC: interpolated", panel)
        self.workflowPhononEpcGroup = epc_group
        epc_form = QFormLayout(epc_group)

        epc_fit_mesh_row = QWidget(epc_group)
        epc_fit_mesh_layout = QHBoxLayout(epc_fit_mesh_row)
        epc_fit_mesh_layout.setContentsMargins(0, 0, 0, 0)
        epc_fit_mesh_layout.setSpacing(8)
        self.workflowPhononEpcFitKmeshSpinBoxes: List[QSpinBox] = []
        for value in (24, 24, 24):
            spin = QSpinBox(epc_fit_mesh_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononEpcFitKmeshSpinBoxes.append(spin)
            epc_fit_mesh_layout.addWidget(spin)
        epc_fit_mesh_layout.addStretch()

        epc_scf_mesh_row = QWidget(epc_group)
        epc_scf_mesh_layout = QHBoxLayout(epc_scf_mesh_row)
        epc_scf_mesh_layout.setContentsMargins(0, 0, 0, 0)
        epc_scf_mesh_layout.setSpacing(8)
        self.workflowPhononEpcScfKmeshSpinBoxes: List[QSpinBox] = []
        for value in (12, 12, 12):
            spin = QSpinBox(epc_scf_mesh_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononEpcScfKmeshSpinBoxes.append(spin)
            epc_scf_mesh_layout.addWidget(spin)
        epc_scf_mesh_layout.addStretch()

        epc_qgrid_row = QWidget(epc_group)
        epc_qgrid_layout = QHBoxLayout(epc_qgrid_row)
        epc_qgrid_layout.setContentsMargins(0, 0, 0, 0)
        epc_qgrid_layout.setSpacing(8)
        self.workflowPhononEpcQGridSpinBoxes: List[QSpinBox] = []
        for value in (3, 3, 3):
            spin = QSpinBox(epc_qgrid_row)
            spin.setRange(1, 1000)
            spin.setValue(value)
            spin.setMaximumWidth(90)
            self.workflowPhononEpcQGridSpinBoxes.append(spin)
            epc_qgrid_layout.addWidget(spin)
        epc_qgrid_layout.addStretch()

        self.workflowPhononEpcFildvscfLineEdit = QLineEdit("dvscf", epc_group)
        self.workflowPhononLambdaEmaxLineEdit = QLineEdit("auto", epc_group)
        self.workflowPhononLambdaEmaxLineEdit.setToolTip(
            "auto: use the maximum phonon band frequency, convert cm-1 to THz, then round up."
        )
        self.workflowPhononLambdaDegaussLineEdit = QLineEdit("0.1", epc_group)
        self.workflowPhononLambdaSmearingMethodSpin = QSpinBox(epc_group)
        self.workflowPhononLambdaSmearingMethodSpin.setRange(1, 100)
        self.workflowPhononLambdaSmearingMethodSpin.setValue(1)
        self.workflowPhononLambdaMuLineEdit = QLineEdit("0.10", epc_group)
        self.workflowPhononLambdaExpectedTcLineEdit = QLineEdit("", epc_group)
        self.workflowPhononLambdaExpectedTcLineEdit.setPlaceholderText("optional, K")
        self.workflowPhononLambdaExpectedTcLineEdit.setToolTip(
            "If set, lambda.x is retried with degauss 0.1 to 0.8 and the closest T_c row is selected."
        )
        self.workflowPhononLambdaTcToleranceSpin = QDoubleSpinBox(epc_group)
        self.workflowPhononLambdaTcToleranceSpin.setRange(0.0, 1000.0)
        self.workflowPhononLambdaTcToleranceSpin.setDecimals(2)
        self.workflowPhononLambdaTcToleranceSpin.setSingleStep(0.5)
        self.workflowPhononLambdaTcToleranceSpin.setValue(1.0)
        self.workflowPhononLambdaTcToleranceSpin.setSuffix(" K")

        epc_form.addRow("Dense fit SCF mesh:", epc_fit_mesh_row)
        epc_form.addRow("Phonon SCF mesh:", epc_scf_mesh_row)
        epc_form.addRow("EPC q-grid nq1/nq2/nq3:", epc_qgrid_row)
        epc_form.addRow("fildvscf:", self.workflowPhononEpcFildvscfLineEdit)
        layout.addWidget(epc_group)

        lambda_group = QGroupBox("lambda.x Tc selection", panel)
        self.workflowPhononLambdaGroup = lambda_group
        lambda_form = QFormLayout(lambda_group)
        lambda_form.addRow("lambda emax:", self.workflowPhononLambdaEmaxLineEdit)
        lambda_form.addRow("lambda degauss:", self.workflowPhononLambdaDegaussLineEdit)
        lambda_form.addRow("lambda smearing method:", self.workflowPhononLambdaSmearingMethodSpin)
        lambda_form.addRow("lambda mu:", self.workflowPhononLambdaMuLineEdit)
        lambda_form.addRow("Expected experimental T_c:", self.workflowPhononLambdaExpectedTcLineEdit)
        lambda_form.addRow("T_c match tolerance:", self.workflowPhononLambdaTcToleranceSpin)
        layout.addWidget(lambda_group)

        note = QLabel(
            "Dispersion mode runs the QE example02-style chain. Single q-point mode skips q2r.x and matdyn.x.",
            panel,
        )
        self.workflowPhononModeNoteLabel = note
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        self.workflowPhononModeCombo.currentIndexChanged.connect(
            lambda _index: self._workflow_refresh_phonon_controls(),
        )
        self.workflowPhononEpcModeCombo.currentIndexChanged.connect(
            lambda _index: self._workflow_refresh_phonon_controls(),
        )
        self.workflowPhononPathRefreshButton.clicked.connect(
            lambda: self._workflow_refresh_phonon_path_options(force_reload=True),
        )
        self.workflowPhononPathList.itemSelectionChanged.connect(
            self._workflow_update_phonon_matdyn_path_preview,
        )
        self.workflowPhononLoadSuggestedPathButton.clicked.connect(
            lambda: self._workflow_load_selected_path_suggestion("phonon"),
        )
        self.workflowPhononPathTable.itemChanged.connect(
            lambda _item: self._workflow_update_phonon_matdyn_path_preview(),
        )
        self.workflowPhononPathTable.rowDropped.connect(
            lambda source, target: self._workflow_drag_move_path_table_point("phonon", source, target),
        )
        self.workflowPhononAddPathPointButton.clicked.connect(
            lambda: self._workflow_add_path_table_point("phonon"),
        )
        self.workflowPhononRemovePathPointButton.clicked.connect(
            lambda: self._workflow_remove_path_table_point("phonon"),
        )
        self.workflowPhononMovePathPointUpButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("phonon", -1),
        )
        self.workflowPhononMovePathPointDownButton.clicked.connect(
            lambda: self._workflow_move_path_table_point("phonon", 1),
        )
        self.workflowPhononPathWeightLineEdit.textChanged.connect(
            self._workflow_update_phonon_matdyn_path_preview,
        )
        self.workflowPhononBandCheckBox.toggled.connect(
            lambda _checked: self._workflow_refresh_phonon_controls(),
        )
        self.workflowPhononDosCheckBox.toggled.connect(
            lambda _checked: self._workflow_refresh_phonon_controls(),
        )

        self._workflow_refresh_phonon_path_options()
        self._workflow_refresh_phonon_controls()
        layout.addStretch()
        return panel

    def _workflow_refresh_bands_tab(self) -> None:
        if not hasattr(self, "workflowBandsPathList"):
            return
        self._workflow_refresh_bands_mode_controls()
        self._workflow_refresh_band_path_options()
        self._workflow_update_band_points_preview()
        self._workflow_fit_options_tab_height()

    def _workflow_refresh_wannier_tab(self) -> None:
        if not hasattr(self, "workflowWannierPathList"):
            return
        self._workflow_refresh_wannier_path_options()
        self._workflow_update_wannier_points_preview()
        self._workflow_update_wannier_kmesh_preview()
        self._workflow_fit_options_tab_height()

    def _workflow_update_wannier_kmesh_preview(self) -> None:
        if not hasattr(self, "workflowWannierKmeshPreviewLabel"):
            return
        mesh = [
            int(spin.value())
            for spin in getattr(self, "workflowWannierKmeshSpinBoxes", [])
        ]
        if len(mesh) < 3:
            self.workflowWannierKmeshPreviewLabel.setText("NSCF crystal k-points: -")
            return
        total = mesh[0] * mesh[1] * mesh[2]
        self.workflowWannierKmeshPreviewLabel.setText(f"NSCF crystal k-points: {total}")

    def _workflow_fat_bands_selected(self) -> bool:
        checkbox = getattr(self, "workflowCalculateFatBandCheckBox", None)
        return bool(checkbox and checkbox.isChecked())

    def _workflow_path_tool_button(
        self,
        parent: QWidget,
        tooltip: str,
        fallback_text: str,
        icon: QStyle.StandardPixmap,
    ) -> QToolButton:
        button = QToolButton(parent)
        button.setToolTip(tooltip)
        button.setText(fallback_text)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        try:
            standard_icon = self.style().standardIcon(icon)
            if not standard_icon.isNull():
                button.setIcon(standard_icon)
        except Exception:
            pass
        self._workflow_style_path_tool_button(button)
        return button

    def _workflow_text_path_tool_button(self, parent: QWidget, text: str, tooltip: str) -> QToolButton:
        button = QToolButton(parent)
        button.setToolTip(tooltip)
        button.setText(text)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._workflow_style_path_tool_button(button)
        return button

    def _workflow_style_path_tool_button(self, button: QToolButton) -> None:
        button.setMinimumHeight(28)
        button.setStyleSheet(
            """
            QToolButton {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                color: #111827;
                padding: 3px 8px;
            }
            QToolButton:hover {
                background-color: #E5E7EB;
                border-color: #94A3B8;
            }
            QToolButton:pressed {
                background-color: #D1D5DB;
            }
            QToolButton:disabled {
                color: #94A3B8;
                background-color: #F1F5F9;
                border-color: #E2E8F0;
            }
            """
        )

    def _workflow_configure_path_table(self, table: QTableWidget) -> None:
        table.setHorizontalHeaderLabels(["Label", "qx", "qy", "qz"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setToolTip("Drag rows by the label column to reorder the selected path.")
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def _workflow_refresh_bands_mode_controls(self) -> None:
        enabled = self._workflow_fat_bands_selected()
        if hasattr(self, "workflowFatBandColorLabel"):
            self.workflowFatBandColorLabel.setVisible(enabled)
        if hasattr(self, "workflowFatBandColorCombo"):
            self.workflowFatBandColorCombo.setVisible(enabled)

    def _workflow_refresh_dos_controls(self) -> None:
        if not hasattr(self, "workflowDosScfKpointSpacingSpin"):
            return
        self.workflowDosScfKpointSpacingSpin.setEnabled(True)

    def _workflow_refresh_elastic_controls(self) -> None:
        if not hasattr(self, "workflowElasticAlgorithmCombo") or not hasattr(
            self,
            "workflowElasticNgeoStrainSpin",
        ):
            return
        algorithm = str(self.workflowElasticAlgorithmCombo.currentData() or "standard")
        target_default = 6 if algorithm == "energy" else 4
        if bool(getattr(self, "_workflowElasticNgeoTouched", False)):
            return
        if int(self.workflowElasticNgeoStrainSpin.value()) == target_default:
            return
        previous_block_state = self.workflowElasticNgeoStrainSpin.blockSignals(True)
        try:
            self.workflowElasticNgeoStrainSpin.setValue(target_default)
        finally:
            self.workflowElasticNgeoStrainSpin.blockSignals(previous_block_state)

    def _workflow_effective_phonon_mode(self) -> str:
        raw_mode = (
            str(self.workflowPhononModeCombo.currentData() or "dispersion")
            if hasattr(self, "workflowPhononModeCombo")
            else "dispersion"
        )
        if raw_mode == "epc":
            return "epc_interpolated"
        if raw_mode == "epc_tetrahedra":
            return "epc_interpolated"
        if raw_mode in {"single_q", "epc_interpolated"}:
            return raw_mode
        return "dispersion"

    def _workflow_refresh_phonon_controls(self) -> None:
        if not hasattr(self, "workflowPhononModeCombo"):
            return
        raw_mode = str(self.workflowPhononModeCombo.currentData() or "dispersion")
        mode = self._workflow_effective_phonon_mode()
        single_q = mode == "single_q"
        epc = mode == "epc_interpolated"
        dispersion = mode == "dispersion"
        epc_parent = raw_mode == "epc" or mode == "epc_interpolated"
        if hasattr(self, "workflowPhononEpcModeCombo"):
            self.workflowPhononEpcModeCombo.setVisible(False)
        if hasattr(self, "workflowPhononEpcModeLabel") and self.workflowPhononEpcModeLabel is not None:
            self.workflowPhononEpcModeLabel.setVisible(False)
        if hasattr(self, "workflowPhononQGridRow"):
            self.workflowPhononQGridRow.setVisible(dispersion)
        if hasattr(self, "workflowPhononQGridLabel") and self.workflowPhononQGridLabel is not None:
            self.workflowPhononQGridLabel.setVisible(dispersion)
        if hasattr(self, "workflowPhononQPointLineEdit"):
            self.workflowPhononQPointLineEdit.setVisible(single_q)
        if hasattr(self, "workflowPhononQPointLabel") and self.workflowPhononQPointLabel is not None:
            self.workflowPhononQPointLabel.setVisible(single_q)
        if hasattr(self, "workflowPhononScfGroup"):
            self.workflowPhononScfGroup.setVisible(not epc)
        if hasattr(self, "workflowPhononInterpolationGroup"):
            self.workflowPhononInterpolationGroup.setVisible(dispersion or epc)
        if hasattr(self, "workflowPhononDosGroup"):
            self.workflowPhononDosGroup.setVisible(dispersion)
        if hasattr(self, "workflowPhononEpcGroup"):
            self.workflowPhononEpcGroup.setVisible(epc)
        if hasattr(self, "workflowPhononLambdaGroup"):
            self.workflowPhononLambdaGroup.setVisible(epc)
        band_enabled = bool(
            getattr(self, "workflowPhononBandCheckBox", None)
            and self.workflowPhononBandCheckBox.isChecked()
        )
        dos_enabled = bool(getattr(self, "workflowPhononDosCheckBox", None) and self.workflowPhononDosCheckBox.isChecked())
        if dispersion and not band_enabled and not dos_enabled and hasattr(self, "workflowPhononBandCheckBox"):
            self.workflowPhononBandCheckBox.blockSignals(True)
            self.workflowPhononBandCheckBox.setChecked(True)
            self.workflowPhononBandCheckBox.blockSignals(False)
            band_enabled = True
        if hasattr(self, "workflowPhononBandCheckBox"):
            self.workflowPhononBandCheckBox.setVisible(dispersion)
        band_path_visible = epc or (dispersion and band_enabled)
        for name in (
            "workflowPhononFlfrqLineEdit",
            "workflowPhononFlfrqLabel",
            "workflowPhononMatdynFlvecLineEdit",
            "workflowPhononMatdynFlvecLabel",
            "workflowPhononPathQInBandWidget",
            "workflowPhononPathHeaderWidget",
            "workflowPhononPathHeaderLabel",
            "workflowPhononPathList",
            "workflowPhononLoadSuggestedPathButton",
            "workflowPhononSelectedPathLabel",
            "workflowPhononPathTable",
            "workflowPhononPathButtonsRow",
            "workflowPhononPathWeightRow",
            "workflowPhononMatdynPathEdit",
            "workflowPhononMatdynPathLabel",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(band_path_visible)
        for name in (
            "workflowPhononFldosLineEdit",
            "workflowPhononPhdosFlvecLineEdit",
            "workflowPhononDosNkRow",
            "workflowPhononDosFldosLabel",
            "workflowPhononDosFlvecLabel",
            "workflowPhononDosNkLabel",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(dos_enabled)
        if hasattr(self, "workflowPhononModeNoteLabel"):
            if epc:
                note = "EPC mode runs the QE example03-style lambda/a2F chain with two SCF stages."
            elif dispersion:
                note = "Dispersion mode runs ph.x on a q-grid, then q2r.x, with optional phonon band structure and DOS."
            else:
                note = "Single q-point mode renders only SCF and ph.x inputs."
            self.workflowPhononModeNoteLabel.setText(note)
        self._workflow_fit_options_tab_height()

    def _workflow_refresh_band_path_options(self, *, force_reload: bool = False) -> None:
        if not hasattr(self, "workflowBandsPathList"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowBandsPathInfoLabel.setText("Select a structure file to suggest high-symmetry paths.")
            self.workflowBandsPathList.clear()
            self._workflow_set_path_table_records("bands", [])
            return

        source = Path(source_path).expanduser()
        if not source.is_file():
            self.workflowBandsPathInfoLabel.setText("Selected source file is unavailable.")
            self.workflowBandsPathList.clear()
            self._workflow_set_path_table_records("bands", [])
            return

        selected_labels = {
            self.workflowBandsPathList.item(index).text()
            for index in range(self.workflowBandsPathList.count())
            if self.workflowBandsPathList.item(index).isSelected()
        }
        cache_source = str(getattr(self, "_workflowBandsPathSource", ""))
        source_changed = force_reload or cache_source != str(source)
        if source_changed:
            try:
                grouped_paths, points_frac = self._workflow_high_symmetry_data_for_source(source)
            except Exception as exc:  # noqa: BLE001 - show in-band status
                self._workflowBandsPathSource = str(source)
                self._workflowBandsPathGroups = []
                self._workflowBandsPathPoints = {}
                self.workflowBandsPathInfoLabel.setText(f"Could not load high-symmetry paths: {exc}")
                self.workflowBandsPathList.clear()
                self._workflow_set_path_table_records("bands", [])
                return

            self._workflowBandsPathSource = str(source)
            self._workflowBandsPathGroups = grouped_paths
            self._workflowBandsPathPoints = points_frac

        grouped_paths = list(getattr(self, "_workflowBandsPathGroups", []))
        points_frac = dict(getattr(self, "_workflowBandsPathPoints", {}))
        self.workflowBandsPathList.clear()
        if not grouped_paths:
            self.workflowBandsPathInfoLabel.setText("No high-symmetry paths are available for this structure.")
            self._workflow_set_path_table_records("bands", [])
            return

        for group in grouped_paths:
            label = " -> ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, list(group))
            self.workflowBandsPathList.addItem(item)
            if label in selected_labels:
                item.setSelected(True)

        if not any(self.workflowBandsPathList.item(i).isSelected() for i in range(self.workflowBandsPathList.count())):
            self.workflowBandsPathList.item(0).setSelected(True)

        self.workflowBandsPathInfoLabel.setText(
            f"Loaded {len(grouped_paths)} path group(s) and {len(points_frac)} high-symmetry points."
        )
        if source_changed or getattr(self, "workflowBandsPathTable", None) is not None and self.workflowBandsPathTable.rowCount() == 0:
            first_group = list(grouped_paths[0]) if grouped_paths else []
            self._workflow_load_path_labels_into_table("bands", first_group)
        else:
            self._workflow_update_band_points_preview()

    def _workflow_high_symmetry_data_for_source(
        self,
        source: Path,
    ) -> Tuple[List[List[str]], Dict[str, Tuple[float, float, float]]]:
        if source.suffix.lower() == ".cif":
            data = load_high_symmetry_path(str(source))
            grouped = [list(path) for path in data.grouped_paths]
            points = {str(label): (float(v[0]), float(v[1]), float(v[2])) for label, v in data.points_frac.items()}
            return grouped, points
        return self._workflow_high_symmetry_from_qe_input(source)

    def _workflow_high_symmetry_from_qe_input(
        self,
        source: Path,
    ) -> Tuple[List[List[str]], Dict[str, Tuple[float, float, float]]]:
        seed = parse_scf_seed(str(source))
        if not seed.atomic_positions:
            raise ValueError("QE input has no ATOMIC_POSITIONS card.")
        cell = self._workflow_cell_from_qe_input(source)
        if cell is None:
            raise ValueError("QE input has no usable CELL_PARAMETERS data.")

        symbols: List[str] = []
        positions: List[List[float]] = []
        mode = str(seed.atomic_positions[0].get("mode") or "").strip().lower()
        if not mode:
            card = seed.raw_cards.get("ATOMIC_POSITIONS", [])
            if card:
                header = str(card[0]).strip()
                if "(" in header and ")" in header:
                    mode = header.split("(", 1)[1].split(")", 1)[0].strip().lower()
                else:
                    parts = header.split(None, 1)
                    if len(parts) > 1:
                        mode = parts[1].strip().lower()
        for position in seed.atomic_positions:
            symbol = normalize_element_symbol(str(position.get("label") or ""))
            x = self._workflow_parse_float(position.get("x", ""))
            y = self._workflow_parse_float(position.get("y", ""))
            z = self._workflow_parse_float(position.get("z", ""))
            if not symbol or x is None or y is None or z is None:
                continue
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        if not symbols:
            raise ValueError("QE input has no parseable atomic positions.")

        from ase import Atoms  # type: ignore

        if mode in {"crystal", "crystal_sg"}:
            atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
        else:
            if mode in {"bohr", "a.u.", "au"}:
                positions = [[component * 0.529177210903 for component in row] for row in positions]
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

        try:
            import seekpath  # type: ignore
        except ModuleNotFoundError:
            fallback = fallback_high_symmetry_path_from_cell(atoms.get_cell())
            grouped = [list(path) for path in fallback.grouped_paths]
            points = {
                str(label): (float(value[0]), float(value[1]), float(value[2]))
                for label, value in fallback.points_frac.items()
            }
            return grouped, points

        structure = (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="seekpath")
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"spglib(\..*)?$",
                message=r".*dict interface \(SpglibDataset\['.*?'\]\) is deprecated.*",
            )
            result = seekpath.get_path(structure)

        segments = result.get("path") or []
        if not segments:
            fallback = fallback_high_symmetry_path_from_cell(atoms.get_cell())
            grouped = [list(path) for path in fallback.grouped_paths]
            points = {
                str(label): (float(value[0]), float(value[1]), float(value[2]))
                for label, value in fallback.points_frac.items()
            }
            return grouped, points

        grouped_paths: List[List[str]] = []
        current_path: List[str] = [segments[0][0]]
        for index, (start, end) in enumerate(segments):
            if current_path[-1] != start:
                grouped_paths.append(current_path)
                current_path = [start]
            current_path.append(end)
            if index == len(segments) - 1:
                grouped_paths.append(current_path)

        points_raw = result.get("point_coords") or {}
        points_frac: Dict[str, Tuple[float, float, float]] = {}
        for label, coords in points_raw.items():
            points_frac[str(label)] = (float(coords[0]), float(coords[1]), float(coords[2]))

        return grouped_paths, points_frac

    def _workflow_selected_band_paths(self) -> List[List[str]]:
        paths: List[List[str]] = []
        if not hasattr(self, "workflowBandsPathList"):
            return paths
        for index in range(self.workflowBandsPathList.count()):
            item = self.workflowBandsPathList.item(index)
            if item is None or not item.isSelected():
                continue
            group = item.data(Qt.ItemDataRole.UserRole) or []
            labels = [str(label).strip() for label in group if str(label).strip()]
            if labels:
                paths.append(labels)
        return paths

    def _workflow_path_widgets(self, kind: str) -> tuple[Optional[QTableWidget], Optional[QLabel], Optional[QListWidget]]:
        if kind == "wannier":
            return (
                getattr(self, "workflowWannierPathTable", None),
                getattr(self, "workflowWannierSelectedPathLabel", None),
                getattr(self, "workflowWannierPathList", None),
            )
        if kind == "phonon":
            return (
                getattr(self, "workflowPhononPathTable", None),
                getattr(self, "workflowPhononSelectedPathLabel", None),
                getattr(self, "workflowPhononPathList", None),
            )
        return (
            getattr(self, "workflowBandsPathTable", None),
            getattr(self, "workflowBandsSelectedPathLabel", None),
            getattr(self, "workflowBandsPathList", None),
        )

    def _workflow_path_points_for_kind(self, kind: str) -> Dict[str, Tuple[float, float, float]]:
        if kind == "phonon":
            attr = "_workflowPhononPathPoints"
        elif kind == "wannier":
            attr = "_workflowWannierPathPoints"
        else:
            attr = "_workflowBandsPathPoints"
        return dict(getattr(self, attr, {}))

    def _workflow_update_path_preview_for_kind(self, kind: str) -> None:
        if kind == "phonon":
            self._workflow_update_phonon_matdyn_path_preview()
        elif kind == "wannier":
            self._workflow_update_wannier_points_preview()
        else:
            self._workflow_update_band_points_preview()

    def _workflow_path_coord_spin(self, value: float = 0.0) -> QDoubleSpinBox:
        spin = TrimmedDoubleSpinBox(self)
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(10)
        spin.setSingleStep(0.01)
        spin.setValue(float(value))
        spin.setMaximumWidth(120)
        return spin

    def _workflow_path_records_from_labels(
        self,
        labels: Sequence[str],
        points_frac: Mapping[str, Sequence[float]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for label in labels:
            cleaned = str(label).strip()
            if not cleaned:
                continue
            coords = points_frac.get(cleaned)
            if coords is None:
                continue
            try:
                qx, qy, qz = float(coords[0]), float(coords[1]), float(coords[2])
            except (IndexError, TypeError, ValueError):
                continue
            records.append({"label": cleaned, "qx": qx, "qy": qy, "qz": qz})
        return records

    def _workflow_path_table_records(self, kind: str) -> List[Dict[str, Any]]:
        table, _label_widget, _list_widget = self._workflow_path_widgets(kind)
        records: List[Dict[str, Any]] = []
        if table is None:
            return records
        for row in range(table.rowCount()):
            label_item = table.item(row, 0)
            label = str(label_item.text() if label_item is not None else "").strip() or f"P{row + 1}"
            values: List[float] = []
            for column in (1, 2, 3):
                widget = table.cellWidget(row, column)
                if isinstance(widget, QDoubleSpinBox):
                    values.append(float(widget.value()))
                else:
                    item = table.item(row, column)
                    try:
                        values.append(float(str(item.text() if item is not None else "0").strip()))
                    except ValueError:
                        values.append(0.0)
            records.append({"label": label, "qx": values[0], "qy": values[1], "qz": values[2]})
        return records

    def _workflow_set_path_table_records(
        self,
        kind: str,
        records: Sequence[Mapping[str, Any]],
        *,
        select_row: int = 0,
    ) -> None:
        table, _label_widget, _list_widget = self._workflow_path_widgets(kind)
        if table is None:
            return
        table.blockSignals(True)
        table.setRowCount(0)
        for record in records:
            row = table.rowCount()
            table.insertRow(row)
            label = str(record.get("label") or f"P{row + 1}").strip() or f"P{row + 1}"
            table.setItem(row, 0, QTableWidgetItem(label))
            for column, key in enumerate(("qx", "qy", "qz"), start=1):
                try:
                    value = float(record.get(key, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                spin = self._workflow_path_coord_spin(value)
                spin.valueChanged.connect(lambda _value, path_kind=kind: self._workflow_update_path_preview_for_kind(path_kind))
                table.setCellWidget(row, column, spin)
        table.blockSignals(False)
        if table.rowCount() > 0:
            table.selectRow(max(0, min(select_row, table.rowCount() - 1)))
        self._workflow_update_path_preview_for_kind(kind)

    def _workflow_load_path_labels_into_table(self, kind: str, labels: Sequence[str]) -> None:
        records = self._workflow_path_records_from_labels(labels, self._workflow_path_points_for_kind(kind))
        if records:
            self._workflow_set_path_table_records(kind, records)

    def _workflow_selected_suggestion_labels(self, kind: str) -> List[str]:
        _table, _label_widget, list_widget = self._workflow_path_widgets(kind)
        labels: List[str] = []
        if list_widget is None:
            return labels
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item is None or not item.isSelected():
                continue
            group = item.data(Qt.ItemDataRole.UserRole) or []
            for label in group:
                cleaned = str(label).strip()
                if cleaned:
                    labels.append(cleaned)
        return labels

    def _workflow_load_selected_path_suggestion(self, kind: str) -> None:
        labels = self._workflow_selected_suggestion_labels(kind)
        if not labels:
            return
        self._workflow_load_path_labels_into_table(kind, labels)

    def _workflow_add_path_table_point(self, kind: str) -> None:
        records = self._workflow_path_table_records(kind)
        records.append({"label": f"P{len(records) + 1}", "qx": 0.0, "qy": 0.0, "qz": 0.0})
        self._workflow_set_path_table_records(kind, records, select_row=len(records) - 1)

    def _workflow_remove_path_table_point(self, kind: str) -> None:
        table, _label_widget, _list_widget = self._workflow_path_widgets(kind)
        if table is None or table.rowCount() <= 0:
            return
        row = table.currentRow()
        if row < 0:
            row = table.rowCount() - 1
        records = self._workflow_path_table_records(kind)
        if 0 <= row < len(records):
            records.pop(row)
        self._workflow_set_path_table_records(kind, records, select_row=max(0, row - 1))

    def _workflow_move_path_table_point(self, kind: str, delta: int) -> None:
        table, _label_widget, _list_widget = self._workflow_path_widgets(kind)
        if table is None or table.rowCount() <= 1:
            return
        row = table.currentRow()
        target = row + int(delta)
        if row < 0 or target < 0 or target >= table.rowCount():
            return
        records = self._workflow_path_table_records(kind)
        records[row], records[target] = records[target], records[row]
        self._workflow_set_path_table_records(kind, records, select_row=target)

    def _workflow_drag_move_path_table_point(self, kind: str, source_row: int, target_row: int) -> None:
        table, _label_widget, _list_widget = self._workflow_path_widgets(kind)
        if table is None or table.rowCount() <= 1:
            return
        if source_row < 0 or target_row < 0 or source_row >= table.rowCount() or target_row >= table.rowCount():
            return
        if source_row == target_row:
            return
        records = self._workflow_path_table_records(kind)
        moved = records.pop(source_row)
        records.insert(target_row, moved)
        self._workflow_set_path_table_records(kind, records, select_row=target_row)

    def _workflow_update_selected_path_label(self, kind: str, records: Sequence[Mapping[str, Any]]) -> None:
        _table, label_widget, _list_widget = self._workflow_path_widgets(kind)
        if label_widget is None:
            return
        labels = [str(record.get("label") or "").strip() for record in records if str(record.get("label") or "").strip()]
        label_widget.setText("Selected path: " + (" -> ".join(labels) if labels else "-"))

    def _workflow_band_crystal_b_from_records(self, records: Sequence[Mapping[str, Any]], weight: int) -> List[str]:
        if not records:
            raise ValueError("At least one high-symmetry point is required.")
        lines = ["K_POINTS crystal_b", str(len(records))]
        for record in records:
            label = str(record.get("label") or "").strip() or "P"
            qx = float(record.get("qx", 0.0))
            qy = float(record.get("qy", 0.0))
            qz = float(record.get("qz", 0.0))
            lines.append(f"{qx:7.3f}  {qy:7.3f}  {qz:7.3f}  {int(weight):g} !{label.lower()}")
        return lines

    def _workflow_matdyn_path_from_records(self, records: Sequence[Mapping[str, Any]], weight: int) -> List[str]:
        if not records:
            raise ValueError("At least one high-symmetry point is required.")
        lines = [str(len(records))]
        for record in records:
            label = str(record.get("label") or "").strip() or "P"
            qx = float(record.get("qx", 0.0))
            qy = float(record.get("qy", 0.0))
            qz = float(record.get("qz", 0.0))
            lines.append(f"{qx:.10g} {qy:.10g} {qz:.10g} {int(weight)} ! {label}")
        return lines

    def _workflow_update_band_points_preview(self) -> None:
        if not hasattr(self, "workflowBandsCrystalBPreview"):
            return
        records = self._workflow_path_table_records("bands")
        self._workflow_update_selected_path_label("bands", records)
        if not records:
            self.workflowBandsCrystalBPreview.setPlainText("Add at least one high-symmetry point.")
            return
        weight_text = (
            self.workflowBandsPathWeightLineEdit.text().strip()
            if hasattr(self, "workflowBandsPathWeightLineEdit")
            else ""
        )
        weight = self._workflow_parse_positive_int(weight_text, default=50, minimum=1)
        if weight is None:
            self.workflowBandsCrystalBPreview.setPlainText("Weight must be a positive integer.")
            return
        try:
            lines = self._workflow_band_crystal_b_from_records(records, weight)
        except Exception as exc:  # noqa: BLE001
            self.workflowBandsCrystalBPreview.setPlainText(f"Could not build K_POINTS crystal_b: {exc}")
            return
        self.workflowBandsCrystalBPreview.setPlainText("\n".join(lines))

    def _workflow_matdyn_path_from_seekpath(
        self,
        selected_paths: Sequence[Sequence[str]],
        points_frac: Dict[str, Tuple[float, float, float]],
        weight: int,
    ) -> List[str]:
        labels: List[str] = []
        for path in selected_paths:
            for label in path:
                cleaned = str(label).strip()
                if cleaned:
                    labels.append(cleaned)
        if not labels:
            raise ValueError("No high-symmetry path labels are selected.")

        lines = [str(len(labels))]
        for label in labels:
            coords = points_frac.get(label)
            if coords is None:
                raise ValueError(f"Missing fractional coordinates for label '{label}'.")
            qx, qy, qz = (float(coords[0]), float(coords[1]), float(coords[2]))
            lines.append(f"{qx:.10g} {qy:.10g} {qz:.10g} {int(weight)} ! {label}")
        return lines

    def _workflow_refresh_phonon_path_options(self, *, force_reload: bool = False) -> None:
        if not hasattr(self, "workflowPhononPathList"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowPhononPathInfoLabel.setText("Select a structure file to suggest high-symmetry paths.")
            self.workflowPhononPathList.clear()
            self._workflow_set_path_table_records("phonon", [])
            self._workflow_update_phonon_matdyn_path_preview()
            return

        source = Path(source_path).expanduser()
        if not source.is_file():
            self.workflowPhononPathInfoLabel.setText("Selected source file is unavailable.")
            self.workflowPhononPathList.clear()
            self._workflow_set_path_table_records("phonon", [])
            self._workflow_update_phonon_matdyn_path_preview()
            return

        selected_labels = {
            self.workflowPhononPathList.item(index).text()
            for index in range(self.workflowPhononPathList.count())
            if self.workflowPhononPathList.item(index).isSelected()
        }
        cache_source = str(getattr(self, "_workflowPhononPathSource", ""))
        source_changed = force_reload or cache_source != str(source)
        if source_changed:
            try:
                grouped_paths, points_frac = self._workflow_high_symmetry_data_for_source(source)
            except Exception as exc:  # noqa: BLE001 - show in-band status
                self._workflowPhononPathSource = str(source)
                self._workflowPhononPathGroups = []
                self._workflowPhononPathPoints = {}
                self.workflowPhononPathInfoLabel.setText(f"Could not load high-symmetry paths: {exc}")
                self.workflowPhononPathList.clear()
                self._workflow_set_path_table_records("phonon", [])
                self._workflow_update_phonon_matdyn_path_preview()
                return

            self._workflowPhononPathSource = str(source)
            self._workflowPhononPathGroups = grouped_paths
            self._workflowPhononPathPoints = points_frac

        grouped_paths = list(getattr(self, "_workflowPhononPathGroups", []))
        points_frac = dict(getattr(self, "_workflowPhononPathPoints", {}))
        self.workflowPhononPathList.clear()
        if not grouped_paths:
            self.workflowPhononPathInfoLabel.setText("No high-symmetry paths are available for this structure.")
            self._workflow_set_path_table_records("phonon", [])
            self._workflow_update_phonon_matdyn_path_preview()
            return

        for group in grouped_paths:
            label = " -> ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, list(group))
            self.workflowPhononPathList.addItem(item)
            if label in selected_labels:
                item.setSelected(True)

        if not any(self.workflowPhononPathList.item(i).isSelected() for i in range(self.workflowPhononPathList.count())):
            self.workflowPhononPathList.item(0).setSelected(True)

        self.workflowPhononPathInfoLabel.setText(
            f"Loaded {len(grouped_paths)} path group(s) and {len(points_frac)} high-symmetry points."
        )
        if source_changed or getattr(self, "workflowPhononPathTable", None) is not None and self.workflowPhononPathTable.rowCount() == 0:
            first_group = list(grouped_paths[0]) if grouped_paths else []
            self._workflow_load_path_labels_into_table("phonon", first_group)
        else:
            self._workflow_update_phonon_matdyn_path_preview()

    def _workflow_refresh_wannier_path_options(self, *, force_reload: bool = False) -> None:
        if not hasattr(self, "workflowWannierPathList"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowWannierPathInfoLabel.setText("Select a structure file to suggest high-symmetry paths.")
            self.workflowWannierPathList.clear()
            self._workflow_set_path_table_records("wannier", [])
            return

        source = Path(source_path).expanduser()
        if not source.is_file():
            self.workflowWannierPathInfoLabel.setText("Selected source file is unavailable.")
            self.workflowWannierPathList.clear()
            self._workflow_set_path_table_records("wannier", [])
            return

        selected_labels = {
            self.workflowWannierPathList.item(index).text()
            for index in range(self.workflowWannierPathList.count())
            if self.workflowWannierPathList.item(index).isSelected()
        }
        cache_source = str(getattr(self, "_workflowWannierPathSource", ""))
        source_changed = force_reload or cache_source != str(source)
        if source_changed:
            try:
                grouped_paths, points_frac = self._workflow_high_symmetry_data_for_source(source)
            except Exception as exc:  # noqa: BLE001 - show in-band status
                self._workflowWannierPathSource = str(source)
                self._workflowWannierPathGroups = []
                self._workflowWannierPathPoints = {}
                self.workflowWannierPathInfoLabel.setText(f"Could not load high-symmetry paths: {exc}")
                self.workflowWannierPathList.clear()
                self._workflow_set_path_table_records("wannier", [])
                return

            self._workflowWannierPathSource = str(source)
            self._workflowWannierPathGroups = grouped_paths
            self._workflowWannierPathPoints = points_frac

        grouped_paths = list(getattr(self, "_workflowWannierPathGroups", []))
        points_frac = dict(getattr(self, "_workflowWannierPathPoints", {}))
        self.workflowWannierPathList.clear()
        if not grouped_paths:
            self.workflowWannierPathInfoLabel.setText("No high-symmetry paths are available for this structure.")
            self._workflow_set_path_table_records("wannier", [])
            return

        for group in grouped_paths:
            label = " -> ".join(group)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, list(group))
            self.workflowWannierPathList.addItem(item)
            if label in selected_labels:
                item.setSelected(True)

        if not any(self.workflowWannierPathList.item(i).isSelected() for i in range(self.workflowWannierPathList.count())):
            self.workflowWannierPathList.item(0).setSelected(True)

        self.workflowWannierPathInfoLabel.setText(
            f"Loaded {len(grouped_paths)} path group(s) and {len(points_frac)} high-symmetry points."
        )
        if source_changed or getattr(self, "workflowWannierPathTable", None) is not None and self.workflowWannierPathTable.rowCount() == 0:
            first_group = list(grouped_paths[0]) if grouped_paths else []
            self._workflow_load_path_labels_into_table("wannier", first_group)
        else:
            self._workflow_update_wannier_points_preview()

    def _workflow_selected_phonon_paths(self) -> List[List[str]]:
        paths: List[List[str]] = []
        if not hasattr(self, "workflowPhononPathList"):
            return paths
        for index in range(self.workflowPhononPathList.count()):
            item = self.workflowPhononPathList.item(index)
            if item is None or not item.isSelected():
                continue
            group = item.data(Qt.ItemDataRole.UserRole) or []
            labels = [str(label).strip() for label in group if str(label).strip()]
            if labels:
                paths.append(labels)
        return paths

    def _workflow_update_phonon_matdyn_path_preview(self) -> None:
        if not hasattr(self, "workflowPhononMatdynPathEdit"):
            return
        records = self._workflow_path_table_records("phonon")
        self._workflow_update_selected_path_label("phonon", records)
        if not records:
            self.workflowPhononMatdynPathEdit.setPlainText("Add at least one high-symmetry point.")
            return

        weight_text = (
            self.workflowPhononPathWeightLineEdit.text().strip()
            if hasattr(self, "workflowPhononPathWeightLineEdit")
            else ""
        )
        weight = self._workflow_parse_positive_int(weight_text, default=50, minimum=1)
        if weight is None:
            self.workflowPhononMatdynPathEdit.setPlainText("Weight must be a positive integer.")
            return

        try:
            lines = self._workflow_matdyn_path_from_records(records, int(weight))
        except Exception as exc:  # noqa: BLE001 - surfaced in the phonon options panel
            self.workflowPhononMatdynPathEdit.setPlainText(f"Could not build matdyn q-path: {exc}")
            return
        self.workflowPhononMatdynPathEdit.setPlainText("\n".join(lines))

    def _workflow_update_wannier_points_preview(self) -> None:
        if not hasattr(self, "workflowWannierCrystalBPreview"):
            return
        records = self._workflow_path_table_records("wannier")
        self._workflow_update_selected_path_label("wannier", records)
        if not records:
            self.workflowWannierCrystalBPreview.setPlainText("Add at least one high-symmetry point.")
            return
        weight_text = (
            self.workflowWannierPathWeightLineEdit.text().strip()
            if hasattr(self, "workflowWannierPathWeightLineEdit")
            else ""
        )
        weight = self._workflow_parse_positive_int(weight_text, default=50, minimum=1)
        if weight is None:
            self.workflowWannierCrystalBPreview.setPlainText("Weight must be a positive integer.")
            return
        try:
            lines = self._workflow_band_crystal_b_from_records(records, weight)
        except Exception as exc:  # noqa: BLE001
            self.workflowWannierCrystalBPreview.setPlainText(f"Could not build K_POINTS crystal_b: {exc}")
            return
        self.workflowWannierCrystalBPreview.setPlainText("\n".join(lines))

    def _workflow_suggest_nbnd_dry_run(self, target: str = "bands") -> None:
        target_key = "wannier" if str(target).strip().lower() in {"wannier", "wannier90", "winner90"} else "bands"
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self._show_error("Select a structure file before running the nbnd dry run.")
            return

        source = Path(source_path).expanduser()
        if not source.is_file():
            self._show_error("Selected structure file is unavailable.")
            return

        try:
            conversion_options = self._workflow_conversion_options()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        pseudo_error = self._workflow_validate_pseudo_selection(conversion_options)
        if pseudo_error:
            self._show_warning("Missing pseudopotentials", pseudo_error)
            return

        kpoint_spacing = float(self.workflowBandScfKpointSpacingSpin.value())
        kmesh = None
        child_workdir = "bands"
        nbnd_widget = getattr(self, "workflowBandsNbndLineEdit", None)
        hint_widget = getattr(self, "workflowBandsNbndHintLabel", None)
        if target_key == "wannier":
            child_workdir = "wannier90"
            nbnd_widget = getattr(self, "workflowWannierNbndLineEdit", None)
            hint_widget = getattr(self, "workflowWannierNbndHintLabel", None)
            kmesh = (
                [int(spin.value()) for spin in self.workflowWannierKmeshSpinBoxes]
                if hasattr(self, "workflowWannierKmeshSpinBoxes")
                else [8, 8, 6]
            )
        spin_orbit = self._workflow_spin_orbit_enabled()
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            with tempfile.TemporaryDirectory(prefix="workflow_nbnd_", dir="/tmp") as tmp_dir:
                workflow_root = Path(tmp_dir) / "workflow"
                source_dir = workflow_root / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                work_dir = workflow_root / child_workdir
                work_dir.mkdir(parents=True, exist_ok=True)
                template_path = source_dir / "QE_input.template"
                suffix = source.suffix.lower()
                if suffix in {".in", ".pwi", ".pw", ".qe", ".txt"} and not bool(conversion_options.get("force_conversion")):
                    shutil.copy2(source, template_path)
                    copy_selected_workflow_pseudos(conversion_options, source_dir / "pseudo")
                else:
                    _generate_workflow_qe_input(source, template_path, conversion_options)

                template_text = template_path.read_text(encoding="utf-8", errors="ignore")
                rendered = render_band_scf_input(
                    template_text,
                    workflow_root=workflow_root,
                    child_workdir=child_workdir,
                    prefix="nbnd_dryrun",
                    kpoint_spacing=kpoint_spacing,
                    kmesh=kmesh,
                    spin_orbit_required=spin_orbit,
                )
                rendered = set_namelist_values(rendered, "control", {"nstep": "0"})

                input_path = work_dir / "nbnd_dryrun_scf.in"
                output_path = work_dir / "nbnd_dryrun_scf.out"
                input_path.write_text(rendered, encoding="utf-8")

                command = self._workflow_qe_command("pw.x", input_path.name, output_path.name)
                output_text = self._workflow_run_command_capture(command, cwd=work_dir)
                if output_path.is_file():
                    output_text = output_path.read_text(encoding="utf-8", errors="ignore")

                nbnd = self._workflow_parse_nbnd_from_output(output_text)
                if nbnd is None:
                    raise RuntimeError("Could not find 'number of Kohn-Sham states' in dry-run output.")
                if isinstance(nbnd_widget, QLineEdit):
                    nbnd_widget.setText(str(int(nbnd)))
                if isinstance(hint_widget, QLabel):
                    hint_widget.setText(f"Suggested nbnd from dry run: {int(nbnd)}")
                self.statusbar.showMessage(f"Suggested nbnd: {int(nbnd)}", 5000)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"nbnd dry run failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _workflow_qe_command(self, executable: str, input_name: str, output_name: str) -> str:
        command_base = self._workflow_qe_command_base(executable)
        return f"{command_base} -in {shlex.quote(input_name)} | tee {shlex.quote(output_name)}"

    def _workflow_qe_command_base(self, executable: str) -> str:
        exe = str(executable or "").strip() or "pw.x"
        if COMMAND_PREFS_PATH.is_file():
            try:
                with COMMAND_PREFS_PATH.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                commands = payload.get("commands") if isinstance(payload, dict) else {}
                if isinstance(commands, dict):
                    command = str(commands.get(exe) or "").strip()
                    if command:
                        return command
            except Exception:
                pass
        return exe

    def _workflow_run_command_capture(self, command: str, *, cwd: Path) -> str:
        bash_command = f"set -o pipefail; {command}"
        process = subprocess.Popen(
            ["/bin/bash", "-lc", bash_command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: List[str] = []
        fatal_error_found = False
        if process.stdout is not None:
            for line in process.stdout:
                lines.append(line)
                if "Error in routine" in line or "MPI_ABORT" in line:
                    fatal_error_found = True
        return_code = process.wait()
        output = "".join(lines)
        if return_code != 0 or fatal_error_found:
            tail = "\n".join(output.splitlines()[-20:]).strip()
            detail = tail or f"exit code {return_code}"
            raise RuntimeError(detail)
        return output

    @staticmethod
    def _workflow_parse_nbnd_from_output(text: str) -> Optional[int]:
        matches = list(
            re.finditer(
                r"number of Kohn-Sham states\s*=\s*(\d+)",
                text or "",
                re.IGNORECASE,
            )
        )
        if not matches:
            return None
        try:
            return int(matches[-1].group(1))
        except Exception:
            return None

    def _workflow_update_energy_delta_labels(self) -> None:
        if hasattr(self, "workflowEcutEnergyDeltaRyLabel"):
            cutoff_ry = self.workflowEcutEnergyDeltaSpin.value() / RY_TO_EV
            self.workflowEcutEnergyDeltaRyLabel.setText(f"= {cutoff_ry:.6g} Ry")
        if hasattr(self, "workflowKpointEnergyDeltaRyLabel"):
            kpoint_ry = self.workflowKpointEnergyDeltaSpin.value() / RY_TO_EV
            self.workflowKpointEnergyDeltaRyLabel.setText(f"= {kpoint_ry:.6g} Ry")

    def _workflow_refresh_convergence_sections(self, _checked: Optional[bool] = None) -> None:
        cutoff_checked = self.workflowCutoffConvergenceCheckbox.isChecked()
        kpoint_checked = self.workflowKpointConvergenceCheckbox.isChecked()
        if not cutoff_checked and not kpoint_checked:
            sender = self.sender()
            checkbox = sender if isinstance(sender, QCheckBox) else self.workflowCutoffConvergenceCheckbox
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
            cutoff_checked = self.workflowCutoffConvergenceCheckbox.isChecked()
            kpoint_checked = self.workflowKpointConvergenceCheckbox.isChecked()

        self.workflowCutoffConvergenceGroup.setVisible(cutoff_checked)
        self.workflowKpointConvergenceGroup.setVisible(kpoint_checked)
        if kpoint_checked:
            self._workflow_update_kpoint_mesh_preview()
        self._workflow_fit_options_tab_height()

    def _workflow_update_kpoint_mesh_preview(self) -> None:
        if not hasattr(self, "workflowKpointMeshPreviewLabel"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowKpointMeshPreviewLabel.setText("Mesh: select a structure file")
            return
        spacing = float(self.workflowKpointStartSpacingSpin.value())
        if spacing <= 0.0:
            self.workflowKpointMeshPreviewLabel.setText("Mesh: waiting for structure-derived spacing")
            return
        mesh = self._workflow_kmesh_preview(source_path, spacing)
        if mesh is None:
            self.workflowKpointMeshPreviewLabel.setText("Mesh: unavailable from selected file")
            return

        total = mesh[0] * mesh[1] * mesh[2]
        limit = int(self.workflowKpointMaxMeshSpin.value())
        suffix = f"total {total}"
        if total > limit:
            suffix += f", above {limit} limit"
        self.workflowKpointMeshPreviewLabel.setText(
            f"Mesh: {mesh[0]} x {mesh[1]} x {mesh[2]} ({suffix})"
        )

    def _workflow_update_fermi_surface_kmesh_preview(self) -> None:
        if not hasattr(self, "workflowFermiSurfaceKmeshPreviewLabel"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowFermiSurfaceKmeshPreviewLabel.setText("Mesh: select a structure file")
            return
        spacing = float(self.workflowFermiSurfaceKpointSpacingSpin.value())
        if spacing <= 0.0:
            self.workflowFermiSurfaceKmeshPreviewLabel.setText("Mesh: waiting for structure-derived spacing")
            return
        mesh = self._workflow_kmesh_preview(source_path, spacing)
        if mesh is None:
            self.workflowFermiSurfaceKmeshPreviewLabel.setText("Mesh: unavailable from selected file")
            return

        total = mesh[0] * mesh[1] * mesh[2]
        self.workflowFermiSurfaceKmeshPreviewLabel.setText(
            f"Mesh: {mesh[0]} x {mesh[1]} x {mesh[2]} (total {total})"
        )

    def _workflow_charge_density_default_prefix(self) -> str:
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if source_path:
            try:
                stem = Path(source_path).expanduser().stem
            except Exception:
                stem = ""
        else:
            stem = ""
        if not stem and hasattr(self, "workflowJobNameLineEdit"):
            stem = self.workflowJobNameLineEdit.text().strip()
        return sanitize_job_name(stem or "charge_density").replace("-", "_").replace(".", "_")

    def _workflow_charge_density_default_fileout(self) -> str:
        return f"{self._workflow_charge_density_default_prefix()}_rho.cube"

    def _workflow_update_charge_density_fileout(self) -> None:
        if not hasattr(self, "workflowChargeDensityFileoutLineEdit"):
            return
        current = self.workflowChargeDensityFileoutLineEdit.text().strip()
        dirty = bool(getattr(self, "_workflowChargeDensityFileoutDirty", False))
        if dirty and "{prefix}" not in current:
            return
        self.workflowChargeDensityFileoutLineEdit.blockSignals(True)
        try:
            self.workflowChargeDensityFileoutLineEdit.setText(self._workflow_charge_density_default_fileout())
        finally:
            self.workflowChargeDensityFileoutLineEdit.blockSignals(False)
        self._workflowChargeDensityFileoutDirty = False

    def _workflow_update_charge_density_kmesh_preview(self) -> None:
        if not hasattr(self, "workflowChargeDensityKmeshPreviewLabel"):
            return
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self.workflowChargeDensityKmeshPreviewLabel.setText("Mesh: select a structure file")
            return
        spacing = float(self.workflowChargeDensityKpointSpacingSpin.value())
        if spacing <= 0.0:
            self.workflowChargeDensityKmeshPreviewLabel.setText("Mesh: waiting for structure-derived spacing")
            return
        mesh = self._workflow_kmesh_preview(source_path, spacing)
        if mesh is None:
            self.workflowChargeDensityKmeshPreviewLabel.setText("Mesh: unavailable from selected file")
            return

        total = mesh[0] * mesh[1] * mesh[2]
        self.workflowChargeDensityKmeshPreviewLabel.setText(
            f"Mesh: {mesh[0]} x {mesh[1]} x {mesh[2]} (total {total})"
        )

    def _workflow_kmesh_preview(self, source_path: str, spacing: float) -> Optional[Tuple[int, int, int]]:
        cell = self._workflow_cell_for_kmesh_preview(source_path)
        if not cell:
            return None
        return self._workflow_kmesh_from_cell(cell, spacing)

    def _workflow_cell_for_kmesh_preview(self, source_path: str) -> Optional[List[List[float]]]:
        path = Path(source_path).expanduser()
        cache = getattr(self, "_workflowKpointPreviewCellCache", None)
        if cache and cache[0] == str(path):
            return cache[1]

        cell: Optional[List[List[float]]] = None
        if path.suffix.lower() in {".in", ".pwi", ".pw", ".qe", ".txt"}:
            cell = self._workflow_cell_from_qe_input(path)
        if cell is None:
            cell = self._workflow_cell_from_ase_structure(path)
        if cell is not None:
            self._workflowKpointPreviewCellCache = (str(path), cell)
        return cell

    def _workflow_cell_from_qe_input(self, path: Path) -> Optional[List[List[float]]]:
        try:
            seed = parse_scf_seed(str(path))
        except Exception:
            return None
        rows = seed.cell_parameters.get("rows") or []
        if len(rows) < 3:
            return None

        raw_cell: List[List[float]] = []
        for row in rows[:3]:
            if len(row) < 3:
                return None
            vector = [self._workflow_parse_float(row[index]) for index in range(3)]
            if any(value is None for value in vector):
                return None
            raw_cell.append([float(value) for value in vector if value is not None])

        scale = self._workflow_qe_cell_scale(seed)
        if scale is None:
            return None
        return [[component * scale for component in row] for row in raw_cell]

    def _workflow_qe_cell_scale(self, seed: ScfSeedParameters) -> Optional[float]:
        mode = str(seed.cell_parameters.get("mode") or "").strip().lower()
        if not mode:
            card = seed.raw_cards.get("CELL_PARAMETERS", [])
            if card:
                header = str(card[0]).strip()
                if "(" in header and ")" in header:
                    mode = header.split("(", 1)[1].split(")", 1)[0].strip().lower()
                else:
                    parts = header.split(None, 1)
                    if len(parts) > 1:
                        mode = parts[1].strip().lower()
        if "angstrom" in mode or mode == "ang":
            return 1.0
        if "bohr" in mode:
            return 0.529177210903

        system = seed.sections.get("system", {})
        celldm = self._workflow_parse_float(system.get("celldm(1)", ""))
        if celldm is not None:
            return float(celldm) * 0.529177210903
        alat = self._workflow_parse_float(system.get("a", ""))
        if alat is not None:
            return float(alat)
        return 1.0 if mode else None

    def _workflow_cell_from_ase_structure(self, path: Path) -> Optional[List[List[float]]]:
        try:
            structure = read_structure(str(path))
            cell_obj = structure.get_cell()
        except Exception:
            return None

        try:
            rows = cell_obj.array.tolist()
        except Exception:
            rows = list(cell_obj)

        cell: List[List[float]] = []
        for row in rows[:3]:
            if len(row) < 3:
                return None
            try:
                cell.append([float(row[0]), float(row[1]), float(row[2])])
            except Exception:
                return None
        if len(cell) != 3:
            return None
        if all(self._workflow_vector_norm(row) <= 1e-12 for row in cell):
            return None
        return cell

    @staticmethod
    def _workflow_parse_float(raw: Any) -> Optional[float]:
        text = str(raw).strip().strip(",").strip("'\"")
        if not text:
            return None
        try:
            return float(text.replace("D", "E").replace("d", "E"))
        except ValueError:
            return None

    @staticmethod
    def _workflow_parse_positive_int(
        raw: Any,
        *,
        default: Optional[int] = None,
        minimum: int = 1,
    ) -> Optional[int]:
        text = str(raw).strip()
        if not text:
            return default
        try:
            value = int(text)
        except ValueError:
            return None
        if value < minimum:
            return None
        return value

    @staticmethod
    def _workflow_cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _workflow_dot(a: Sequence[float], b: Sequence[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _workflow_vector_norm(vector: Sequence[float]) -> float:
        return math.sqrt(sum(component * component for component in vector[:3]))

    def _workflow_kmesh_from_cell(
        self,
        cell: Sequence[Sequence[float]],
        spacing: float,
    ) -> Optional[Tuple[int, int, int]]:
        if spacing <= 0 or len(cell) < 3:
            return None
        a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
        volume = abs(self._workflow_dot(a_vec, self._workflow_cross(b_vec, c_vec)))
        if volume <= 1e-12:
            return None

        reciprocal_vectors = [
            [2.0 * math.pi * component / volume for component in self._workflow_cross(b_vec, c_vec)],
            [2.0 * math.pi * component / volume for component in self._workflow_cross(c_vec, a_vec)],
            [2.0 * math.pi * component / volume for component in self._workflow_cross(a_vec, b_vec)],
        ]
        return tuple(
            max(1, int(math.ceil(self._workflow_vector_norm(vector) / spacing)))
            for vector in reciprocal_vectors
        )

    def _build_workflow_conversion_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        basic_group = QGroupBox("Basic", panel)
        basic_layout = QGridLayout(basic_group)
        basic_layout.setHorizontalSpacing(12)
        basic_layout.setVerticalSpacing(8)

        electronic_control, self.workflowElectronicTypeButtonGroup, self.workflowElectronicTypeButtons = (
            self._workflow_segmented_control(
                basic_group,
                [("metal", "Metal"), ("insulator", "Insulator")],
                "metal",
            )
        )
        spin_orbit_control, self.workflowSpinOrbitButtonGroup, self.workflowSpinOrbitButtons = (
            self._workflow_segmented_control(
                basic_group,
                [("off", "Off"), ("on", "On")],
                "off",
            )
        )
        protocol_control, self.workflowProtocolButtonGroup, self.workflowProtocolButtons = (
            self._workflow_segmented_control(
                basic_group,
                [("fast", "Fast"), ("balanced", "Balanced"), ("stringent", "Stringent")],
                "balanced",
            )
        )

        basic_layout.addWidget(QLabel("Electronic type:", basic_group), 0, 0)
        basic_layout.addWidget(electronic_control, 0, 1)
        basic_layout.addWidget(QLabel("Spin-orbit coupling:", basic_group), 1, 0)
        basic_layout.addWidget(spin_orbit_control, 1, 1)
        basic_layout.addWidget(QLabel("Protocol:", basic_group), 2, 0)
        basic_layout.addWidget(protocol_control, 2, 1)
        self.workflowManualEcutwfcLineEdit = QLineEdit(basic_group)
        self.workflowManualEcutwfcLineEdit.setPlaceholderText("Protocol/convergence default")
        self.workflowManualEcutwfcLineEdit.setToolTip(
            "Optional &SYSTEM ecutwfc override in Ry. When set, ecutrho is 8 * ecutwfc."
        )
        basic_layout.addWidget(QLabel("Manual ecutwfc override (Ry):", basic_group), 3, 0)
        basic_layout.addWidget(self.workflowManualEcutwfcLineEdit, 3, 1)
        self.workflowUseMaterialsCloudServicesCheckBox = QCheckBox(
            "Use Materials Cloud services",
            basic_group,
        )
        self.workflowUseMaterialsCloudServicesCheckBox.setChecked(True)
        basic_layout.addWidget(QLabel("Conversion service:", basic_group), 4, 0)
        basic_layout.addWidget(self.workflowUseMaterialsCloudServicesCheckBox, 4, 1)
        self.workflowRelaxPressureLabel = QLabel("Pressure:", basic_group)
        self.workflowRelaxPressureLineEdit = QLineEdit("0.D0", basic_group)
        self.workflowRelaxPressureLineEdit.setToolTip("&CELL: press. Accepts decimal, E, or D notation.")
        self.workflowRelaxPressureConvThrLabel = QLabel("Pressure convergence threshold:", basic_group)
        self.workflowRelaxPressureConvThrLineEdit = QLineEdit("0.5D0", basic_group)
        self.workflowRelaxPressureConvThrLineEdit.setToolTip(
            "&CELL: press_conv_thr. Accepts decimal, E, or D notation."
        )
        basic_layout.addWidget(self.workflowRelaxPressureLabel, 5, 0)
        basic_layout.addWidget(self.workflowRelaxPressureLineEdit, 5, 1)
        basic_layout.addWidget(self.workflowRelaxPressureConvThrLabel, 6, 0)
        basic_layout.addWidget(self.workflowRelaxPressureConvThrLineEdit, 6, 1)
        basic_layout.setColumnStretch(1, 1)
        layout.addWidget(basic_group)
        for button in self.workflowSpinOrbitButtons.values():
            button.toggled.connect(lambda _checked: self._workflow_pseudo_requirements_changed())
        self.workflowUseMaterialsCloudServicesCheckBox.toggled.connect(
            lambda _checked: self._workflow_conversion_service_changed(),
        )

        self.workflowConversionAdvancedToggle = QPushButton("Advanced options", panel)
        self.workflowConversionAdvancedToggle.setObjectName("workflowSegmentButton")
        self.workflowConversionAdvancedToggle.setCheckable(True)
        self.workflowConversionAdvancedToggle.setChecked(False)
        self.workflowConversionAdvancedToggle.setMinimumHeight(36)
        self.workflowConversionAdvancedToggle.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self.workflowConversionAdvancedToggle)

        advanced_group = QGroupBox("Advanced", panel)
        self.workflowConversionAdvancedGroup = advanced_group
        advanced_form = QFormLayout(advanced_group)

        self.workflowFileFormatCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.workflowFileFormatCombo,
            [
                ("CIF File (.cif) [parser: ase]", "cif-ase"),
                ("CIF File (.cif) [parser: pymatgen]", "cif-pymatgen"),
                ("Quantum ESPRESSO input [parser: qetools]", "qeinp-qetools"),
                ("XCrySDen (.xsf) [parser: ase]", "xsf-ase"),
                ("VASP POSCAR [parser: ase]", "vasp-ase"),
                ("CASTEP (.cell) [parser: ase]", "castep-ase"),
                ("Protein Data Bank format (.pdb)", "pdb-ase"),
                ("Auto-detect", "auto"),
            ],
        )

        self.workflowKpointsCombo = QComboBox(advanced_group)
        self.workflowKpointsCombo.addItem("fine (0.10 1/Angstrom, 0.0125 Ry)", ("fine", 0.10))
        self.workflowKpointsCombo.addItem("medium (0.15 1/Angstrom, 0.0200 Ry)", ("medium", 0.15))
        self.workflowKpointsCombo.addItem("coarse (0.30 1/Angstrom, 0.0275 Ry)", ("coarse", 0.30))
        self.workflowKpointsCombo.addItem("Protocol default (local: 0.10 1/Angstrom)", ("", 0.10))

        self.workflowRefineCellCombo = QComboBox(advanced_group)
        self._set_combo_items(self.workflowRefineCellCombo, [("Yes", "yes"), ("No", "no")])

        self.workflowSymprecCombo = QComboBox(advanced_group)
        for exponent in range(-2, -9, -1):
            self.workflowSymprecCombo.addItem(f"10^{exponent} Angstrom", str(exponent))

        self.workflowXcCombo = QComboBox(advanced_group)
        self._set_combo_items(self.workflowXcCombo, [("PBEsol", "PBEsol"), ("PBE", "PBE")])

        self.workflowMagnetismCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.workflowMagnetismCombo,
            [
                ("non-magnetic metal (fractional occupations)", "non-magnetic-fractional"),
                ("non-magnetic insulator (fixed occupations)", "non-magnetic-fixed"),
                ("magnetic (fractional occupations)", "magnetic"),
            ],
        )

        self.workflowPseudoOverrideCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.workflowPseudoOverrideCombo,
            [
                ("Protocol default", ""),
                ("SSSP 1.3: Efficiency", "efficiency"),
                ("SSSP 1.3: Precision", "precision"),
            ],
        )

        self.workflowThresholdOverrideCombo = QComboBox(advanced_group)
        self._set_combo_items(
            self.workflowThresholdOverrideCombo,
            [
                ("Protocol default", ""),
                ("loose", "loose"),
                ("intermediate", "intermediate"),
                ("tight", "tight"),
            ],
        )

        for combo in [
            self.workflowFileFormatCombo,
            self.workflowKpointsCombo,
            self.workflowRefineCellCombo,
            self.workflowSymprecCombo,
            self.workflowXcCombo,
            self.workflowMagnetismCombo,
            self.workflowPseudoOverrideCombo,
            self.workflowThresholdOverrideCombo,
        ]:
            self._workflow_prepare_advanced_combo(combo)

        advanced_form.addRow("Select the file format:", self.workflowFileFormatCombo)
        advanced_form.addRow("K-point distance:", self.workflowKpointsCombo)
        advanced_form.addRow("Refine cell:", self.workflowRefineCellCombo)
        advanced_form.addRow("Symmetry tolerance:", self.workflowSymprecCombo)
        advanced_form.addRow("Select the XC functional:", self.workflowXcCombo)
        advanced_form.addRow("Select the magnetism:", self.workflowMagnetismCombo)
        advanced_form.addRow("Override the pseudopotential:", self.workflowPseudoOverrideCombo)
        advanced_form.addRow("Override energy/forces threshold:", self.workflowThresholdOverrideCombo)
        advanced_group.setVisible(False)
        self.workflowConversionAdvancedToggle.toggled.connect(
            self._workflow_toggle_conversion_advanced,
        )
        self._workflow_refresh_conversion_service_controls()
        self._workflow_refresh_relaxation_pressure_controls()
        layout.addWidget(advanced_group)
        layout.addStretch()
        return panel

    def _build_workflow_pseudos_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        source_group = QGroupBox("Structure species", panel)
        source_layout = QVBoxLayout(source_group)
        self.workflowPseudoSpeciesLabel = QLabel("", source_group)
        self.workflowPseudoSpeciesLabel.setWordWrap(True)
        source_layout.addWidget(self.workflowPseudoSpeciesLabel)
        layout.addWidget(source_group)

        self.workflowPseudosAssignmentsGroup = QGroupBox("Pseudopotential assignments", panel)
        assignments_layout = QVBoxLayout(self.workflowPseudosAssignmentsGroup)
        self.workflowPseudosSummaryLabel = QLabel("", self.workflowPseudosAssignmentsGroup)
        self.workflowPseudosSummaryLabel.setWordWrap(True)
        assignments_layout.addWidget(self.workflowPseudosSummaryLabel)

        self.workflowPseudosTable = QTableWidget(0, 3, self.workflowPseudosAssignmentsGroup)
        self.workflowPseudosTable.setHorizontalHeaderLabels(
            ["Species", "Pseudopotential", "Full-relativistic pseudopotential"]
        )
        self.workflowPseudosTable.verticalHeader().setVisible(False)
        self.workflowPseudosTable.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.workflowPseudosTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.workflowPseudosTable.setAlternatingRowColors(True)
        table_header = self.workflowPseudosTable.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.workflowPseudosTable.setMaximumHeight(180)
        assignments_layout.addWidget(self.workflowPseudosTable)

        pseudo_actions = QHBoxLayout()
        self.workflowSelectPseudosButton = QPushButton(
            "Select pseudopotentials...",
            self.workflowPseudosAssignmentsGroup,
        )
        self.workflowClearPseudosButton = QPushButton("Clear", self.workflowPseudosAssignmentsGroup)
        pseudo_actions.addWidget(self.workflowSelectPseudosButton)
        pseudo_actions.addWidget(self.workflowClearPseudosButton)
        pseudo_actions.addStretch()
        assignments_layout.addLayout(pseudo_actions)
        layout.addWidget(self.workflowPseudosAssignmentsGroup)

        self.workflowSelectPseudosButton.clicked.connect(self._workflow_open_pseudo_selector)
        self.workflowClearPseudosButton.clicked.connect(self._workflow_clear_pseudo_selection)

        self._workflow_refresh_pseudos_tab()
        layout.addStretch()
        return panel

    def _workflow_refresh_pseudos_tab(self) -> None:
        if not hasattr(self, "workflowUseMaterialsCloudServicesCheckBox"):
            return

        species = self._workflow_structure_species()
        self._workflow_species_order = [
            normalize_element_symbol(entry.element)
            for entry in species
            if normalize_element_symbol(entry.element)
        ]
        self._workflow_local_pseudo_files = self._workflow_pruned_pseudo_mapping(
            self._workflow_local_pseudo_files,
        )
        self._workflow_full_relativistic_pseudo_files = self._workflow_pruned_pseudo_mapping(
            self._workflow_full_relativistic_pseudo_files,
        )

        species_text = ", ".join(self._workflow_species_order)
        if species_text:
            self.workflowPseudoSpeciesLabel.setText(f"Detected species: {species_text}")
        else:
            self.workflowPseudoSpeciesLabel.setText("Detected species will appear after a structure is selected.")

        use_materials_cloud = self._workflow_materials_cloud_services_enabled()
        spin_orbit_enabled = self._workflow_spin_orbit_applies_to_selected_workflows()
        self.workflowPseudosSummaryLabel.setText(
            self._workflow_pseudo_assignment_summary(
                require_pseudopotentials=not use_materials_cloud,
                require_full_relativistic=spin_orbit_enabled,
            )
        )
        self._workflow_refresh_pseudo_assignment_table(
            require_pseudopotentials=not use_materials_cloud,
            require_full_relativistic=spin_orbit_enabled,
        )
        self.workflowClearPseudosButton.setEnabled(
            bool(self._workflow_local_pseudo_files or self._workflow_full_relativistic_pseudo_files)
        )

    def _workflow_pseudo_requirements_changed(self) -> None:
        self._workflow_refresh_pseudos_tab()
        self._workflow_refresh_option_tabs()

    def _workflow_conversion_service_changed(self) -> None:
        self._workflow_refresh_conversion_service_controls()
        self._workflow_pseudo_requirements_changed()

    def _workflow_pseudos_tab_needed(self) -> bool:
        return (
            not self._workflow_materials_cloud_services_enabled()
            or self._workflow_spin_orbit_applies_to_selected_workflows()
        )

    def _workflow_materials_cloud_services_enabled(self) -> bool:
        checkbox = getattr(self, "workflowUseMaterialsCloudServicesCheckBox", None)
        return bool(checkbox and checkbox.isChecked())

    def _workflow_pseudo_assignment_summary(
        self,
        *,
        require_pseudopotentials: bool,
        require_full_relativistic: bool,
    ) -> str:
        if require_pseudopotentials and require_full_relativistic:
            return (
                "Select one pseudopotential and one full-relativistic pseudopotential "
                "for each detected species."
            )
        if require_full_relativistic:
            return (
                "Materials Cloud services provide the workflow pseudopotentials. "
                "Select one full-relativistic pseudopotential for each detected species."
            )
        return "Select one pseudopotential for each detected species."

    def _workflow_refresh_pseudo_assignment_table(
        self,
        *,
        require_pseudopotentials: bool,
        require_full_relativistic: bool,
    ) -> None:
        table = self.workflowPseudosTable
        table.setRowCount(len(self._workflow_species_order))
        table.setColumnHidden(1, not require_pseudopotentials)
        table.setColumnHidden(2, not require_full_relativistic)
        for row, element in enumerate(self._workflow_species_order):
            species_item = QTableWidgetItem(element)
            species_item.setFlags(species_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, species_item)
            table.setItem(
                row,
                1,
                self._workflow_pseudo_path_item(self._workflow_local_pseudo_files.get(element)),
            )
            table.setItem(
                row,
                2,
                self._workflow_pseudo_path_item(self._workflow_full_relativistic_pseudo_files.get(element)),
            )
        table.resizeRowsToContents()

    @staticmethod
    def _workflow_pseudo_path_item(raw_path: Optional[str]) -> QTableWidgetItem:
        path = Path(raw_path).expanduser() if raw_path else None
        item = QTableWidgetItem(path.name if path is not None else "Not selected")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if path is not None:
            item.setToolTip(str(path))
        else:
            item.setForeground(QColor("#b42318"))
        return item

    def _workflow_open_pseudo_selector(self) -> None:
        species = self._workflow_structure_species()
        if not species:
            self._show_error("Select a structure with detectable atomic species first.")
            return

        require_pseudopotentials = not self._workflow_materials_cloud_services_enabled()
        require_full_relativistic = self._workflow_spin_orbit_applies_to_selected_workflows()
        dialog = WorkflowPseudoSelectorDialog(
            self.system_config,
            species,
            require_pseudopotentials=require_pseudopotentials,
            require_full_relativistic=require_full_relativistic,
            existing_pseudopotentials=self._workflow_local_pseudo_files,
            existing_full_relativistic=self._workflow_full_relativistic_pseudo_files,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if require_pseudopotentials:
            self._workflow_local_pseudo_files = self._workflow_pruned_pseudo_mapping(
                dialog.selected_pseudopotentials(),
            )
        if require_full_relativistic:
            self._workflow_full_relativistic_pseudo_files = self._workflow_pruned_pseudo_mapping(
                dialog.selected_full_relativistic_pseudopotentials(),
            )
        self._workflow_refresh_pseudos_tab()
        selected_count = len(self._workflow_local_pseudo_files) + len(self._workflow_full_relativistic_pseudo_files)
        self.statusbar.showMessage(f"Selected {selected_count} workflow pseudopotential file(s).", 4000)

    def _workflow_clear_pseudo_selection(self) -> None:
        self._workflow_local_pseudo_files.clear()
        self._workflow_full_relativistic_pseudo_files.clear()
        self._workflow_refresh_pseudos_tab()

    def _workflow_pruned_pseudo_mapping(self, mapping: Dict[str, str]) -> Dict[str, str]:
        allowed = set(self._workflow_species_order)
        pruned: Dict[str, str] = {}
        for element, raw_path in (mapping or {}).items():
            symbol = normalize_element_symbol(element)
            if not symbol or (allowed and symbol not in allowed) or not raw_path:
                continue
            pruned[symbol] = str(raw_path)

        if not self._workflow_species_order:
            return pruned
        ordered: Dict[str, str] = {}
        for element in self._workflow_species_order:
            if element in pruned:
                ordered[element] = pruned[element]
        return ordered

    def _workflow_structure_species(self) -> List[AtomicSpeciesEntry]:
        raw_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not raw_path:
            return []

        path = Path(raw_path).expanduser()
        if not path.is_file():
            return []

        try:
            qe_species = self._extract_atomic_species(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            qe_species = []
        if qe_species:
            return qe_species

        try:
            structure = read_structure(str(path))
            symbols = structure.get_chemical_symbols()
        except Exception:
            return []

        species: List[AtomicSpeciesEntry] = []
        seen: set[str] = set()
        for raw_symbol in symbols:
            symbol = normalize_element_symbol(str(raw_symbol))
            if not symbol or symbol in seen:
                continue
            species.append(AtomicSpeciesEntry(element=symbol))
            seen.add(symbol)
        return species

    def _workflow_spin_orbit_enabled(self) -> bool:
        return self._workflow_checked_button_key(
            getattr(self, "workflowSpinOrbitButtons", {}),
            "off",
        ) == "on"

    def _workflow_spin_orbit_applies_to_selected_workflows(self) -> bool:
        if not self._workflow_spin_orbit_enabled():
            return False
        if self._selected_relaxation_mode() in {"atomic_positions", "full_geometry"}:
            return True
        selected = set(self._selected_workflow_keys())
        return bool(
            selected
            & {
                "convergence",
                "bands",
                "band",
                "fat_bands",
                "fat_band",
                "fatbands",
                "pdos",
                "dos",
                "wannier",
                "wannier90",
                "winner90",
            }
        )

    @staticmethod
    def _workflow_non_full_relativistic_pseudos(mapping: Dict[str, str]) -> List[Tuple[str, str]]:
        invalid: List[Tuple[str, str]] = []
        for element, raw_path in mapping.items():
            path = Path(raw_path).expanduser()
            if not path.is_file() or pseudo_relativistic_type(path).lower() != "full":
                invalid.append((element, raw_path))
        return invalid

    def _workflow_toggle_conversion_advanced(self, checked: bool) -> None:
        checked = bool(checked and self._workflow_materials_cloud_services_enabled())
        if hasattr(self, "workflowConversionAdvancedGroup"):
            self.workflowConversionAdvancedGroup.setVisible(checked)
        if hasattr(self, "workflowConversionAdvancedToggle"):
            self.workflowConversionAdvancedToggle.setText(
                "Hide advanced options" if checked else "Advanced options"
            )
        self._workflow_fit_options_tab_height()

    def _workflow_refresh_conversion_service_controls(self) -> None:
        if not hasattr(self, "workflowConversionAdvancedToggle"):
            return
        materials_cloud_enabled = self._workflow_materials_cloud_services_enabled()
        self.workflowConversionAdvancedToggle.setEnabled(materials_cloud_enabled)
        if not materials_cloud_enabled:
            self.workflowConversionAdvancedToggle.setChecked(False)
        self._workflow_toggle_conversion_advanced(
            materials_cloud_enabled and self.workflowConversionAdvancedToggle.isChecked(),
        )

    def _workflow_relaxation_pressure_controls_visible(self) -> bool:
        return self._selected_relaxation_mode() in {"atomic_positions", "full_geometry"}

    def _workflow_refresh_relaxation_pressure_controls(self) -> None:
        visible = self._workflow_relaxation_pressure_controls_visible()
        for widget_name in (
            "workflowRelaxPressureLabel",
            "workflowRelaxPressureLineEdit",
            "workflowRelaxPressureConvThrLabel",
            "workflowRelaxPressureConvThrLineEdit",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(visible)
        self._workflow_fit_options_tab_height()

    def _workflow_relaxation_pressure_options(self) -> Dict[str, str]:
        if not self._workflow_relaxation_pressure_controls_visible():
            return {}
        pressure_text = (
            self.workflowRelaxPressureLineEdit.text()
            if hasattr(self, "workflowRelaxPressureLineEdit")
            else ""
        )
        pressure_conv_thr_text = (
            self.workflowRelaxPressureConvThrLineEdit.text()
            if hasattr(self, "workflowRelaxPressureConvThrLineEdit")
            else ""
        )
        return {
            "relax_pressure": normalize_qe_fortran_number(
                pressure_text,
                default="0.D0",
                label="pressure",
            ),
            "relax_pressure_conv_thr": normalize_qe_fortran_number(
                pressure_conv_thr_text,
                default="0.5D0",
                label="pressure convergence threshold",
            ),
        }

    def _workflow_prepare_advanced_combo(self, combo: QComboBox) -> None:
        combo.setObjectName("workflowAdvancedCombo")
        combo.insertItem(0, "Select option...", "__placeholder__")
        combo.setCurrentIndex(0)
        self._workflow_style_combo_popup(combo)
        self._workflow_update_combo_state(combo)
        combo.currentIndexChanged.connect(lambda _index, watched=combo: self._workflow_update_combo_state(watched))

    def _workflow_style_combo_popup(self, combo: QComboBox) -> None:
        view = combo.view()
        if view is None:
            return
        palette = view.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_DARK))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(PRIMARY_COLOR))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_LIGHT))
        view.setPalette(palette)
        view.setMouseTracking(True)
        delegate = WorkflowComboPopupDelegate(view)
        view.setItemDelegate(delegate)
        combo._workflow_popup_delegate = delegate  # keep delegate alive for the popup view
        view.setStyleSheet(
            f"""
            QAbstractItemView {{
                background-color: #FFFFFF;
                color: {TEXT_DARK};
                border: 1px solid {NEUTRAL_BORDER};
                outline: 0;
            }}
            QAbstractItemView::item {{
                color: {TEXT_DARK};
                background-color: #FFFFFF;
                min-height: 26px;
                padding: 4px 8px;
            }}
            QAbstractItemView::item:hover,
            QAbstractItemView::item:selected {{
                background-color: {PRIMARY_COLOR};
                color: {TEXT_LIGHT};
            }}
            """
        )

    def _workflow_update_combo_state(self, combo: QComboBox) -> None:
        has_selection = combo.currentData() != "__placeholder__"
        combo.setProperty("hasSelection", has_selection)
        combo.style().unpolish(combo)
        combo.style().polish(combo)
        combo.update()

    @staticmethod
    def _workflow_combo_value(combo: QComboBox, default: Any = "") -> Any:
        data = combo.currentData()
        if data == "__placeholder__":
            return default
        return data if data is not None else combo.currentText()

    def _workflow_segmented_control(
        self,
        parent: QWidget,
        options: Sequence[Tuple[str, str]],
        default_key: str,
    ) -> tuple[QWidget, QButtonGroup, Dict[str, QPushButton]]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        group = QButtonGroup(container)
        group.setExclusive(True)
        buttons: Dict[str, QPushButton] = {}
        for index, (key, label) in enumerate(options):
            button = QPushButton(label, container)
            button.setObjectName("workflowSegmentButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if key == default_key:
                button.setChecked(True)
            group.addButton(button, index)
            buttons[key] = button
            layout.addWidget(button)
        return container, group, buttons

    @staticmethod
    def _workflow_checked_button_key(buttons: Dict[str, QPushButton], default: str) -> str:
        for key, button in buttons.items():
            if button.isChecked():
                return key
        return default

    def _handle_workflow_structure_browse(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select workflow structure",
            str(PROJECT_ROOT),
            "Structure or QE input (*.cif *.in *.txt);;CIF files (*.cif);;QE input (*.in *.txt);;All files (*)",
        )
        if not file_path:
            return
        self._set_workflow_structure_file(file_path)

    def _set_workflow_structure_file(self, file_path: str) -> None:
        path = Path(file_path).expanduser()
        if not path.is_file():
            self._show_error("The selected structure file does not exist.")
            return

        self.workflowStructurePathLineEdit.setText(str(path))
        self.workflowSourceInputPath = str(path)
        self._workflowKpointPreviewCellCache = None
        self._workflowBandsPathSource = ""
        self._workflowBandsPathGroups = []
        self._workflowBandsPathPoints = {}
        self._workflowPhononPathSource = ""
        self._workflowPhononPathGroups = []
        self._workflowPhononPathPoints = {}
        self._workflowWannierPathSource = ""
        self._workflowWannierPathGroups = []
        self._workflowWannierPathPoints = {}
        if not self.workflowJobNameLineEdit.text().strip():
            self.workflowJobNameLineEdit.setText(path.stem)
        if hasattr(self, "workflowOutputDirectoryLineEdit") and not self.workflowOutputDirectoryLineEdit.text().strip():
            self.workflowOutputDirectoryLineEdit.setText(str(path.parent))
        self.workflowStructureStatusLabel.clear()
        self._workflow_refresh_option_tabs()
        self._workflow_refresh_pseudos_tab()
        self._workflow_update_kpoint_mesh_preview()
        self._workflow_update_fermi_surface_kmesh_preview()
        self._workflow_update_charge_density_fileout()
        self._workflow_update_charge_density_kmesh_preview()
        self.statusbar.showMessage("Workflow structure selected", 3000)

    def _workflow_convert_cif_to_source(self, cif_path: Path) -> None:
        output_path = self._default_output_path(str(cif_path))
        self.workflowStructureStatusLabel.setText("Converting structure to pw.x input...")
        conversion_options = self._workflow_conversion_options()

        try:
            output_path = _generate_workflow_qe_input(
                source_path=cif_path,
                output_path=output_path,
                options=conversion_options,
            )
            content = output_path.read_text(encoding="utf-8", errors="ignore")
            trimmed = trim_qe_input(content)
            if trimmed != content:
                save_text_to_file(output_path, trimmed)
        except CalledProcessError as exc:
            details = exc.stderr or exc.stdout or ""
            self._show_error(f"Workflow conversion failed with exit code {exc.returncode}.\n{details}")
            self.workflowStructureStatusLabel.setText("Conversion failed.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Workflow conversion failed: {exc}")
            self.workflowStructureStatusLabel.setText("Conversion failed.")
            return

        self.workflowSourceInputPath = str(output_path)
        if not self.workflowJobNameLineEdit.text().strip():
            self.workflowJobNameLineEdit.setText(output_path.stem)
        self.workflowStructureStatusLabel.clear()
        self._workflow_refresh_option_tabs()
        self._workflow_refresh_pseudos_tab()
        self._workflow_update_fermi_surface_kmesh_preview()
        self._workflow_update_charge_density_fileout()
        self._workflow_update_charge_density_kmesh_preview()
        self.statusbar.showMessage("Workflow conversion finished", 4000)

    def _handle_workflow_output_directory_browse(self) -> None:
        start_dir = self.workflowOutputDirectoryLineEdit.text().strip()
        if not start_dir:
            source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
            if source_path:
                try:
                    start_dir = str(Path(source_path).expanduser().parent)
                except Exception:
                    start_dir = ""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select workflow output directory",
            start_dir or str(PROJECT_ROOT),
        )
        if directory:
            self.workflowOutputDirectoryLineEdit.setText(directory)
            self.statusbar.showMessage("Workflow output directory selected", 3000)

    def _handle_workflow_save_template(self) -> None:
        suggested_name = sanitize_job_name(self.workflowJobNameLineEdit.text().strip() or "workflow")
        suggested_dir = self.workflowOutputDirectoryLineEdit.text().strip()
        if not suggested_dir:
            suggested_dir = str(PROJECT_ROOT)
        suggested_path = Path(suggested_dir).expanduser() / f"{suggested_name}.workflow.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save workflow draft",
            str(suggested_path),
            "Workflow draft (*.workflow.json *.json);;All files (*)",
        )
        if not file_path:
            return

        payload = {
            "schema": "qe_gui.workflow_draft",
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "workflow": self._workflow_current_draft_payload(),
        }
        try:
            Path(file_path).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not save workflow draft: {exc}")
            return
        self.statusbar.showMessage(f"Workflow draft saved: {file_path}", 5000)

    def _handle_workflow_load_template(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load workflow draft or manifest",
            str(PROJECT_ROOT),
            "Workflow JSON (*.workflow.json workflow_manifest.json *.json);;All files (*)",
        )
        if not file_path:
            return
        self._workflow_load_recipe_from_file(Path(file_path).expanduser())

    def _workflow_current_draft_payload(self) -> Dict[str, Any]:
        return {
            "source_structure": str(getattr(self, "workflowSourceInputPath", "") or self.workflowStructurePathLineEdit.text()).strip(),
            "job_name": self.workflowJobNameLineEdit.text().strip(),
            "output_directory": self.workflowOutputDirectoryLineEdit.text().strip(),
            "description": self.workflowDescriptionEdit.toPlainText().strip(),
            "relaxation_mode": self._selected_relaxation_mode(),
            "selected_workflows": self._selected_workflow_keys(),
            "conversion": self._workflow_draft_conversion_options(),
            "convergence": self._workflow_convergence_options(),
            "bands": self._workflow_band_options(),
            "wannier90": self._workflow_wannier_options(),
            "dos": self._workflow_dos_options(),
            "fermi_surface": self._workflow_fermi_surface_options(),
            "charge_density": self._workflow_charge_density_options(),
            "elastic": self._workflow_elastic_options(),
            "phonon": self._workflow_phonon_options(),
        }

    def _workflow_draft_conversion_options(self) -> Dict[str, Any]:
        kpoint_data = self.workflowKpointsCombo.currentData()
        kmesh_override = ""
        kpoint_distance = None
        if kpoint_data != "__placeholder__":
            if isinstance(kpoint_data, tuple):
                kmesh_override = str(kpoint_data[0] or "")
                try:
                    kpoint_distance = float(kpoint_data[1])
                except (TypeError, ValueError):
                    kpoint_distance = None
            else:
                try:
                    kpoint_distance = float(kpoint_data)
                except (TypeError, ValueError):
                    kpoint_distance = None
        tolerance = ""
        tolerance_data = self.workflowSymprecCombo.currentData()
        if tolerance_data != "__placeholder__":
            tolerance = str(tolerance_data)
        manual_text = self.workflowManualEcutwfcLineEdit.text().strip() if hasattr(self, "workflowManualEcutwfcLineEdit") else ""
        manual_value = None
        if manual_text:
            try:
                manual_value = float(manual_text)
            except ValueError:
                manual_value = None
        return {
            "electronic_type": self._workflow_checked_button_key(getattr(self, "workflowElectronicTypeButtons", {}), "metal"),
            "spin_orbit": self._workflow_checked_button_key(getattr(self, "workflowSpinOrbitButtons", {}), "off"),
            "protocol": self._workflow_checked_button_key(getattr(self, "workflowProtocolButtons", {}), "balanced"),
            "use_materials_cloud": self._workflow_materials_cloud_services_enabled(),
            "file_format": self._workflow_combo_value(self.workflowFileFormatCombo, ""),
            "kpoint_distance": kpoint_distance,
            "kmesh_override": kmesh_override,
            "refine_cell": self._workflow_combo_value(self.workflowRefineCellCombo, ""),
            "tolerance": tolerance,
            "xc": self._workflow_combo_value(self.workflowXcCombo, DEFAULT_XC) or DEFAULT_XC,
            "magnetism": self._workflow_combo_value(self.workflowMagnetismCombo, ""),
            "pseudo_override": self._workflow_combo_value(self.workflowPseudoOverrideCombo, ""),
            "threshold_override": self._workflow_combo_value(self.workflowThresholdOverrideCombo, ""),
            "manual_ecutwfc": manual_value,
            "manual_ecutwfc_text": manual_text,
            "relax_pressure": self.workflowRelaxPressureLineEdit.text() if hasattr(self, "workflowRelaxPressureLineEdit") else "0.D0",
            "relax_pressure_conv_thr": (
                self.workflowRelaxPressureConvThrLineEdit.text()
                if hasattr(self, "workflowRelaxPressureConvThrLineEdit")
                else "0.5D0"
            ),
            "local_pseudo_files": dict(getattr(self, "_workflow_local_pseudo_files", {})),
            "full_relativistic_pseudo_files": dict(getattr(self, "_workflow_full_relativistic_pseudo_files", {})),
            "advanced_open": bool(
                getattr(self, "workflowConversionAdvancedToggle", None)
                and self.workflowConversionAdvancedToggle.isChecked()
            ),
        }

    def _workflow_load_recipe_from_file(self, path: Path) -> None:
        if not path.is_file():
            self._show_error("The selected workflow file does not exist.")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not read workflow JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self._show_error("Workflow JSON must contain an object.")
            return
        try:
            recipe = self._workflow_recipe_from_payload(payload, source_path=path)
            warnings_list = self._workflow_apply_recipe_to_creator(recipe)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not load workflow: {exc}")
            return
        self._workflow_show_creator_tab()
        self.statusbar.showMessage(f"Workflow loaded from {path.name}", 6000)
        if warnings_list:
            self._show_warning("Workflow loaded with warnings", "\n".join(warnings_list))

    def _workflow_recipe_from_payload(self, payload: Dict[str, Any], *, source_path: Path) -> Dict[str, Any]:
        if payload.get("schema") == "qe_gui.workflow_draft" and isinstance(payload.get("workflow"), dict):
            return dict(payload["workflow"])
        if isinstance(payload.get("workflow"), dict) and "workflow_root" not in payload:
            return dict(payload["workflow"])
        if "workflow_root" in payload or "tasks" in payload or "selected_workflows" in payload:
            return self._workflow_recipe_from_manifest(payload, manifest_path=source_path)
        raise ValueError("Unrecognized workflow JSON. Select a saved draft or workflow_manifest.json.")

    def _workflow_recipe_from_manifest(self, manifest: Dict[str, Any], *, manifest_path: Path) -> Dict[str, Any]:
        root_text = str(manifest.get("workflow_root") or "").strip()
        manifest_path = manifest_path.expanduser()
        declared_root = Path(root_text).expanduser() if root_text else None
        workflow_root = declared_root if declared_root is not None else manifest_path.parent
        if declared_root is None or not (declared_root / "workflow_manifest.json").is_file():
            workflow_root = manifest_path.parent
        else:
            try:
                if (declared_root / "workflow_manifest.json").resolve() != manifest_path.resolve():
                    workflow_root = manifest_path.parent
            except OSError:
                workflow_root = manifest_path.parent
        source_info = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        original_source = str(source_info.get("original") or source_info.get("original_path") or "").strip()
        stored_source = str(source_info.get("stored_original") or "").strip()
        source_candidates = [original_source]
        if stored_source:
            source_candidates.append(str((workflow_root / stored_source).expanduser()))
        source_candidates.append(str(workflow_root / "source" / "QE_input.template"))
        source_structure = next((candidate for candidate in source_candidates if candidate and Path(candidate).expanduser().is_file()), "")
        if not source_structure:
            source_structure = original_source or (source_candidates[1] if len(source_candidates) > 1 else "")

        selected = self._workflow_gui_selected_from_manifest(manifest.get("selected_workflows") or [])
        bands = dict(manifest.get("bands") if isinstance(manifest.get("bands"), dict) else {})
        if self._workflow_manifest_has_workflow(manifest, "fat_bands"):
            bands["calculate_fat_band"] = True
        return {
            "source_structure": source_structure,
            "job_name": str(manifest.get("name") or "").strip(),
            "output_directory": str(workflow_root.parent if workflow_root else "").strip(),
            "description": str(manifest.get("description") or "").strip(),
            "relaxation_mode": str((manifest.get("relaxation") or {}).get("mode") or "as_is"),
            "selected_workflows": selected,
            "conversion": dict(manifest.get("conversion") if isinstance(manifest.get("conversion"), dict) else {}),
            "convergence": dict(manifest.get("convergence") if isinstance(manifest.get("convergence"), dict) else {}),
            "bands": bands,
            "wannier90": dict(manifest.get("wannier90") if isinstance(manifest.get("wannier90"), dict) else {}),
            "dos": dict(manifest.get("dos") if isinstance(manifest.get("dos"), dict) else {}),
            "fermi_surface": dict(manifest.get("fermi_surface") if isinstance(manifest.get("fermi_surface"), dict) else {}),
            "charge_density": dict(manifest.get("charge_density") if isinstance(manifest.get("charge_density"), dict) else {}),
            "elastic": dict(manifest.get("elastic") if isinstance(manifest.get("elastic"), dict) else {}),
            "phonon": dict(manifest.get("phonon") if isinstance(manifest.get("phonon"), dict) else {}),
        }

    def _workflow_gui_selected_from_manifest(self, selected_workflows: Sequence[Any]) -> List[str]:
        selected: List[str] = []
        aliases = {
            "convergence": "convergence",
            "bands": "bands",
            "band": "bands",
            "fat_bands": "bands",
            "fat_band": "bands",
            "fatbands": "bands",
            "dos": "pdos",
            "pdos": "pdos",
            "fermi_surface": "fermi_surface",
            "charge_density": "charge_density",
            "elastic": "elastic",
            "phonon": "phonon",
            "wannier90": "wannier",
            "wannier": "wannier",
            "winner90": "wannier",
        }
        raw_items: Sequence[Any]
        if isinstance(selected_workflows, str):
            raw_items = selected_workflows.replace(";", ",").split(",")
        else:
            raw_items = selected_workflows or []
        for raw in raw_items:
            key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
            mapped = aliases.get(key)
            if mapped and mapped not in selected:
                selected.append(mapped)
        return selected

    def _workflow_manifest_has_workflow(self, manifest: Mapping[str, Any], key: str) -> bool:
        raw_selected = manifest.get("selected_workflows") or []
        if isinstance(raw_selected, str):
            raw_selected = raw_selected.replace(";", ",").split(",")
        selected = [str(entry).strip().lower() for entry in raw_selected]
        if key in selected:
            return True
        for task in manifest.get("tasks") or []:
            if isinstance(task, dict) and key in {
                str(task.get("id") or "").strip().lower(),
                str(task.get("type") or "").strip().lower(),
            }:
                return True
        return False

    def _workflow_apply_recipe_to_creator(self, recipe: Mapping[str, Any]) -> List[str]:
        warnings_list: List[str] = []
        source_text = str(recipe.get("source_structure") or "").strip()
        if source_text and Path(source_text).expanduser().is_file():
            self._set_workflow_structure_file(source_text)
        else:
            self.workflowStructurePathLineEdit.setText(source_text)
            self.workflowSourceInputPath = source_text
            if source_text:
                self.workflowStructureStatusLabel.setText("Loaded workflow source path, but the file is unavailable.")
                warnings_list.append(f"Source structure is unavailable: {source_text}")

        self.workflowJobNameLineEdit.setText(str(recipe.get("job_name") or "").strip())
        self.workflowOutputDirectoryLineEdit.setText(str(recipe.get("output_directory") or "").strip())
        self.workflowDescriptionEdit.setPlainText(str(recipe.get("description") or ""))
        self._workflow_set_checked_button(getattr(self, "workflowRelaxationButtons", {}), str(recipe.get("relaxation_mode") or "as_is"), "as_is")

        raw_selected = recipe.get("selected_workflows") or []
        if isinstance(raw_selected, str):
            raw_selected = raw_selected.replace(";", ",").split(",")
        selected = set(str(key).strip() for key in raw_selected if str(key).strip())
        for key, checkbox in getattr(self, "workflowSelectionCheckboxes", {}).items():
            checkbox.setChecked(key in selected)

        self._workflow_apply_conversion_options(recipe.get("conversion") if isinstance(recipe.get("conversion"), Mapping) else {})
        self._workflow_apply_convergence_options(recipe.get("convergence") if isinstance(recipe.get("convergence"), Mapping) else {})
        self._workflow_apply_band_options(recipe.get("bands") if isinstance(recipe.get("bands"), Mapping) else {}, warnings_list)
        self._workflow_apply_wannier_options(recipe.get("wannier90") if isinstance(recipe.get("wannier90"), Mapping) else {}, warnings_list)
        self._workflow_apply_dos_options(recipe.get("dos") if isinstance(recipe.get("dos"), Mapping) else {})
        self._workflow_apply_fermi_surface_options(recipe.get("fermi_surface") if isinstance(recipe.get("fermi_surface"), Mapping) else {})
        self._workflow_apply_charge_density_options(recipe.get("charge_density") if isinstance(recipe.get("charge_density"), Mapping) else {})
        self._workflow_apply_elastic_options(recipe.get("elastic") if isinstance(recipe.get("elastic"), Mapping) else {})
        self._workflow_apply_phonon_options(recipe.get("phonon") if isinstance(recipe.get("phonon"), Mapping) else {}, warnings_list)

        self._workflow_refresh_option_tabs()
        self._workflow_refresh_pseudos_tab()
        self._workflow_update_kpoint_mesh_preview()
        self._workflow_update_fermi_surface_kmesh_preview()
        self._workflow_update_charge_density_fileout()
        self._workflow_update_charge_density_kmesh_preview()
        self.workflowSubmitStatusLabel.setText("Workflow loaded. Adjust parameters and submit to queue a fresh run.")
        return warnings_list

    @staticmethod
    def _workflow_set_checked_button(buttons: Dict[str, QPushButton], value: str, default: str) -> None:
        key = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
        if key not in buttons:
            key = default
        button = buttons.get(key)
        if button is not None:
            button.setChecked(True)

    @staticmethod
    def _workflow_set_spinbox_values(spinboxes: Sequence[QSpinBox], values: Any) -> None:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return
        for spin, value in zip(spinboxes, values):
            try:
                spin.setValue(int(value))
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _workflow_set_spinbox_value(spinbox: QSpinBox, value: Any) -> None:
        try:
            spinbox.setValue(int(value))
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _workflow_set_double_spinbox_value(spinbox: QDoubleSpinBox, value: Any) -> None:
        try:
            spinbox.setValue(float(value))
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _workflow_set_combo_data(combo: QComboBox, value: Any) -> None:
        target = str(value or "")
        if not target and combo.count() > 0 and combo.itemData(0) == "__placeholder__":
            combo.setCurrentIndex(0)
            return
        for index in range(combo.count()):
            data = combo.itemData(index)
            if str(data) == target:
                combo.setCurrentIndex(index)
                return
        if combo.count() > 0 and combo.itemData(0) == "__placeholder__":
            combo.setCurrentIndex(0)

    def _workflow_set_kpoints_combo_from_options(self, options: Mapping[str, Any]) -> None:
        target_mesh = str(options.get("kmesh_override") or "")
        target_distance = options.get("kpoint_distance")
        if not target_mesh and target_distance is None and self.workflowKpointsCombo.count() > 0:
            self.workflowKpointsCombo.setCurrentIndex(0)
            return
        for index in range(self.workflowKpointsCombo.count()):
            data = self.workflowKpointsCombo.itemData(index)
            if not isinstance(data, tuple):
                continue
            mesh = str(data[0] or "")
            try:
                distance = float(data[1])
            except (TypeError, ValueError):
                distance = None
            if target_mesh and mesh == target_mesh:
                self.workflowKpointsCombo.setCurrentIndex(index)
                return
            if target_distance is not None and distance is not None and abs(float(target_distance) - distance) < 1e-9:
                self.workflowKpointsCombo.setCurrentIndex(index)
                return
        if self.workflowKpointsCombo.count() > 0:
            self.workflowKpointsCombo.setCurrentIndex(0)

    def _workflow_apply_conversion_options(self, options: Mapping[str, Any]) -> None:
        protocol = str(options.get("protocol") or "").strip()
        if not protocol:
            local_protocol = str(options.get("local_protocol") or "").strip()
            protocol = "fast" if local_protocol == "sssp_efficiency" else "stringent" if local_protocol == "sssp_precision" else "balanced"
        self._workflow_set_checked_button(getattr(self, "workflowElectronicTypeButtons", {}), str(options.get("electronic_type") or "metal"), "metal")
        self._workflow_set_checked_button(getattr(self, "workflowSpinOrbitButtons", {}), str(options.get("spin_orbit") or "off"), "off")
        self._workflow_set_checked_button(getattr(self, "workflowProtocolButtons", {}), protocol, "balanced")
        if hasattr(self, "workflowUseMaterialsCloudServicesCheckBox"):
            self.workflowUseMaterialsCloudServicesCheckBox.setChecked(bool(options.get("use_materials_cloud", True)))
        manual_text = str(options.get("manual_ecutwfc_text") or "")
        if not manual_text and options.get("manual_ecutwfc") is not None:
            manual_text = f"{options.get('manual_ecutwfc')}"
        if hasattr(self, "workflowManualEcutwfcLineEdit"):
            self.workflowManualEcutwfcLineEdit.setText(manual_text)
        if hasattr(self, "workflowRelaxPressureLineEdit") and options.get("relax_pressure") is not None:
            self.workflowRelaxPressureLineEdit.setText(str(options.get("relax_pressure")))
        if hasattr(self, "workflowRelaxPressureConvThrLineEdit") and options.get("relax_pressure_conv_thr") is not None:
            self.workflowRelaxPressureConvThrLineEdit.setText(str(options.get("relax_pressure_conv_thr")))
        for widget_name, option_key in (
            ("workflowFileFormatCombo", "file_format"),
            ("workflowRefineCellCombo", "refine_cell"),
            ("workflowSymprecCombo", "tolerance"),
            ("workflowXcCombo", "xc"),
            ("workflowMagnetismCombo", "magnetism"),
            ("workflowPseudoOverrideCombo", "pseudo_override"),
            ("workflowThresholdOverrideCombo", "threshold_override"),
        ):
            combo = getattr(self, widget_name, None)
            if isinstance(combo, QComboBox) and options.get(option_key) is not None:
                self._workflow_set_combo_data(combo, options.get(option_key))
        if hasattr(self, "workflowKpointsCombo"):
            self._workflow_set_kpoints_combo_from_options(options)
        self._workflow_local_pseudo_files = {
            normalize_element_symbol(str(element)): str(path)
            for element, path in dict(options.get("local_pseudo_files") or {}).items()
            if normalize_element_symbol(str(element)) and str(path).strip()
        }
        self._workflow_full_relativistic_pseudo_files = {
            normalize_element_symbol(str(element)): str(path)
            for element, path in dict(options.get("full_relativistic_pseudo_files") or {}).items()
            if normalize_element_symbol(str(element)) and str(path).strip()
        }
        if hasattr(self, "workflowConversionAdvancedToggle"):
            self.workflowConversionAdvancedToggle.setChecked(bool(options.get("advanced_open", False)))
        self._workflow_conversion_service_changed()

    def _workflow_apply_convergence_options(self, options: Mapping[str, Any]) -> None:
        cutoff = options.get("cutoff") if isinstance(options.get("cutoff"), Mapping) else {}
        kpoints = options.get("kpoints") if isinstance(options.get("kpoints"), Mapping) else {}
        self.workflowCutoffConvergenceCheckbox.setChecked(bool(cutoff.get("enabled", True)))
        self.workflowKpointConvergenceCheckbox.setChecked(bool(kpoints.get("enabled", False)))
        if cutoff.get("start") is not None:
            self._workflow_set_spinbox_value(self.workflowEcutMinSpin, cutoff.get("start"))
        self._workflow_set_spinbox_value(self.workflowEcutStepSpin, cutoff.get("step", self.workflowEcutStepSpin.value()))
        self._workflow_set_spinbox_value(self.workflowEcutMaxSpin, cutoff.get("maximum", self.workflowEcutMaxSpin.value()))
        if cutoff.get("energy_delta_ev") is not None:
            self._workflow_set_double_spinbox_value(self.workflowEcutEnergyDeltaSpin, cutoff.get("energy_delta_ev"))
        elif cutoff.get("energy_delta_ry") is not None:
            self._workflow_set_double_spinbox_value(self.workflowEcutEnergyDeltaSpin, float(cutoff.get("energy_delta_ry")) * RY_TO_EV)
        if kpoints.get("start_spacing") is not None:
            self._workflow_set_double_spinbox_value(self.workflowKpointStartSpacingSpin, kpoints.get("start_spacing"))
        self._workflow_set_double_spinbox_value(self.workflowKpointSpacingStepSpin, kpoints.get("spacing_step", self.workflowKpointSpacingStepSpin.value()))
        self._workflow_set_spinbox_value(self.workflowKpointMaxMeshSpin, kpoints.get("max_mesh_size", self.workflowKpointMaxMeshSpin.value()))
        if kpoints.get("energy_delta_ev") is not None:
            self._workflow_set_double_spinbox_value(self.workflowKpointEnergyDeltaSpin, kpoints.get("energy_delta_ev"))
        elif kpoints.get("energy_delta_ry") is not None:
            self._workflow_set_double_spinbox_value(self.workflowKpointEnergyDeltaSpin, float(kpoints.get("energy_delta_ry")) * RY_TO_EV)
        self._workflow_refresh_convergence_sections()

    def _workflow_apply_band_options(self, options: Mapping[str, Any], warnings_list: List[str]) -> None:
        self._workflow_set_double_spinbox_value(self.workflowBandScfKpointSpacingSpin, options.get("kpoint_spacing", self.workflowBandScfKpointSpacingSpin.value()))
        if options.get("nbnd") is not None:
            self.workflowBandsNbndLineEdit.setText(str(options.get("nbnd")))
        self.workflowBandsPathWeightLineEdit.setText(str(options.get("kpoint_weight") or 50))
        self.workflowCalculateFatBandCheckBox.setChecked(bool(options.get("calculate_fat_band", False)))
        self.workflowBandsFilbandLineEdit.setText(str(options.get("filband") or "bands.out"))
        if hasattr(self, "workflowFatBandColorCombo"):
            self._workflow_set_combo_data(self.workflowFatBandColorCombo, options.get("fatband_color_mode") or "dominant")
        self._workflow_apply_path_options("bands", options, warnings_list)
        self._workflow_refresh_bands_mode_controls()

    def _workflow_apply_wannier_options(self, options: Mapping[str, Any], warnings_list: List[str]) -> None:
        kmesh = options.get("scf_kmesh", options.get("kmesh", options.get("mp_grid")))
        self._workflow_set_spinbox_values(getattr(self, "workflowWannierKmeshSpinBoxes", []), kmesh)
        if options.get("wannier_nbnd") is not None or options.get("nbnd") is not None:
            self.workflowWannierNbndLineEdit.setText(str(options.get("wannier_nbnd", options.get("nbnd"))))
        self.workflowWannierPathWeightLineEdit.setText(str(options.get("kpoint_weight") or 50))
        self.workflowWannierWriteUnkCheckBox.setChecked(bool(options.get("write_unk", False)))
        self._workflow_set_spinbox_value(self.workflowWannierNumIterSpin, options.get("num_iter", self.workflowWannierNumIterSpin.value()))
        self._workflow_set_spinbox_value(self.workflowWannierDisNumIterSpin, options.get("dis_num_iter", self.workflowWannierDisNumIterSpin.value()))
        self._workflow_apply_path_options("wannier", options, warnings_list)
        self._workflow_update_wannier_kmesh_preview()

    def _workflow_apply_dos_options(self, options: Mapping[str, Any]) -> None:
        self._workflow_set_double_spinbox_value(self.workflowDosScfKpointSpacingSpin, options.get("scf_kpoint_spacing", self.workflowDosScfKpointSpacingSpin.value()))
        self._workflow_set_double_spinbox_value(self.workflowDosNscfKpointSpacingSpin, options.get("nscf_kpoint_spacing", self.workflowDosNscfKpointSpacingSpin.value()))
        self.workflowDosUseConvergedKmeshCheckBox.setChecked(bool(options.get("use_converged_kmesh", True)))

    def _workflow_apply_fermi_surface_options(self, options: Mapping[str, Any]) -> None:
        self._workflow_set_double_spinbox_value(self.workflowFermiSurfaceKpointSpacingSpin, options.get("kpoint_spacing", self.workflowFermiSurfaceKpointSpacingSpin.value()))

    def _workflow_apply_charge_density_options(self, options: Mapping[str, Any]) -> None:
        self._workflow_set_double_spinbox_value(self.workflowChargeDensityKpointSpacingSpin, options.get("kpoint_spacing", self.workflowChargeDensityKpointSpacingSpin.value()))
        if options.get("fileout") is not None:
            self.workflowChargeDensityFileoutLineEdit.setText(str(options.get("fileout")))
            self._workflowChargeDensityFileoutDirty = True
        self._workflow_set_spinbox_value(self.workflowChargeDensityNxSpin, options.get("nx", self.workflowChargeDensityNxSpin.value()))
        self._workflow_set_spinbox_value(self.workflowChargeDensityNySpin, options.get("ny", self.workflowChargeDensityNySpin.value()))
        self._workflow_set_spinbox_value(self.workflowChargeDensityNzSpin, options.get("nz", self.workflowChargeDensityNzSpin.value()))

    def _workflow_apply_elastic_options(self, options: Mapping[str, Any]) -> None:
        self._workflow_set_combo_data(self.workflowElasticWhatCombo, options.get("what") or "scf_elastic_constants")
        self._workflow_set_combo_data(self.workflowElasticAlgorithmCombo, options.get("elastic_algorithm") or "standard")
        self._workflow_set_spinbox_value(self.workflowElasticNgeoStrainSpin, options.get("ngeo_strain", self.workflowElasticNgeoStrainSpin.value()))
        self._workflowElasticNgeoTouched = True
        self.workflowElasticFrozenIonsCheckBox.setChecked(bool(options.get("frozen_ions", False)))
        self._workflow_refresh_elastic_controls()

    def _workflow_apply_phonon_options(self, options: Mapping[str, Any], warnings_list: List[str]) -> None:
        mode = str(options.get("mode") or options.get("calculation_mode") or "dispersion").strip().lower()
        if mode in {"epc", "epc_interpolated", "electron_phonon", "lambda", "a2f"}:
            self._workflow_set_combo_data(self.workflowPhononModeCombo, "epc")
            self._workflow_set_combo_data(self.workflowPhononEpcModeCombo, "epc_interpolated")
        else:
            self._workflow_set_combo_data(self.workflowPhononModeCombo, "single_q" if mode == "single_q" else "dispersion")
        self._workflow_set_spinbox_values(self.workflowPhononScfKmeshSpinBoxes, options.get("scf_kmesh"))
        self._workflow_set_spinbox_values(self.workflowPhononQGridSpinBoxes, options.get("nq", options.get("q_grid")))
        q_point = options.get("q_point")
        if isinstance(q_point, Sequence) and not isinstance(q_point, (str, bytes)) and len(q_point) >= 3:
            self.workflowPhononQPointLineEdit.setText(" ".join(str(q_point[index]) for index in range(3)))
        self.workflowPhononTr2PhLineEdit.setText(str(options.get("tr2_ph") or "1.0d-12"))
        self._workflow_set_double_spinbox_value(self.workflowPhononAlphaMixSpin, options.get("alpha_mix", self.workflowPhononAlphaMixSpin.value()))
        self.workflowPhononFildynLineEdit.setText(str(options.get("fildyn") or "phonon.dyn"))
        self._workflow_set_spinbox_value(self.workflowPhononNmixSpin, options.get("nmix_ph", self.workflowPhononNmixSpin.value()))
        self._workflow_set_combo_data(self.workflowPhononVerbosityCombo, options.get("verbosity") or "default")
        self.workflowPhononTransCheckBox.setChecked(bool(options.get("trans", True)))
        self.workflowPhononEpsilCheckBox.setChecked(bool(options.get("epsil", False)))
        self.workflowPhononFlfrcLineEdit.setText(str(options.get("flfrc") or "phonon.fc"))
        self.workflowPhononFlfrqLineEdit.setText(str(options.get("flfrq") or "phonon.freq.band"))
        self.workflowPhononMatdynFlvecLineEdit.setText(str(options.get("matdyn_flvec") or "phonon_band.modes"))
        self._workflow_set_combo_data(self.workflowPhononQ2rZasrCombo, options.get("q2r_zasr") or "crystal")
        self._workflow_set_combo_data(self.workflowPhononMatdynAsrCombo, options.get("matdyn_asr") or "crystal")
        self.workflowPhononBandCheckBox.setChecked(bool(options.get("band_enabled", True)))
        self.workflowPhononMatdynQInBandFormCheckBox.setChecked(bool(options.get("matdyn_q_in_band_form", True)))
        self.workflowPhononDosCheckBox.setChecked(bool(options.get("dos_enabled", True)))
        self.workflowPhononFldosLineEdit.setText(str(options.get("fldos") or "phonon.phdos"))
        self.workflowPhononPhdosFlvecLineEdit.setText(str(options.get("phdos_flvec") or "phonon_dos.modes"))
        self._workflow_set_spinbox_values(self.workflowPhononDosNkSpinBoxes, options.get("phdos_nk"))
        self._workflow_set_spinbox_values(self.workflowPhononEpcFitKmeshSpinBoxes, options.get("epc_fit_kmesh"))
        self._workflow_set_spinbox_values(self.workflowPhononEpcScfKmeshSpinBoxes, options.get("epc_scf_kmesh"))
        self._workflow_set_spinbox_values(self.workflowPhononEpcQGridSpinBoxes, options.get("epc_q_grid"))
        self.workflowPhononEpcFildvscfLineEdit.setText(str(options.get("fildvscf") or "dvscf"))
        self.workflowPhononLambdaEmaxLineEdit.setText(str(options.get("lambda_emax") or "auto"))
        self.workflowPhononLambdaDegaussLineEdit.setText(str(options.get("lambda_degauss") or "0.1"))
        self._workflow_set_spinbox_value(self.workflowPhononLambdaSmearingMethodSpin, options.get("lambda_smearing_method", 1))
        self.workflowPhononLambdaMuLineEdit.setText(str(options.get("lambda_mu") or "0.10"))
        expected_tc = options.get("lambda_expected_tc")
        self.workflowPhononLambdaExpectedTcLineEdit.setText("" if expected_tc is None else str(expected_tc))
        self._workflow_set_double_spinbox_value(self.workflowPhononLambdaTcToleranceSpin, options.get("lambda_tc_tolerance", 1.0))
        self._workflow_apply_path_options("phonon", options, warnings_list)
        self._workflow_refresh_phonon_controls()

    def _workflow_apply_path_options(self, kind: str, options: Mapping[str, Any], warnings_list: List[str]) -> None:
        records = self._workflow_records_from_options(kind, options)
        if records:
            self._workflow_set_path_table_records(kind, records)
            return
        if kind in {"bands", "wannier"} and options.get("kpoints_crystal_b"):
            warnings_list.append(f"Could not restore {kind} high-symmetry path; reselect it before submitting.")
        if kind == "phonon" and (options.get("matdyn_path_points") or options.get("matdyn_q_path")):
            warnings_list.append("Could not restore phonon high-symmetry path; reselect it before submitting.")

    def _workflow_records_from_options(self, kind: str, options: Mapping[str, Any]) -> List[Dict[str, Any]]:
        for key in ("selected_path_points", "matdyn_path_points"):
            raw_records = options.get(key)
            if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)):
                records = self._workflow_normalize_path_records(raw_records)
                if records:
                    return records
        if kind in {"bands", "wannier"}:
            return self._workflow_records_from_kpoint_lines(options.get("kpoints_crystal_b"), skip_header=True)
        if kind == "phonon":
            return self._workflow_records_from_kpoint_lines(options.get("matdyn_q_path"), skip_header=False)
        return []

    def _workflow_normalize_path_records(self, raw_records: Sequence[Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for index, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                continue
            try:
                qx = float(record.get("qx", record.get("x", 0.0)))
                qy = float(record.get("qy", record.get("y", 0.0)))
                qz = float(record.get("qz", record.get("z", 0.0)))
            except (TypeError, ValueError):
                continue
            label = str(record.get("label") or f"P{index + 1}").strip() or f"P{index + 1}"
            records.append({"label": label, "qx": qx, "qy": qy, "qz": qz})
        return records

    def _workflow_records_from_kpoint_lines(self, raw_lines: Any, *, skip_header: bool) -> List[Dict[str, Any]]:
        if isinstance(raw_lines, str):
            lines = [line.strip() for line in raw_lines.splitlines() if line.strip()]
        elif isinstance(raw_lines, Sequence):
            lines = [str(line).strip() for line in raw_lines if str(line).strip()]
        else:
            return []
        records: List[Dict[str, Any]] = []
        for line in lines:
            lower = line.lower()
            if skip_header and lower.startswith("k_points"):
                continue
            parts = line.split()
            if len(parts) == 1:
                try:
                    int(float(parts[0]))
                    continue
                except ValueError:
                    pass
            if len(parts) < 3:
                continue
            try:
                qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            label = f"P{len(records) + 1}"
            if "!" in line:
                comment = line.split("!", 1)[1].strip()
                if comment:
                    label = comment.split()[0].strip() or label
            records.append({"label": label, "qx": qx, "qy": qy, "qz": qz})
        return records

    def _workflow_show_creator_tab(self) -> None:
        for index in range(self.tabWidget.count()):
            if self.tabWidget.tabText(index) == "Workflow Creator":
                self.tabWidget.setCurrentIndex(index)
                return

    def _selected_workflow_manifest_path(self) -> Optional[Path]:
        record = self._selected_job_record()
        if not self._is_workflow_record(record) or record is None:
            return None
        root = self._workflow_root_for_record(record)
        return self._workflow_manifest_path_for_record(record, root)

    def _handle_load_selected_workflow_in_creator(self) -> None:
        manifest_path = self._selected_workflow_manifest_path()
        if manifest_path is None:
            self.statusbar.showMessage("Workflow manifest is not available for the selected job.", 5000)
            return
        self._workflow_load_recipe_from_file(manifest_path)

    def _workflow_conversion_options(self) -> Dict[str, Any]:
        protocol = self._workflow_checked_button_key(
            getattr(self, "workflowProtocolButtons", {}),
            "balanced",
        )
        protocol_map = {
            "fast": "sssp_efficiency",
            "balanced": DEFAULT_PROTOCOL,
            "stringent": "sssp_precision",
        }
        kpoint_distance = None
        kmesh_override = ""
        kpoint_data = self.workflowKpointsCombo.currentData()
        if kpoint_data != "__placeholder__":
            if isinstance(kpoint_data, tuple):
                kmesh_override = str(kpoint_data[0] or "")
                try:
                    kpoint_distance = float(kpoint_data[1])
                except (TypeError, ValueError):
                    kpoint_distance = None
            else:
                try:
                    kpoint_distance = float(kpoint_data)
                except (TypeError, ValueError):
                    kpoint_distance = None

        tolerance = ""
        symprec = None
        tolerance_data = self.workflowSymprecCombo.currentData()
        if tolerance_data != "__placeholder__":
            tolerance = str(tolerance_data)
            try:
                symprec = 10 ** int(tolerance_data)
            except (TypeError, ValueError):
                symprec = CIF_DEFAULT_SYMPREC

        refine_cell = ""
        refine_data = self.workflowRefineCellCombo.currentData()
        if refine_data != "__placeholder__":
            refine_cell = str(refine_data)

        use_materials_cloud = self._workflow_materials_cloud_services_enabled()

        spin_orbit = self._workflow_checked_button_key(
            getattr(self, "workflowSpinOrbitButtons", {}),
            "off",
        )
        manual_ecutwfc_text = (
            self.workflowManualEcutwfcLineEdit.text().strip()
            if hasattr(self, "workflowManualEcutwfcLineEdit")
            else ""
        )
        manual_ecutwfc = None
        if manual_ecutwfc_text:
            try:
                manual_ecutwfc = float(manual_ecutwfc_text)
            except ValueError as exc:
                raise ValueError("Manual ecutwfc override must be a positive number in Ry.") from exc
            if manual_ecutwfc <= 0:
                raise ValueError("Manual ecutwfc override must be a positive number in Ry.")

        options = {
            "electronic_type": self._workflow_checked_button_key(
                getattr(self, "workflowElectronicTypeButtons", {}),
                "metal",
            ),
            "spin_orbit": spin_orbit,
            "use_materials_cloud": use_materials_cloud,
            "protocol": protocol,
            "local_protocol": protocol_map.get(protocol, DEFAULT_PROTOCOL),
            "file_format": self._workflow_combo_value(self.workflowFileFormatCombo, ""),
            "kpoint_distance": kpoint_distance,
            "kmesh_override": kmesh_override,
            "refine_cell": refine_cell,
            "symprec": symprec,
            "tolerance": tolerance,
            "xc": self._workflow_combo_value(self.workflowXcCombo, DEFAULT_XC) or DEFAULT_XC,
            "magnetism": self._workflow_combo_value(self.workflowMagnetismCombo, ""),
            "pseudo_override": self._workflow_combo_value(self.workflowPseudoOverrideCombo, ""),
            "threshold_override": self._workflow_combo_value(self.workflowThresholdOverrideCombo, ""),
            "manual_ecutwfc": manual_ecutwfc,
            "local_pseudo_files": {} if use_materials_cloud else dict(self._workflow_local_pseudo_files),
            "full_relativistic_pseudo_files": (
                dict(self._workflow_full_relativistic_pseudo_files)
                if self._workflow_spin_orbit_applies_to_selected_workflows()
                else {}
            ),
        }
        options.update(self._workflow_relaxation_pressure_options())
        return options

    def _selected_workflow_keys(self) -> List[str]:
        return [
            key
            for key, checkbox in getattr(self, "workflowSelectionCheckboxes", {}).items()
            if checkbox.isChecked()
        ]

    def _selected_relaxation_mode(self) -> str:
        for key, button in getattr(self, "workflowRelaxationButtons", {}).items():
            if button.isChecked():
                return key
        return "as_is"

    def _workflow_convergence_options(self) -> Dict[str, Any]:
        cutoff_enabled = bool(self.workflowCutoffConvergenceCheckbox.isChecked())
        kpoint_enabled = bool(self.workflowKpointConvergenceCheckbox.isChecked())
        ecut_start = int(self.workflowEcutMinSpin.value())
        kpoint_start_spacing = float(self.workflowKpointStartSpacingSpin.value())
        return {
            "cutoff": {
                "enabled": cutoff_enabled,
                "start": ecut_start,
                "step": int(self.workflowEcutStepSpin.value()),
                "maximum": int(self.workflowEcutMaxSpin.value()),
                "window_size": 5,
                "expand_by": 2,
                "energy_delta_ev": float(self.workflowEcutEnergyDeltaSpin.value()),
            },
            "kpoints": {
                "enabled": kpoint_enabled,
                "start_spacing": kpoint_start_spacing,
                "spacing_step": float(self.workflowKpointSpacingStepSpin.value()),
                "max_mesh_size": int(self.workflowKpointMaxMeshSpin.value()),
                "window_size": 5,
                "expand_by": 2,
                "energy_delta_ev": float(self.workflowKpointEnergyDeltaSpin.value()),
            },
        }

    def _workflow_band_options(self) -> Dict[str, Any]:
        path_records = self._workflow_path_table_records("bands") if hasattr(self, "workflowBandsPathTable") else []
        selected_labels = [str(record.get("label") or "").strip() for record in path_records if str(record.get("label") or "").strip()]
        weight_text = (
            self.workflowBandsPathWeightLineEdit.text().strip()
            if hasattr(self, "workflowBandsPathWeightLineEdit")
            else ""
        )
        weight = self._workflow_parse_positive_int(weight_text, default=50, minimum=1)
        if weight is None:
            weight = 50
        nbnd_text = self.workflowBandsNbndLineEdit.text().strip() if hasattr(self, "workflowBandsNbndLineEdit") else ""
        nbnd_value = self._workflow_parse_positive_int(nbnd_text, default=64, minimum=1)
        if nbnd_value is None:
            nbnd_value = 64
        kpoints_crystal_b: List[str] = []
        if path_records:
            try:
                kpoints_crystal_b = self._workflow_band_crystal_b_from_records(path_records, weight)
            except Exception:
                kpoints_crystal_b = []

        return {
            "calculate_fat_band": self._workflow_fat_bands_selected(),
            "kpoint_spacing": float(self.workflowBandScfKpointSpacingSpin.value()),
            "nbnd": int(nbnd_value),
            "kpoint_weight": weight,
            "selected_paths": [selected_labels] if selected_labels else [],
            "selected_path_points": path_records,
            "kpoints_crystal_b": kpoints_crystal_b,
            "filband": self.workflowBandsFilbandLineEdit.text().strip() or "bands.out",
            "fatband_color_mode": (
                str(self.workflowFatBandColorCombo.currentData() or "dominant")
                if hasattr(self, "workflowFatBandColorCombo")
                else "dominant"
            ),
        }

    def _workflow_wannier_options(self) -> Dict[str, Any]:
        path_records = self._workflow_path_table_records("wannier") if hasattr(self, "workflowWannierPathTable") else []
        selected_labels = [str(record.get("label") or "").strip() for record in path_records if str(record.get("label") or "").strip()]
        weight_text = (
            self.workflowWannierPathWeightLineEdit.text().strip()
            if hasattr(self, "workflowWannierPathWeightLineEdit")
            else ""
        )
        weight = self._workflow_parse_positive_int(weight_text, default=50, minimum=1)
        if weight is None:
            weight = 50
        nbnd_text = self.workflowWannierNbndLineEdit.text().strip() if hasattr(self, "workflowWannierNbndLineEdit") else ""
        nbnd_value = self._workflow_parse_positive_int(nbnd_text, default=64, minimum=1)
        if nbnd_value is None:
            nbnd_value = 64
        kmesh = (
            [int(spin.value()) for spin in self.workflowWannierKmeshSpinBoxes]
            if hasattr(self, "workflowWannierKmeshSpinBoxes")
            else [8, 8, 6]
        )
        kpoints_crystal_b: List[str] = []
        if path_records:
            try:
                kpoints_crystal_b = self._workflow_band_crystal_b_from_records(path_records, weight)
            except Exception:
                kpoints_crystal_b = []
        return {
            "scf_kmesh": kmesh,
            "kmesh": kmesh,
            "wannier_nbnd": int(nbnd_value),
            "nbnd": int(nbnd_value),
            "kpoint_weight": weight,
            "selected_paths": [selected_labels] if selected_labels else [],
            "selected_path_points": path_records,
            "kpoints_crystal_b": kpoints_crystal_b,
            "write_unk": (
                bool(self.workflowWannierWriteUnkCheckBox.isChecked())
                if hasattr(self, "workflowWannierWriteUnkCheckBox")
                else False
            ),
            "num_iter": (
                int(self.workflowWannierNumIterSpin.value())
                if hasattr(self, "workflowWannierNumIterSpin")
                else 500
            ),
            "dis_num_iter": (
                int(self.workflowWannierDisNumIterSpin.value())
                if hasattr(self, "workflowWannierDisNumIterSpin")
                else 500
            ),
            "filband": "bands.out",
        }

    def _workflow_dos_options(self) -> Dict[str, Any]:
        return {
            "scf_kpoint_spacing": float(self.workflowDosScfKpointSpacingSpin.value()),
            "nscf_kpoint_spacing": float(self.workflowDosNscfKpointSpacingSpin.value()),
            "use_converged_kmesh": bool(self.workflowDosUseConvergedKmeshCheckBox.isChecked()),
            "reuse_band_scf": False,
        }

    def _workflow_fermi_surface_options(self) -> Dict[str, Any]:
        return {
            "kpoint_spacing": float(self.workflowFermiSurfaceKpointSpacingSpin.value()),
        }

    def _workflow_charge_density_options(self) -> Dict[str, Any]:
        fileout = self.workflowChargeDensityFileoutLineEdit.text().strip()
        if not fileout:
            fileout = self._workflow_charge_density_default_fileout()
        if "{prefix}" in fileout:
            fileout = fileout.replace("{prefix}", self._workflow_charge_density_default_prefix())
        return {
            "kpoint_spacing": float(self.workflowChargeDensityKpointSpacingSpin.value()),
            "fileout": fileout,
            "nx": int(self.workflowChargeDensityNxSpin.value()),
            "ny": int(self.workflowChargeDensityNySpin.value()),
            "nz": int(self.workflowChargeDensityNzSpin.value()),
        }

    def _workflow_elastic_options(self) -> Dict[str, Any]:
        return {
            "what": (
                str(self.workflowElasticWhatCombo.currentData() or "scf_elastic_constants")
                if hasattr(self, "workflowElasticWhatCombo")
                else "scf_elastic_constants"
            ),
            "elastic_algorithm": (
                str(self.workflowElasticAlgorithmCombo.currentData() or "standard")
                if hasattr(self, "workflowElasticAlgorithmCombo")
                else "standard"
            ),
            "ngeo_strain": (
                int(self.workflowElasticNgeoStrainSpin.value())
                if hasattr(self, "workflowElasticNgeoStrainSpin")
                else 4
            ),
            "frozen_ions": (
                bool(self.workflowElasticFrozenIonsCheckBox.isChecked())
                if hasattr(self, "workflowElasticFrozenIonsCheckBox")
                else False
            ),
        }

    def _workflow_phonon_q_point(self) -> List[float]:
        raw = (
            self.workflowPhononQPointLineEdit.text().strip()
            if hasattr(self, "workflowPhononQPointLineEdit")
            else ""
        )
        parts = raw.replace(",", " ").split()
        if len(parts) < 3:
            return [0.0, 0.0, 0.0]
        try:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
        except (TypeError, ValueError):
            return [0.0, 0.0, 0.0]

    def _workflow_phonon_matdyn_q_path(self) -> List[str]:
        raw = (
            self.workflowPhononMatdynPathEdit.toPlainText()
            if hasattr(self, "workflowPhononMatdynPathEdit")
            else ""
        )
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            try:
                int(float(lines[0].split()[0]))
                return lines
            except (IndexError, TypeError, ValueError):
                pass
        return [
            "5",
            "0.0 0.0 0.0 40",
            "0.5 0.0 0.0 40",
            "0.3333333333 0.3333333333 0.0 40",
            "0.0 0.0 0.0 40",
            "0.0 0.0 0.5 1",
        ]

    def _workflow_phonon_options(self) -> Dict[str, Any]:
        mode = self._workflow_effective_phonon_mode()
        q_grid = (
            [int(spin.value()) for spin in self.workflowPhononQGridSpinBoxes]
            if hasattr(self, "workflowPhononQGridSpinBoxes")
            else [3, 3, 3]
        )
        scf_kmesh = (
            [int(spin.value()) for spin in self.workflowPhononScfKmeshSpinBoxes]
            if hasattr(self, "workflowPhononScfKmeshSpinBoxes")
            else [16, 16, 16]
        )
        epc_fit_kmesh = (
            [int(spin.value()) for spin in self.workflowPhononEpcFitKmeshSpinBoxes]
            if hasattr(self, "workflowPhononEpcFitKmeshSpinBoxes")
            else [24, 24, 24]
        )
        epc_scf_kmesh = (
            [int(spin.value()) for spin in self.workflowPhononEpcScfKmeshSpinBoxes]
            if hasattr(self, "workflowPhononEpcScfKmeshSpinBoxes")
            else [12, 12, 12]
        )
        epc_q_grid = (
            [int(spin.value()) for spin in self.workflowPhononEpcQGridSpinBoxes]
            if hasattr(self, "workflowPhononEpcQGridSpinBoxes")
            else [3, 3, 3]
        )
        q_point = self._workflow_phonon_q_point()
        tr2_ph = (
            self.workflowPhononTr2PhLineEdit.text().strip()
            if hasattr(self, "workflowPhononTr2PhLineEdit")
            else ""
        ) or "1.0d-12"
        alpha_mix = (
            f"{float(self.workflowPhononAlphaMixSpin.value()):.10g}"
            if hasattr(self, "workflowPhononAlphaMixSpin")
            else "0.7"
        )
        fildyn = (
            self.workflowPhononFildynLineEdit.text().strip()
            if hasattr(self, "workflowPhononFildynLineEdit")
            else ""
        ) or "phonon.dyn"
        flfrc = (
            self.workflowPhononFlfrcLineEdit.text().strip()
            if hasattr(self, "workflowPhononFlfrcLineEdit")
            else ""
        ) or "phonon.fc"
        flfrq = (
            self.workflowPhononFlfrqLineEdit.text().strip()
            if hasattr(self, "workflowPhononFlfrqLineEdit")
            else ""
        ) or "phonon.freq.band"
        matdyn_flvec = (
            self.workflowPhononMatdynFlvecLineEdit.text().strip()
            if hasattr(self, "workflowPhononMatdynFlvecLineEdit")
            else ""
        ) or "phonon_band.modes"
        q2r_zasr = (
            str(self.workflowPhononQ2rZasrCombo.currentData() or "crystal")
            if hasattr(self, "workflowPhononQ2rZasrCombo")
            else "crystal"
        )
        matdyn_asr = (
            str(self.workflowPhononMatdynAsrCombo.currentData() or "crystal")
            if hasattr(self, "workflowPhononMatdynAsrCombo")
            else "crystal"
        )
        verbosity = (
            str(self.workflowPhononVerbosityCombo.currentData() or "default")
            if hasattr(self, "workflowPhononVerbosityCombo")
            else "default"
        )
        dos_enabled = (
            bool(self.workflowPhononDosCheckBox.isChecked())
            if hasattr(self, "workflowPhononDosCheckBox")
            else True
        )
        band_enabled = (
            bool(self.workflowPhononBandCheckBox.isChecked())
            if hasattr(self, "workflowPhononBandCheckBox")
            else True
        )
        if mode == "dispersion" and not band_enabled and not dos_enabled:
            band_enabled = True
        fldos = (
            self.workflowPhononFldosLineEdit.text().strip()
            if hasattr(self, "workflowPhononFldosLineEdit")
            else ""
        ) or "phonon.phdos"
        phdos_flvec = (
            self.workflowPhononPhdosFlvecLineEdit.text().strip()
            if hasattr(self, "workflowPhononPhdosFlvecLineEdit")
            else ""
        ) or "phonon_dos.modes"
        phdos_nk = (
            [int(spin.value()) for spin in self.workflowPhononDosNkSpinBoxes]
            if hasattr(self, "workflowPhononDosNkSpinBoxes")
            else [20, 20, 20]
        )
        fildvscf = (
            self.workflowPhononEpcFildvscfLineEdit.text().strip()
            if hasattr(self, "workflowPhononEpcFildvscfLineEdit")
            else ""
        ) or "dvscf"
        lambda_emax = (
            self.workflowPhononLambdaEmaxLineEdit.text().strip()
            if hasattr(self, "workflowPhononLambdaEmaxLineEdit")
            else ""
        ) or "auto"
        lambda_degauss = (
            self.workflowPhononLambdaDegaussLineEdit.text().strip()
            if hasattr(self, "workflowPhononLambdaDegaussLineEdit")
            else ""
        ) or "0.1"
        lambda_mu = (
            self.workflowPhononLambdaMuLineEdit.text().strip()
            if hasattr(self, "workflowPhononLambdaMuLineEdit")
            else ""
        ) or "0.10"
        lambda_expected_tc_text = (
            self.workflowPhononLambdaExpectedTcLineEdit.text().strip()
            if hasattr(self, "workflowPhononLambdaExpectedTcLineEdit")
            else ""
        )
        lambda_expected_tc = self._workflow_parse_float(lambda_expected_tc_text) if lambda_expected_tc_text else None
        lambda_tc_tolerance = (
            float(self.workflowPhononLambdaTcToleranceSpin.value())
            if hasattr(self, "workflowPhononLambdaTcToleranceSpin")
            else 1.0
        )
        return {
            "mode": mode,
            "calculation_mode": mode,
            "q_point": q_point,
            "ldisp": mode != "single_q",
            "nq": q_grid,
            "q_grid": q_grid,
            "scf_kmesh": scf_kmesh,
            "epc_fit_kmesh": epc_fit_kmesh,
            "epc_scf_kmesh": epc_scf_kmesh,
            "epc_q_grid": epc_q_grid,
            "tr2_ph": tr2_ph,
            "epc_tr2_ph": tr2_ph,
            "alpha_mix": alpha_mix,
            "fildyn": fildyn,
            "flfrc": flfrc,
            "flfrq": flfrq,
            "phdos_flfrq": "phonon.freq.dos",
            "fldos": fldos,
            "matdyn_flvec": matdyn_flvec,
            "phdos_flvec": phdos_flvec,
            "epc_fldos": "phonon.dos",
            "band_enabled": bool(band_enabled and mode == "dispersion"),
            "dos_enabled": bool(dos_enabled and mode == "dispersion"),
            "phonon_projection_plot_enabled": bool(mode == "dispersion" and band_enabled and dos_enabled),
            "phdos_nk": phdos_nk,
            "fildvscf": fildvscf,
            "electron_phonon": "interpolated",
            "lambda_emax": lambda_emax,
            "lambda_degauss": lambda_degauss,
            "lambda_smearing_method": (
                int(self.workflowPhononLambdaSmearingMethodSpin.value())
                if hasattr(self, "workflowPhononLambdaSmearingMethodSpin")
                else 1
            ),
            "lambda_mu": lambda_mu,
            "lambda_expected_tc": lambda_expected_tc,
            "lambda_tc_tolerance": lambda_tc_tolerance,
            "q2r_zasr": q2r_zasr,
            "matdyn_asr": matdyn_asr,
            "matdyn_q_in_band_form": (
                bool(self.workflowPhononMatdynQInBandFormCheckBox.isChecked())
                if hasattr(self, "workflowPhononMatdynQInBandFormCheckBox")
                else True
            ),
            "matdyn_q_path": self._workflow_phonon_matdyn_q_path(),
            "matdyn_path_points": (
                self._workflow_path_table_records("phonon")
                if hasattr(self, "workflowPhononPathTable")
                else []
            ),
            "trans": (
                bool(self.workflowPhononTransCheckBox.isChecked())
                if hasattr(self, "workflowPhononTransCheckBox")
                else True
            ),
            "epsil": (
                bool(self.workflowPhononEpsilCheckBox.isChecked())
                if hasattr(self, "workflowPhononEpsilCheckBox")
                else False
            ),
            "nmix_ph": (
                int(self.workflowPhononNmixSpin.value())
                if hasattr(self, "workflowPhononNmixSpin")
                else 4
            ),
            "verbosity": verbosity,
            "use_converged_kmesh": False,
            "scf_kpoint_spacing": 0.0,
        }

    def _workflow_runner_command(self, manifest_path: Path) -> str:
        return " ".join(
            [
                shlex.quote(sys.executable),
                shlex.quote(str(PROJECT_ROOT / "workflow_runner.py")),
                "run",
                shlex.quote(str(manifest_path)),
            ]
        )

    def _handle_workflow_submit(self) -> None:
        source_path = str(getattr(self, "workflowSourceInputPath", "")).strip()
        if not source_path:
            self._show_error("Select a CIF or pw.x input file before submitting the workflow.")
            return
        if not Path(source_path).expanduser().is_file():
            self._show_error("The workflow source input file does not exist.")
            return

        selected_workflows = self._selected_workflow_keys()
        relaxation_mode = self._selected_relaxation_mode()
        if not selected_workflows and relaxation_mode == "as_is":
            self._show_error("Select at least one workflow or relaxation mode to run.")
            return

        job_name = self.workflowJobNameLineEdit.text().strip()
        if not job_name:
            self._show_error("Enter a workflow job name.")
            return

        output_directory = self.workflowOutputDirectoryLineEdit.text().strip()
        if not output_directory:
            self._show_error("Select an output directory for the workflow.")
            return

        try:
            conversion_options = self._workflow_conversion_options()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        pseudo_error = self._workflow_validate_pseudo_selection(conversion_options)
        if pseudo_error:
            self._show_warning("Missing pseudopotentials", pseudo_error)
            return
        convergence_options = self._workflow_convergence_options()
        if "convergence" in selected_workflows:
            cutoff_enabled = bool(convergence_options.get("cutoff", {}).get("enabled"))
            kpoints_enabled = bool(convergence_options.get("kpoints", {}).get("enabled"))
            manual_ecutwfc = conversion_options.get("manual_ecutwfc")
            if not cutoff_enabled and not kpoints_enabled:
                self._show_error("Enable at least one convergence routine.")
                return
            if manual_ecutwfc is not None and cutoff_enabled and not kpoints_enabled:
                has_other_work = bool(
                    relaxation_mode != "as_is"
                    or any(workflow != "convergence" for workflow in selected_workflows)
                )
                if not has_other_work:
                    self._show_error(
                        "Manual ecutwfc override skips cutoff convergence. "
                        "Enable k-point convergence or select another workflow/relaxation step."
                    )
                    return
        band_options = self._workflow_band_options()
        wannier_options = self._workflow_wannier_options()
        dos_options = self._workflow_dos_options()
        fermi_surface_options = self._workflow_fermi_surface_options()
        charge_density_options = self._workflow_charge_density_options()
        elastic_options = self._workflow_elastic_options()
        phonon_options = self._workflow_phonon_options()
        if "bands" in selected_workflows:
            if not band_options.get("kpoints_crystal_b"):
                self._show_error("Select at least one high-symmetry path for the band workflow.")
                return
        if "wannier" in selected_workflows:
            if not wannier_options.get("kpoints_crystal_b"):
                self._show_error("Select at least one high-symmetry path for the Wannier workflow.")
                return

        request = WorkflowBuildRequest(
            name=job_name,
            output_directory=Path(output_directory).expanduser(),
            source_structure=Path(source_path).expanduser(),
            selected_workflows=selected_workflows,
            relaxation_mode=relaxation_mode,
            description=self.workflowDescriptionEdit.toPlainText().strip(),
            queue="default",
            conversion_options=conversion_options,
            convergence_options=convergence_options,
            band_options=band_options,
            wannier_options=wannier_options,
            dos_options=dos_options,
            fermi_surface_options=fermi_surface_options,
            charge_density_options=charge_density_options,
            elastic_options=elastic_options,
            phonon_options=phonon_options,
        )

        try:
            workflow_files = create_workflow_files(request, copy_source=True, overwrite=False)
        except FileExistsError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not create workflow files: {exc}")
            return

        command = self._workflow_runner_command(workflow_files.manifest_path)
        config = JobConfig(
            name=job_name,
            calculation="workflow",
            mpi_procs=1,
            queue="default",
            working_directory=str(workflow_files.workflow_root),
            pseudo_directory=None,
            input_files=[str(workflow_files.manifest_path)],
            pseudo_files={},
        )
        metadata = {
            "workflow_root": str(workflow_files.workflow_root),
            "workflow_manifest": str(workflow_files.manifest_path),
            "workflow_state": str(workflow_files.state_path),
            "workflow_id": workflow_files.manifest.get("workflow_id", ""),
        }
        self._enqueue_job(
            config,
            [command],
            origin="workflow_creator",
            log_header=f"Workflow manifest: {workflow_files.manifest_path}",
            metadata=metadata,
        )
        relaxation = relaxation_mode.replace("_", " ")
        workflow_labels = ", ".join(selected_workflows) if selected_workflows else "relaxation"
        notices = workflow_files.manifest.get("notices") if isinstance(workflow_files.manifest.get("notices"), list) else []
        notice_text = f" {notices[0]}" if notices else ""
        self.workflowSubmitStatusLabel.setText(
            f"Queued '{job_name}' as a {relaxation} workflow: {workflow_labels}.{notice_text}"
        )
        status_text = notices[0] if notices else "Workflow queued for daemon execution"
        self.statusbar.showMessage(status_text, 7000 if notices else 5000)

    def _workflow_validate_pseudo_selection(self, conversion_options: Dict[str, Any]) -> str:
        species = self._workflow_structure_species()
        species_order = [
            normalize_element_symbol(entry.element)
            for entry in species
            if normalize_element_symbol(entry.element)
        ]
        if not species_order:
            return "Could not detect atomic species for workflow pseudopotential selection."

        if not bool(conversion_options.get("use_materials_cloud")):
            error = self._workflow_pseudo_mapping_error(
                self._workflow_local_pseudo_files,
                species_order,
                "pseudopotentials",
            )
            if error:
                return error

        if self._workflow_spin_orbit_applies_to_selected_workflows():
            error = self._workflow_pseudo_mapping_error(
                self._workflow_full_relativistic_pseudo_files,
                species_order,
                "full-relativistic spin-orbit pseudopotentials",
            )
            if error:
                return error
            invalid = self._workflow_non_full_relativistic_pseudos(
                self._workflow_full_relativistic_pseudo_files,
            )
            if invalid:
                labels = ", ".join(f"{element}: {Path(path).name}" for element, path in invalid)
                return (
                    "Spin-orbit workflow pseudos must have PP_HEADER relativistic=\"full\". "
                    f"Check: {labels}"
                )
        return ""

    @staticmethod
    def _workflow_pseudo_mapping_error(
        mapping: Dict[str, str],
        species_order: Sequence[str],
        label: str,
    ) -> str:
        missing = [element for element in species_order if not mapping.get(element)]
        if missing:
            return f"Select {label} for: {', '.join(missing)}."
        unavailable = [
            f"{element}: {Path(raw_path).name}"
            for element, raw_path in mapping.items()
            if not Path(raw_path).expanduser().is_file()
        ]
        if unavailable:
            return f"Selected {label} files are missing: {', '.join(unavailable)}."
        return ""

    def _build_job_status_tab(self) -> None:
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setContentsMargins(0, 0, 0, 0)

        # Daemon status section at the top
        daemon_layout = QHBoxLayout()
        daemon_layout.addWidget(QLabel("Job Daemon:", status_tab))
        self.daemonStatusLabel = QLabel("Unknown", status_tab)
        self.daemonStatusLabel.setStyleSheet("color: gray;")
        daemon_layout.addWidget(self.daemonStatusLabel)
        daemon_layout.addStretch()
        self.startDaemonButton = QPushButton("Start Daemon", status_tab)
        self.stopDaemonButton = QPushButton("Stop Daemon", status_tab)
        daemon_layout.addWidget(self.startDaemonButton)
        daemon_layout.addWidget(self.stopDaemonButton)
        status_layout.addLayout(daemon_layout)

        toolbar_layout = QHBoxLayout()
        self.jobStatusFilterLineEdit = QLineEdit(status_tab)
        self.jobStatusFilterLineEdit.setPlaceholderText("Filter jobs by name or status...")
        self.refreshJobStatusButton = QPushButton("Refresh", status_tab)
        toolbar_layout.addWidget(QLabel("Jobs: ", status_tab))
        toolbar_layout.addWidget(self.jobStatusFilterLineEdit)
        toolbar_layout.addWidget(self.refreshJobStatusButton)
        status_layout.addLayout(toolbar_layout)

        self.jobOverallProgressBar = QProgressBar(status_tab)
        self.jobOverallProgressBar.setRange(0, 100)
        self.jobOverallProgressBar.setValue(0)
        self.jobOverallProgressBar.setFormat("No jobs running")
        status_layout.addWidget(self.jobOverallProgressBar)

        status_splitter = QSplitter(Qt.Orientation.Horizontal, status_tab)

        tree_panel = QWidget(status_splitter)
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(0, 0, 6, 0)
        self.jobStatusTree = QTreeWidget(tree_panel)
        self.jobStatusTree.setObjectName("jobStatusTree")
        self.jobStatusTree.setHeaderHidden(True)
        self.jobStatusTree.setIconSize(QSize(18, 18))
        self.jobStatusTree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.jobStatusTree.setAlternatingRowColors(False)
        self.jobStatusTree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.jobStatusTree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tree_layout.addWidget(self.jobStatusTree)
        status_splitter.addWidget(tree_panel)

        detail_group = QGroupBox("Selected output", status_splitter)
        detail_layout = QVBoxLayout(detail_group)
        self.jobDetailPreview = QPlainTextEdit(detail_group)
        self.jobDetailPreview.setReadOnly(True)
        self.jobDetailPreview.setPlaceholderText("Select a workflow step or job to see its output.")
        self.jobDetailPreview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.jobDetailPreview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.jobDetailOutputStack = QStackedWidget(detail_group)
        self.jobDetailOutputStack.addWidget(self.jobDetailPreview)

        output_view_layout = QHBoxLayout()
        output_view_layout.setContentsMargins(0, 0, 0, 0)
        output_view_layout.setSpacing(4)
        output_view_layout.addWidget(self.jobDetailOutputStack, 1)

        output_controls = QWidget(detail_group)
        output_controls.setFixedWidth(34)
        output_controls_layout = QVBoxLayout(output_controls)
        output_controls_layout.setContentsMargins(0, 0, 0, 0)
        output_controls_layout.setSpacing(3)
        self.jumpToJobOutputEndButton = QToolButton(output_controls)
        self.jumpToJobOutputEndButton.setObjectName("jobOutputControlButton")
        self.jumpToJobOutputEndButton.setIcon(self._job_output_end_icon())
        self.jumpToJobOutputEndButton.setIconSize(QSize(24, 24))
        self.jumpToJobOutputEndButton.setToolTip("Go to the end of the output")
        self.jumpToJobOutputEndButton.setFixedSize(32, 32)
        output_controls_layout.addWidget(self.jumpToJobOutputEndButton, 0, Qt.AlignmentFlag.AlignTop)
        output_controls_layout.addStretch()
        output_view_layout.addWidget(output_controls)
        detail_layout.addLayout(output_view_layout)

        self.phononProjectedPreviewFrame = QFrame(detail_group)
        self.phononProjectedPreviewFrame.setObjectName("phononProjectedPreviewFrame")
        preview_layout = QVBoxLayout(self.phononProjectedPreviewFrame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        self.phononProjectedPreviewTitle = QLabel("Phonon band + PhDOS plot", self.phononProjectedPreviewFrame)
        self.phononProjectedPreviewTitle.setStyleSheet("font-weight: 600;")
        self.phononProjectedPreviewImage = QLabel(self.phononProjectedPreviewFrame)
        self.phononProjectedPreviewImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phononProjectedPreviewImage.setMinimumHeight(170)
        self.phononProjectedPreviewImage.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1;")
        preview_layout.addWidget(self.phononProjectedPreviewTitle)
        preview_layout.addWidget(self.phononProjectedPreviewImage)

        preview_actions = QHBoxLayout()
        self.openPhononProjectedImageButton = QPushButton("Open image", self.phononProjectedPreviewFrame)
        self.openPhononProjectedFolderButton = QPushButton("Open folder", self.phononProjectedPreviewFrame)
        self.editPhononProjectedScriptButton = QPushButton("Edit script", self.phononProjectedPreviewFrame)
        self.runPhononProjectedScriptButton = QPushButton("Run script", self.phononProjectedPreviewFrame)
        preview_actions.addWidget(self.openPhononProjectedImageButton)
        preview_actions.addWidget(self.openPhononProjectedFolderButton)
        preview_actions.addWidget(self.editPhononProjectedScriptButton)
        preview_actions.addWidget(self.runPhononProjectedScriptButton)
        preview_actions.addStretch()
        preview_layout.addLayout(preview_actions)
        self.jobDetailOutputStack.addWidget(self.phononProjectedPreviewFrame)
        self.phononProjectedPreviewFrame.hide()
        self._workflowPlotPreviewPaths: Dict[str, Optional[Path]] = {}
        self._phononProjectedPreviewPaths: Dict[str, Optional[Path]] = {}

        self.openPhononProjectedImageButton.clicked.connect(self._open_phonon_projected_image)
        self.openPhononProjectedFolderButton.clicked.connect(self._open_phonon_projected_folder)
        self.editPhononProjectedScriptButton.clicked.connect(self._edit_phonon_projected_script)
        self.runPhononProjectedScriptButton.clicked.connect(self._run_phonon_projected_replot_script)

        self.workflowLambdaOutputFrame = QFrame(detail_group)
        self.workflowLambdaOutputFrame.setObjectName("workflowLambdaOutputFrame")
        lambda_layout = QVBoxLayout(self.workflowLambdaOutputFrame)
        lambda_layout.setContentsMargins(8, 8, 8, 8)
        lambda_layout.setSpacing(6)
        self.workflowLambdaOutputTitle = QLabel("lambda.x outputs", self.workflowLambdaOutputFrame)
        self.workflowLambdaOutputTitle.setStyleSheet("font-weight: 600;")
        lambda_layout.addWidget(self.workflowLambdaOutputTitle)

        lambda_nav = QHBoxLayout()
        self.workflowLambdaPrevButton = QPushButton("Previous", self.workflowLambdaOutputFrame)
        self.workflowLambdaNextButton = QPushButton("Next", self.workflowLambdaOutputFrame)
        self.workflowLambdaOutputCombo = QComboBox(self.workflowLambdaOutputFrame)
        self.workflowLambdaOpenFolderButton = QPushButton("Open folder", self.workflowLambdaOutputFrame)
        lambda_nav.addWidget(self.workflowLambdaPrevButton)
        lambda_nav.addWidget(self.workflowLambdaNextButton)
        lambda_nav.addWidget(self.workflowLambdaOutputCombo, 1)
        lambda_nav.addWidget(self.workflowLambdaOpenFolderButton)
        lambda_layout.addLayout(lambda_nav)

        self.workflowLambdaOutputText = QPlainTextEdit(self.workflowLambdaOutputFrame)
        self.workflowLambdaOutputText.setReadOnly(True)
        self.workflowLambdaOutputText.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.workflowLambdaOutputText.setPlaceholderText("lambda.x output will appear here.")
        lambda_layout.addWidget(self.workflowLambdaOutputText, 1)

        lambda_rerun = QHBoxLayout()
        lambda_rerun.addWidget(QLabel("Rerun with mu:", self.workflowLambdaOutputFrame))
        self.workflowLambdaMuLineEdit = QLineEdit("0.10", self.workflowLambdaOutputFrame)
        self.workflowLambdaMuLineEdit.setMaximumWidth(120)
        self.workflowLambdaRerunButton = QPushButton("Rerun lambda.x", self.workflowLambdaOutputFrame)
        lambda_rerun.addWidget(self.workflowLambdaMuLineEdit)
        lambda_rerun.addWidget(self.workflowLambdaRerunButton)
        lambda_rerun.addStretch()
        lambda_layout.addLayout(lambda_rerun)

        self.jobDetailOutputStack.addWidget(self.workflowLambdaOutputFrame)
        self.workflowLambdaOutputFrame.hide()
        self._workflowLambdaOutputPaths: List[Path] = []
        self._workflowLambdaContext: Dict[str, Any] = {}
        self._workflowLambdaRerunProcess: Optional[subprocess.Popen] = None
        self._workflowLambdaMuDirty = False
        self._workflowLambdaPendingMu = ""
        self._workflowLambdaMuContextKey = ""
        self._workflowLambdaDisplayedPath: Optional[Path] = None
        self._workflowLambdaDisplayedText = ""
        self.workflowLambdaOutputCombo.currentIndexChanged.connect(self._workflow_lambda_output_combo_changed)
        self.workflowLambdaPrevButton.clicked.connect(self._workflow_lambda_previous_output)
        self.workflowLambdaNextButton.clicked.connect(self._workflow_lambda_next_output)
        self.workflowLambdaOpenFolderButton.clicked.connect(self._workflow_lambda_open_folder)
        self.workflowLambdaRerunButton.clicked.connect(self._workflow_lambda_rerun_with_mu)
        self.workflowLambdaMuLineEdit.textEdited.connect(self._mark_workflow_lambda_mu_edited)

        status_splitter.addWidget(detail_group)
        status_splitter.setStretchFactor(0, 2)
        status_splitter.setStretchFactor(1, 3)
        status_splitter.setSizes([420, 620])
        status_layout.addWidget(status_splitter, 1)

        status_actions_layout = QHBoxLayout()
        self.clearCompletedJobsButton = QPushButton("Clear completed", status_tab)
        self.clearStoppedJobsButton = QPushButton("Clear stopped", status_tab)
        status_actions_layout.addStretch()
        status_actions_layout.addWidget(self.clearCompletedJobsButton)
        status_actions_layout.addWidget(self.clearStoppedJobsButton)
        status_layout.addLayout(status_actions_layout)

        self.tabWidget.addTab(status_tab, "Job Status")

    # ---------------------- Job Queue helpers ---------------------------
    def _load_saved_job_queue(self) -> None:
        if not self.job_queue_file.exists():
            return

        # Read job queue file under a file lock to avoid races with the daemon
        data = None
        lock = None
        if FileLock is not None:
            lock = FileLock(str(self.job_queue_file) + ".lock")
        try:
            if lock is not None:
                try:
                    with lock.acquire(timeout=5):
                        with self.job_queue_file.open("r", encoding="utf-8") as fh:
                            data = json.load(fh)
                except Timeout:
                    # Could not acquire lock; skip reading for now
                    return
            else:
                with self.job_queue_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.statusbar.showMessage(f"Could not load saved jobs: {exc}", 5000)
            return

        restored: List[Dict[str, Any]] = []
        for entry in data or []:
            if not isinstance(entry, dict):
                continue
            config_data = entry.get("config")
            if not isinstance(config_data, dict):
                continue
            try:
                config = JobConfig.from_dict(config_data)
            except Exception:
                continue

            # Preserve the status reported by the daemon; do not override running states
            status = str(entry.get("status", "Queued")) or "Queued"

            try:
                progress = int(entry.get("progress", 0))
            except Exception:
                progress = 0

            restored.append(
                {
                    "config": config,
                    "commands": list(entry.get("commands", [])),
                    "status": status,
                    "progress": max(0, min(100, progress)),
                    "log": str(entry.get("log", "")),
                    "origin": entry.get("origin", "restored"),
                    "metadata": entry.get("metadata", {}),
                }
            )

        self.job_queue = restored

    def _persist_job_queue(self) -> None:
        payload = []
        for record in self.job_queue:
            config = record.get("config")
            if not isinstance(config, JobConfig):
                continue
            payload.append(
                {
                    "config": config.to_dict(),
                    "commands": list(record.get("commands", [])),
                    "status": record.get("status", "Queued"),
                    "progress": int(record.get("progress", 0)),
                    "log": record.get("log", ""),
                    "origin": record.get("origin", "manual"),
                    "metadata": record.get("metadata", {}),
                }
            )

        try:
            self.job_queue_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.job_queue_file.with_suffix(".json.tmp")
            lock = None
            if FileLock is not None:
                lock = FileLock(str(self.job_queue_file) + ".lock")
            if lock is not None:
                try:
                    with lock.acquire(timeout=5):
                        with tmp.open("w", encoding="utf-8") as fh:
                            json.dump(payload, fh, indent=2)
                        os.replace(str(tmp), str(self.job_queue_file))
                except Timeout:
                    self.statusbar.showMessage("Could not acquire job queue lock to save jobs", 5000)
            else:
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                os.replace(str(tmp), str(self.job_queue_file))
        except Exception as exc:  # noqa: BLE001
            self.statusbar.showMessage(f"Could not save job queue: {exc}", 5000)

    def _poll_job_queue_file(self) -> None:
        """Called periodically to refresh the in-memory job queue from disk.

        Uses the same locked loader to avoid races with the daemon.
        """
        if self._workflow_lambda_output_popup_open():
            return
        try:
            self._load_saved_job_queue()
            self._sync_job_status_table()
            self._update_daemon_status()
        except Exception:
            # Non-fatal; we'll try again at next poll
            return

    def _enqueue_job(
        self,
        config: JobConfig,
        commands: List[str],
        *,
        origin: str = "manual",
        log_header: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        meta = dict(metadata or {})
        created_at = meta.get("created_at")
        if not created_at:
            now = datetime.now()
            created_at = now.isoformat(timespec="microseconds")
            meta["created_at"] = created_at
        else:
            try:
                now = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            except Exception:
                now = datetime.now()

        base_name = sanitize_job_name(config.name or config.sanitized_name())
        timestamp_slug = now.strftime("%Y%m%d_%H%M%S_%f")
        job_id = str(meta.get("job_id") or f"{base_name}_{timestamp_slug}")
        meta["job_id"] = sanitize_job_name(job_id)
        if not meta.get("log_file"):
            working_dir = self._resolve_working_directory(config)
            meta["log_file"] = str(working_dir / f"{meta['job_id']}.log")

        record = {
            "config": config,
            "commands": list(commands),
            "status": "Queued",
            "progress": 0,
            "log": f"{log_header}\n" if log_header else "",
            "origin": origin,
            "metadata": meta,
        }
        self.job_queue.append(record)
        self._persist_job_queue()
        self._sync_job_status_table()
        self._select_job_index(len(self.job_queue) - 1)
        # Note: job execution is handled by the external job daemon.
        # Do not start local JobRunner here.

    def _format_job_timestamp(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return text

    def _job_display_name(self, record: Dict[str, Any]) -> str:
        config = record.get("config")
        if not isinstance(config, JobConfig):
            return ""
        job_name = config.name or config.sanitized_name()
        metadata = record.get("metadata") or {}
        created_at_text = self._format_job_timestamp(metadata.get("created_at"))
        return f"{job_name} ({created_at_text})" if created_at_text else job_name

    def _sync_job_status_table(self) -> None:
        if not hasattr(self, "jobStatusTree"):
            return

        filter_text = ""
        if hasattr(self, "jobStatusFilterLineEdit"):
            filter_text = (self.jobStatusFilterLineEdit.text() or "").strip().lower()

        selected_key = self._current_status_tree_key()
        selected_index = self._current_job_index()
        expanded_keys = self._expanded_job_status_tree_keys()
        previous_signals_blocked = self.jobStatusTree.blockSignals(True)
        previous_updates_enabled = self.jobStatusTree.updatesEnabled()
        previous_scroll_value = self.jobStatusTree.verticalScrollBar().value()
        self.jobStatusTree.setUpdatesEnabled(False)
        self.jobStatusTree.clear()

        item_by_key: Dict[str, QTreeWidgetItem] = {}
        visible_indices: List[int] = []
        for idx, record in enumerate(self.job_queue):
            config = record.get("config")
            if not isinstance(config, JobConfig):
                continue
            if not self._job_record_matches_filter(record, filter_text):
                continue

            job_tone = len(visible_indices) % 2
            key = f"job:{idx}"
            if self._is_workflow_record(record):
                state, state_path, error = self._load_workflow_state_for_record(record)
                if state is None:
                    item = self._workflow_tree_item(
                        f"Workflow: {self._job_display_name(record) or config.name or config.sanitized_name()}",
                        record.get("status", "Queued"),
                        {
                            "kind": "workflow",
                            "key": key,
                            "job_index": idx,
                            "job_tone": job_tone,
                            "state_path": str(state_path or ""),
                            "state_error": error,
                        },
                    )
                    self.jobStatusTree.addTopLevelItem(item)
                    item_by_key[key] = item
                else:
                    self._populate_workflow_tree(
                        state,
                        record,
                        state_path,
                        item_by_key,
                        job_index=idx,
                        job_tone=job_tone,
                    )
            else:
                item = self._workflow_tree_item(
                    self._job_display_name(record) or config.name or config.sanitized_name(),
                    record.get("status", "Queued"),
                    {"kind": "job", "key": key, "job_index": idx, "job_tone": job_tone},
                )
                self.jobStatusTree.addTopLevelItem(item)
                item_by_key[key] = item
            visible_indices.append(idx)

        if selected_index is not None and selected_index not in visible_indices:
            self.jobDetailPreview.clear()

        selected = item_by_key.get(selected_key)
        if selected is None and selected_index is not None:
            selected = item_by_key.get(f"job:{selected_index}")
        if selected is None and self.jobStatusTree.topLevelItemCount() > 0:
            selected = self.jobStatusTree.topLevelItem(0)
        if selected is not None:
            self.jobStatusTree.setCurrentItem(selected)

        if expanded_keys:
            for key in expanded_keys:
                item = item_by_key.get(key)
                if item is not None:
                    item.setExpanded(True)

        self.jobStatusTree.blockSignals(previous_signals_blocked)
        self.jobStatusTree.setUpdatesEnabled(previous_updates_enabled)

        self._update_job_overall_progress()
        self._update_job_action_states()
        self._handle_job_table_selection()
        self.jobStatusTree.verticalScrollBar().setValue(previous_scroll_value)

    def _job_record_matches_filter(self, record: Dict[str, Any], filter_text: str) -> bool:
        if not filter_text:
            return True
        config = record.get("config")
        if not isinstance(config, JobConfig):
            return False
        values = [
            config.name,
            config.sanitized_name(),
            config.calculation,
            config.queue,
            self._job_display_name(record),
            record.get("status", ""),
        ]
        return any(filter_text in str(value or "").lower() for value in values)

    def _select_job_index(self, index: int) -> None:
        if index < 0 or not hasattr(self, "jobStatusTree"):
            return
        item = self._find_status_tree_item(f"job:{index}")
        if item is not None:
            self.jobStatusTree.setCurrentItem(item)
            self._handle_job_table_selection()

    def _update_job_overall_progress(self) -> None:
        if not hasattr(self, "jobOverallProgressBar"):
            return

        total = len(self.job_queue)
        if total == 0:
            self.jobOverallProgressBar.setValue(0)
            self.jobOverallProgressBar.setFormat("No jobs running")
            return

        progress_sum = sum(int(record.get("progress", 0)) for record in self.job_queue)
        average_progress = int(progress_sum / total)
        running = sum(1 for record in self.job_queue if str(record.get("status", "")).startswith("Running"))

        self.jobOverallProgressBar.setValue(max(0, min(100, average_progress)))
        if running:
            self.jobOverallProgressBar.setFormat(f"{running} running / {total} total")
        else:
            self.jobOverallProgressBar.setFormat(f"{total} job(s) queued")

    def _current_job_index(self) -> Optional[int]:
        item = self.jobStatusTree.currentItem() if hasattr(self, "jobStatusTree") else None
        while item is not None:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            index = data.get("job_index") if isinstance(data, dict) else None
            if index is not None:
                try:
                    return int(index)
                except Exception:
                    return None
            item = item.parent()
        return None

    def _handle_job_filter_changed(self, _text: str) -> None:
        self._sync_job_status_table()

    def _handle_refresh_job_status(self) -> None:
        if self._workflow_lambda_output_popup_open():
            return
        self._sync_job_status_table()
        # The daemon owns execution; refresh only updates the view.

    def _handle_job_table_selection(self) -> None:
        data = self._current_status_tree_data()
        if isinstance(data, dict):
            self._statusTreeSelectionKey = str(data.get("key") or "")
        if self._selected_job_is_workflow():
            self._update_workflow_tree_detail()
        else:
            self._update_job_detail_preview()
        self._update_job_action_states()

    def _current_status_tree_data(self) -> Dict[str, Any]:
        item = self.jobStatusTree.currentItem() if hasattr(self, "jobStatusTree") else None
        data = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else {}
        return data if isinstance(data, dict) else {}

    def _current_status_tree_key(self) -> str:
        data = self._current_status_tree_data()
        return str(data.get("key") or getattr(self, "_statusTreeSelectionKey", "") or "")

    def _expanded_job_status_tree_keys(self) -> set[str]:
        if not hasattr(self, "jobStatusTree"):
            return set()
        expanded: set[str] = set()

        def collect(item: QTreeWidgetItem) -> None:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if item.isExpanded() and isinstance(data, dict) and data.get("key"):
                expanded.add(str(data["key"]))
            for child_index in range(item.childCount()):
                collect(item.child(child_index))

        for root_index in range(self.jobStatusTree.topLevelItemCount()):
            collect(self.jobStatusTree.topLevelItem(root_index))
        return expanded

    def _handle_job_tree_item_collapsed(self, item: QTreeWidgetItem) -> None:
        current = self.jobStatusTree.currentItem() if hasattr(self, "jobStatusTree") else None
        parent = current.parent() if current is not None else None
        while parent is not None:
            if parent is item:
                self.jobStatusTree.setCurrentItem(item)
                return
            parent = parent.parent()

    def _find_status_tree_item(self, key: str) -> Optional[QTreeWidgetItem]:
        if not key or not hasattr(self, "jobStatusTree"):
            return None

        def find(item: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and str(data.get("key") or "") == key:
                return item
            for child_index in range(item.childCount()):
                found = find(item.child(child_index))
                if found is not None:
                    return found
            return None

        for root_index in range(self.jobStatusTree.topLevelItemCount()):
            found = find(self.jobStatusTree.topLevelItem(root_index))
            if found is not None:
                return found
        return None

    def _selected_job_record(self) -> Optional[Dict[str, Any]]:
        index = self._current_job_index()
        if index is None or index < 0 or index >= len(self.job_queue):
            return None
        record = self.job_queue[index]
        return record if isinstance(record, dict) else None

    def _selected_job_is_workflow(self) -> bool:
        record = self._selected_job_record()
        return self._is_workflow_record(record)

    def _is_workflow_record(self, record: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(record, dict):
            return False
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        config = record.get("config")
        return bool(
            metadata.get("workflow_state")
            or metadata.get("workflow_manifest")
            or record.get("origin") == "workflow_creator"
            or (isinstance(config, JobConfig) and str(config.calculation).strip().lower() == "workflow")
        )

    def _workflow_state_path_for_record(self, record: Dict[str, Any]) -> Optional[Path]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        state_path = str(metadata.get("workflow_state") or "").strip()
        if state_path:
            return Path(state_path).expanduser()
        root = self._workflow_root_for_record(record)
        return root / "workflow_state.json" if root is not None else None

    def _workflow_root_for_record(self, record: Dict[str, Any]) -> Optional[Path]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        root = str(metadata.get("workflow_root") or "").strip()
        if root:
            return Path(root).expanduser()
        config = record.get("config")
        if isinstance(config, JobConfig) and config.working_directory:
            return Path(config.working_directory).expanduser()
        return None

    def _load_workflow_state_for_record(self, record: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[Path], str]:
        state_path = self._workflow_state_path_for_record(record)
        if state_path is None:
            return None, None, "Workflow state path is not available."
        if not state_path.is_file():
            return None, state_path, f"Workflow state file is not available yet: {state_path}"
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return None, state_path, f"Workflow state is not a JSON object: {state_path}"
            return payload, state_path, ""
        except Exception as exc:  # noqa: BLE001
            return None, state_path, f"Could not read workflow state: {exc}"

    def _populate_workflow_tree(
        self,
        state: Dict[str, Any],
        record: Dict[str, Any],
        state_path: Optional[Path],
        item_by_key: Dict[str, QTreeWidgetItem],
        *,
        job_index: int,
        job_tone: int,
    ) -> None:
        config = record.get("config")
        display_name = self._job_display_name(record)
        if not display_name:
            display_name = config.name if isinstance(config, JobConfig) and config.name else str(state.get("workflow_id") or "workflow")
        name = f"Workflow: {display_name}"
        root_status = str(state.get("status") or record.get("status") or "queued")
        root_key = f"job:{job_index}"
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        root = self._workflow_tree_item(
            name,
            root_status,
            {
                "kind": "workflow",
                "key": root_key,
                "job_index": job_index,
                "job_tone": job_tone,
                "progress_label": self._workflow_tasks_progress_text(tasks.values()),
                "state_path": str(state_path or ""),
            },
        )
        self.jobStatusTree.addTopLevelItem(root)
        item_by_key[root_key] = root

        source_task = tasks.get("prepare_source") if isinstance(tasks.get("prepare_source"), dict) else None
        if source_task:
            source_key = f"{root_key}:group:source"
            source_group = self._workflow_tree_item(
                "Source preparation",
                self._workflow_task_status(source_task),
                {
                    "kind": "group",
                    "key": source_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([source_task]),
                    "task_ids": ["prepare_source"],
                },
            )
            root.addChild(source_group)
            item_by_key[source_key] = source_group
            self._add_workflow_task_children(
                source_group,
                "prepare_source",
                source_task,
                item_by_key,
                job_index=job_index,
                job_tone=job_tone,
            )

        relaxation_ids = ["initial_relaxation", "relaxation", "final_relaxation"]
        relaxation_tasks = [(task_id, tasks.get(task_id)) for task_id in relaxation_ids if isinstance(tasks.get(task_id), dict)]
        if relaxation_tasks:
            relaxation_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in relaxation_tasks]
            )
            relaxation_key = f"{root_key}:group:relaxation"
            relaxation_group = self._workflow_tree_item(
                "Relaxation",
                relaxation_status,
                {
                    "kind": "group",
                    "key": relaxation_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in relaxation_tasks]),
                    "task_ids": [task_id for task_id, _ in relaxation_tasks],
                },
            )
            root.addChild(relaxation_group)
            item_by_key[relaxation_key] = relaxation_group
            for task_id, task in relaxation_tasks:
                self._add_workflow_task_children(
                    relaxation_group,
                    task_id,
                    task,
                    item_by_key,
                    direct=True,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        convergence_ids = ["cutoff_convergence", "kpoint_convergence"]
        convergence_tasks = [(task_id, tasks.get(task_id)) for task_id in convergence_ids if isinstance(tasks.get(task_id), dict)]
        if convergence_tasks:
            convergence_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in convergence_tasks]
            )
            convergence_key = f"{root_key}:group:convergence"
            convergence_group = self._workflow_tree_item(
                "Convergence",
                convergence_status,
                {
                    "kind": "group",
                    "key": convergence_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in convergence_tasks]),
                    "task_ids": [task_id for task_id, _ in convergence_tasks],
                },
            )
            root.addChild(convergence_group)
            item_by_key[convergence_key] = convergence_group
            for task_id, task in convergence_tasks:
                label = self._workflow_task_display_label(task_id, task)
                task_key = f"{root_key}:task:{task_id}"
                task_item = self._workflow_tree_item(
                    label,
                    self._workflow_task_status(task),
                    {
                        "kind": "task",
                        "key": task_key,
                        "job_index": job_index,
                        "job_tone": job_tone,
                        "progress_label": self._workflow_task_progress_text(task),
                        "task_id": task_id,
                    },
                )
                convergence_group.addChild(task_item)
                item_by_key[task_key] = task_item
                self._add_workflow_task_children(
                    task_item,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        band_ids = ["bands_workflow", "fat_bands_workflow", "fatband_csv_extraction", "fatband_plotting"]
        band_tasks = [(task_id, tasks.get(task_id)) for task_id in band_ids if isinstance(tasks.get(task_id), dict)]
        if band_tasks:
            band_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in band_tasks]
            )
            band_key = f"{root_key}:group:bands"
            band_group = self._workflow_tree_item(
                "Band structure workflow",
                band_status,
                {
                    "kind": "group",
                    "key": band_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in band_tasks]),
                    "task_ids": [task_id for task_id, _ in band_tasks],
                },
            )
            root.addChild(band_group)
            item_by_key[band_key] = band_group
            for task_id, task in band_tasks:
                self._add_workflow_task_children(
                    band_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        dos_ids = ["dos_workflow"]
        dos_tasks = [(task_id, tasks.get(task_id)) for task_id in dos_ids if isinstance(tasks.get(task_id), dict)]
        if dos_tasks:
            dos_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in dos_tasks]
            )
            dos_key = f"{root_key}:group:dos"
            dos_group = self._workflow_tree_item(
                "Projected density of states workflow",
                dos_status,
                {
                    "kind": "group",
                    "key": dos_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in dos_tasks]),
                    "task_ids": [task_id for task_id, _ in dos_tasks],
                },
            )
            root.addChild(dos_group)
            item_by_key[dos_key] = dos_group
            for task_id, task in dos_tasks:
                self._add_workflow_task_children(
                    dos_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        fermi_surface_ids = ["fermi_surface_workflow"]
        fermi_surface_tasks = [
            (task_id, tasks.get(task_id))
            for task_id in fermi_surface_ids
            if isinstance(tasks.get(task_id), dict)
        ]
        if fermi_surface_tasks:
            fermi_surface_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in fermi_surface_tasks]
            )
            fermi_surface_key = f"{root_key}:group:fermi_surface"
            fermi_surface_group = self._workflow_tree_item(
                "Fermi surface workflow",
                fermi_surface_status,
                {
                    "kind": "group",
                    "key": fermi_surface_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in fermi_surface_tasks]),
                    "task_ids": [task_id for task_id, _ in fermi_surface_tasks],
                },
            )
            root.addChild(fermi_surface_group)
            item_by_key[fermi_surface_key] = fermi_surface_group
            for task_id, task in fermi_surface_tasks:
                self._add_workflow_task_children(
                    fermi_surface_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        charge_density_ids = ["charge_density_workflow"]
        charge_density_tasks = [
            (task_id, tasks.get(task_id))
            for task_id in charge_density_ids
            if isinstance(tasks.get(task_id), dict)
        ]
        if charge_density_tasks:
            charge_density_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in charge_density_tasks]
            )
            charge_density_key = f"{root_key}:group:charge_density"
            charge_density_group = self._workflow_tree_item(
                "Charge density workflow",
                charge_density_status,
                {
                    "kind": "group",
                    "key": charge_density_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in charge_density_tasks]),
                    "task_ids": [task_id for task_id, _ in charge_density_tasks],
                },
            )
            root.addChild(charge_density_group)
            item_by_key[charge_density_key] = charge_density_group
            for task_id, task in charge_density_tasks:
                self._add_workflow_task_children(
                    charge_density_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        elastic_ids = ["elastic_workflow"]
        elastic_tasks = [(task_id, tasks.get(task_id)) for task_id in elastic_ids if isinstance(tasks.get(task_id), dict)]
        if elastic_tasks:
            elastic_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in elastic_tasks]
            )
            elastic_key = f"{root_key}:group:elastic"
            elastic_group = self._workflow_tree_item(
                "Elastic properties workflow",
                elastic_status,
                {
                    "kind": "group",
                    "key": elastic_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in elastic_tasks]),
                    "task_ids": [task_id for task_id, _ in elastic_tasks],
                },
            )
            root.addChild(elastic_group)
            item_by_key[elastic_key] = elastic_group
            for task_id, task in elastic_tasks:
                self._add_workflow_task_children(
                    elastic_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        phonon_ids = ["phonon_workflow"]
        phonon_tasks = [(task_id, tasks.get(task_id)) for task_id in phonon_ids if isinstance(tasks.get(task_id), dict)]
        if phonon_tasks:
            phonon_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in phonon_tasks]
            )
            phonon_key = f"{root_key}:group:phonon"
            phonon_group = self._workflow_tree_item(
                "Phonon workflow",
                phonon_status,
                {
                    "kind": "group",
                    "key": phonon_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in phonon_tasks]),
                    "task_ids": [task_id for task_id, _ in phonon_tasks],
                },
            )
            root.addChild(phonon_group)
            item_by_key[phonon_key] = phonon_group
            for task_id, task in phonon_tasks:
                self._add_workflow_task_children(
                    phonon_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        wannier_ids = ["wannier90_workflow"]
        wannier_tasks = [(task_id, tasks.get(task_id)) for task_id in wannier_ids if isinstance(tasks.get(task_id), dict)]
        if wannier_tasks:
            wannier_status = self._derive_group_status(
                [self._workflow_task_status(task) for _, task in wannier_tasks]
            )
            wannier_key = f"{root_key}:group:wannier90"
            wannier_group = self._workflow_tree_item(
                "Wannier90 + SCDM workflow routine",
                wannier_status,
                {
                    "kind": "group",
                    "key": wannier_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in wannier_tasks]),
                    "task_ids": [task_id for task_id, _ in wannier_tasks],
                },
            )
            root.addChild(wannier_group)
            item_by_key[wannier_key] = wannier_group
            for task_id, task in wannier_tasks:
                self._add_workflow_task_children(
                    wannier_group,
                    task_id,
                    task,
                    item_by_key,
                    job_index=job_index,
                    job_tone=job_tone,
                )

        known = {
            "prepare_source",
            *relaxation_ids,
            *convergence_ids,
            *band_ids,
            *dos_ids,
            *fermi_surface_ids,
            *charge_density_ids,
            *elastic_ids,
            *phonon_ids,
            *wannier_ids,
        }
        other_tasks = [(task_id, task) for task_id, task in tasks.items() if task_id not in known and isinstance(task, dict)]
        if other_tasks:
            other_key = f"{root_key}:group:other"
            other_group = self._workflow_tree_item(
                "Other workflow routines",
                self._derive_group_status([self._workflow_task_status(task) for _, task in other_tasks]),
                {
                    "kind": "group",
                    "key": other_key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_tasks_progress_text([task for _, task in other_tasks]),
                    "task_ids": [task_id for task_id, _ in other_tasks],
                },
            )
            root.addChild(other_group)
            item_by_key[other_key] = other_group
            for task_id, task in other_tasks:
                self._add_workflow_task_children(
                    other_group,
                    task_id,
                    task,
                    item_by_key,
                    direct=True,
                    job_index=job_index,
                    job_tone=job_tone,
                )

    def _add_workflow_task_children(
        self,
        parent: QTreeWidgetItem,
        task_id: str,
        task: Dict[str, Any],
        item_by_key: Dict[str, QTreeWidgetItem],
        *,
        direct: bool = False,
        job_index: int,
        job_tone: int,
    ) -> None:
        children = [child for child in (task.get("children") or []) if isinstance(child, dict)]
        root_key = f"job:{job_index}"
        display_task_id = str(task_id or task.get("id") or "")
        if not children:
            key = f"{root_key}:task:{task_id}"
            label = (
                self._workflow_task_progress_label(task, display_task_id)
                if not direct
                else self._workflow_task_display_label(display_task_id, task)
            )
            item = self._workflow_tree_item(
                label,
                self._workflow_task_status(task),
                {
                    "kind": "task",
                    "key": key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "progress_label": self._workflow_task_progress_text(task),
                    "task_id": task_id,
                },
            )
            parent.addChild(item)
            item_by_key[key] = item
            return

        for index, child in enumerate(children):
            child_id = str(child.get("id") or index)
            key = f"{root_key}:child:{task_id}:{child_id}"
            item = self._workflow_tree_item(
                self._workflow_child_display_label(task_id, task, child),
                self._workflow_child_status(task, child),
                {
                    "kind": "child",
                    "key": key,
                    "job_index": job_index,
                    "job_tone": job_tone,
                    "task_id": task_id,
                    "child_id": child_id,
                },
            )
            parent.addChild(item)
            item_by_key[key] = item

    def _workflow_child_display_label(self, task_id: str, task: Dict[str, Any], child: Dict[str, Any]) -> str:
        child_id = str(child.get("id") or "")
        if task_id == "prepare_source":
            return "Convert structure" if child_id == "convert_structure" else "Prepare source"
        if task_id == "initial_relaxation":
            return "Initial relaxation"
        if task_id == "final_relaxation":
            return "Final relaxation"
        if task_id == "relaxation":
            return "Relaxation"
        if task_id == "cutoff_convergence":
            ecut = (child.get("params") or {}).get("ecutwfc")
            if ecut is not None:
                return f"ecutwfc {ecut} Ry SCF"
            if child_id.startswith("ecut"):
                return f"{child_id} SCF"
        if task_id == "kpoint_convergence":
            kmesh = (child.get("params") or {}).get("kmesh")
            if isinstance(kmesh, list) and len(kmesh) >= 3:
                return f"kmesh {int(kmesh[0])}x{int(kmesh[1])}x{int(kmesh[2])} SCF"
            if child_id.startswith("kmesh_"):
                return f"{child_id.replace('kmesh_', 'kmesh ')} SCF"
        if task_id in {"bands_workflow", "fat_bands_workflow"}:
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            stage_labels = {
                "scf": "SCF calculation",
                "pwbands": "Bands calculation",
                "bands": "Bands calculation",
                "bandx": "bands.x post-processing",
                "projwfc": "projwfc.x projections",
            }
            if stage in stage_labels:
                return stage_labels[stage]
        if task_id == "dos_workflow":
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            stage_labels = {
                "scf": "SCF calculation",
                "nscf": "NSCF calculation",
                "projwfc": "projwfc.x projections",
                "pdos_sumpdos": "sumpdos.x aggregation",
                "pdos_plotting": "PDOS plotting",
                "pdos_postprocess": "PDOS post-processing",
            }
            if stage in stage_labels:
                return stage_labels[stage]
        if task_id == "fermi_surface_workflow":
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            stage_labels = {
                "scf": "SCF calculation",
                "fermi_velocity": "fermi_velocity.x Fermi surface",
            }
            if stage in stage_labels:
                return stage_labels[stage]
        if task_id == "charge_density_workflow":
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            stage_labels = {
                "scf": "SCF calculation",
                "pp": "pp.x charge density",
            }
            if stage in stage_labels:
                return stage_labels[stage]
        if task_id == "elastic_workflow":
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            if stage == "elastic":
                return "thermo_pw.x elastic properties"
        if task_id == "phonon_workflow":
            stage = str((child.get("params") or {}).get("stage") or child_id).lower()
            stage_labels = {
                "scf": "SCF calculation",
                "ph": "ph.x phonon calculation",
                "q2r": "q2r.x force constants",
                "matdyn": "matdyn.x dispersion",
                "phdos": "matdyn.x phonon DOS",
                "ploting": "ploting band + PhDOS post-processing",
                "projected": "ploting band + PhDOS post-processing",
                "epc_fit_scf": "Dense SCF for a2F",
                "epc_scf": "Phonon SCF",
                "epc_ph": "ph.x interpolated EPC",
                "epc_q2r": "q2r.x EPC force constants",
                "epc_matdyn": "matdyn.x EPC frequencies",
                "epc_phdos": "matdyn.x EPC DOS",
                "lambda": "lambda.x EPC summary",
                "lambda_ploting": "lambda ploting post-processing",
            }
            if stage in stage_labels:
                return stage_labels[stage]
        if task_id == "fatband_csv_extraction":
            return "Extract projection CSV"
        if task_id == "fatband_plotting":
            return "Plot fat-band figures"

        label = str(child.get("label") or child_id or task.get("label") or "Workflow child")
        label = label.replace(".workflow", "")
        label = label.replace("_", " ")
        return label.strip() or "Workflow child"

    def _workflow_tree_item(self, label: str, status: str, data: Dict[str, Any]) -> QTreeWidgetItem:
        normalized = self._normalize_workflow_status(status)
        job_index = data.get("job_index") if isinstance(data, dict) else None
        try:
            job_record = self.job_queue[int(job_index)] if job_index is not None else None
        except Exception:
            job_record = None
        if isinstance(job_record, dict):
            job_status = self._normalize_workflow_status(job_record.get("status"))
            if job_status in {"stopped", "failed"} and normalized in {"running", "queued", "waiting", "retrying"}:
                normalized = job_status
        status_text = self._workflow_status_label(normalized)
        progress = str(data.get("progress_label") or "").strip()
        text = f"{label} | {status_text}"
        if progress:
            text = f"{text} | {progress}"
        item = QTreeWidgetItem([text])
        item_data = dict(data)
        item_data.setdefault("label", label)
        item.setData(0, Qt.ItemDataRole.UserRole, item_data)
        item.setIcon(0, self._workflow_status_icon(normalized))
        self._apply_job_tree_tone(item, item_data)
        return item

    def _apply_job_tree_tone(self, item: QTreeWidgetItem, item_data: Dict[str, Any]) -> None:
        try:
            tone_index = int(item_data.get("job_tone", 0)) % 2
        except Exception:
            tone_index = 0
        block_colors = [QColor("#d8e3ef"), QColor("#f9fbfe")]
        root_colors = [QColor("#b8cbdf"), QColor("#e7eef7")]
        kind = str(item_data.get("kind") or "")
        is_root = kind in {"workflow", "job"} and str(item_data.get("key") or "").count(":") == 1
        item.setBackground(0, root_colors[tone_index] if is_root else block_colors[tone_index])

    def _workflow_status_label(self, status: str) -> str:
        normalized = self._normalize_workflow_status(status)
        labels = {
            "completed": "finished",
            "running": "running",
            "failed": "failed",
            "continued": "continued",
            "retrying": "retrying",
            "queued": "queued",
            "waiting": "waiting",
            "stopped": "stopped",
        }
        return labels.get(normalized, normalized or "queued")

    def _normalize_workflow_status(self, status: Any) -> str:
        text = str(status or "").strip().lower()
        if text.startswith("running"):
            return "running"
        if text in {"complete", "completed", "done"}:
            return "completed"
        if text in {"fail", "failed", "error"}:
            return "failed"
        if text in {"continued", "warning", "warn"}:
            return "continued"
        if text in {"retry", "retrying"}:
            return "retrying"
        if text in {"stop", "stopped", "stopping"}:
            return "stopped"
        if text in {"queue", "queued"}:
            return "queued"
        if text in {"wait", "waiting"}:
            return "waiting"
        if text in {"start", "starting"}:
            return "running"
        return text or "queued"

    def _workflow_status_icon(self, status: str) -> QIcon:
        normalized = self._normalize_workflow_status(status)
        mapping = {
            "completed": ("✓", QColor("#ffffff"), QColor("#61ad45")),
            "running": ("▶", QColor("#ffffff"), QColor("#ff9800")),
            "failed": ("✕", QColor("#d64040"), None),
            "continued": ("!", QColor("#ffffff"), QColor("#b26a00")),
            "retrying": ("↻", QColor("#b26a00"), None),
            "queued": ("⏱", QColor("#6b7280"), None),
            "waiting": ("⏱", QColor("#6b7280"), None),
            "stopped": ("■", QColor("#4b5563"), None),
        }
        symbol, symbol_color, background = mapping.get(normalized, ("⏱", QColor("#6b7280"), None))
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if background is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(1, 1, 16, 16, 2, 2)
        painter.setPen(symbol_color)
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(14)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        painter.end()
        return QIcon(pixmap)

    def _job_output_end_icon(self) -> QIcon:
        pixmap = QPixmap(28, 28)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        outline_pen = QPen(QColor("#111111"), 2.2)
        outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        outline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRect(2, 2, 24, 24))
        self._draw_job_output_chevron(painter, 8, 9, 20, 5)
        self._draw_job_output_chevron(painter, 8, 15, 20, 5)

        painter.end()
        return QIcon(pixmap)

    def _draw_job_output_chevron(
        self,
        painter: QPainter,
        left_x: int,
        top_y: int,
        right_x: int,
        depth: int,
    ) -> None:
        middle_x = (left_x + right_x) // 2
        painter.drawLine(left_x, top_y, middle_x, top_y + depth)
        painter.drawLine(middle_x, top_y + depth, right_x, top_y)

    def _workflow_task_status(self, task: Dict[str, Any]) -> str:
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        if str(task.get("status") or "").strip().lower() == "failed" and result.get("continued_with_last_tried"):
            return "continued"
        return self._normalize_workflow_status(task.get("status") or "queued")

    def _workflow_child_status(self, task: Dict[str, Any], child: Dict[str, Any]) -> str:
        status = self._normalize_workflow_status(child.get("status") or task.get("status") or "queued")
        if status == "failed":
            result = child.get("result") if isinstance(child.get("result"), dict) else {}
            if result.get("continued_with_last_tried"):
                return "continued"
        return status

    def _derive_group_status(self, statuses: Sequence[Any]) -> str:
        normalized = [self._normalize_workflow_status(status) for status in statuses if str(status or "").strip()]
        if not normalized:
            return "waiting"
        for candidate in ["failed", "continued", "retrying", "running", "queued", "waiting", "stopped"]:
            if candidate in normalized:
                return candidate
        if all(status == "completed" for status in normalized):
            return "completed"
        return normalized[0]

    def _workflow_task_progress_label(self, task: Dict[str, Any], task_id: str = "") -> str:
        label = self._workflow_task_display_label(task_id or str(task.get("id") or ""), task)
        progress = self._workflow_task_progress_text(task)
        if progress:
            return f"{label} ({progress.replace(' jobs', '').replace(' job', '')})"
        return label

    def _workflow_task_display_label(self, task_id: str, task: Dict[str, Any]) -> str:
        normalized = str(task_id or task.get("id") or "").strip().lower()
        if normalized in {"bands_workflow", "fat_bands_workflow"}:
            return "Band structure workflow"
        if normalized == "dos_workflow":
            return "Projected density of states workflow"
        if normalized == "fermi_surface_workflow":
            return "Fermi surface workflow"
        if normalized == "charge_density_workflow":
            return "Charge density workflow"
        if normalized == "elastic_workflow":
            return "Elastic properties workflow"
        if normalized == "phonon_workflow":
            return "Phonon workflow"
        if normalized == "wannier90_workflow":
            return "Wannier90 + SCDM workflow routine"
        if normalized == "fatband_csv_extraction":
            return "Fat-band CSV extraction"
        if normalized == "fatband_plotting":
            return "Fat-band plotting"
        label = str(task.get("label") or "").strip()
        if label:
            return label
        if normalized:
            return normalized.replace("_", " ")
        return "Workflow task"

    def _workflow_task_progress_text(self, task: Dict[str, Any]) -> str:
        try:
            completed = int(task.get("completed") or 0)
        except Exception:
            completed = 0
        try:
            planned = int(task.get("planned_total") or len(task.get("children") or []))
        except Exception:
            planned = 0
        status = self._normalize_workflow_status(task.get("status"))
        star = "*" if task.get("may_expand") and status not in {"completed", "failed"} else ""
        if planned > 0:
            noun = "job" if planned == 1 else "jobs"
            return f"{completed}/{planned}{star} {noun}"
        return ""

    def _workflow_tasks_progress_text(self, tasks: Sequence[Any]) -> str:
        completed = 0
        planned = 0
        may_expand = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            try:
                completed += int(task.get("completed") or 0)
            except Exception:
                pass
            try:
                planned += int(task.get("planned_total") or len(task.get("children") or []))
            except Exception:
                pass
            status = self._normalize_workflow_status(task.get("status"))
            may_expand = may_expand or bool(task.get("may_expand") and status not in {"completed", "failed"})
        if planned <= 0:
            return ""
        noun = "job" if planned == 1 else "jobs"
        star = "*" if may_expand else ""
        return f"{completed}/{planned}{star} {noun}"

    def _update_workflow_tree_detail(self) -> None:
        record = self._selected_job_record()
        if not self._is_workflow_record(record):
            return
        item = self.jobStatusTree.currentItem() if hasattr(self, "jobStatusTree") else None
        data = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else {}
        if not isinstance(data, dict):
            self._update_job_detail_preview()
            return

        state, state_path, error = self._load_workflow_state_for_record(record)
        if state is None:
            self._set_job_detail_text(error)
            self._hide_phonon_projected_preview()
            self._hide_workflow_lambda_output()
            return
        root = self._workflow_root_for_record(record)
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        kind = str(data.get("kind") or "workflow")
        lines: List[str] = []
        task_id = ""
        stage = ""

        if kind == "child":
            task_id = str(data.get("task_id") or "")
            child_id = str(data.get("child_id") or "")
            task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
            child = self._workflow_child_by_id(task, child_id)
            stage = str((child.get("params") or {}).get("stage") or child.get("id") or "").strip().lower()
            lines = self._workflow_child_detail_lines(state, state_path, root, task_id, task, child)
        elif kind == "task":
            task_id = str(data.get("task_id") or "")
            task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
            lines = self._workflow_task_detail_lines(state, state_path, root, task_id, task)
        elif kind == "group":
            task_ids = [str(task_id) for task_id in data.get("task_ids") or []]
            lines = self._workflow_group_detail_lines(
                state,
                state_path,
                root,
                str(data.get("label") or (item.text(0) if item else "Workflow group")),
                task_ids,
            )
        else:
            lines = self._workflow_root_detail_lines(record, state, state_path, root)

        self._set_job_detail_text("\n".join(lines))
        if kind == "child" and task_id == "phonon_workflow" and stage == "lambda":
            self._hide_workflow_plot_preview()
            self._update_workflow_lambda_output(root, state, state_path, task, child, record)
            return
        self._hide_workflow_lambda_output()
        preview_paths = self._workflow_plot_preview_paths_from_state(root, state, task_id=task_id, stage=stage, kind=kind)
        if preview_paths.get("image"):
            self._update_workflow_plot_preview(preview_paths)
        else:
            self._hide_workflow_plot_preview()

    def _workflow_child_by_id(self, task: Dict[str, Any], child_id: str) -> Dict[str, Any]:
        for child in task.get("children") or []:
            if isinstance(child, dict) and str(child.get("id") or "") == child_id:
                return child
        return {}

    def _workflow_root_detail_lines(
        self,
        record: Dict[str, Any],
        state: Dict[str, Any],
        state_path: Optional[Path],
        root: Optional[Path],
    ) -> List[str]:
        config = record.get("config")
        name = config.name if isinstance(config, JobConfig) and config.name else str(state.get("workflow_id") or "workflow")
        lines = [
            f"Workflow: {name}",
            f"Status: {state.get('status', record.get('status', 'Queued'))}",
            f"Current task: {state.get('current_task') or '-'}",
            f"Workflow root: {root or '<unset>'}",
            f"Workflow state: {state_path or '<unset>'}",
        ]
        notices = state.get("notices") if isinstance(state.get("notices"), list) else []
        if notices:
            lines.extend(["", "Notices:"])
            for notice in notices:
                text = str(notice or "").strip()
                if text:
                    lines.append(f"  - {text}")
        lines.extend(["", "Tasks:"])
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        for task_id, task in tasks.items():
            if isinstance(task, dict):
                lines.append(f"  - {task.get('label', task_id)}: {self._workflow_task_status(task)}")
        return lines

    def _workflow_group_detail_lines(
        self,
        state: Dict[str, Any],
        state_path: Optional[Path],
        root: Optional[Path],
        label: str,
        task_ids: Sequence[str],
    ) -> List[str]:
        clean_label = re.sub(r"^[^\w]+\s*", "", label).strip()
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        lines = [
            f"Workflow group: {clean_label}",
            f"Workflow state: {state_path or '<unset>'}",
            "",
            "Steps:",
        ]
        for task_id in task_ids:
            task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
            if not task:
                continue
            lines.append(f"  - {self._workflow_task_progress_label(task)}: {self._workflow_task_status(task)}")
            for child in task.get("children") or []:
                if isinstance(child, dict):
                    lines.append(
                        f"      {self._workflow_child_display_label(task_id, task, child)}: "
                        f"{self._workflow_child_status(task, child)}"
                    )
        return lines

    def _workflow_task_detail_lines(
        self,
        state: Dict[str, Any],
        state_path: Optional[Path],
        root: Optional[Path],
        task_id: str,
        task: Dict[str, Any],
    ) -> List[str]:
        task_status = self._workflow_task_status(task)
        lines = [
            f"Workflow routine: {self._workflow_task_display_label(task_id, task)}",
            f"Task id: {task_id}",
            f"Type: {task.get('type', '-')}",
            f"Status: {task_status}",
            f"Progress: {task.get('completed', 0)}/{task.get('planned_total', len(task.get('children') or []))}",
            f"Workflow state: {state_path or '<unset>'}",
        ]
        result = task.get("result")
        if task_status == "continued":
            lines.append("Continued with fallback: yes")
        lines.extend(["", "Children:"])
        for child in task.get("children") or []:
            if isinstance(child, dict):
                lines.append(
                    f"  - {self._workflow_child_display_label(task_id, task, child)}: "
                    f"{self._workflow_child_status(task, child)}"
                )
        if result:
            lines.extend(["", "Result:", json.dumps(result, indent=2)])
        running_child = str(task.get("running_child") or "")
        child = self._workflow_child_by_id(task, running_child) if running_child else {}
        if child:
            lines.extend(["", "Running child output:", self._workflow_child_output_text(root, child) or "(no output yet)"])
        return lines

    def _workflow_child_detail_lines(
        self,
        state: Dict[str, Any],
        state_path: Optional[Path],
        root: Optional[Path],
        task_id: str,
        task: Dict[str, Any],
        child: Dict[str, Any],
    ) -> List[str]:
        label = self._workflow_child_display_label(task_id, task, child)
        lines = [
            f"Workflow child: {label}",
            f"Task: {task.get('label', task_id)}",
            f"Status: {self._workflow_child_status(task, child)}",
            f"Workflow state: {state_path or '<unset>'}",
            f"Workdir: {self._workflow_abs_path(root, child.get('workdir')) or '<unset>'}",
            f"Input: {self._workflow_abs_path(root, child.get('input')) or '<unset>'}",
            f"Output: {self._workflow_abs_path(root, child.get('output')) or '<unset>'}",
        ]
        result = child.get("result")
        if result:
            lines.extend(["", "Result:", json.dumps(result, indent=2)])
        lines.extend(["", "Calculation output:", self._workflow_child_output_text(root, child) or "(no calculation output yet)"])
        return lines

    def _workflow_child_output_text(self, root: Optional[Path], child: Dict[str, Any]) -> str:
        return self._workflow_log_text(root, child.get("output"))

    def _workflow_log_text(self, root: Optional[Path], relative_or_absolute: Any) -> str:
        path = self._workflow_abs_path(root, relative_or_absolute)
        if path is None or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _workflow_abs_path(self, root: Optional[Path], relative_or_absolute: Any) -> Optional[Path]:
        text = str(relative_or_absolute or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if path.is_absolute():
            return path
        return (root / path) if root is not None else path

    def _first_existing_workflow_path(self, root: Optional[Path], candidates: Sequence[Any]) -> Optional[Path]:
        for candidate in candidates:
            path = self._workflow_abs_path(root, candidate)
            if path is not None and path.is_file():
                return path
        return None

    def _workflow_result_mapping(self, state: Dict[str, Any], key: str) -> Dict[str, Any]:
        results = state.get("results") if isinstance(state.get("results"), dict) else {}
        value = results.get(key) if isinstance(results, dict) else {}
        return value if isinstance(value, dict) else {}

    def _workflow_task_result_mapping(self, state: Dict[str, Any], task_id: str, key: str = "") -> Dict[str, Any]:
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        if not key:
            return result
        value = result.get(key) if isinstance(result, dict) else {}
        return value if isinstance(value, dict) else {}

    def _workflow_plot_preview_paths_from_state(
        self,
        root: Optional[Path],
        state: Dict[str, Any],
        *,
        task_id: str = "",
        stage: str = "",
        kind: str = "",
    ) -> Dict[str, Optional[Path]]:
        if root is None:
            return {}
        task_id = str(task_id or "").strip()
        stage = str(stage or "").strip().lower()
        kind = str(kind or "").strip().lower()
        if task_id == "phonon_workflow" and stage in {"ploting", "projected"}:
            return self._phonon_projected_paths_from_state(root, state)
        if task_id == "phonon_workflow" and stage == "lambda_ploting":
            return self._epc_lambda_ploting_paths_from_state(root, state)
        if task_id in {"bands_workflow", "fat_bands_workflow", "fatband_plotting"}:
            return self._band_plot_paths_from_state(root, state, task_id=task_id, stage=stage, kind=kind)
        if task_id == "dos_workflow" and (
            kind == "task" or stage in {"pdos_plotting", "pdos_postprocess"}
        ):
            return self._pdos_plot_paths_from_state(root, state)
        return {}

    def _band_plot_paths_from_state(
        self,
        root: Path,
        state: Dict[str, Any],
        *,
        task_id: str,
        stage: str,
        kind: str,
    ) -> Dict[str, Optional[Path]]:
        data_dir = root / "data_and_plots"
        results = state.get("results") if isinstance(state.get("results"), dict) else {}
        fatband_result = self._workflow_result_mapping(state, "fatband")
        fatband_task_result = self._workflow_task_result_mapping(state, "fat_bands_workflow", "fatband")
        fatband_plots = (
            results.get("fatband_plots")
            or fatband_result.get("plots")
            or fatband_task_result.get("plots")
            or {}
        )
        has_fatband = task_id in {"fat_bands_workflow", "fatband_plotting"} or bool(fatband_plots)
        if has_fatband:
            plot_rel = fatband_plots.get("atom-orbital-panels") if isinstance(fatband_plots, dict) else ""
            script_rel = (
                fatband_result.get("replot_script")
                or fatband_task_result.get("replot_script")
                or "data_and_plots/replot_fatbands.py"
            )
            folder = data_dir
            image_path = self._first_existing_workflow_path(
                root,
                [
                    plot_rel,
                    "data_and_plots/fatband_atom_orbital_panels.png",
                ],
            )
            script_path = self._workflow_abs_path(root, script_rel)
            return {
                "image": image_path,
                "plot": self._workflow_abs_path(root, plot_rel) or data_dir / "fatband_atom_orbital_panels.png",
                "folder": folder if folder.is_dir() else None,
                "script": script_path if script_path is not None and script_path.is_file() else None,
                "title": "Fat-band plot",
            }

        band_result = self._workflow_result_mapping(state, "band_structure")
        band_task_result = self._workflow_task_result_mapping(state, "bands_workflow", "band_structure")
        plot_rel = (
            results.get("band_structure_plot")
            or band_result.get("plot")
            or band_task_result.get("plot")
            or "data_and_plots/band_structure.png"
        )
        script_rel = (
            band_result.get("replot_script")
            or band_task_result.get("replot_script")
            or "data_and_plots/replot_band_structure.py"
        )
        image_path = self._first_existing_workflow_path(
            root,
            [
                plot_rel,
                "data_and_plots/band_structure.png",
            ],
        )
        script_path = self._workflow_abs_path(root, script_rel)
        return {
            "image": image_path,
            "plot": self._workflow_abs_path(root, plot_rel),
            "folder": data_dir if data_dir.is_dir() else None,
            "script": script_path if script_path is not None and script_path.is_file() else None,
            "title": "Band structure plot",
        }

    def _pdos_plot_paths_from_state(self, root: Path, state: Dict[str, Any]) -> Dict[str, Optional[Path]]:
        data_dir = root / "data_and_plots"
        results = state.get("results") if isinstance(state.get("results"), dict) else {}
        dos_result = self._workflow_result_mapping(state, "dos")
        pdos_result = dos_result.get("pdos_postprocess") if isinstance(dos_result.get("pdos_postprocess"), dict) else {}
        combined_rel = (
            results.get("pdos_combined_plot")
            or pdos_result.get("combined_band_dos_plot")
            or dos_result.get("combined_band_dos_plot")
        )
        total_rel = (
            results.get("dos_summary_plot")
            or pdos_result.get("dos_summary_plot")
            or dos_result.get("dos_summary_plot")
            or "data_and_plots/density_of_states.png"
        )
        script_rel = (
            results.get("pdos_replot_script")
            or pdos_result.get("replot_script")
            or dos_result.get("replot_script")
            or "data_and_plots/replot_pdos.py"
        )
        image_path = self._first_existing_workflow_path(
            root,
            [
                combined_rel,
                "data_and_plots/fatband_plus_dos.png",
                "data_and_plots/band_plus_dos.png",
                total_rel,
                "data_and_plots/density_of_states.png",
            ],
        )
        script_path = self._workflow_abs_path(root, script_rel)
        return {
            "image": image_path,
            "plot": self._workflow_abs_path(root, combined_rel or total_rel),
            "folder": data_dir if data_dir.is_dir() else None,
            "script": script_path if script_path is not None and script_path.is_file() else None,
            "title": "PDOS plot",
        }

    def _phonon_projected_paths_from_state(self, root: Optional[Path], state: Dict[str, Any]) -> Dict[str, Optional[Path]]:
        if root is None:
            return {}
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        phonon_task = tasks.get("phonon_workflow") if isinstance(tasks.get("phonon_workflow"), dict) else {}
        task_result = phonon_task.get("result") if isinstance(phonon_task.get("result"), dict) else {}
        results = state.get("results") if isinstance(state.get("results"), dict) else {}
        projected_child: Dict[str, Any] = {}
        for child in phonon_task.get("children") or []:
            if isinstance(child, dict) and str(child.get("id") or "") in {"ploting", "projected"}:
                projected_child = child
                break
        child_result = projected_child.get("result") if isinstance(projected_child.get("result"), dict) else {}

        plot_rel = (
            child_result.get("combined_plot")
            or results.get("phonon_projected_combined_plot")
            or task_result.get("projected_combined_plot")
        )
        script_rel = (
            child_result.get("replot_script")
            or results.get("phonon_projected_replot_script")
            or task_result.get("projected_replot_script")
        )
        default_dir = root / "data_and_plots"
        plot_path = self._workflow_abs_path(root, plot_rel) if plot_rel else default_dir / "phonon_projected_band_dos.png"
        script_path = self._workflow_abs_path(root, script_rel) if script_rel else default_dir / "replot_phonon_projected.py"
        folder = script_path.parent if script_path is not None else default_dir
        image_path = plot_path
        return {
            "image": image_path if image_path is not None and image_path.is_file() else None,
            "plot": plot_path if plot_path is not None and plot_path.is_file() else None,
            "folder": folder if folder is not None and folder.is_dir() else None,
            "script": script_path if script_path is not None and script_path.is_file() else None,
            "title": "Phonon band + PhDOS plot",
        }

    def _epc_lambda_ploting_paths_from_state(self, root: Path, state: Dict[str, Any]) -> Dict[str, Optional[Path]]:
        data_dir = root / "data_and_plots"
        results = state.get("results") if isinstance(state.get("results"), dict) else {}
        lambda_result = results.get("phonon_lambda_ploting") if isinstance(results.get("phonon_lambda_ploting"), dict) else {}
        plot_rel = lambda_result.get("combined_plot") or results.get("phonon_epc_alpha2f_lambda_plot")
        script_rel = lambda_result.get("replot_script") or results.get("phonon_epc_alpha2f_lambda_replot_script")
        plot_path = self._workflow_abs_path(root, plot_rel) if plot_rel else data_dir / "epc_phonon_linewidth_alpha2f_lambda.png"
        script_path = self._workflow_abs_path(root, script_rel) if script_rel else data_dir / "replot_epc_phonon_alpha2f_lambda.py"
        image_path = plot_path
        return {
            "image": image_path if image_path is not None and image_path.is_file() else None,
            "plot": plot_path if plot_path is not None and plot_path.is_file() else None,
            "folder": data_dir if data_dir.is_dir() else None,
            "script": script_path if script_path is not None and script_path.is_file() else None,
            "title": "EPC linewidth + alpha2F/lambda plot",
        }

    def _update_workflow_plot_preview(self, paths: Dict[str, Optional[Path]]) -> None:
        if not hasattr(self, "phononProjectedPreviewFrame"):
            return
        image_path = paths.get("image")
        if image_path is None:
            self._hide_workflow_plot_preview()
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._hide_workflow_plot_preview()
            return
        target_size = self.phononProjectedPreviewImage.size()
        if target_size.width() < 20 or target_size.height() < 20:
            target_size = QSize(520, 220)
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        title = str(paths.get("title") or "Workflow plot")
        self.phononProjectedPreviewImage.setPixmap(scaled)
        self.phononProjectedPreviewTitle.setText(f"{title}: {image_path.name}")
        self._workflowPlotPreviewPaths = paths
        self._phononProjectedPreviewPaths = paths
        self.openPhononProjectedImageButton.setEnabled(paths.get("image") is not None)
        self.openPhononProjectedFolderButton.setEnabled(paths.get("folder") is not None)
        self.editPhononProjectedScriptButton.setEnabled(paths.get("script") is not None)
        self.runPhononProjectedScriptButton.setEnabled(paths.get("script") is not None)
        self.phononProjectedPreviewFrame.show()
        if hasattr(self, "jobDetailOutputStack"):
            self.jobDetailOutputStack.setCurrentWidget(self.phononProjectedPreviewFrame)
        if hasattr(self, "jumpToJobOutputEndButton"):
            self.jumpToJobOutputEndButton.setEnabled(False)

    def _hide_workflow_plot_preview(self) -> None:
        if not hasattr(self, "phononProjectedPreviewFrame"):
            return
        self._workflowPlotPreviewPaths = {}
        self._phononProjectedPreviewPaths = {}
        self.phononProjectedPreviewImage.clear()
        self.phononProjectedPreviewFrame.hide()
        if hasattr(self, "jobDetailOutputStack"):
            self.jobDetailOutputStack.setCurrentWidget(self.jobDetailPreview)
        if hasattr(self, "jumpToJobOutputEndButton"):
            self.jumpToJobOutputEndButton.setEnabled(True)

    def _lambda_output_sort_key(self, path: Path) -> Tuple[int, float, str]:
        if path.name == "lambda.out":
            return (0, 0.0, path.name)
        match = re.search(r"lambda_degauss_([0-9]+)p([0-9]+)\.out$", path.name)
        if match:
            try:
                value = float(f"{match.group(1)}.{match.group(2)}")
            except ValueError:
                value = 0.0
            return (1, value, path.name)
        return (2, 0.0, path.name)

    def _workflow_lambda_output_files(self, root: Optional[Path]) -> List[Path]:
        if root is None:
            return []
        phonon_dir = root / "phonon"
        if not phonon_dir.is_dir():
            return []
        files = [path for path in phonon_dir.glob("lambda*.out") if path.is_file()]
        return sorted(files, key=self._lambda_output_sort_key)

    def _workflow_manifest_path_for_record(self, record: Dict[str, Any], root: Optional[Path]) -> Optional[Path]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        candidates = [
            metadata.get("workflow_manifest"),
            (root / "workflow_manifest.json") if root is not None else None,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if path.is_file():
                return path
        return None

    def _update_workflow_lambda_output(
        self,
        root: Optional[Path],
        state: Dict[str, Any],
        state_path: Optional[Path],
        task: Dict[str, Any],
        child: Dict[str, Any],
        record: Dict[str, Any],
    ) -> None:
        if not hasattr(self, "workflowLambdaOutputFrame"):
            return
        popup_open = self._workflow_lambda_output_popup_open()
        previous = ""
        current_index = self.workflowLambdaOutputCombo.currentIndex()
        old_paths = list(getattr(self, "_workflowLambdaOutputPaths", []))
        if 0 <= current_index < len(old_paths):
            previous = old_paths[current_index].name
        files = self._workflow_lambda_output_files(root)
        if not files:
            output_path = self._workflow_abs_path(root, child.get("output"))
            if output_path is not None:
                files = [output_path]
        self._workflowLambdaOutputPaths = files
        self._workflowLambdaContext = {
            "root": root,
            "state_path": state_path,
            "manifest": self._workflow_manifest_path_for_record(record, root),
        }

        context_key = str(state_path or root or "")
        if context_key != getattr(self, "_workflowLambdaMuContextKey", ""):
            self._workflowLambdaMuContextKey = context_key
            self._workflowLambdaMuDirty = False
            self._workflowLambdaPendingMu = ""
        phonon_options = ((task.get("params") or {}).get("phonon_options") or {})
        if isinstance(phonon_options, dict):
            mu = str(phonon_options.get("lambda_mu") or "0.10")
            current_mu = self.workflowLambdaMuLineEdit.text().strip()
            pending_mu = str(getattr(self, "_workflowLambdaPendingMu", "") or "").strip()
            dirty = bool(getattr(self, "_workflowLambdaMuDirty", False))
            preserve_user_mu = bool(current_mu) and (
                (dirty and current_mu != mu)
                or (pending_mu and current_mu == pending_mu and mu != pending_mu)
            )
            if not preserve_user_mu and not self.workflowLambdaMuLineEdit.hasFocus():
                self.workflowLambdaMuLineEdit.setText(mu)
                current_mu = mu
            if current_mu == mu:
                self._workflowLambdaMuDirty = False
                if pending_mu == mu:
                    self._workflowLambdaPendingMu = ""

        old_names = [path.name for path in old_paths]
        new_names = [path.name for path in files]
        selected = 0
        if previous:
            for index, path in enumerate(files):
                if path.name == previous:
                    selected = index
                    break
        elif files:
            for index, path in enumerate(files):
                if path.name == "lambda.out":
                    selected = index
                    break
        if not popup_open:
            self.workflowLambdaOutputCombo.blockSignals(True)
            try:
                if old_names != new_names:
                    self.workflowLambdaOutputCombo.clear()
                    for path in files:
                        self.workflowLambdaOutputCombo.addItem(path.name)
                if files and self.workflowLambdaOutputCombo.currentIndex() != selected:
                    self.workflowLambdaOutputCombo.setCurrentIndex(selected)
            finally:
                self.workflowLambdaOutputCombo.blockSignals(False)

        result = child.get("result") if isinstance(child.get("result"), dict) else {}
        lambda_result = result.get("lambda") if isinstance(result.get("lambda"), dict) else {}
        tc_match = lambda_result.get("tc_match") if isinstance(lambda_result.get("tc_match"), dict) else {}
        selected_gamma = str(tc_match.get("elph_gamma_file") or "-")
        self.workflowLambdaOutputTitle.setText(f"lambda.x outputs | selected elph gamma: {selected_gamma}")
        self.workflowLambdaPrevButton.setEnabled(len(files) > 1)
        self.workflowLambdaNextButton.setEnabled(len(files) > 1)
        self.workflowLambdaOpenFolderButton.setEnabled(root is not None and (root / "phonon").is_dir())
        self.workflowLambdaRerunButton.setEnabled(self._workflowLambdaContext.get("manifest") is not None)
        if not popup_open:
            self._display_selected_lambda_output()
        self.workflowLambdaOutputFrame.show()
        if not popup_open:
            self.jobDetailOutputStack.setCurrentWidget(self.workflowLambdaOutputFrame)
        self.jumpToJobOutputEndButton.setEnabled(False)

    def _workflow_lambda_output_popup_open(self) -> bool:
        if not hasattr(self, "workflowLambdaOutputCombo"):
            return False
        try:
            view = self.workflowLambdaOutputCombo.view()
            return bool(view is not None and view.isVisible())
        except Exception:
            return False

    def _mark_workflow_lambda_mu_edited(self, _text: str) -> None:
        self._workflowLambdaMuDirty = True

    def _display_selected_lambda_output(self) -> None:
        files = getattr(self, "_workflowLambdaOutputPaths", [])
        index = self.workflowLambdaOutputCombo.currentIndex() if hasattr(self, "workflowLambdaOutputCombo") else -1
        if index < 0 or index >= len(files):
            placeholder = "(no lambda output yet)"
            if self.workflowLambdaOutputText.toPlainText() != placeholder:
                self.workflowLambdaOutputText.setPlainText(placeholder)
            self._workflowLambdaDisplayedPath = None
            self._workflowLambdaDisplayedText = placeholder
            return
        path = files[index]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = f"Could not read {path}: {exc}"
        if self._workflowLambdaDisplayedPath == path and self._workflowLambdaDisplayedText == text:
            return
        vertical = self.workflowLambdaOutputText.verticalScrollBar()
        horizontal = self.workflowLambdaOutputText.horizontalScrollBar()
        keep_scroll = self._workflowLambdaDisplayedPath == path
        old_vertical = vertical.value()
        old_horizontal = horizontal.value()
        was_at_bottom = vertical.value() >= max(0, vertical.maximum() - 2)
        self.workflowLambdaOutputText.setPlainText(text)
        if keep_scroll:
            if was_at_bottom:
                vertical.setValue(vertical.maximum())
            else:
                vertical.setValue(min(old_vertical, vertical.maximum()))
            horizontal.setValue(min(old_horizontal, horizontal.maximum()))
        self._workflowLambdaDisplayedPath = path
        self._workflowLambdaDisplayedText = text

    def _workflow_lambda_output_combo_changed(self, _index: int) -> None:
        self._display_selected_lambda_output()

    def _workflow_lambda_previous_output(self) -> None:
        count = self.workflowLambdaOutputCombo.count()
        if count <= 0:
            return
        self.workflowLambdaOutputCombo.setCurrentIndex((self.workflowLambdaOutputCombo.currentIndex() - 1) % count)

    def _workflow_lambda_next_output(self) -> None:
        count = self.workflowLambdaOutputCombo.count()
        if count <= 0:
            return
        self.workflowLambdaOutputCombo.setCurrentIndex((self.workflowLambdaOutputCombo.currentIndex() + 1) % count)

    def _workflow_lambda_open_folder(self) -> None:
        root = (getattr(self, "_workflowLambdaContext", {}) or {}).get("root")
        folder = (root / "phonon") if isinstance(root, Path) else None
        self._browse_local_path(folder, "lambda output folder")

    def _workflow_lambda_rerun_with_mu(self) -> None:
        context = getattr(self, "_workflowLambdaContext", {}) or {}
        manifest = context.get("manifest")
        state_path = context.get("state_path")
        if not isinstance(manifest, Path) or not manifest.is_file():
            self.statusbar.showMessage("Workflow manifest is not available for lambda rerun.", 5000)
            return
        mu = self.workflowLambdaMuLineEdit.text().strip()
        if not mu:
            self.statusbar.showMessage("Enter a mu value before rerunning lambda.x.", 5000)
            return
        process = getattr(self, "_workflowLambdaRerunProcess", None)
        if process is not None and process.poll() is None:
            self.statusbar.showMessage("lambda.x rerun is already running.", 5000)
            return
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "workflow_runner.py"),
            "rerun-lambda",
            str(manifest),
            "--mu",
            mu,
        ]
        if isinstance(state_path, Path):
            cmd.extend(["--state", str(state_path)])
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self.statusbar.showMessage(f"Could not rerun lambda.x: {exc}", 7000)
            return
        self._workflowLambdaRerunProcess = proc
        self._workflowLambdaPendingMu = mu
        self._workflowLambdaMuDirty = True
        self.workflowLambdaRerunButton.setEnabled(False)
        self.statusbar.showMessage("Rerunning lambda.x with updated mu...", 5000)
        QTimer.singleShot(700, lambda proc=proc: self._poll_workflow_lambda_rerun_process(proc))

    def _poll_workflow_lambda_rerun_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            if self._selected_job_is_workflow():
                self._update_workflow_tree_detail()
            QTimer.singleShot(1000, lambda proc=proc: self._poll_workflow_lambda_rerun_process(proc))
            return
        stdout, stderr = proc.communicate()
        self._workflowLambdaRerunProcess = None
        self.workflowLambdaRerunButton.setEnabled(True)
        if proc.returncode == 0:
            message = (stdout or "").strip().splitlines()[-1] if (stdout or "").strip() else "lambda.x rerun finished."
            self.statusbar.showMessage(message, 7000)
            if self.workflowLambdaMuLineEdit.text().strip() == str(getattr(self, "_workflowLambdaPendingMu", "") or "").strip():
                self._workflowLambdaMuDirty = False
                self._workflowLambdaPendingMu = ""
        else:
            message = (stderr or stdout or "lambda.x rerun failed.").strip().splitlines()[-1]
            self.statusbar.showMessage(message, 9000)
        if self._selected_job_is_workflow():
            self._sync_job_status_table()
            self._update_workflow_tree_detail()

    def _hide_workflow_lambda_output(self) -> None:
        if not hasattr(self, "workflowLambdaOutputFrame"):
            return
        self._workflowLambdaOutputPaths = []
        self._workflowLambdaContext = {}
        self._workflowLambdaMuDirty = False
        self._workflowLambdaPendingMu = ""
        self._workflowLambdaMuContextKey = ""
        self._workflowLambdaDisplayedPath = None
        self._workflowLambdaDisplayedText = ""
        self.workflowLambdaOutputText.clear()
        self.workflowLambdaOutputCombo.clear()
        self.workflowLambdaOutputFrame.hide()
        if hasattr(self, "jobDetailOutputStack"):
            self.jobDetailOutputStack.setCurrentWidget(self.jobDetailPreview)
        if hasattr(self, "jumpToJobOutputEndButton"):
            self.jumpToJobOutputEndButton.setEnabled(True)

    def _update_phonon_projected_preview(self, root: Optional[Path], state: Dict[str, Any]) -> None:
        if not hasattr(self, "phononProjectedPreviewFrame"):
            return
        paths = self._phonon_projected_paths_from_state(root, state)
        self._update_workflow_plot_preview(paths)

    def _hide_phonon_projected_preview(self) -> None:
        self._hide_workflow_plot_preview()

    def _open_phonon_projected_image(self) -> None:
        paths = getattr(self, "_workflowPlotPreviewPaths", {}) or getattr(self, "_phononProjectedPreviewPaths", {}) or {}
        self._open_local_path(paths.get("image"), "workflow plot")

    def _open_phonon_projected_folder(self) -> None:
        paths = getattr(self, "_workflowPlotPreviewPaths", {}) or getattr(self, "_phononProjectedPreviewPaths", {}) or {}
        self._browse_local_path(paths.get("folder"), "workflow plot folder")

    def _edit_phonon_projected_script(self) -> None:
        paths = getattr(self, "_workflowPlotPreviewPaths", {}) or getattr(self, "_phononProjectedPreviewPaths", {}) or {}
        self._open_local_path(paths.get("script"), "workflow replot script")

    def _run_phonon_projected_replot_script(self) -> None:
        paths = getattr(self, "_workflowPlotPreviewPaths", {}) or getattr(self, "_phononProjectedPreviewPaths", {}) or {}
        script = paths.get("script")
        if script is None or not script.is_file():
            self.statusbar.showMessage("Replot script is not available yet.", 4000)
            return
        process = getattr(self, "_phononProjectedReplotProcess", None)
        if process is not None and process.poll() is None:
            self.statusbar.showMessage("Replot script is already running.", 4000)
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self.statusbar.showMessage(f"Could not run replot script: {exc}", 6000)
            return
        self._phononProjectedReplotProcess = proc
        self.runPhononProjectedScriptButton.setEnabled(False)
        self.statusbar.showMessage("Running replot script...", 4000)
        QTimer.singleShot(500, lambda proc=proc: self._poll_phonon_projected_replot_process(proc))

    def _poll_phonon_projected_replot_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            QTimer.singleShot(500, lambda proc=proc: self._poll_phonon_projected_replot_process(proc))
            return
        stdout, stderr = proc.communicate()
        self._phononProjectedReplotProcess = None
        self.runPhononProjectedScriptButton.setEnabled(True)
        if proc.returncode == 0:
            message = (stdout or "").strip().splitlines()[-1] if (stdout or "").strip() else "Replot finished."
            self.statusbar.showMessage(message, 6000)
        else:
            message = (stderr or stdout or "Replot failed.").strip().splitlines()[-1]
            self.statusbar.showMessage(message, 8000)
        if self._selected_job_is_workflow():
            self._update_workflow_tree_detail()

    def _update_job_detail_preview(self) -> None:
        self._hide_workflow_plot_preview()
        self._hide_workflow_lambda_output()
        index = self._current_job_index()
        if index is None or index >= len(self.job_queue):
            self.jobDetailPreview.clear()
            return

        record = self.job_queue[index]
        config = record.get("config")
        if not isinstance(config, JobConfig):
            self.jobDetailPreview.clear()
            return

        pseudo_dir = config.pseudo_directory or "<unset>"
        metadata = record.get("metadata") or {}
        display_name = self._job_display_name(record) or (config.name or config.sanitized_name())
        lines = [
            f"Job: {display_name}",
            f"Calculation: {config.calculation}",
            f"Queue: {config.queue or '-'}",
            f"MPI processes: {config.mpi_procs}",
            f"Working directory: {config.working_directory or '<current>'}",
            f"Pseudo directory: {pseudo_dir}",
        ]
        if metadata.get("log_file"):
            lines.append(f"Log file: {metadata.get('log_file')}")
        if metadata.get("workflow_root"):
            lines.extend(
                [
                    f"Workflow root: {metadata.get('workflow_root')}",
                    f"Workflow manifest: {metadata.get('workflow_manifest') or '<unset>'}",
                    f"Workflow state: {metadata.get('workflow_state') or '<unset>'}",
                ]
            )

        if getattr(config, "pseudo_files", None):
            lines.append("Pseudopotentials:")
            for element in sorted(config.pseudo_files):
                path = Path(config.pseudo_files[element])
                lines.append(f"  - {element}: {path.name}")

        lines.extend(
            [
                f"Status: {record.get('status', 'Queued')}",
                f"Progress: {record.get('progress', 0)}%",
                "",
                "Commands:",
            ]
        )

        commands = record.get("commands", [])
        if commands:
            lines.extend(f"  - {cmd}" for cmd in commands)
        else:
            lines.append("  (no commands generated)")

        log_text = self._read_job_log_text(record).strip()
        lines.append("")
        lines.append("Recent output:")
        lines.append(log_text if log_text else "(no log yet)")

        self._set_job_detail_text("\n".join(lines))

    def _read_job_log_text(self, record: Dict[str, Any]) -> str:
        metadata = record.get("metadata") or {}
        log_file = metadata.get("log_file")
        if log_file:
            try:
                path = Path(str(log_file))
                if path.exists():
                    return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        config = record.get("config")
        if isinstance(config, JobConfig):
            working_dir = Path(config.working_directory).expanduser() if config.working_directory else None
            if working_dir is not None:
                candidate = working_dir / f"{config.sanitized_name()}.log"
                try:
                    if candidate.exists():
                        return candidate.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
        return str(record.get("log", "") or "")

    def _set_job_detail_text(self, text: str) -> None:
        current_text = self.jobDetailPreview.toPlainText()
        if current_text == text:
            return

        scroll_bar = self.jobDetailPreview.verticalScrollBar()
        scroll_value = scroll_bar.value()
        scroll_max = scroll_bar.maximum()

        self.jobDetailPreview.blockSignals(True)
        try:
            self.jobDetailPreview.setPlainText(text)
        finally:
            self.jobDetailPreview.blockSignals(False)

        new_max = scroll_bar.maximum()
        if scroll_max > 0 and scroll_value >= scroll_max - 2:
            scroll_bar.setValue(new_max)
        else:
            scroll_bar.setValue(min(scroll_value, new_max))

    def _scroll_job_detail_to_end(self) -> None:
        if not hasattr(self, "jobDetailPreview"):
            return
        scroll_bar = self.jobDetailPreview.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        cursor = self.jobDetailPreview.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.jobDetailPreview.setTextCursor(cursor)

    def _ensure_job_runner(self) -> None:
        # Execution is handled by the external daemon.
        self._update_job_action_states()

    def _start_job_runner(self, index: int) -> None:
        if index < 0 or index >= len(self.job_queue):
            return

        record = self.job_queue[index]
        commands = list(record.get("commands", []))
        if not commands:
            record["status"] = "Skipped (no commands)"
            record["progress"] = 100
            self._persist_job_queue()
            self._sync_job_status_table()
            self._ensure_job_runner()
            return

        config = record.get("config")
        if not isinstance(config, JobConfig):
            return

        working_directory = self._resolve_working_directory(config)
        runner = JobRunner(commands, working_directory, parent=self)
        self.active_runner = runner
        self._active_job_index = index

        runner.progress_updated.connect(lambda value, idx=index: self._on_job_progress(idx, value))
        runner.status_changed.connect(lambda status, idx=index: self._on_job_status_changed(idx, status))
        runner.log_appended.connect(lambda text, idx=index: self._append_job_log(idx, text))
        runner.finished_with_success.connect(lambda success, idx=index: self._on_job_finished(idx, success))

        record["status"] = "Queued"
        self._sync_job_status_table()

        runner.start()
        self._update_job_action_states()

    def _resolve_working_directory(self, config: JobConfig) -> Path:
        if config.working_directory:
            path = Path(config.working_directory).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path

        if config.input_files:
            first = Path(config.input_files[0]).expanduser()
            if first.exists():
                return first.parent

        return PROJECT_ROOT

    def _on_job_progress(self, index: int, value: int) -> None:
        if index < 0 or index >= len(self.job_queue):
            return
        self.job_queue[index]["progress"] = max(0, min(100, int(value)))
        self._sync_job_status_table()
        if self._current_job_index() == index:
            self._update_job_detail_preview()

    def _on_job_status_changed(self, index: int, status: str) -> None:
        if index < 0 or index >= len(self.job_queue):
            return
        self.job_queue[index]["status"] = status
        record = self.job_queue[index]
        config = record.get("config")
        if isinstance(config, JobConfig):
            label = config.name or config.sanitized_name()
            self.statusbar.showMessage(f"Job '{label}' → {status}", 4000)
        self._sync_job_status_table()
        self._update_job_detail_preview()

    def _append_job_log(self, index: int, text: str) -> None:
        if index < 0 or index >= len(self.job_queue):
            return
        log = self.job_queue[index].setdefault("log", "")
        self.job_queue[index]["log"] = f"{log}{text}\n" if text else log
        current = self._current_job_index()
        if current == index:
            self._update_job_detail_preview()

    def _on_job_finished(self, index: int, success: bool) -> None:
        if index < 0 or index >= len(self.job_queue):
            return

        record = self.job_queue[index]
        self._active_job_index = None
        if success and record.get("status") != "Completed":
            record["status"] = "Completed"
            record["progress"] = 100
        elif not success and record.get("status") not in {"Stopped", "Failed"}:
            record["status"] = "Failed"

        self.active_runner = None
        self._persist_job_queue()
        self._sync_job_status_table()
        self._ensure_job_runner()

    def _handle_stop_job(self) -> None:
        index = self._current_job_index()
        if index is None or index >= len(self.job_queue):
            return
        record = self.job_queue[index]
        status = str(record.get("status", ""))

        # For queued jobs, mark them as Stopped
        if status == "Queued":
            record["status"] = "Stopped"
            self._persist_job_queue()
            self._sync_job_status_table()
            self.statusbar.showMessage("Job marked as stopped", 4000)
        # For running jobs, request the daemon to stop
        elif self._is_running_status(status):
            record["status"] = "Stopping"
            self._persist_job_queue()
            self._request_job_process_termination(record, force=False)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            job_id = str(metadata.get("job_id") or "").strip()
            QTimer.singleShot(5000, lambda target_job_id=job_id: self._force_stop_job_if_still_running(target_job_id))
            self._sync_job_status_table()
            self._update_job_detail_preview()
            self.statusbar.showMessage("Stop requested for running job; terminating process group", 5000)
        else:
            self.statusbar.showMessage("Cannot stop a job that is not queued or running", 3000)

    def _request_job_process_termination(self, record: Dict[str, Any], *, force: bool) -> bool:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        signal_to_send = signal.SIGKILL if force else signal.SIGTERM
        sent = False

        pgid = metadata.get("proc_pgid")
        if pgid is not None and sys.platform != "win32":
            try:
                os.killpg(int(pgid), signal_to_send)
                sent = True
            except (ProcessLookupError, ValueError, OSError):
                pass

        pid = metadata.get("proc_pid")
        if pid is not None:
            try:
                os.kill(int(pid), signal_to_send)
                sent = True
            except (ProcessLookupError, ValueError, OSError):
                pass

        return sent

    def _force_stop_job_if_still_running(self, job_id: str) -> None:
        if not job_id:
            return
        index = next(
            (
                idx
                for idx, record in enumerate(self.job_queue)
                if str((record.get("metadata") or {}).get("job_id") or "").strip() == job_id
            ),
            -1,
        )
        if index < 0:
            return
        record = self.job_queue[index]
        if str(record.get("status", "")) != "Stopping":
            return
        if self._request_job_process_termination(record, force=True):
            self.statusbar.showMessage("Force-killed job process group after stop request", 5000)

    def _handle_start_job(self) -> None:
        """Restart a stopped job by marking it as Queued."""
        index = self._current_job_index()
        if index is None or index >= len(self.job_queue):
            return
        record = self.job_queue[index]
        status = str(record.get("status", ""))
        
        if status == "Stopped":
            record["status"] = "Queued"
            has_active_job = any(
                self._is_running_status(str(other.get("status", "")))
                for i, other in enumerate(self.job_queue)
                if i != index
            )
            self._persist_job_queue()
            self._sync_job_status_table()
            self._select_job_index(index)
            if has_active_job:
                self.statusbar.showMessage("Job requeued; it will start after active job finishes", 4000)
            else:
                self.statusbar.showMessage("Job requeued and will start with daemon", 4000)
        else:
            self.statusbar.showMessage("Can only start stopped jobs", 3000)

    def _handle_open_job_folder(self) -> None:
        index = self._current_job_index()
        if index is None or index >= len(self.job_queue):
            return
        config = self.job_queue[index].get("config")
        if not isinstance(config, JobConfig):
            return
        path = self._resolve_working_directory(config)
        self._browse_local_path(path, "job folder")

    def _show_job_status_context_menu(self, position: Any) -> None:
        if not hasattr(self, "jobStatusTree"):
            return

        item = self.jobStatusTree.itemAt(position)
        if item is None:
            return
        self.jobStatusTree.setCurrentItem(item)

        data = self._current_status_tree_data()
        kind = str(data.get("kind") or "")
        is_queue_root = kind in {"job", "workflow"} and item.parent() is None
        record = self._selected_job_record()
        is_workflow = self._is_workflow_record(record)
        menu = QMenu(self.jobStatusTree)
        menu.setStyleSheet(
            """
            QMenu {
                background: #FFFFFF;
                color: #111827;
                border: 1px solid #94A3B8;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 26px 6px 24px;
            }
            QMenu::item:selected {
                background: #DBEAFE;
                color: #1D4ED8;
            }
            QMenu::item:disabled {
                color: #9CA3AF;
            }
            """
        )

        start_action = stop_action = move_up_action = move_down_action = remove_action = open_folder_action = None
        if record is not None:
            capabilities = self._job_action_capabilities()
            title_action = menu.addAction("Workflow job actions" if is_workflow else "Job actions")
            title_action.setEnabled(False)
            start_action = menu.addAction("Start workflow job" if is_workflow else "Start job")
            stop_action = menu.addAction("Stop workflow job" if is_workflow else "Stop job")
            if is_queue_root:
                menu.addSeparator()
                move_up_action = menu.addAction("Move up")
                move_down_action = menu.addAction("Move down")
                remove_action = menu.addAction("Remove")
            menu.addSeparator()
            open_folder_action = menu.addAction("Open folder")

            start_action.setEnabled(capabilities["start"])
            stop_action.setEnabled(capabilities["stop"])
            if move_up_action is not None:
                move_up_action.setEnabled(capabilities["move_up"])
            if move_down_action is not None:
                move_down_action.setEnabled(capabilities["move_down"])
            if remove_action is not None:
                remove_action.setEnabled(capabilities["remove"])
            open_folder_action.setEnabled(capabilities["open_folder"])

        open_output_action = open_input_action = browse_step_action = browse_workflow_action = load_workflow_action = None
        paths: Dict[str, Optional[Path]] = {}
        if is_workflow:
            if not menu.isEmpty():
                menu.addSeparator()
            paths = self._selected_workflow_open_paths()
            can_open_files = bool(paths.get("single_child"))
            if can_open_files:
                open_output_action = menu.addAction("Open output")
                open_input_action = menu.addAction("Open input")
                menu.addSeparator()
            load_workflow_action = menu.addAction("Load workflow in Creator")
            browse_step_action = menu.addAction("Browse step folder")
            browse_workflow_action = menu.addAction("Browse workflow folder")

            output = paths.get("output")
            input_path = paths.get("input")
            workdir = paths.get("workdir")
            workflow_root = paths.get("workflow_root")
            manifest_path = self._selected_workflow_manifest_path()
            if open_output_action is not None:
                open_output_action.setEnabled(output is not None and output.is_file())
            if open_input_action is not None:
                open_input_action.setEnabled(input_path is not None and input_path.is_file())
            load_workflow_action.setEnabled(manifest_path is not None and manifest_path.is_file())
            browse_step_action.setEnabled(workdir is not None and workdir.is_dir())
            browse_workflow_action.setEnabled(workflow_root is not None and workflow_root.is_dir())

        if menu.isEmpty():
            return
        chosen = menu.exec(self.jobStatusTree.viewport().mapToGlobal(position))
        if chosen == start_action:
            self._handle_start_job()
        elif chosen == stop_action:
            self._handle_stop_job()
        elif chosen == move_up_action:
            self._handle_move_job_up()
        elif chosen == move_down_action:
            self._handle_move_job_down()
        elif chosen == remove_action:
            self._handle_remove_job()
        elif chosen == open_folder_action:
            self._handle_open_job_folder()
        elif chosen == open_output_action:
            self._open_local_path(paths.get("output"), "calculation output")
        elif chosen == open_input_action:
            self._open_local_path(paths.get("input"), "input file")
        elif chosen == load_workflow_action:
            self._handle_load_selected_workflow_in_creator()
        elif chosen == browse_step_action:
            self._browse_local_path(paths.get("workdir"), "step folder")
        elif chosen == browse_workflow_action:
            self._browse_local_path(paths.get("workflow_root"), "workflow folder")

    def _job_action_capabilities(self, index: Optional[int] = None) -> Dict[str, bool]:
        selected_index = self._current_job_index() if index is None else index
        capabilities = {
            "open_folder": False,
            "start": False,
            "stop": False,
            "move_up": False,
            "move_down": False,
            "remove": False,
        }
        if selected_index is None or selected_index < 0 or selected_index >= len(self.job_queue):
            return capabilities

        capabilities["open_folder"] = True
        record = self.job_queue[selected_index]
        status = str(record.get("status", ""))
        running_job_index = self._active_job_index

        capabilities["stop"] = status == "Queued" or self._is_running_status(status)
        capabilities["start"] = status == "Stopped"
        capabilities["remove"] = not self._is_running_status(status) and (
            running_job_index is None or selected_index > running_job_index
        )

        if status != "Queued":
            return capabilities

        if selected_index > 0:
            prev_status = str(self.job_queue[selected_index - 1].get("status", ""))
            capabilities["move_up"] = (
                not self._is_running_status(prev_status)
                and (running_job_index is None or selected_index - 1 != running_job_index)
            )
        if selected_index < len(self.job_queue) - 1:
            next_status = str(self.job_queue[selected_index + 1].get("status", ""))
            capabilities["move_down"] = (
                not self._is_running_status(next_status)
                and (running_job_index is None or selected_index + 1 != running_job_index)
            )
        return capabilities

    def _selected_workflow_open_paths(self) -> Dict[str, Any]:
        record = self._selected_job_record()
        is_workflow = self._is_workflow_record(record)
        root = self._workflow_root_for_record(record) if is_workflow else None
        paths: Dict[str, Any] = {
            "workflow_root": root,
            "workdir": None,
            "input": None,
            "output": None,
            "single_child": False,
        }
        if not is_workflow or record is None:
            return paths
        item = self.jobStatusTree.currentItem() if hasattr(self, "jobStatusTree") else None
        data = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else {}
        if not isinstance(data, dict):
            return paths

        state, _state_path, _error = self._load_workflow_state_for_record(record)
        if state is None:
            return paths
        tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
        kind = str(data.get("kind") or "")
        task_id = str(data.get("task_id") or "")
        task = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
        selected_tasks: List[Dict[str, Any]] = []
        child: Dict[str, Any] = {}
        if kind == "child":
            child = self._workflow_child_by_id(task, str(data.get("child_id") or ""))
            if task:
                selected_tasks = [task]
        elif kind == "task" and task:
            selected_tasks = [task]
        elif kind == "group":
            task_ids = [str(selected_id) for selected_id in data.get("task_ids") or []]
            selected_tasks = [
                selected_task
                for selected_id in task_ids
                for selected_task in [tasks.get(selected_id)]
                if isinstance(selected_task, dict)
            ]
        elif kind == "workflow":
            selected_tasks = [selected_task for selected_task in tasks.values() if isinstance(selected_task, dict)]

        if kind == "workflow":
            paths["workdir"] = root
        elif kind == "child" and child:
            paths["workdir"] = self._workflow_abs_path(root, child.get("workdir"))
        else:
            paths["workdir"] = self._workflow_step_workdir_for_tree_item(root, data, selected_tasks)

        if not child and len(selected_tasks) == 1:
            selected_children = [entry for entry in selected_tasks[0].get("children") or [] if isinstance(entry, dict)]
            if len(selected_children) == 1:
                child = selected_children[0]
        if child:
            paths["input"] = self._workflow_abs_path(root, child.get("input"))
            paths["output"] = self._workflow_abs_path(root, child.get("output"))
            paths["single_child"] = True
        return paths

    def _workflow_step_workdir_for_tree_item(
        self,
        root: Optional[Path],
        data: Dict[str, Any],
        tasks: Sequence[Dict[str, Any]],
    ) -> Optional[Path]:
        kind = str(data.get("kind") or "")
        key = str(data.get("key") or "")
        if root is not None and kind == "group":
            if key.endswith(":group:source"):
                return root / "source"
            if key.endswith(":group:relaxation"):
                return root / "relaxation"
            if key.endswith(":group:convergence"):
                return root / "convergence"
            if key.endswith(":group:bands"):
                return root / "bands"
            if key.endswith(":group:dos"):
                return root / "dos"
            if key.endswith(":group:phonon"):
                return root / "phonon"

        if root is not None and kind == "task" and tasks:
            task = tasks[0]
            task_type = str(task.get("type") or "")
            task_id = str(task.get("id") or data.get("task_id") or "")
            if task_id == "prepare_source":
                return root / "source"
            if task_type == "convergence.cutoff" or task_id == "cutoff_convergence":
                return root / "convergence" / "cutoff_convergence"
            if task_type == "convergence.kpoints" or task_id == "kpoint_convergence":
                return root / "convergence" / "kpoint_convergence"
            if task_type in {"qe.bands_workflow", "qe.fat_bands_workflow"} or task_id in {
                "bands_workflow",
                "fat_bands_workflow",
                "fatband_csv_extraction",
                "fatband_plotting",
            }:
                return root / "bands"
            if task_type == "qe.dos_workflow" or task_id == "dos_workflow":
                return root / "dos"
            if task_type == "qe.phonon_workflow" or task_id == "phonon_workflow":
                return root / "phonon"

        children = [
            child
            for task in tasks
            for child in task.get("children") or []
            if isinstance(child, dict)
        ]
        workdirs = [
            path
            for child in children
            for path in [self._workflow_abs_path(root, child.get("workdir"))]
            if path is not None
        ]
        if not workdirs:
            return None
        if len(workdirs) == 1:
            return workdirs[0]
        try:
            return Path(os.path.commonpath([str(path) for path in workdirs]))
        except Exception:
            return workdirs[0]

    def _open_local_path(self, path: Optional[Path], label: str) -> None:
        if path is None or not path.is_file():
            self.statusbar.showMessage(f"{label.capitalize()} is not available yet.", 4000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _browse_local_path(self, path: Optional[Path], label: str) -> None:
        if path is None or not path.is_dir():
            self.statusbar.showMessage(f"{label.capitalize()} is not available yet.", 4000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _is_running_status(status: str) -> bool:
        text = (status or "").strip()
        return bool(text) and (text.startswith("Running") or text in {"Stopping", "Starting"})

    @staticmethod
    def _is_terminal_status(status: str) -> bool:
        text = (status or "").strip()
        if not text:
            return False
        if text in {"Completed", "Failed", "Stopped"}:
            return True
        return text.startswith("Skipped")

    def _handle_clear_completed_jobs(self) -> None:
        before = len(self.job_queue)
        remaining: List[Dict[str, Any]] = []
        for record in self.job_queue:
            status = str(record.get("status", ""))
            if self._is_terminal_status(status):
                continue
            remaining.append(record)

        removed = before - len(remaining)
        if removed == 0:
            return

        self.job_queue = remaining
        self._persist_job_queue()
        self._sync_job_status_table()
        self.statusbar.showMessage(f"Cleared {removed} finished job(s)", 4000)
        self._ensure_job_runner()

    def _handle_clear_stopped_jobs(self) -> None:
        before = len(self.job_queue)
        remaining: List[Dict[str, Any]] = []
        for record in self.job_queue:
            status = str(record.get("status", ""))
            if status == "Stopped":
                continue
            remaining.append(record)

        removed = before - len(remaining)
        if removed == 0:
            return

        self.job_queue = remaining
        self._persist_job_queue()
        self._sync_job_status_table()
        self.statusbar.showMessage(f"Cleared {removed} stopped job(s)", 4000)

    def _is_daemon_running(self) -> bool:
        """Check if the job daemon is currently running."""
        try:
            if self.job_daemon_pid_file.exists():
                pid_text = self.job_daemon_pid_file.read_text(encoding="utf-8").strip()
                try:
                    pid = int(pid_text)
                    os.kill(pid, 0)  # Signal 0 checks if process exists
                    return True
                except (ValueError, ProcessLookupError, OSError):
                    return False
        except Exception:
            pass
        return False

    def _update_daemon_status(self) -> None:
        """Update the daemon status label and button states."""
        if not hasattr(self, "daemonStatusLabel"):
            return
        is_running = self._is_daemon_running()
        if is_running:
            self.daemonStatusLabel.setText("Running")
            self.daemonStatusLabel.setStyleSheet("color: green;")
            self.startDaemonButton.setEnabled(False)
            self.stopDaemonButton.setEnabled(True)
        else:
            self.daemonStatusLabel.setText("Stopped")
            self.daemonStatusLabel.setStyleSheet("color: red;")
            self.startDaemonButton.setEnabled(True)
            self.stopDaemonButton.setEnabled(False)

    def _handle_start_daemon(self) -> None:
        """Start the job daemon in the background."""
        if self._is_daemon_running():
            self.statusbar.showMessage("Job daemon is already running", 3000)
            return
        try:
            # Mark any "Stopping" jobs as "Stopped" since daemon was offline
            for record in self.job_queue:
                if str(record.get("status", "")) == "Stopping":
                    record["status"] = "Stopped"
            self._persist_job_queue()
            self._load_saved_job_queue()
            self._sync_job_status_table()
            
            cmd = [
                sys.executable,
                str(self.job_daemon_script),
                "--queue-file", str(self.job_queue_file),
                "--poll-interval", "1.0"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.job_daemon_pid_file.write_text(str(proc.pid), encoding="utf-8")
            self.statusbar.showMessage(f"Started job daemon (PID {proc.pid})", 4000)
            self._update_daemon_status()
        except Exception as exc:
            self.statusbar.showMessage(f"Could not start daemon: {exc}", 5000)

    def _handle_stop_daemon(self) -> None:
        """Stop the running job daemon immediately."""
        try:
            # Step 1: Kill any running subprocesses from jobs (by killing their process groups)
            for record in self.job_queue:
                metadata = record.get("metadata") or {}
                pgid = metadata.get("proc_pgid")
                if pgid is not None:
                    try:
                        # Kill the entire process group
                        if sys.platform != "win32":
                            os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass  # Process group already dead
            
            # Step 2: Kill the daemon process itself
            if self.job_daemon_pid_file.exists():
                pid_text = self.job_daemon_pid_file.read_text(encoding="utf-8").strip()
                try:
                    pid = int(pid_text)
                    os.kill(pid, signal.SIGKILL)  # Use SIGKILL to ensure immediate termination
                    self.job_daemon_pid_file.unlink(missing_ok=True)
                except (ValueError, ProcessLookupError, OSError):
                    self.job_daemon_pid_file.unlink(missing_ok=True)
            
            # Step 3: Now mark all running/queued jobs as "Stopped"
            for record in self.job_queue:
                status = str(record.get("status", ""))
                if status in ["Running", "Queued", "Stopping", "Starting"] or self._is_running_status(status):
                    record["status"] = "Stopped"
            
            self._persist_job_queue()
            self._load_saved_job_queue()
            self._sync_job_status_table()
            self.statusbar.showMessage("Daemon stopped, all processes killed, jobs marked as stopped", 4000)
        except Exception as exc:
            self.statusbar.showMessage(f"Could not stop daemon: {exc}", 5000)
        finally:
            self._update_daemon_status()


    def _handle_move_job_up(self) -> None:
        index = self._current_job_index()
        if index is None or index <= 0 or index >= len(self.job_queue):
            return

        record = self.job_queue[index]
        status = str(record.get("status", ""))
        if status != "Queued":
            self.statusbar.showMessage("Only queued jobs can be moved.", 4000)
            return

        if self._active_job_index is not None and index - 1 == self._active_job_index:
            self.statusbar.showMessage("Cannot move a job ahead of the running job.", 4000)
            return

        prev_status = str(self.job_queue[index - 1].get("status", ""))
        if self._is_running_status(prev_status):
            self.statusbar.showMessage("Cannot move a job ahead of a running job.", 4000)
            return

        self.job_queue[index - 1], self.job_queue[index] = self.job_queue[index], self.job_queue[index - 1]
        self._persist_job_queue()
        self._sync_job_status_table()
        self._select_job_index(index - 1)
        self._ensure_job_runner()

    def _handle_move_job_down(self) -> None:
        index = self._current_job_index()
        if index is None or index >= len(self.job_queue) - 1 or index < 0:
            return

        record = self.job_queue[index]
        status = str(record.get("status", ""))
        if status != "Queued":
            self.statusbar.showMessage("Only queued jobs can be moved.", 4000)
            return

        if self._active_job_index is not None and index + 1 == self._active_job_index:
            self.statusbar.showMessage("Cannot move a job behind the running job.", 4000)
            return

        next_status = str(self.job_queue[index + 1].get("status", ""))
        if self._is_running_status(next_status):
            self.statusbar.showMessage("Cannot reorder a running job.", 4000)
            return

        self.job_queue[index], self.job_queue[index + 1] = self.job_queue[index + 1], self.job_queue[index]
        self._persist_job_queue()
        self._sync_job_status_table()
        self._select_job_index(index + 1)
        self._ensure_job_runner()

    def _handle_remove_job(self) -> None:
        index = self._current_job_index()
        if index is None or index < 0 or index >= len(self.job_queue):
            return

        record = self.job_queue[index]
        status = str(record.get("status", ""))
        if self._is_running_status(status) or (self._active_job_index is not None and index == self._active_job_index):
            self.statusbar.showMessage("Cannot remove a running job. Use Stop instead.", 5000)
            return

        if self._active_job_index is not None and index <= self._active_job_index:
            self.statusbar.showMessage("Cannot remove jobs ahead of the running job.", 5000)
            return

        if 0 <= index < len(self.job_queue):
            self.job_queue.pop(index)

        new_index = min(index, len(self.job_queue) - 1)
        self._persist_job_queue()
        self._sync_job_status_table()
        if new_index >= 0:
            self._select_job_index(new_index)
        self._ensure_job_runner()
        self.statusbar.showMessage("Removed job from queue.", 4000)

    def _update_job_action_states(self) -> None:
        has_clearable = any(self._is_terminal_status(str(record.get("status", ""))) for record in self.job_queue)
        self.clearCompletedJobsButton.setEnabled(has_clearable)

        has_stopped = any(str(record.get("status", "")) == "Stopped" for record in self.job_queue)
        self.clearStoppedJobsButton.setEnabled(has_stopped)

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self.actionQuit = QAction("Quit", self)
        file_menu.addAction(self.actionQuit)

        settings_menu = menubar.addMenu("Settings")
        self.actionSystemConfig = QAction("System configuration...", self)
        self.actionCommandPreferences = QAction("Command preferences...", self)
        settings_menu.addAction(self.actionSystemConfig)
        settings_menu.addAction(self.actionCommandPreferences)

        help_menu = menubar.addMenu("help")
        self.actioninfo_2 = QAction("info", self)
        help_menu.addAction(self.actioninfo_2)

    def _configure_status_bar(self) -> None:
        current_status = getattr(self, "statusbar", None)
        if not isinstance(current_status, QStatusBar):
            self.statusbar = QStatusBar(self)
            self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready", 3000)

    def _system_mpi_procs(self) -> int:
        cfg = getattr(self, "system_config", None)
        try:
            return max(1, int(getattr(cfg, "mpi_procs", 1) or 1))
        except Exception:
            return 1

    def _apply_system_config_defaults(self, *, force: bool = False) -> None:
        cfg = getattr(self, "system_config", None)
        if cfg is None:
            return

        self._job_mpi_procs = self._system_mpi_procs()
        if force:
            self._refresh_pseudo_selection_view()

    def _default_output_path(self, cif_path: str) -> Path:
        """Return a Path pointing to `<stem>.scf.in`, stripping common CIF suffixes."""

        candidate = Path(cif_path)
        trimmed = candidate
        while trimmed.suffix and trimmed.suffix.lower() in {".cif"}:
            trimmed = trimmed.with_suffix("")

        base_name = trimmed.name or candidate.stem
        return candidate.with_name(f"{base_name}.scf.in")

    def _current_cif_path(self) -> Optional[str]:
        widget = getattr(self, "workflowStructurePathLineEdit", None)
        if not isinstance(widget, QLineEdit):
            return None

        candidate_text = widget.text().strip()
        if not candidate_text:
            return None

        candidate = Path(candidate_text).expanduser()
        if candidate.suffix.lower() != ".cif" or not candidate.is_file():
            return None

        return str(candidate)

    def _connect_signals(self) -> None:
        self.qeGeneratorInputBrowseButton.clicked.connect(self._handle_qe_generator_input_browse)
        self.qeGeneratorOutputBrowseButton.clicked.connect(self._handle_qe_generator_output_browse)
        self.qeGeneratorInputLineEdit.textChanged.connect(self._sync_materials_cloud_file_format)
        self.materialsCloudCheckBox.toggled.connect(self._handle_materials_cloud_toggled)
        self.generatePWscfButton.clicked.connect(self._handle_generate_pwscf)
        self.qeInputGeneratorSavePreviewButton.clicked.connect(self._handle_qe_generator_save_preview)

        self.outputDirectoryBrowseButton.clicked.connect(self._select_working_directory)
        self.pseudoSelectButton.clicked.connect(self._open_pseudo_selector)
        self.pseudoClearButton.clicked.connect(self._clear_pseudo_selection)
        self.jobInputBrowseButton.clicked.connect(self._handle_browse_job_input)
        self.jobInputPathLineEdit.textChanged.connect(self._mark_job_preview_stale)
        self.generatedFilesListWidget.currentRowChanged.connect(self._handle_generated_file_selection_changed)
        self.generatedFilePreviewEdit.textChanged.connect(self._handle_generated_preview_edited)
        self.saveGeneratedPreviewButton.clicked.connect(self._handle_save_generated_preview)
        self.jobCalcTypeComboBox.currentTextChanged.connect(self._on_calculation_type_changed)
        self.jobQueueComboBox.currentTextChanged.connect(self._mark_job_preview_stale)
        self.jobsInputFilesGenerateButton.clicked.connect(self._handle_prepare_job_preview)
        self.jobsInputFilesSaveButton.clicked.connect(self._handle_save_job_script)
        self.addJobsToQueueButton.clicked.connect(self._handle_enqueue_job)

        self.workflowStructureBrowseButton.clicked.connect(self._handle_workflow_structure_browse)
        self.workflowLoadButton.clicked.connect(self._handle_workflow_load_template)
        self.workflowSaveButton.clicked.connect(self._handle_workflow_save_template)
        self.workflowOutputDirectoryBrowseButton.clicked.connect(self._handle_workflow_output_directory_browse)
        for button in self.workflowRelaxationButtons.values():
            button.toggled.connect(
                lambda _checked=False: (
                    self._workflow_refresh_relaxation_pressure_controls(),
                    self._workflow_refresh_option_tabs(),
                )
            )
        for checkbox in self.workflowSelectionCheckboxes.values():
            checkbox.toggled.connect(lambda _checked=False: self._workflow_refresh_option_tabs())
        self.workflowSubmitButton.clicked.connect(self._handle_workflow_submit)

        self.actionQuit.triggered.connect(self.close)
        self.actionSystemConfig.triggered.connect(self._open_system_configuration_dialog)
        self.actionCommandPreferences.triggered.connect(self._open_command_preferences_dialog)
        self.actioninfo_2.triggered.connect(self._show_about)

        # Job status tab
        self.refreshJobStatusButton.clicked.connect(self._handle_refresh_job_status)
        self.jobStatusFilterLineEdit.textChanged.connect(self._handle_job_filter_changed)
        self.jobStatusTree.itemSelectionChanged.connect(self._handle_job_table_selection)
        self.jobStatusTree.customContextMenuRequested.connect(self._show_job_status_context_menu)
        self.jobStatusTree.itemCollapsed.connect(self._handle_job_tree_item_collapsed)
        self.jumpToJobOutputEndButton.clicked.connect(self._scroll_job_detail_to_end)
        self.clearCompletedJobsButton.clicked.connect(self._handle_clear_completed_jobs)
        self.clearStoppedJobsButton.clicked.connect(self._handle_clear_stopped_jobs)
        self.startDaemonButton.clicked.connect(self._handle_start_daemon)
        self.stopDaemonButton.clicked.connect(self._handle_stop_daemon)

    def _open_system_configuration_dialog(self) -> None:
        dialog = SystemConfigDialog(self.system_config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        updated = dialog.result_config()
        try:
            save_system_config(updated)
        except IOError as exc:
            self._show_error(str(exc))
            return

        self.system_config = updated
        self._apply_system_config_defaults(force=True)
        self._mark_job_preview_stale()
        self.statusbar.showMessage("System configuration updated", 4000)
        # no automatic update of Preferences dialog here

    def _open_command_preferences_dialog(self) -> None:
        dialog = CommandPreferencesDialog(self)
        dialog.exec()

    # GUI Event Handlers -------------------------------------------------
    def _handle_qe_generator_input_browse(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select structure file",
            str(PROJECT_ROOT),
            (
                "Structure files (*.cif *.xsf *.cell *.pdb *.in *.pwi *.pw *.qe *.vasp *.poscar POSCAR CONTCAR);;"
                "All files (*)"
            ),
        )
        if not file_path:
            return

        path = Path(file_path).expanduser()
        self.qeGeneratorInputLineEdit.setText(str(path))
        self.qeGeneratorOutputLineEdit.setText(str(_default_pwscf_output_path(path)))
        self.qeInputGeneratorStatusLabel.setText("Structure file selected.")
        self.statusbar.showMessage("Structure file selected", 3000)

    def _handle_qe_generator_output_browse(self) -> None:
        suggested = self.qeGeneratorOutputLineEdit.text().strip()
        if not suggested:
            input_text = self.qeGeneratorInputLineEdit.text().strip()
            if input_text:
                suggested = str(_default_pwscf_output_path(Path(input_text).expanduser()))

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose PWscf output file",
            suggested or str(PROJECT_ROOT),
            "Quantum ESPRESSO input (*.in *.txt);;All files (*)",
        )
        if output_path:
            self.qeGeneratorOutputLineEdit.setText(output_path)
            self.qeInputGeneratorStatusLabel.setText("Output path selected.")
            self.statusbar.showMessage("PWscf output path selected", 3000)

    def _handle_materials_cloud_toggled(self, checked: bool) -> None:
        self.materialsCloudOptionsGroup.setVisible(bool(checked))
        if checked:
            self.qeInputGeneratorStatusLabel.setText("Materials Cloud mode enabled. Internet connection required.")
            self._sync_materials_cloud_file_format()
        else:
            self.qeInputGeneratorStatusLabel.setText("Local CIF generator mode enabled.")

    def _sync_materials_cloud_file_format(self, *_args: Any) -> None:
        if not hasattr(self, "mcFileFormatCombo"):
            return

        input_text = self.qeGeneratorInputLineEdit.text().strip()
        if not input_text:
            return
        self._set_combo_current_data(
            self.mcFileFormatCombo,
            self._infer_materials_cloud_file_format(Path(input_text).expanduser()),
        )

    def _handle_generate_pwscf(self) -> None:
        if self._qe_input_generator_thread and self._qe_input_generator_thread.isRunning():
            self.qeInputGeneratorStatusLabel.setText("Generation is already running.")
            return

        input_text = self.qeGeneratorInputLineEdit.text().strip()
        if not input_text:
            self._show_error("Select an input structure file before generating.")
            return

        input_path = Path(input_text).expanduser()
        if not input_path.is_file():
            self._show_error("The selected input file does not exist.")
            return

        output_text = self.qeGeneratorOutputLineEdit.text().strip()
        if not output_text:
            output_path = _default_pwscf_output_path(input_path)
            self.qeGeneratorOutputLineEdit.setText(str(output_path))
        else:
            output_path = Path(output_text).expanduser()

        use_materials_cloud = self.materialsCloudCheckBox.isChecked()
        if not use_materials_cloud and input_path.suffix.lower() != ".cif":
            self._show_error("The local generator accepts CIF files only. Enable Materials Cloud for this file format.")
            return

        options = self._materials_cloud_options()
        self.generatePWscfButton.setEnabled(False)
        self.qeInputGeneratorStatusLabel.setText("Generating PWscf input...")
        self.statusbar.showMessage("Generating PWscf input", 3000)

        thread = QeInputGeneratorThread(
            str(input_path),
            str(output_path),
            use_materials_cloud,
            options,
            self,
        )
        thread.log_appended.connect(self._handle_qe_generator_log)
        thread.finished_with_result.connect(self._handle_qe_generator_finished)
        thread.finished.connect(thread.deleteLater)
        self._qe_input_generator_thread = thread
        thread.start()

    def _handle_qe_generator_log(self, message: str) -> None:
        text = (message or "").strip()
        if not text:
            return
        last_line = text.splitlines()[-1].strip()
        if last_line:
            self.qeInputGeneratorStatusLabel.setText(last_line[:300])

    def _handle_qe_generator_finished(self, success: bool, message: str, output_path_text: str) -> None:
        self.generatePWscfButton.setEnabled(True)
        self._qe_input_generator_thread = None

        if not success:
            self.qeInputGeneratorStatusLabel.setText("Generation failed.")
            self._show_error(f"PWscf generation failed: {message}")
            return

        output_path = Path(output_path_text).expanduser()
        try:
            content = output_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.qeInputGeneratorStatusLabel.setText("Generated, but preview could not be loaded.")
            self._show_error(f"Generated file could not be opened for preview: {exc}")
            return

        self.qeInputGeneratorPreviewEdit.setPlainText(content)
        self.qeGeneratorOutputLineEdit.setText(str(output_path))
        if hasattr(self, "jobInputPathLineEdit") and not self.jobInputPathLineEdit.text().strip():
            self.jobInputPathLineEdit.setText(str(output_path))
            self._mark_job_preview_stale()

        status = f"Generated {output_path.name}"
        if self.materialsCloudCheckBox.isChecked():
            status += " and copied Materials Cloud pseudopotentials when available"
        self.qeInputGeneratorStatusLabel.setText(status)
        self.statusbar.showMessage(status, 5000)

    def _handle_qe_generator_save_preview(self) -> None:
        content = self.qeInputGeneratorPreviewEdit.toPlainText()
        if not content.strip():
            self._show_error("There is no generated content to save.")
            return

        output_path = self.qeGeneratorOutputLineEdit.text().strip()
        if not output_path:
            input_text = self.qeGeneratorInputLineEdit.text().strip()
            suggested = str(_default_pwscf_output_path(Path(input_text).expanduser())) if input_text else str(PROJECT_ROOT)
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Choose PWscf output file",
                suggested,
                "Quantum ESPRESSO input (*.in *.txt);;All files (*)",
            )
            if not output_path:
                return
            self.qeGeneratorOutputLineEdit.setText(output_path)

        try:
            save_text_to_file(output_path, content)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not save generated input: {exc}")
            return

        self.qeInputGeneratorStatusLabel.setText(f"Saved to {output_path}")
        self.statusbar.showMessage("Generated input saved", 4000)

    def _materials_cloud_options(self) -> Dict[str, str]:
        kmesh_override, kpoint_spacing = self._selected_kpoint_settings()
        tolerance = self._combo_current_data(self.qeGeneratorSymprecCombo)
        return {
            "file_format": self._combo_current_data(self.mcFileFormatCombo),
            "protocol": self._combo_current_data(self.mcProtocolCombo),
            "xc": self._combo_current_data(self.mcXcCombo),
            "magnetization": self._combo_current_data(self.mcMagnetizationCombo),
            "refine_cell": self._combo_current_data(self.qeGeneratorRefineCellCombo),
            "tolerance": tolerance,
            "symprec": f"{10 ** int(tolerance):.8g}",
            "kpoint_spacing": f"{kpoint_spacing:.8g}",
            "pseudo_override": self._combo_current_data(self.mcPseudoOverrideCombo),
            "kmesh_override": kmesh_override,
            "threshold_override": self._combo_current_data(self.mcThresholdOverrideCombo),
        }

    def _selected_kpoint_settings(self) -> Tuple[str, float]:
        data = self.qeGeneratorKpointsCombo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            try:
                return str(data[0]), float(data[1])
            except (TypeError, ValueError):
                pass
        return "fine", 0.10

    @staticmethod
    def _combo_current_data(combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data) if data is not None else combo.currentText()

    @staticmethod
    def _set_combo_current_data(combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if str(combo.itemData(index)) == value:
                combo.setCurrentIndex(index)
                return

    @staticmethod
    def _infer_materials_cloud_file_format(path: Path) -> str:
        suffix = path.suffix.lower()
        name_upper = path.name.upper()
        if suffix == ".cif":
            return "cif-ase"
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
        return "auto"

    def _select_working_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select working directory",
            self.outputDirectoryLineEdit.text().strip() or str(PROJECT_ROOT),
        )
        if directory:
            self.outputDirectoryLineEdit.setText(directory)
            self.statusbar.showMessage("Working directory selected", 3000)

    def _collect_job_config(self) -> JobConfig:
        input_path = self.jobInputPathLineEdit.text().strip()
        input_files = [input_path] if input_path else []

        calculation_raw = self.jobCalcTypeComboBox.currentText().strip()
        calc_lower = calculation_raw.lower()

        pseudo_files = {} if calc_lower == "bandx" else dict(self._pseudo_files)
        pseudo_dir: Optional[str] = None
        if calc_lower != "bandx" and pseudo_files:
            try:
                parent_dirs = {str(Path(path).expanduser().parent) for path in pseudo_files.values() if path}
                if len(parent_dirs) == 1:
                    pseudo_dir = parent_dirs.pop()
            except Exception:
                pseudo_dir = None

        if calc_lower != "bandx" and not pseudo_dir and getattr(self, "system_config", None):
            try:
                dirs = getattr(self.system_config, "pseudo_directories", None)
                if dirs:
                    pseudo_dir = dirs[0]
                elif self.system_config.pseudo_directory:
                    pseudo_dir = self.system_config.pseudo_directory
            except Exception:
                pseudo_dir = None

        return JobConfig(
            name=self.jobNameLineEdit.text().strip(),
            calculation=calculation_raw,
            mpi_procs=self._job_mpi_procs,
            queue=self.jobQueueComboBox.currentText().strip(),
            working_directory=self.outputDirectoryLineEdit.text().strip() or (str(Path(input_path).parent) if input_path else None),
            pseudo_directory=pseudo_dir or None,
            input_files=input_files,
            pseudo_files=pseudo_files,
        )

    def _sync_working_directory_with_input(self) -> None:
        """Default the working directory to the source input file directory when empty."""
        if self.outputDirectoryLineEdit.text().strip():
            return
        input_path = self.jobInputPathLineEdit.text().strip()
        if not input_path:
            return
        try:
            self.outputDirectoryLineEdit.setText(str(Path(input_path).expanduser().parent))
        except Exception:
            pass


    def _handle_browse_job_input(self) -> None:
        start_dir = self.jobInputPathLineEdit.text().strip() or self.outputDirectoryLineEdit.text().strip()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select QE input file",
            start_dir or str(PROJECT_ROOT),
            "Quantum Espresso input (*.in *.txt);;All files (*)",
        )
        if file_path:
            self.jobInputPathLineEdit.setText(file_path)
            self._sync_working_directory_with_input()
            self.statusbar.showMessage("Input file selected", 3000)
            self._mark_job_preview_stale()

    def _open_pseudo_selector(self) -> None:
        source_text = self._get_input_source(show_errors=False) or ""
        species = self._extract_atomic_species(source_text)
        if not species and self._pseudo_files:
            fallback: List[AtomicSpeciesEntry] = []
            for element, path in self._pseudo_files.items():
                symbol = normalize_element_symbol(element)
                if not symbol:
                    continue
                fallback.append(AtomicSpeciesEntry(element=symbol, pseudo=Path(path).name))
            species = fallback

        if species:
            ordered_unique: List[str] = []
            seen_symbols: set[str] = set()
            for entry in species:
                symbol = normalize_element_symbol(entry.element)
                if symbol and symbol not in seen_symbols:
                    ordered_unique.append(symbol)
                    seen_symbols.add(symbol)
            allowed_symbols = ordered_unique
            self._current_species_order = ordered_unique
            self._prune_pseudo_assignments(allowed_symbols)
        else:
            allowed_symbols = []
            self._current_species_order = []

        dialog = PseudoBrowserDialog(
            self.system_config,
            species,
            existing=self._pseudo_files,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selections = dialog.selected_files()
        allowed_set = {symbol for symbol in allowed_symbols}
        normalized_selections: Dict[str, str] = {}
        for element, path in selections.items():
            symbol = normalize_element_symbol(element)
            if not symbol:
                continue
            if allowed_symbols and symbol not in allowed_set:
                continue
            normalized_selections[symbol] = path

        if self._current_species_order:
            ordered_mapping: Dict[str, str] = {}
            for symbol in self._current_species_order:
                if symbol in normalized_selections and symbol not in ordered_mapping:
                    ordered_mapping[symbol] = normalized_selections[symbol]
            for symbol, path in normalized_selections.items():
                if symbol not in ordered_mapping:
                    ordered_mapping[symbol] = path
            selections = ordered_mapping
        else:
            selections = normalized_selections

        self._pseudo_files = dict(selections)
        self._refresh_pseudo_selection_view()
        self._mark_job_preview_stale()
        if self._pseudo_files:
            self.statusbar.showMessage(
                f"Selected {len(self._pseudo_files)} pseudopotential file(s).",
                4000,
            )
        else:
            self.statusbar.showMessage("No pseudopotentials selected", 3000)

    def _mark_job_preview_stale(self, *_args) -> None:
        if not hasattr(self, "jobStatusLabel"):
            return

        calc = ""
        if hasattr(self, "jobCalcTypeComboBox"):
            calc = (self.jobCalcTypeComboBox.currentText() or "").strip().lower()
        if calc == "custom":
            self.jobStatusLabel.setText("Custom job ready for manual editing.")
            self._job_preview_dirty = False
            return

        self._sync_working_directory_with_input()

        if not self.jobInputPathLineEdit.text().strip():
            self.generatedFilesListWidget.clear()
            self.generatedFilePreviewEdit.clear()
            self._set_command_preview_text("", force=True)
            self.jobStatusLabel.setText("Awaiting input file")
            self.statusbar.showMessage("Awaiting input file", 2000)
            self._job_preview_dirty = True
            return

        self.jobStatusLabel.setText('Configuration changed; click "Generate input" to refresh preview.')
        self.statusbar.showMessage("Configuration changed; regenerate when ready.", 2000)
        self._job_preview_dirty = True

    def _on_calculation_type_changed(self, text: str) -> None:
        self._apply_calculation_type_side_effects(text)
        self._mark_job_preview_stale()

    def _apply_calculation_type_side_effects(self, calc_text: str) -> None:
        calc = (calc_text or "").strip().lower()
        was_custom = self._custom_job_active
        is_custom = calc == "custom"

        allowed_pseudo_calc = {"scf", "nscf", "vc-relax", "relax", "band", "bands", "custom"}

        reason_map = {
            "bandx": "Bandx post-processing reuses SCF wavefunctions; pseudopotentials stay managed by the SCF run.",
            "pdos": "PDOS (projwfc.x) post-processing reuses SCF results and does not require selecting pseudopotentials here.",
            "optics": "Optics calculations reuse wavefunctions from prior SCF/NSCF steps; pseudopotentials stay with those runs.",
            "phonon": "Phonon calculations reuse SCF artifacts (charge density and wavefunctions); pick pseudopotentials in the SCF step.",
            "matdyn": "matdyn.x consumes force constants from q2r.x; reuse SCF pseudopotentials from earlier steps.",
            "matdyn.x": "matdyn.x consumes force constants from q2r.x; reuse SCF pseudopotentials from earlier steps.",
        }

        disable_pseudo = calc not in allowed_pseudo_calc
        group = getattr(self, "pseudoGroupBox", None)
        if group is not None:
            tooltip_base = getattr(self, "_pseudo_group_tooltip", "") or ""
            if disable_pseudo:
                message = reason_map.get(
                    calc,
                    "This calculation reuses SCF artefacts; configure pseudopotentials via the SCF/NSCF steps.",
                )
                group.setEnabled(False)
                group.setToolTip(message)
                if hasattr(self, "pseudoSelectButton"):
                    self.pseudoSelectButton.setEnabled(False)
                if hasattr(self, "pseudoClearButton"):
                    self.pseudoClearButton.setEnabled(False)
            else:
                group.setEnabled(True)
                group.setToolTip(tooltip_base)
                if hasattr(self, "pseudoSelectButton"):
                    self.pseudoSelectButton.setEnabled(True)
        if not disable_pseudo:
            # Refresh button enabled state when reactivated
            self._refresh_pseudo_selection_view()

        if is_custom:
            self._ensure_custom_preview_ready(show_status=False)
        elif was_custom:
            self._deactivate_custom_mode()

    def _default_custom_input_name(self) -> str:
        base = self.jobNameLineEdit.text().strip() or "custom"
        return f"{sanitize_job_name(base)}.in"

    def _default_custom_preview_text(self) -> str:
        label = self.jobNameLineEdit.text().strip() or "custom job"
        lines = [
            f"! Manual Quantum ESPRESSO input for {label}",
            "! Update namelists and cards as needed.",
            "",
            "&CONTROL",
            "    calculation = 'scf'",
            "    prefix = 'custom'",
            "/",
            "",
            "! Add remaining sections (SYSTEM, ELECTRONS, ATOMIC_SPECIES, etc.).",
        ]
        return "\n".join(lines)

    def _ensure_custom_preview_ready(self, *, show_status: bool) -> None:
        if self._custom_preview_cache:
            name, content, stored_path = self._custom_preview_cache
        else:
            name = self._default_custom_input_name()
            content = self._default_custom_preview_text()
            stored_path = None
        if not name:
            name = self._default_custom_input_name()
        if not content:
            content = self._default_custom_preview_text()

        self._custom_job_active = True

        self.generatedFilesListWidget.clear()
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, content)
        item.setData(Qt.ItemDataRole.UserRole + 1, stored_path)
        if stored_path:
            item.setToolTip(stored_path)
        else:
            item.setToolTip("Unsaved preview")
        self.generatedFilesListWidget.addItem(item)
        self.generatedFilesListWidget.setCurrentRow(0)
        self._generated_preview_paths = [stored_path]

        if show_status and hasattr(self, "statusbar"):
            self.statusbar.showMessage("Manual custom job editing enabled.", 3000)
        self.jobStatusLabel.setText("Custom job ready for manual editing.")
        self._job_preview_dirty = False

        command_text = self._custom_command_cache
        if not command_text.strip():
            input_hint = name or "custom.in"
            output_hint = f"{Path(input_hint).stem or 'custom'}.out"
            command_text = (
                "# Add commands to run your custom job\n"
                f"# Example: mpirun -np {self._job_mpi_procs} pw.x -in {input_hint} | tee {output_hint}\n"
            )

        self._updating_command_preview = True
        try:
            self.commandPreviewEdit.setPlainText(command_text)
        finally:
            self._updating_command_preview = False
        self._custom_commands_override = True
        self._custom_command_cache = command_text
        self._custom_preview_cache = (name, content, stored_path)

    def _cache_custom_preview_state(self) -> None:
        if not self._custom_job_active:
            return

        default_name = self._default_custom_input_name()
        name = default_name
        content = self.generatedFilePreviewEdit.toPlainText()
        stored_path: Optional[str] = None
        if self.generatedFilesListWidget.count() > 0:
            item = self.generatedFilesListWidget.item(0)
            if item is not None:
                label = item.text().strip()
                if label:
                    name = label
                stored = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(stored, str) and stored:
                    content = stored
                path_data = item.data(Qt.ItemDataRole.UserRole + 1)
                if isinstance(path_data, str) and path_data:
                    stored_path = path_data
        if not content:
            content = self._default_custom_preview_text()

        self._custom_preview_cache = (name, content, stored_path)
        self._custom_command_cache = self.commandPreviewEdit.toPlainText()

    def _deactivate_custom_mode(self) -> None:
        if not self._custom_job_active:
            return

        self._cache_custom_preview_state()
        self._custom_job_active = False
        self._custom_commands_override = False
        self.generatedFilesListWidget.clear()
        self.generatedFilePreviewEdit.clear()
        self._generated_preview_paths = []
        self._set_command_preview_text("", force=True)
        self._job_preview_dirty = True

    def _clear_pseudo_selection(self) -> None:
        if not self._pseudo_files:
            return
        self._pseudo_files.clear()
        self._refresh_pseudo_selection_view()
        self._mark_job_preview_stale()
        self.statusbar.showMessage("Cleared pseudopotential selections", 4000)

    def _refresh_pseudo_selection_view(self) -> None:
        self.pseudoFilesListWidget.clear()
        if not self._pseudo_files:
            self.pseudoSummaryLabel.setText("No pseudopotentials selected.")
            self.pseudoClearButton.setEnabled(False)
            return

        ordered_elements: List[str] = []
        seen: set[str] = set()
        for element in self._current_species_order:
            if element in self._pseudo_files and element not in seen:
                ordered_elements.append(element)
                seen.add(element)
        for element in self._pseudo_files:
            if element not in seen:
                ordered_elements.append(element)
                seen.add(element)

        for element in ordered_elements:
            path = Path(self._pseudo_files[element])
            item = QListWidgetItem(f"{element}: {path.name}")
            item.setToolTip(str(path))
            self.pseudoFilesListWidget.addItem(item)

        directories = {str(Path(p).parent) for p in self._pseudo_files.values() if p}
        summary = f"Selected {len(self._pseudo_files)} pseudopotential file(s)."
        if len(directories) == 1:
            summary += f" Directory: {directories.pop()}."
        self.pseudoSummaryLabel.setText(summary)
        self.pseudoClearButton.setEnabled(True)

    def _prune_pseudo_assignments(self, elements: List[str]) -> None:
        if not self._pseudo_files or not elements:
            return

        allowed = {normalize_element_symbol(el) for el in elements if el}
        new_mapping: Dict[str, str] = {}
        updated = False
        for element, path in self._pseudo_files.items():
            symbol = normalize_element_symbol(element)
            if symbol and symbol in allowed:
                new_mapping[symbol] = path
            else:
                updated = True

        if updated or len(new_mapping) != len(self._pseudo_files):
            self._pseudo_files = new_mapping
            self._refresh_pseudo_selection_view()

    @staticmethod
    def _extract_atomic_species(text: str) -> List[AtomicSpeciesEntry]:
        species: List[AtomicSpeciesEntry] = []
        if not text:
            return species

        in_block = False
        seen: set[str] = set()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if in_block:
                    break
                continue

            upper = line.upper()
            if not in_block:
                if upper.startswith("ATOMIC_SPECIES"):
                    in_block = True
                continue

            if upper.startswith("ATOMIC_POSITIONS") or upper.startswith("K_POINTS") or upper.startswith("CELL_PARAMETERS") or line.startswith("&"):
                break

            parts = line.split()
            if not parts:
                continue

            symbol = normalize_element_symbol(parts[0])
            if not symbol or symbol in seen:
                continue

            mass = parts[1] if len(parts) > 1 else None
            pseudo = parts[2] if len(parts) > 2 else None
            species.append(AtomicSpeciesEntry(element=symbol, mass=mass, pseudo=pseudo))
            seen.add(symbol)

        return species

    def _get_input_source(self, show_errors: bool) -> Optional[str]:
        path = self.jobInputPathLineEdit.text().strip()
        if not path:
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            if show_errors:
                self._show_error("The selected input file does not exist.")
            else:
                self.statusbar.showMessage("Selected input file not found", 4000)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not read input file: {exc}")
        return None

    @staticmethod
    def _strip_quotes_text(value: Optional[str]) -> str:
        text = (value or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1]
        return text

    @staticmethod
    def _qe_quote(value: str) -> str:
        cleaned = (value or "").strip()
        escaped = cleaned.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _bandx_base_parts(source_name: str) -> tuple[str, bool]:
        name = Path((source_name or "").strip()).name
        if not name:
            return "", False
        lowered = name.lower()
        if lowered.endswith("scf.in"):
            prefix = name[: -len("scf.in")].rstrip(".-_ ")
            return prefix, True
        return Path(name).stem or "", False

    def _bandx_dialog_defaults(self, seed: ScfSeedParameters, source_path: str) -> Dict[str, Any]:
        control = seed.sections.get("control", {}) if seed else {}
        prefix_clean = self._strip_quotes_text(control.get("prefix"))
        base_name, is_scf_pattern = self._bandx_base_parts(Path(source_path).name)
        if not prefix_clean:
            fallback_prefix = base_name or Path(source_path).stem or "pwscf"
            prefix_clean = fallback_prefix or "pwscf"

        outdir_clean = self._strip_quotes_text(control.get("outdir"))
        if not outdir_clean:
            outdir_clean = self.outputDirectoryLineEdit.text().strip() or "./"
        if not outdir_clean:
            outdir_clean = "./"

        if is_scf_pattern and base_name:
            input_name = f"{base_name}.bandx.in"
            filband_name = f"{base_name}.bandx.dat"
            filp_name = f"{base_name}.bandx.p"
        else:
            input_name = "bands.in"
            filband_name = "bands.dat"
            filp_name = "bands.p"

        optional_order = [
            "filp",
            "spin_component",
            "lsigma",
            "lp",
            "lsym",
            "no_overlap",
            "plot_2d",
            "firstk",
            "lastk",
        ]
        optional_meta: Dict[str, Dict[str, Any]] = {
            "filp": {"type": "string", "default": filp_name, "label": "filp"},
            "spin_component": {"type": "int", "default": 0, "label": "spin_component", "minimum": -3, "maximum": 3},
            "lsigma": {"type": "bool", "default": False, "label": "lsigma"},
            "lp": {"type": "bool", "default": False, "label": "lp"},
            "lsym": {"type": "bool", "default": True, "label": "lsym"},
            "no_overlap": {"type": "bool", "default": False, "label": "no_overlap"},
            "plot_2d": {"type": "bool", "default": False, "label": "plot_2d"},
            "firstk": {"type": "int", "default": 1, "label": "firstk", "minimum": 1, "maximum": 100000},
            "lastk": {"type": "int", "default": -1, "label": "lastk", "minimum": -1, "maximum": 100000},
        }

        return {
            "base_name": base_name,
            "is_scf_pattern": is_scf_pattern,
            "input_name": input_name,
            "prefix": prefix_clean,
            "outdir": outdir_clean,
            "filband": filband_name,
            "optional": optional_meta,
            "optional_order": optional_order,
        }

    def _format_bandx_value(self, value: Any, meta: Dict[str, Any]) -> str:
        field_type = (meta.get("type") or "string").lower()
        if field_type == "bool":
            return ".true." if bool(value) else ".false."
        if field_type == "int":
            try:
                return str(int(value))
            except Exception:
                return str(value)
        if field_type == "string":
            return self._qe_quote(str(value))
        return str(value)

    def _pdos_dialog_defaults(self, seed: ScfSeedParameters, source_path: str) -> Dict[str, Any]:
        control = seed.sections.get("control", {}) if seed else {}
        base_name, is_scf_pattern = self._bandx_base_parts(Path(source_path).name)

        prefix_clean = self._strip_quotes_text(control.get("prefix"))
        if not prefix_clean:
            fallback_prefix = base_name or Path(source_path).stem or "pwscf"
            prefix_clean = fallback_prefix or "pwscf"

        outdir_clean = self._strip_quotes_text(control.get("outdir"))
        if not outdir_clean:
            outdir_clean = self.outputDirectoryLineEdit.text().strip() or "./"
        if not outdir_clean:
            outdir_clean = "./"

        if is_scf_pattern and base_name:
            input_name = f"{base_name}.projwfc.in"
        else:
            input_name = "projwfc.in"

        return {
            "input_name": input_name,
            "prefix": prefix_clean,
            "outdir": outdir_clean,
            "filpdos": f"{prefix_clean or 'projwfc'}.pdos",
            "base_name": base_name,
            "is_scf_pattern": is_scf_pattern,
        }

    def _render_pdos_input(
        self,
        defaults: Dict[str, Any],
        options: Dict[str, Any],
    ) -> tuple[str, str]:
        prefix = options.get("prefix", defaults.get("prefix", ""))
        outdir = options.get("outdir", defaults.get("outdir", ""))
        filpdos = options.get("filpdos", defaults.get("filpdos", ""))

        optional = options.get("optional", {}) or {}
        advanced = options.get("advanced", {}) or {}

        lines: List[str] = [
            "&PROJWFC",
            f"    prefix = {self._qe_quote(prefix)}",
            f"    outdir = {self._qe_quote(outdir)}",
            f"    filpdos = {self._qe_quote(filpdos)}",
        ]

        if "ngauss" in optional:
            lines.append(f"    ngauss = {optional['ngauss']}")
        if "degauss" in optional:
            lines.append(f"    degauss = {optional['degauss']}")
        if "Emin" in optional:
            lines.append(f"    Emin = {optional['Emin']}")
        if "Emax" in optional:
            lines.append(f"    Emax = {optional['Emax']}")
        if "DeltaE" in optional:
            lines.append(f"    DeltaE = {optional['DeltaE']}")
        if "filproj" in optional:
            lines.append(f"    filproj = {self._qe_quote(optional['filproj'])}")

        advanced_order = [
            "lsym",
            "diag_basis",
            "pawproj",
            "lbinary_data",
            "kresolveddos",
            "plotboxes",
        ]
        for key in advanced_order:
            if key not in advanced:
                continue
            value = advanced[key]
            lines.append(f"    {key} = {'.true.' if value else '.false.'}")

        lines.extend(["/", ""])
        return defaults.get("input_name", "projwfc.in"), "\n".join(lines)

    def _phonon_dialog_defaults(self, seed: ScfSeedParameters, source_path: str) -> Dict[str, Any]:
        control = seed.sections.get("control", {}) if seed else {}
        prefix_clean = self._strip_quotes_text(control.get("prefix"))
        if not prefix_clean:
            prefix_clean = Path(source_path).stem or "pwscf"

        outdir_clean = self._strip_quotes_text(control.get("outdir"))
        if not outdir_clean:
            outdir_clean = self.outputDirectoryLineEdit.text().strip() or "./"
        if not outdir_clean:
            outdir_clean = "./"

        tr2_default = control.get("tr2_ph", "1.0d-12")
        if isinstance(tr2_default, (int, float)):
            tr2_default = f"{tr2_default}"
        else:
            tr2_default = str(tr2_default or "1.0d-12")

        nmix_default = control.get("nmix_ph", 4)
        try:
            nmix_default_int = int(float(nmix_default))
        except Exception:
            nmix_default_int = 4

        verbosity_default = self._strip_quotes_text(control.get("verbosity")) or "default"

        fildyn_default = self._strip_quotes_text(control.get("fildyn"))
        if not fildyn_default:
            fildyn_default = f"{prefix_clean}.dyn"

        fildvscf_default = self._strip_quotes_text(control.get("fildvscf"))

        flags = {
            "epsil": False,
            "trans": True,
            "lmultipole": False,
            "recover": False,
            "low_directory_check": False,
            "ldisp": False,
            "qplot": False,
            "q2d": False,
            "q_in_band_form": False,
            "lqdir": False,
            "search_sym": True,
        }

        base_namelist: Dict[str, str] = {
            "prefix": self._qe_quote(prefix_clean),
            "outdir": self._qe_quote(outdir_clean),
            "tr2_ph": tr2_default,
            "nmix_ph": str(nmix_default_int),
            "verbosity": self._qe_quote(verbosity_default),
            "fildyn": self._qe_quote(fildyn_default),
        }

        if fildvscf_default:
            base_namelist["fildvscf"] = self._qe_quote(fildvscf_default)

        k_points = seed.k_points if seed else None
        grid_defaults = (4, 4, 4)
        offset_defaults = (0, 0, 0)
        if isinstance(k_points, dict):
            grid = k_points.get("grid")
            offset = k_points.get("offset")
            if isinstance(grid, (list, tuple)) and len(grid) == 3:
                try:
                    grid_defaults = tuple(int(float(x)) for x in grid)  # type: ignore[assignment]
                except Exception:
                    grid_defaults = (4, 4, 4)
            if isinstance(offset, (list, tuple)) and len(offset) == 3:
                try:
                    offset_defaults = tuple(int(float(x)) for x in offset)  # type: ignore[assignment]
                except Exception:
                    offset_defaults = (0, 0, 0)

        base_namelist["nq1"] = str(grid_defaults[0])
        base_namelist["nq2"] = str(grid_defaults[1])
        base_namelist["nq3"] = str(grid_defaults[2])
        base_namelist["nk1"] = str(grid_defaults[0])
        base_namelist["nk2"] = str(grid_defaults[1])
        base_namelist["nk3"] = str(grid_defaults[2])
        base_namelist["k1"] = str(offset_defaults[0])
        base_namelist["k2"] = str(offset_defaults[1])
        base_namelist["k3"] = str(offset_defaults[2])

        defaults = {
            "input_name": self._suggest_calc_io_names("phonon", Path(source_path).name)[0],
            "prefix": prefix_clean,
            "outdir": outdir_clean,
            "tr2_ph": tr2_default,
            "nmix_ph": nmix_default_int,
            "verbosity": verbosity_default,
            "fildyn": fildyn_default,
            "fildvscf": fildvscf_default,
            "flags": flags,
            "electron_phonon": "",
            "el_ph_nsigma": 10,
            "el_ph_sigma": "0.02",
            "diagonalization": "david",
            "nq": grid_defaults,
            "nk": grid_defaults,
            "kshift": offset_defaults,
            "q_weight": 1,
            "namelist": base_namelist,
            "cif_path": self._current_cif_path() or "",
        }

        return defaults

    def _render_phonon_input(self, namelist: Dict[str, str], q_lines: Sequence[str]) -> str:
        if not namelist:
            raise ValueError("Phonon options produced an empty namelist.")

        width = max(len(key) for key in namelist)
        lines = ["&inputph"]
        for key, value in namelist.items():
            lines.append(f"    {key.ljust(width)} = {value}")
        lines.append("/")

        if q_lines:
            lines.extend(q_lines)

        text = "\n".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        return text

    def _matdyn_dialog_defaults(self, seed: ScfSeedParameters, source_path: str) -> Dict[str, Any]:
        base_name, is_scf_pattern = self._bandx_base_parts(Path(source_path).name)

        flfrc_default = ""
        if is_scf_pattern and base_name:
            flfrc_default = f"{base_name}.fc"
        else:
            control = seed.sections.get("control", {}) if seed else {}
            prefix_clean = self._strip_quotes_text(control.get("prefix"))
            if prefix_clean:
                flfrc_default = f"{prefix_clean}.fc"

        nk_defaults = (0, 0, 0)
        k_points = seed.k_points if seed else None
        if isinstance(k_points, dict):
            grid = k_points.get("grid")
            if isinstance(grid, (list, tuple)) and len(grid) == 3:
                try:
                    nk_defaults = tuple(int(float(v)) for v in grid)  # type: ignore[assignment]
                except Exception:
                    nk_defaults = (0, 0, 0)

        return {
            "flfrc": "",
            "flfrc_hint": flfrc_default,
            "fldos": "matdyn.dos",
            "flfrq": "matdyn.freq",
            "fldyn": "",
            "asr": "no",
            "deltaE": "",
            "ndos": 1000,
            "nk": nk_defaults,
            "dos": False,
            "la2f": False,
            "q_in_band_form": False,
            "weight": 50,
            "cif_path": self._current_cif_path() or "",
        }

    def _render_matdyn_input(self, options: Dict[str, Any]) -> str:
        flfrc = (options.get("flfrc") or "").strip()
        if not flfrc:
            raise ValueError("flfrc is required.")

        flfrq = (options.get("flfrq") or "matdyn.freq").strip() or "matdyn.freq"
        fldos = (options.get("fldos") or "matdyn.dos").strip() or "matdyn.dos"
        fldyn = (options.get("fldyn") or "").strip()
        asr_value = (options.get("asr") or "no").strip().lower() or "no"
        deltaE_text = (options.get("deltaE") or "").strip()
        ndos_value = int(options.get("ndos") or 0)
        nk_values = tuple(options.get("nk") or (0, 0, 0))
        dos_flag = bool(options.get("dos"))
        la2f_flag = bool(options.get("la2f"))
        q_band_flag = bool(options.get("q_in_band_form"))
        q_points = list(options.get("q_points") or [])

        entries: List[tuple[str, str]] = []
        entries.append(("asr", self._qe_quote(asr_value)))
        entries.append(("flfrc", self._qe_quote(flfrc)))
        entries.append(("flfrq", self._qe_quote(flfrq)))
        entries.append(("fldos", self._qe_quote(fldos)))
        if fldyn:
            entries.append(("fldyn", self._qe_quote(fldyn)))
        entries.append(("dos", ".true." if dos_flag else ".false."))

        if len(nk_values) == 3:
            for key, value in zip(("nk1", "nk2", "nk3"), nk_values):
                try:
                    intval = int(value)
                except Exception:
                    continue
                if intval > 0:
                    entries.append((key, str(intval)))

        if deltaE_text:
            entries.append(("deltaE", deltaE_text))

        if ndos_value > 0:
            entries.append(("ndos", str(ndos_value)))

        if la2f_flag:
            entries.append(("la2F", ".true."))

        if q_band_flag:
            entries.append(("q_in_band_form", ".true."))

        pad = max(len(key) for key, _ in entries) if entries else 0
        lines: List[str] = ["&input"]
        for key, value in entries:
            lines.append(f"    {key.ljust(pad)} = {value}")
        lines.append("/")

        include_q_section = q_band_flag and not dos_flag and q_points
        if include_q_section:
            lines.append(str(len(q_points)))
            for point in q_points:
                coords = point.get("coords") or (0.0, 0.0, 0.0)
                try:
                    x1, x2, x3 = (float(coords[0]), float(coords[1]), float(coords[2]))
                except Exception as exc:  # noqa: BLE001 - propagate nicely
                    raise ValueError("Invalid q-point coordinates for band path.") from exc
                weight = int(point.get("weight") or 1)
                label = str(point.get("label", "")).strip()
                row = f"{x1:.10f}\t{x2:.10f}\t{x3:.10f}\t{weight}"
                if label:
                    row += f" !{label}"
                lines.append(row)

        text = "\n".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        return text

    def _suggest_matdyn_input_name(self, source_name: str, band_path: bool) -> str:
        lowered = (source_name or "").strip().lower()
        if lowered.endswith("matdyn.in.dos") or lowered.endswith("matdyn.in.fre"):
            return source_name

        if lowered.endswith("scf.in") and source_name:
            prefix = source_name[: -len("scf.in")]
            suffix = "matdyn.in.fre" if band_path else "matdyn.in.dos"
            return f"{prefix}{suffix}"

        base_name, is_scf_pattern = self._bandx_base_parts(source_name)
        if is_scf_pattern and base_name:
            suffix = "matdyn.in.fre" if band_path else "matdyn.in.dos"
            return f"{base_name}.{suffix}"

        return "matdyn.in.fre" if band_path else "matdyn.in.dos"

    @staticmethod
    def _matdyn_output_name(input_name: str) -> str:
        lowered = input_name.lower()
        idx = lowered.find(".in")
        if idx == -1:
            return f"{Path(input_name).stem}.out"
        return f"{input_name[:idx]}.out{input_name[idx + 3:]}"


    def _epsilon_dialog_defaults(self, seed: ScfSeedParameters, source_path: str) -> Dict[str, Any]:
        control = seed.sections.get("control", {}) if seed else {}
        base_name, is_scf_pattern = self._bandx_base_parts(Path(source_path).name)

        prefix_clean = self._strip_quotes_text(control.get("prefix"))
        if not prefix_clean:
            fallback_prefix = base_name or Path(source_path).stem or "pwscf"
            prefix_clean = fallback_prefix or "pwscf"

        outdir_clean = self._strip_quotes_text(control.get("outdir"))
        if not outdir_clean:
            outdir_clean = self.outputDirectoryLineEdit.text().strip() or "./"
        if not outdir_clean:
            outdir_clean = "./"

        if is_scf_pattern and base_name:
            input_name = f"{base_name}.eps.in"
        else:
            input_name = "eps.in"

        return {
            "input_name": input_name,
            "prefix": prefix_clean,
            "outdir": outdir_clean,
            "smeartype": "lorentzian",
            "intersmear": "0.1d0",
            "intrasmear": "0.0d0",
            "wmax": "30.0d0",
            "wmin": "0.02d0",
            "nw": "500",
            "shift": "0.0d0",
        }

    def _render_epsilon_input(self, defaults: Dict[str, Any], options: Dict[str, Any]) -> tuple[str, str]:
        prefix = options.get("prefix", defaults.get("prefix", ""))
        outdir = options.get("outdir", defaults.get("outdir", ""))
        smeartype = options.get("smeartype", defaults.get("smeartype", "lorentzian"))
        intersmear = options.get("intersmear", defaults.get("intersmear", ""))
        intrasmear = options.get("intrasmear", defaults.get("intrasmear", ""))
        wmax = options.get("wmax", defaults.get("wmax", ""))
        wmin = options.get("wmin", defaults.get("wmin", ""))
        nw = options.get("nw", defaults.get("nw", ""))
        shift = options.get("shift", defaults.get("shift", ""))

        lines = [
            "&INPUTPP",
            "    calculation = 'eps'",
            f"    prefix = {self._qe_quote(prefix)}",
            f"    outdir = {self._qe_quote(outdir)}",
            "/",
            "",
            "&ENERGY_GRID",
            f"    smeartype = {self._qe_quote(smeartype)}",
            f"    intersmear = {intersmear}",
            f"    intrasmear = {intrasmear}",
            f"    wmax = {wmax}",
            f"    wmin = {wmin}",
            f"    nw = {nw}",
            f"    shift = {shift}",
            "/",
            "",
        ]

        return defaults.get("input_name", "eps.in"), "\n".join(lines)

    def _post_process_generated_files(self, config: JobConfig, files: Dict[str, str]) -> Dict[str, str]:
        if not files:
            return {}

        calc = (config.calculation or "").strip().lower()
        processed: Dict[str, str] = {}
        for name, content in files.items():
            text = content
            if calc not in {"bandx", "pdos", "optics", "phonon", "matdyn", "matdyn.x"}:
                text = apply_pseudo_settings_to_text(
                    text,
                    config.pseudo_files,
                    pseudo_dir="./",
                )
                # Match INPUT_PW guidance by forcing high verbosity inside &CONTROL.
                text = ensure_control_verbosity(text, level="'high'")
            processed[name] = text
        return processed

    def _resolve_output_directory(self, config: JobConfig) -> Path:
        path_text = (config.working_directory or "").strip()
        if path_text:
            try:
                base = Path(path_text).expanduser()
            except Exception:
                base = Path(path_text)
            try:
                base.mkdir(parents=True, exist_ok=True)
                return base
            except Exception as exc:  # noqa: BLE001
                self.statusbar.showMessage(f"Could not use working directory '{path_text}': {exc}", 6000)

        source_dir = None
        if config.input_files:
            try:
                source_dir = Path(config.input_files[0]).expanduser().parent
            except Exception:
                try:
                    source_dir = Path(config.input_files[0]).parent
                except Exception:
                    source_dir = None
        return source_dir or PROJECT_ROOT

    def _save_generated_files(self, destination: Path, files: Dict[str, str]) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        if not files:
            return saved

        try:
            destination.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory may already exist; ignore errors here and rely on save failures to warn.
            pass

        for name, content in files.items():
            target_path = destination / name
            try:
                save_text_to_file(target_path, content)
            except Exception as exc:  # noqa: BLE001
                self.statusbar.showMessage(f"Failed to save {name}: {exc}", 6000)
                continue
            saved[name] = str(target_path)
        if saved:
            display_dir = destination if len(saved) > 1 else Path(next(iter(saved.values()))).parent
            self.statusbar.showMessage(f"Saved generated input(s) to {display_dir}", 4000)
        return saved

    def _suggest_calc_io_names(self, calculation: str, source_name: str) -> tuple[str, str]:
        calc = (calculation or "").strip().lower()
        if calc == "band":
            calc = "bands"
        base_name = (source_name or "").strip()
        lowered = base_name.lower()

        if calc == "nscf":
            if lowered.endswith("nscf.in") and base_name:
                stem = Path(base_name).stem or "nscf"
                return base_name, f"{stem}.out"
            if lowered.endswith("scf.in") and base_name:
                prefix = base_name[: -len("scf.in")]
                input_name = f"{prefix}nscf.in"
                stem = Path(input_name).stem or "nscf"
                return input_name, f"{stem}.out"
            input_name = "nscf.in"
            return input_name, "nscf.out"

        if calc == "bands":
            if lowered.endswith("pwbands.in") and base_name:
                stem = Path(base_name).stem or "pwbands"
                return base_name, f"{stem}.out"
            if lowered.endswith("scf.in") and base_name:
                prefix = base_name[: -len("scf.in")]
                input_name = f"{prefix}pwbands.in"
                stem = Path(input_name).stem or "pwbands"
                return input_name, f"{stem}.out"
            input_name = "pwbands.in"
            return input_name, "pwbands.out"

        if calc == "bandx":
            if lowered.endswith("bandx.in") and base_name:
                stem = Path(base_name).stem or "bandx"
                return base_name, f"{stem}.out"
            base_prefix, is_scf_pattern = self._bandx_base_parts(base_name)
            if is_scf_pattern and base_prefix:
                input_name = f"{base_prefix}.bandx.in"
                stem = Path(input_name).stem or "bandx"
                return input_name, f"{stem}.out"
            input_name = "bands.in"
            return input_name, "bands.out"

        if calc == "pdos":
            if lowered.endswith("projwfc.in") and base_name:
                stem = Path(base_name).stem or "projwfc"
                return base_name, f"{stem}.out"
            base_prefix, is_scf_pattern = self._bandx_base_parts(base_name)
            if is_scf_pattern and base_prefix:
                input_name = f"{base_prefix}.projwfc.in"
                stem = Path(input_name).stem or "projwfc"
                return input_name, f"{stem}.out"
            input_name = "projwfc.in"
            return input_name, "projwfc.out"

        if calc == "optics":
            if lowered.endswith("eps.in") and base_name:
                stem = Path(base_name).stem or "eps"
                return base_name, f"{stem}.out"
            base_prefix, is_scf_pattern = self._bandx_base_parts(base_name)
            if is_scf_pattern and base_prefix:
                input_name = f"{base_prefix}.eps.in"
                stem = Path(input_name).stem or "eps"
                return input_name, f"{stem}.out"
            input_name = "eps.in"
            return input_name, "eps.out"

        if calc == "phonon":
            if lowered.endswith("ph.in") and base_name:
                stem = Path(base_name).stem or "ph"
                return base_name, f"{stem}.out"
            base_prefix, is_scf_pattern = self._bandx_base_parts(base_name)
            if is_scf_pattern and base_prefix:
                input_name = f"{base_prefix}.ph.in"
                stem = Path(input_name).stem or "ph"
                return input_name, f"{stem}.out"
            if lowered.endswith("scf.in") and base_name:
                prefix = base_name[: -len("scf.in")]
                input_name = f"{prefix}ph.in"
            else:
                input_name = "ph.in"
            stem = Path(input_name).stem or "ph"
            return input_name, f"{stem}.out"

        if calc in {"matdyn", "matdyn.x"}:
            use_band = lowered.endswith("matdyn.in.fre")
            input_name = self._suggest_matdyn_input_name(base_name, use_band)
            output_name = self._matdyn_output_name(input_name)
            return input_name, output_name

        if calc in {"vc-relax", "relax"}:
            if lowered.endswith("vc-relax.in") and base_name:
                stem = Path(base_name).stem or "vc-relax"
                return base_name, f"{stem}.out"
            if lowered.endswith("scf.in") and base_name:
                prefix = base_name[: -len("scf.in")]
                input_name = f"{prefix}vc-relax.in"
                stem = Path(input_name).stem or "vc-relax"
                return input_name, f"{stem}.out"
            input_name = "vc-relax.in"
            return input_name, "vc-relax.out"

        if not base_name:
            input_name = "qe_job.in"
        else:
            input_name = base_name
        stem = Path(input_name).stem or "qe_job"
        return input_name, f"{stem}.out"

    def _build_commands_for_config(self, config: JobConfig) -> List[str]:
        calc_raw = (config.calculation or "").strip()
        calc = calc_raw.lower()
        if calc == "band":
            calc = "bands"
        if calc == "custom":
            return []
        if not config.input_files or not calc:
            return build_job_commands(config)

        commands: List[str] = []
        for raw_path in config.input_files:
            if not raw_path:
                continue
            source_name = Path(raw_path).name
            input_name, output_name = self._suggest_calc_io_names(calc, source_name)
            try:
                command = create_run_command(
                    calc_raw,
                    input_name,
                    output_name,
                    config.mpi_procs,
                    validate_input=False,
                )
            except Exception:
                commands = []
                break
            commands.append(command)

        if not commands:
            return build_job_commands(config)
        return commands

    def _copy_pseudo_files_to_workdir(self, config: JobConfig) -> None:
        if (config.calculation or "").strip().lower() == "bandx":
            return

        if not config.pseudo_files:
            return

        workdir_text = (config.working_directory or "").strip()
        if not workdir_text:
            if not self._warned_missing_pseudo_dir:
                self.statusbar.showMessage(
                    "Set a working directory to copy pseudopotentials before running.",
                    6000,
                )
                self._warned_missing_pseudo_dir = True
            return

        destination = Path(workdir_text).expanduser()
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.statusbar.showMessage(f"Could not prepare working directory: {exc}", 6000)
            return

        self._warned_missing_pseudo_dir = False
        copied = 0
        failures: List[str] = []
        seen: set[str] = set()

        for raw_path in config.pseudo_files.values():
            if not raw_path:
                continue
            try:
                src = Path(raw_path).expanduser()
            except Exception:
                src = Path(raw_path)

            try:
                resolved_src = src.resolve(strict=True)
            except FileNotFoundError:
                failures.append(f"{src.name} (missing)")
                continue
            except Exception:
                resolved_src = src.absolute()

            canonical = str(resolved_src)
            if canonical in seen:
                continue
            seen.add(canonical)

            dest_path = destination / resolved_src.name
            try:
                if dest_path.exists():
                    try:
                        if dest_path.resolve() == resolved_src:
                            continue
                    except Exception:
                        pass
                shutil.copy2(resolved_src, dest_path)
                copied += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{resolved_src.name} ({exc})")

        if failures:
            summary = "; ".join(failures[:3])
            if len(failures) > 3:
                summary += "; …"
            self.statusbar.showMessage(f"Failed to copy pseudopotentials: {summary}", 8000)
            return

        if copied:
            self.statusbar.showMessage(
                f"Copied {copied} pseudopotential file(s) to {destination}",
                5000,
            )

    def _apply_job_preview(
        self,
        config: JobConfig,
        generated_files: dict[str, str],
        commands: list[str],
        *,
        show_status: bool,
    ) -> None:
        processed_files = self._post_process_generated_files(config, generated_files)
        output_dir = self._resolve_output_directory(config)
        saved_paths = self._save_generated_files(output_dir, processed_files)
        self._copy_pseudo_files_to_workdir(config)

        self.generatedFilesListWidget.clear()
        self._generated_preview_paths = []
        first_saved_path: Optional[str] = None
        for name, content in processed_files.items():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, content)
            saved_path = saved_paths.get(name)
            if saved_path:
                item.setToolTip(saved_path)
                item.setData(Qt.ItemDataRole.UserRole + 1, saved_path)
                if first_saved_path is None:
                    first_saved_path = saved_path
            else:
                item.setToolTip("Unsaved preview")
            self.generatedFilesListWidget.addItem(item)
            self._generated_preview_paths.append(saved_path)

        if processed_files:
            self.generatedFilesListWidget.setCurrentRow(0)
            self._handle_generated_file_selection_changed(0)
        else:
            self.generatedFilePreviewEdit.setPlainText("No generated files yet.")

        command_text = "\n".join(commands) if commands else "No commands generated."
        self._set_command_preview_text(command_text, force=True)

        label_name = config.name or config.sanitized_name()
        self.jobStatusLabel.setText(f"Preview ready for '{label_name}'")
        if show_status:
            self.statusbar.showMessage("Job preview updated", 3000)

        self._job_preview_dirty = False

    def _set_command_preview_text(self, text: str, *, force: bool = False) -> None:
        if not force and self._custom_commands_override:
            return
        self._updating_command_preview = True
        try:
            self.commandPreviewEdit.setPlainText(text)
        finally:
            self._updating_command_preview = False
        if force:
            self._custom_commands_override = False

    def _handle_command_preview_text_changed(self) -> None:
        if self._updating_command_preview:
            return
        if self._custom_job_active:
            self._custom_commands_override = True
            self._custom_command_cache = self.commandPreviewEdit.toPlainText()
            return
        if self._custom_commands_override:
            return
        self._custom_commands_override = True
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(
                "Custom command edits active; regenerate to restore defaults.",
                4000,
            )

    def _collect_command_preview_lines(self) -> List[str]:
        content = self.commandPreviewEdit.toPlainText()
        lines: List[str] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

    def _update_command_preview(self) -> None:
        config = self._collect_job_config()
        if (config.calculation or "").strip().lower() == "custom":
            return
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001 - surface issue and keep preview text predictable
            self._set_command_preview_text("No commands generated.", force=True)
            self.statusbar.showMessage(f"Could not refresh commands: {exc}", 6000)
            return

        command_text = "\n".join(commands) if commands else "No commands generated."
        self._set_command_preview_text(command_text, force=True)

    def _refresh_job_views(self, show_errors: bool = True) -> bool:
        config = self._collect_job_config()
        calculation = (config.calculation or "").strip().lower()

        if calculation == "custom":
            self._ensure_custom_preview_ready(show_status=show_errors)
            return True

        source_text = self._get_input_source(show_errors)

        if not config.input_files:
            self.generatedFilesListWidget.clear()
            self.generatedFilePreviewEdit.clear()
            self._set_command_preview_text("Select or generate a source input file to see previews.", force=True)
            self.jobStatusLabel.setText("Awaiting input file")
            self._job_preview_dirty = True
            return False

        if not self._job_preview_dirty and self.generatedFilesListWidget.count() > 0:
            try:
                commands = self._build_commands_for_config(config)
            except Exception as exc:  # noqa: BLE001 - surface to user
                if show_errors:
                    self._show_error(f"Could not prepare job preview: {exc}")
                return False
            command_text = "\n".join(commands) if commands else "No commands generated."
            self._set_command_preview_text(command_text)
            label_name = config.name or config.sanitized_name()
            self.jobStatusLabel.setText(f"Preview ready for '{label_name}'")
            if show_errors:
                self.statusbar.showMessage("Job preview updated", 3000)
            return True

        if calculation == "bandx":
            return self._prepare_bandx_inputs(config, show_errors=show_errors, show_status=show_errors)
        if calculation == "phonon":
            return self._prepare_phonon_inputs(config, show_errors=show_errors, show_status=show_errors)

        try:
            generated_files = generate_dummy_inputs(
                config,
                base_text=source_text,
                system_config=self.system_config,
            )
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001 - surface to user
            if show_errors:
                self._show_error(f"Could not prepare job preview: {exc}")
            return False

        self._apply_job_preview(config, generated_files, commands, show_status=show_errors)
        return True

    def _handle_generated_file_selection_changed(self, row: int) -> None:
        self._updating_generated_preview = True
        try:
            if row < 0:
                self.generatedFilePreviewEdit.clear()
                return

            item = self.generatedFilesListWidget.item(row)
            if item is None:
                self.generatedFilePreviewEdit.clear()
                return

            content = item.data(Qt.ItemDataRole.UserRole) or ""
            self.generatedFilePreviewEdit.setPlainText(content)
            if self._custom_job_active:
                cached_name = item.text().strip() or self._default_custom_input_name()
                path_hint = item.data(Qt.ItemDataRole.UserRole + 1)
                stored_path = path_hint if isinstance(path_hint, str) and path_hint else None
                self._custom_preview_cache = (cached_name, content, stored_path)
        finally:
            self._updating_generated_preview = False

    def _handle_generated_preview_edited(self) -> None:
        if self._updating_generated_preview:
            return

        row = self.generatedFilesListWidget.currentRow()
        if row < 0:
            return

        item = self.generatedFilesListWidget.item(row)
        if item is None:
            return

        content = self.generatedFilePreviewEdit.toPlainText()
        item.setData(Qt.ItemDataRole.UserRole, content)
        if self._custom_job_active:
            cached_name = item.text().strip() or self._default_custom_input_name()
            path_hint = item.data(Qt.ItemDataRole.UserRole + 1)
            stored_path = path_hint if isinstance(path_hint, str) and path_hint else None
            self._custom_preview_cache = (cached_name, content, stored_path)

    def _handle_prepare_job_preview(self) -> None:
        config = self._collect_job_config()

        calculation = (config.calculation or "scf").strip().lower()
        if calculation == "custom":
            self._ensure_custom_preview_ready(show_status=True)
            return

        if not config.input_files:
            self._show_error("Select a source input file before generating.")
            return

        if calculation == "scf":
            self._generate_scf_preview(config)
        elif calculation == "nscf":
            self._generate_nscf_preview(config)
        elif calculation == "band":
            self._generate_bands_preview(config)
        elif calculation == "bandx":
            self._generate_bandx_preview(config)
        elif calculation == "pdos":
            self._generate_pdos_preview(config)
        elif calculation == "optics":
            self._prepare_optics_inputs(config, show_errors=True, show_status=True)
        elif calculation == "vc-relax":
            self._generate_vc_relax_preview(config)
        elif calculation == "phonon":
            self._generate_phonon_preview(config)
        elif calculation in {"matdyn", "matdyn.x"}:
            self._generate_matdyn_preview(config)
        else:
            if self._refresh_job_views():
                self.statusbar.showMessage("Job preview refreshed", 3000)

    def _generate_scf_preview(self, config: JobConfig) -> None:
        source_path = config.input_files[0]

        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            self._show_error("The selected input file does not exist.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not parse SCF seed: {exc}")
            return

        cif_path = self._current_cif_path()
        dialog = CalculationOptionsDialog("scf", seed, config, self.system_config, self, cif_path=cif_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        overrides = dialog.result_overrides() or {}
        sections = overrides.setdefault("sections", {})
        control_section = sections.setdefault("control", {})
        control_section["verbosity"] = "'high'"
        try:
            rendered = render_scf_template(seed, config=self.system_config, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not assemble SCF input: {exc}")
            return

        generated_files = {Path(source_path).name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not prepare commands: {exc}")
            return

        self._apply_job_preview(config, generated_files, commands, show_status=True)

    def _generate_vc_relax_preview(self, config: JobConfig) -> None:
        source_path = config.input_files[0]

        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            self._show_error("The selected input file does not exist.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not parse vc-relax seed: {exc}")
            return

        dialog = CalculationOptionsDialog("vc-relax", seed, config, self.system_config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        overrides = dialog.result_overrides() or {}
        sections = overrides.setdefault("sections", {})
        control_section = sections.setdefault("control", {})
        control_section["verbosity"] = "'high'"
        try:
            rendered = render_scf_template(seed, config=self.system_config, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not assemble vc-relax input: {exc}")
            return

        source_name = Path(source_path).name
        input_name, _ = self._suggest_calc_io_names("vc-relax", source_name)
        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not prepare commands: {exc}")
            return

        self._apply_job_preview(config, generated_files, commands, show_status=True)

    def _generate_phonon_preview(self, config: JobConfig) -> None:
        self._prepare_phonon_inputs(config, show_errors=True, show_status=True)

    def _generate_matdyn_preview(self, config: JobConfig) -> None:
        source_path = config.input_files[0]

        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            self._show_error("The selected input file does not exist.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not parse SCF seed for matdyn: {exc}")
            return

        defaults = self._matdyn_dialog_defaults(seed, source_path)
        dialog = MatdynOptionsDialog(defaults, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        options = dialog.result_options() or {}
        try:
            rendered = self._render_matdyn_input(options)
        except ValueError as exc:
            self._show_error(f"Could not assemble matdyn input: {exc}")
            return

        source_name = Path(source_path).name
        use_band_path = bool(options.get("q_in_band_form")) and not bool(options.get("dos"))
        input_name = self._suggest_matdyn_input_name(source_name, use_band_path)
        generated_files = {input_name: rendered}

        output_name = self._matdyn_output_name(input_name)
        try:
            command = create_run_command(
                config.calculation,
                input_name,
                output_name,
                config.mpi_procs,
                validate_input=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not prepare commands: {exc}")
            return

        self._apply_job_preview(config, generated_files, [command], show_status=True)

    def _generate_nscf_preview(self, config: JobConfig) -> None:
        source_path = config.input_files[0]

        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            self._show_error("The selected input file does not exist.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not parse NSCF seed: {exc}")
            return

        cif_path = self._current_cif_path()
        dialog = CalculationOptionsDialog("nscf", seed, config, self.system_config, self, cif_path=cif_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        overrides = dialog.result_overrides() or {}
        sections = overrides.setdefault("sections", {})
        control_section = sections.setdefault("control", {})
        control_section["verbosity"] = "'high'"
        try:
            rendered = render_scf_template(seed, config=self.system_config, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not assemble NSCF input: {exc}")
            return

        source_name = Path(source_path).name
        input_name, _output_name = self._suggest_calc_io_names("nscf", source_name)
        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not prepare commands: {exc}")
            return

        self._apply_job_preview(config, generated_files, commands, show_status=True)

    def _generate_bands_preview(self, config: JobConfig) -> None:
        source_path = config.input_files[0]

        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            self._show_error("The selected input file does not exist.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not parse bands seed: {exc}")
            return

        cif_path = self._current_cif_path()
        dialog = CalculationOptionsDialog("bands", seed, config, self.system_config, self, cif_path=cif_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        overrides = dialog.result_overrides() or {}
        sections = overrides.setdefault("sections", {})
        control_section = sections.setdefault("control", {})
        control_section["verbosity"] = "'high'"
        try:
            rendered = render_scf_template(seed, config=self.system_config, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not assemble bands input: {exc}")
            return

        source_name = Path(source_path).name
        input_name, _ = self._suggest_calc_io_names("bands", source_name)
        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not prepare commands: {exc}")
            return

        self._apply_job_preview(config, generated_files, commands, show_status=True)

    def _generate_bandx_preview(self, config: JobConfig) -> None:
        self._prepare_bandx_inputs(config, show_errors=True, show_status=True)

    def _generate_pdos_preview(self, config: JobConfig) -> None:
        self._prepare_pdos_inputs(config, show_errors=True, show_status=True)

    def _prepare_pdos_inputs(self, config: JobConfig, *, show_errors: bool, show_status: bool) -> bool:
        if not config.input_files:
            if show_errors:
                self._show_error("Select a source input file before generating.")
            return False

        source_path = config.input_files[0]
        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            if show_errors:
                self._show_error("The selected input file does not exist.")
            return False
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not parse SCF seed for pdos: {exc}")
            return False

        defaults = self._pdos_dialog_defaults(seed, source_path)
        dialog = PdosOptionsDialog(defaults, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        options = dialog.result_options()
        if not options:
            return False

        try:
            input_name, rendered = self._render_pdos_input(defaults, options)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not assemble pdos input: {exc}")
            return False

        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not prepare commands: {exc}")
            return False

        self._apply_job_preview(config, generated_files, commands, show_status=show_status)
        return True

    def _prepare_phonon_inputs(self, config: JobConfig, *, show_errors: bool, show_status: bool) -> bool:
        if not config.input_files:
            if show_errors:
                self._show_error("Select a source input file before generating.")
            return False

        source_path = config.input_files[0]
        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            if show_errors:
                self._show_error("The selected input file does not exist.")
            return False
        except Exception as exc:  # noqa: BLE001 - surface to user
            if show_errors:
                self._show_error(f"Could not parse SCF seed for phonon: {exc}")
            return False

        defaults = self._phonon_dialog_defaults(seed, source_path)
        dialog = PhononOptionsDialog(defaults, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        options = dialog.result_options() or {}
        base_namelist = dict(defaults.get("namelist", {}))
        namelist_updates = options.get("namelist") or {}
        namelist = dict(base_namelist)
        namelist.update(namelist_updates)

        q_lines = options.get("q_lines") or []

        try:
            rendered = self._render_phonon_input(namelist, q_lines)
        except Exception as exc:  # noqa: BLE001 - surface to user
            if show_errors:
                self._show_error(f"Could not assemble phonon input: {exc}")
            return False

        source_name = Path(source_path).name
        input_name, _ = self._suggest_calc_io_names("phonon", source_name)
        generated_files = {input_name: rendered}

        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001 - surface to user
            if show_errors:
                self._show_error(f"Could not prepare commands: {exc}")
            return False

        self._apply_job_preview(config, generated_files, commands, show_status=show_status)
        return True

    def _prepare_bandx_inputs(self, config: JobConfig, *, show_errors: bool, show_status: bool) -> bool:
        if not config.input_files:
            if show_errors:
                self._show_error("Select a source input file before generating.")
            return False

        source_path = config.input_files[0]
        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            if show_errors:
                self._show_error("The selected input file does not exist.")
            return False
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not parse SCF seed for bandx: {exc}")
            return False

        defaults = self._bandx_dialog_defaults(seed, source_path)
        dialog = BandxOptionsDialog(defaults, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        options = dialog.result_options()
        if not options:
            return False

        try:
            input_name, rendered = self._render_bandx_input(defaults, options)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not assemble bandx input: {exc}")
            return False

        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not prepare commands: {exc}")
            return False

        self._apply_job_preview(config, generated_files, commands, show_status=show_status)
        return True

    def _prepare_optics_inputs(self, config: JobConfig, *, show_errors: bool, show_status: bool) -> bool:
        if not config.input_files:
            if show_errors:
                self._show_error("Select a source input file before generating.")
            return False

        source_path = config.input_files[0]
        try:
            seed = parse_scf_seed(source_path)
        except FileNotFoundError:
            if show_errors:
                self._show_error("The selected input file does not exist.")
            return False
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not parse SCF seed for optics: {exc}")
            return False

        defaults = self._epsilon_dialog_defaults(seed, source_path)
        dialog = EpsilonOptionsDialog(defaults, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        options = dialog.result_options()
        if not options:
            return False

        try:
            input_name, rendered = self._render_epsilon_input(defaults, options)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not assemble optics input: {exc}")
            return False

        generated_files = {input_name: rendered}
        try:
            commands = self._build_commands_for_config(config)
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                self._show_error(f"Could not prepare commands: {exc}")
            return False

        self._apply_job_preview(config, generated_files, commands, show_status=show_status)
        return True

    def _render_bandx_input(self, defaults: Dict[str, Any], options: Dict[str, Any]) -> tuple[str, str]:
        prefix = options.get("prefix", defaults.get("prefix", ""))
        outdir = options.get("outdir", defaults.get("outdir", ""))
        filband = options.get("filband", defaults.get("filband", ""))
        extras = options.get("extra", {}) or {}

        lines: List[str] = [
            "&BANDS",
            f"    prefix = {self._qe_quote(prefix)}",
            f"    outdir = {self._qe_quote(outdir)}",
            f"    filband = {self._qe_quote(filband)}",
        ]

        optional_meta = defaults.get("optional", {})
        optional_order = defaults.get("optional_order", list(optional_meta.keys()))
        for key in optional_order:
            if key not in extras:
                continue
            meta = optional_meta.get(key, {})
            formatted = self._format_bandx_value(extras[key], meta)
            lines.append(f"    {key} = {formatted}")

        lines.extend(["/", ""])
        return defaults.get("input_name", "bands.in"), "\n".join(lines)

    def _handle_save_job_script(self) -> None:
        if not self._refresh_job_views(show_errors=False):
            self._show_error("Provide a valid input file before saving a script.")
            return

        config = self._collect_job_config()
        if self._custom_commands_override:
            commands = self._collect_command_preview_lines()
            if not commands:
                self._show_error("Enter at least one command before saving a script.")
                return
        else:
            try:
                commands = self._build_commands_for_config(config)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"Could not prepare commands: {exc}")
                return
        try:
            script = generate_job_script(config, commands=commands)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not assemble job script: {exc}")
            return

        default_dir = Path(config.working_directory or PROJECT_ROOT)
        default_path = default_dir / config.script_filename()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save job script",
            str(default_path),
            "Shell script (*.sh);;All files (*)",
        )
        if not file_path:
            return

        try:
            save_text_to_file(file_path, script)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not save job script: {exc}")
            return

        self.statusbar.showMessage(f"Job script saved to {file_path}", 5000)

    def _handle_save_generated_preview(self) -> None:
        row = self.generatedFilesListWidget.currentRow()
        if row < 0:
            self._show_error("Select a generated file before saving the preview.")
            return

        item = self.generatedFilesListWidget.item(row)
        if item is None:
            self._show_error("Select a generated file before saving the preview.")
            return

        content = self.generatedFilePreviewEdit.toPlainText()

        config = self._collect_job_config()
        default_dir = Path(config.working_directory or PROJECT_ROOT)

        suggested_name = item.text().strip() or "generated.in"
        stored_path = None
        if 0 <= row < len(self._generated_preview_paths):
            stored_path = self._generated_preview_paths[row]
        if stored_path:
            default_path = Path(stored_path)
        else:
            default_path = default_dir / suggested_name

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save generated preview",
            str(default_path),
            "Quantum Espresso input (*.in *.txt);;All files (*)",
        )
        if not file_path:
            return

        try:
            save_text_to_file(file_path, content)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not save preview: {exc}")
            return

        path_obj = Path(file_path).expanduser()
        try:
            resolved_path = str(path_obj.resolve())
        except Exception:  # noqa: BLE001 - best effort; fallback to expanded path
            resolved_path = str(path_obj)
        base_name = Path(resolved_path).name
        item.setText(base_name)
        item.setData(Qt.ItemDataRole.UserRole + 1, resolved_path)
        item.setToolTip(resolved_path)
        if 0 <= row < len(self._generated_preview_paths):
            self._generated_preview_paths[row] = resolved_path

        if row == 0:
            self.jobInputPathLineEdit.setText(resolved_path)

        if self._custom_job_active and row == 0:
            self._custom_preview_cache = (base_name, content, resolved_path)

        self._update_command_preview()
        self.statusbar.showMessage(f"Generated preview saved to {resolved_path}", 5000)

    def _handle_enqueue_job(self) -> None:
        if not self._refresh_job_views():
            return

        config = self._collect_job_config()
        if self._custom_commands_override:
            commands = self._collect_command_preview_lines()
            if not commands:
                self._show_error("Enter at least one command before adding to the queue.")
                return
        else:
            try:
                commands = self._build_commands_for_config(config)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"Could not prepare commands: {exc}")
                return
        self._enqueue_job(config, commands, origin="job_creator")
        label_name = config.name or config.sanitized_name()
        self.jobStatusLabel.setText(f"Job '{label_name}' added to queue")
        self.statusbar.showMessage("Job enqueued", 3000)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "Quantum ESPRESSO helper GUI\n\nPrepare input decks and manage jobs.",
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        if hasattr(self, "statusbar") and isinstance(self.statusbar, QStatusBar):
            self.statusbar.showMessage(message, 5000)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
        if hasattr(self, "statusbar") and isinstance(self.statusbar, QStatusBar):
            self.statusbar.showMessage(message, 5000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API requirement)
        self._persist_job_queue()
        if hasattr(self, "statusbar") and isinstance(self.statusbar, QStatusBar):
            self.statusbar.showMessage("Closing...", 1000)
        super().closeEvent(event)


# Color theme constants
PRIMARY_COLOR = "#992327"
PRIMARY_LIGHT = "#F4D3D7"
SECONDARY_COLOR = "#334155"
NEUTRAL_BG = "#F4F5F7"
NEUTRAL_BORDER = "#E2E8F0"
TEXT_DARK = "#0F172A"
TEXT_LIGHT = "#FFFFFF"
TEXT_MUTED = "#475569"

APP_QSS = f"""
QMainWindow {{
    background: #CCCCCC;
}}

QFrame#headerBar {{
    background: {PRIMARY_COLOR};
    border-radius: 12px;
}}

QLabel#headerTitle {{
    color: {TEXT_LIGHT};
    font-size: 28px;
    font-weight: 700;
}}

QLabel#headerSubtitle {{
    color: {PRIMARY_LIGHT};
    font-size: 12px;
}}

QMenuBar {{
    background: {NEUTRAL_BG};
    color: {SECONDARY_COLOR};
    padding: 4px 6px;
    border: 1px solid {NEUTRAL_BORDER};
    border-radius: 10px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 10px;
    margin: 2px 4px;
    border-radius: 6px;
}}

QMenuBar::item:selected {{
    background: {PRIMARY_LIGHT};
    color: {PRIMARY_COLOR};
}}

QMenu {{
    background: {TEXT_LIGHT};
    color: {TEXT_DARK};
    border: 1px solid {NEUTRAL_BORDER};
}}

QMenu::item:selected {{
    background: {PRIMARY_LIGHT};
    color: {PRIMARY_COLOR};
}}

QTabWidget::pane {{
    border: none;
    background: transparent;
}}

QTabBar::tab {{
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 6px;
    color: {TEXT_MUTED};
    min-height: 36px;
    min-width: 140px;
    text-align: left;
}}

QTabBar::tab:selected {{
    background: {PRIMARY_COLOR};
    color: {TEXT_LIGHT};
}}

QPushButton {{
    background-color: {PRIMARY_COLOR};
    color: {TEXT_LIGHT};
    border: 1px solid {PRIMARY_COLOR};
    border-radius: 8px;
    padding: 6px 12px;
}}

QPushButton:hover {{
    background-color: {PRIMARY_COLOR};
}}

QPushButton:pressed {{
    background-color: {PRIMARY_COLOR};
}}

QPushButton:disabled {{
    background-color: {PRIMARY_COLOR};
    color: {PRIMARY_LIGHT};
    border-color: {PRIMARY_COLOR};
}}

QToolButton#jobOutputControlButton {{
    background-color: #F4F6F8;
    border: 1px solid #CBD5E1;
    border-radius: 4px;
    padding: 3px;
}}

QToolButton#jobOutputControlButton:hover {{
    background-color: #E5EAF0;
    border-color: #94A3B8;
}}

QTreeWidget#jobStatusTree {{
    selection-background-color: #C9D8E8;
    selection-color: {TEXT_DARK};
}}

QTreeWidget#jobStatusTree::item:selected {{
    background-color: #C9D8E8;
    color: {TEXT_DARK};
}}

QTreeWidget#jobStatusTree::item:selected:!active {{
    background-color: #D8E4F0;
    color: {TEXT_DARK};
}}

QPushButton#workflowSegmentButton {{
    background-color: #cfd2d4;
    color: {TEXT_DARK};
    border: 1px solid {NEUTRAL_BORDER};
    border-radius: 8px;
    padding: 7px 12px;
}}

QPushButton#workflowSegmentButton:hover {{
    background-color: #E8EEF6;
    border-color: {PRIMARY_LIGHT};
}}

QPushButton#workflowSegmentButton:checked {{
    background-color: {PRIMARY_COLOR};
    color: {TEXT_LIGHT};
    border-color: {PRIMARY_COLOR};
}}

QPushButton#workflowSegmentButton:checked:hover {{
    background-color: {PRIMARY_COLOR};
}}

QComboBox#workflowAdvancedCombo[hasSelection="false"] {{
    background-color: #F1F5F9;
    color: {TEXT_MUTED};
    border: 1px solid {NEUTRAL_BORDER};
    border-radius: 6px;
    padding: 5px 10px;
}}

QComboBox#workflowAdvancedCombo[hasSelection="false"]:hover {{
    background-color: #E8EEF6;
    border-color: {PRIMARY_LIGHT};
}}

QComboBox#workflowAdvancedCombo[hasSelection="true"] {{
    background-color: {PRIMARY_COLOR};
    color: {TEXT_LIGHT};
    border: 1px solid {PRIMARY_COLOR};
    border-radius: 6px;
    padding: 5px 10px;
}}

QComboBox#workflowAdvancedCombo[hasSelection="true"]:hover {{
    background-color: {PRIMARY_COLOR};
}}

QComboBox#workflowAdvancedCombo QAbstractItemView {{
    background-color: #FFFFFF;
    color: {TEXT_DARK};
    border: 1px solid {NEUTRAL_BORDER};
    selection-background-color: {PRIMARY_COLOR};
    selection-color: {TEXT_LIGHT};
    outline: 0;
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid #64748B;
    border-radius: 4px;
    background: #FFFFFF;
}}

QCheckBox::indicator:hover {{
    border-color: {PRIMARY_COLOR};
    background: {PRIMARY_LIGHT};
}}

QCheckBox::indicator:checked {{
    border-color: {PRIMARY_COLOR};
    background: {PRIMARY_COLOR};
}}

QCheckBox::indicator:disabled {{
    border-color: #CBD5E1;
    background: #F1F5F9;
}}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    font = QFont("Inter", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#F4F5F7"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#0F172A"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#E2E8F0"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#0F172A"))
    palette.setColor(QPalette.ColorRole.Button, QColor(PRIMARY_COLOR))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1F2937"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(APP_QSS)


class RedesignMainUI(MainUI):
    def _refresh_pseudo_selection_view(self) -> None:
        """Fallback wrapper to ensure the method exists on this subclass.

        Prefer the implementation from the superclass if available; otherwise
        perform a minimal safe update to the pseudopotential widgets so callers
        won't raise AttributeError.
        """
        # Try superclass implementation first
        try:
            super_impl = getattr(super(RedesignMainUI, self), "_refresh_pseudo_selection_view", None)
            if callable(super_impl):
                return super_impl()
        except Exception:
            pass

        # Fallback: attempt to update known widget elements if present
        try:
            if hasattr(self, "pseudoFilesListWidget"):
                self.pseudoFilesListWidget.clear()
            if hasattr(self, "pseudoSummaryLabel"):
                self.pseudoSummaryLabel.setText("No pseudopotentials selected.")
            if hasattr(self, "pseudoClearButton"):
                self.pseudoClearButton.setEnabled(False)
        except Exception:
            # last resort: ignore to avoid crashing UI construction
            return
    def _build_ui(self) -> None:
        self.setWindowTitle("Quantum ESPRESSO Toolkits")
        self.setMinimumSize(900, 640)
        self.resize(1200, 820)

        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(18, 18, 18, 18)
        central_layout.setSpacing(14)

        header = QFrame(self)
        header.setObjectName("headerBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(8)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        header_title = QLabel("Quantum ESPRESSO Toolkits", header)
        header_title.setObjectName("headerTitle")
        header_subtitle = QLabel("Job setup · workflow automation", header)
        header_subtitle.setObjectName("headerSubtitle")
        title_stack.addWidget(header_title)
        title_stack.addWidget(header_subtitle)

        header_layout.addLayout(title_stack)
        header_layout.setAlignment(
            title_stack, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        header_layout.addStretch()
        header_logo = QLabel(header)
        header_logo.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        logo_movie = QMovie("logo.gif")
        if logo_movie.isValid():
            frame_rect = logo_movie.frameRect()
            target_height = 90
            target_width = target_height
            if frame_rect.height() > 0:
                target_width = max(
                    1, int(frame_rect.width() * (target_height / frame_rect.height()))
                )
            logo_movie.setScaledSize(QSize(target_width, target_height))
            header_logo.setMovie(logo_movie)
            logo_movie.start()
            self._header_logo_movie = logo_movie
        header_layout.addWidget(header_logo)
        self._header_bar = header

        self.tabWidget = NoWheelTabWidget(central_widget)
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabWidget.setDocumentMode(True)
        central_layout.addWidget(self.tabWidget)
        self.setCentralWidget(central_widget)

        self._build_qe_input_generator_tab()
        self._build_job_creator_tab()
        self._build_workflow_tab()
        self._build_job_status_tab()
        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        menubar = QMenuBar(self)

        file_menu = menubar.addMenu("File")
        self.actionQuit = QAction("Quit", self)
        file_menu.addAction(self.actionQuit)

        settings_menu = menubar.addMenu("Settings")
        self.actionSystemConfig = QAction("System configuration...", self)
        self.actionCommandPreferences = QAction("Command preferences...", self)
        settings_menu.addAction(self.actionSystemConfig)
        settings_menu.addAction(self.actionCommandPreferences)

        help_menu = menubar.addMenu("help")
        self.actioninfo_2 = QAction("info", self)
        help_menu.addAction(self.actioninfo_2)

        top_widget = QWidget(self)
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        header_wrapper = QWidget(top_widget)
        header_wrapper_layout = QVBoxLayout(header_wrapper)
        header_wrapper_layout.setContentsMargins(18, 18, 18, 8)
        header_wrapper_layout.setSpacing(0)
        header_wrapper_layout.addWidget(self._header_bar)

        top_layout.addWidget(header_wrapper)
        top_layout.addWidget(menubar)
        self.setMenuWidget(top_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)
    ui = RedesignMainUI()
    ui.show()
    app.exec()

    
        
