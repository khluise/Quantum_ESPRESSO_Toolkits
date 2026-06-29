"""System configuration helpers for Quantum ESPRESSO Helper."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "settings.json"


def resolve_executable_bin_dir(path_value: Optional[str], executable_name: str) -> tuple[Optional[str], Optional[str]]:
    if not path_value:
        return None, None
    raw = str(path_value).strip()
    if not raw:
        return None, None
    try:
        path = Path(raw).expanduser()
    except Exception:
        return None, None

    try:
        if path.exists() or path.is_symlink():
            if path.is_dir():
                executable_candidates = [path / executable_name, path / "bin" / executable_name]
            else:
                executable_candidates = [path]
            seen: set[Path] = set()
            for executable_candidate in executable_candidates:
                if executable_candidate in seen:
                    continue
                seen.add(executable_candidate)
                bin_dir, resolved_executable = _validate_executable_candidate(
                    executable_candidate,
                    executable_name,
                )
                if bin_dir:
                    return bin_dir, resolved_executable
    except Exception:
        return None, None

    return None, None


def resolve_qe_bin_dir(qe_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    return resolve_executable_bin_dir(qe_path, "pw.x")


def resolve_thermo_pw_bin_dir(thermo_pw_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    return resolve_executable_bin_dir(thermo_pw_path, "thermo_pw.x")


def _validate_executable_candidate(
    executable_candidate: Path,
    executable_name: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        if executable_candidate.name != executable_name:
            return None, None
        if not (executable_candidate.exists() or executable_candidate.is_symlink()):
            return None, None
        resolved_executable = executable_candidate.resolve(strict=True)
        if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
            return None, None
        return str(executable_candidate.parent), str(resolved_executable)
    except Exception:
        return None, None


def _normalize_executable_bin_dir(path_value: Optional[str], executable_name: str) -> Optional[str]:
    bin_dir, _resolved_executable = resolve_executable_bin_dir(path_value, executable_name)
    if bin_dir:
        return bin_dir
    if not path_value:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
    except Exception:
        return None

    try:
        if path.exists() or path.is_symlink():
            return None
    except Exception:
        return None

    raw = str(path)
    executable_suffix = f"/bin/{executable_name}"
    if raw.endswith(executable_suffix):
        return raw[: -len(f"/{executable_name}")]
    if raw.endswith("/bin"):
        return raw.rstrip("/")
    return None


def _normalize_qe_bin_dir(qe_path: Optional[str]) -> Optional[str]:
    return _normalize_executable_bin_dir(qe_path, "pw.x")


def _normalize_thermo_pw_bin_dir(thermo_pw_path: Optional[str]) -> Optional[str]:
    return _normalize_executable_bin_dir(thermo_pw_path, "thermo_pw.x")


@dataclass
class SystemConfig:
    """User-configurable system defaults for QE executions."""

    qe_path: Optional[str] = None
    thermo_pw_path: Optional[str] = None
    mpi_procs: int = 4
    # Backwards-compatible single-directory field. New code should prefer
    # `pseudo_directories` which may contain multiple paths.
    pseudo_directory: Optional[str] = None
    pseudo_directories: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> "SystemConfig":
        detected = shutil.which("pw.x")
        detected_thermo = shutil.which("thermo_pw.x")
        return cls(
            qe_path=_normalize_qe_bin_dir(detected),
            thermo_pw_path=_normalize_thermo_pw_bin_dir(detected_thermo),
            mpi_procs=4,
            pseudo_directory=None,
            pseudo_directories=[],
        )

    def to_dict(self) -> dict[str, Any]:
        # Persist both formats for compatibility: `pseudo_directories` is the
        # preferred modern representation (list), while `pseudo_directory` is
        # kept for older consumers.
        data: dict[str, Any] = {
            "qe_path": _normalize_qe_bin_dir(self.qe_path),
            "thermo_pw_path": _normalize_thermo_pw_bin_dir(self.thermo_pw_path),
            "mpi_procs": int(self.mpi_procs),
        }
        if self.pseudo_directories:
            data["pseudo_directories"] = list(self.pseudo_directories)
            # also write the first as legacy key
            data["pseudo_directory"] = self.pseudo_directories[0]
        else:
            data["pseudo_directory"] = self.pseudo_directory
            data["pseudo_directories"] = [self.pseudo_directory] if self.pseudo_directory else []
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemConfig":
        qe_path = _normalize_qe_bin_dir(data.get("qe_path") or None)
        thermo_pw_path = _normalize_thermo_pw_bin_dir(data.get("thermo_pw_path") or None)
        # Accept either the legacy single-directory key or the new list key.
        pseudo_directory = data.get("pseudo_directory") or None
        pseudo_directories = data.get("pseudo_directories") or []
        if isinstance(pseudo_directories, str):
            pseudo_directories = [pseudo_directories]
        # If no explicit list provided, but a legacy single value exists, use it.
        if not pseudo_directories and pseudo_directory:
            pseudo_directories = [pseudo_directory]
        try:
            mpi_procs = int(data.get("mpi_procs", 4))
        except Exception:  # noqa: BLE001
            mpi_procs = 4
        mpi_procs = max(1, mpi_procs)
        return cls(
            qe_path=qe_path,
            thermo_pw_path=thermo_pw_path,
            mpi_procs=mpi_procs,
            pseudo_directory=pseudo_directory,
            pseudo_directories=list(pseudo_directories) if pseudo_directories else [],
        )


def load_system_config() -> SystemConfig:
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return SystemConfig.from_dict(data)
        except Exception:  # noqa: BLE001
            pass
    return SystemConfig.default()


def save_system_config(config: SystemConfig) -> None:
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise IOError(f"Could not save system configuration: {exc}")

    except Exception as exc:  # noqa: BLE001
        raise IOError(f"Could not save system configuration: {exc}")
