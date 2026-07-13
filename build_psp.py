#!/usr/bin/env python3
"""build_psp.py

Python version of the build_psp helper. Provides a small API you can import
into another Python project and a CLI that mirrors the existing bash script.

Functions:
- download_file(url, dest_path)
- extract_tar_gz(archive_path, target_dir)
- clone_or_update_repo(git_url, dest_dir)
- write_qe_path(pslib_dir, qe_path)
- write_make_all_ps(pslib_dir)
- run_make_all_ps(pslib_dir)

CLI flags: --qe-path, --dest, --run
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("build_psp")


SSSP_URLS = {
    "SSSP_1.3.0_PBE_precision.tar.gz": (
        "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_precision.tar.gz/content"
    ),
    "SSSP_1.3.0_PBE_efficiency.tar.gz": (
        "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_efficiency.tar.gz/content"
    ),
    "SSSP_1.3.0_PBEsol_precision.tar.gz": (
        "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBEsol_precision.tar.gz/content"
    ),
    "SSSP_1.3.0_PBEsol_efficiency.tar.gz": (
        "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBEsol_efficiency.tar.gz/content"
    ),
}

PSLIB_GIT = "https://github.com/dalcorso/pslibrary.git"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download URL to dest_path (overwrites if exists)."""
    LOGGER.info("Downloading %s -> %s", url, dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)


def extract_tar_gz(archive_path: Path, target_dir: Path) -> None:
    """Extract a .tar.gz into target_dir/NAME_without_ext/ (mirrors bash script)."""
    LOGGER.info("Extracting %s into %s", archive_path, target_dir)
    name = archive_path.name
    base = name[:-7] if name.endswith(".tar.gz") else name
    dest = target_dir / base
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=dest)


def clone_or_update_repo(git_url: str, dest_dir: Path) -> None:
    """Clone the git_url into dest_dir, or pull if already present."""
    if dest_dir.exists() and (dest_dir / ".git").exists():
        LOGGER.info("pslibrary exists at %s: pulling latest", dest_dir)
        subprocess.run(["git", "-C", str(dest_dir), "pull", "--ff-only"], check=False)
    else:
        LOGGER.info("Cloning %s -> %s", git_url, dest_dir)
        subprocess.run(["git", "clone", git_url, str(dest_dir)], check=True)


def write_qe_path(pslib_dir: Path, qe_path: str) -> None:
    """Write the QE_path script inside pslibrary to set PWDIR to qe_path/"""
    qe_path = qe_path.rstrip("/") + "/"
    p = pslib_dir / "QE_path"
    content = f"#!/bin/bash\nPWDIR='{qe_path}'\n"
    LOGGER.info("Writing QE_path in %s", p)
    p.write_text(content)
    p.chmod(0o755)


def write_make_all_ps(pslib_dir: Path) -> None:
    """Write the requested make_all_ps content into pslibrary/make_all_ps."""
    p = pslib_dir / "make_all_ps"
    LOGGER.info("Writing make_all_ps in %s", p)
    content = r'''#!/bin/bash
#
# Copyright (C) 2010-2013 Andrea Dal Corso 
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
#   This file makes only the pz, pbe, rel-pz and rel-pbe tests.
#   If you are interested in some other functional enter in the directory
#   with the functional name and type ../make_test
#
cd ./bp
. ../make_ps
cd ..
cd ./pbe
. ../make_ps
cd ..
cd ./pbesol
. ../make_ps
cd ..
cd ./pw91
. ../make_ps
cd ..
cd ./pz
. ../make_ps
cd ..
cd ./rel-bp
. ../make_ps
cd ..
cd ./rel-pbe
. ../make_ps
cd ..
cd ./rel-pbesol
. ../make_ps
cd ..
cd ./rel-pw91
. ../make_ps
cd ..
cd ./rel-pz
. ../make_ps
cd ..
cd ./rel-revpbe
. ../make_ps
cd ..
cd ./rel-wc
. ../make_ps
cd ..
cd ./revpbe
. ../make_ps
cd ..
cd ./wc
. ../make_ps
cd ..
'''
    p.write_text(content)
    p.chmod(0o755)


def run_make_all_ps(pslib_dir: Path) -> int:
    """Run the make_all_ps script in pslibrary (returns subprocess exit code)."""
    script = pslib_dir / "make_all_ps"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found")
    LOGGER.info("Running %s (this may take a long time)", script)
    return subprocess.run([str(script)], cwd=str(pslib_dir)).returncode


def prepare_all(qe_path: str, dest: Path, run_flag: bool) -> None:
    """High-level orchestration: download SSSP, clone pslibrary, patch files, optional run."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Download and extract each SSSP archive
    for fname, url in SSSP_URLS.items():
        archive_path = dest / fname
        if not archive_path.exists():
            download_file(url, archive_path)
        else:
            LOGGER.info("Archive already exists: %s", archive_path)
        extract_tar_gz(archive_path, dest)

    # Clone/update pslibrary
    pslib_dir = dest / "pslibrary"
    clone_or_update_repo(PSLIB_GIT, pslib_dir)

    write_qe_path(pslib_dir, qe_path)
    write_make_all_ps(pslib_dir)

    if run_flag:
        code = run_make_all_ps(pslib_dir)
        if code != 0:
            LOGGER.warning("make_all_ps exited with code %s", code)
        else:
            LOGGER.info("make_all_ps finished successfully")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SSSP and prepare pslibrary for QE")
    p.add_argument("--qe-path", required=True, help="Path to Quantum ESPRESSO install")
    p.add_argument("--dest", default="./Pseudopotentials", help="Destination directory (default ./Pseudopotentials)")
    p.add_argument("--run", action="store_true", help="Run patched make_all_ps after preparing pslibrary")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        prepare_all(args.qe_path, Path(args.dest), args.run)
    except Exception:
        LOGGER.exception("Error during prepare_all")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
