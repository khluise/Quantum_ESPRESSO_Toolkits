#!/usr/bin/env python3
"""Simple job daemon to execute jobs from job_queue.json.

Features:
- Uses `filelock.FileLock` for robust cross-process locking
- Resumes multi-command jobs where they left off via metadata['last_completed_index']
- Writes per-job log files in the job's working directory
- Configurable polling interval
"""
from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from filelock import FileLock, Timeout
except Exception:  # pragma: no cover
    print("Missing dependency: filelock. Install with pip install filelock", file=sys.stderr)
    raise

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QUEUE_FILE = PROJECT_ROOT / "job_queue.json"
QUEUE_UPDATE_INTERVAL_SECONDS = 1.0
STOP_CHECK_INTERVAL_SECONDS = 0.25
STOP_GRACE_SECONDS = 5.0


def _load_queue_unlocked(path: Path) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh) or []
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def read_queue(path: Path, lock: FileLock) -> List[Dict[str, Any]]:
    try:
        with lock.acquire(timeout=5):
            return _load_queue_unlocked(path)
    except Timeout:
        return []


def write_queue(path: Path, lock: FileLock, payload: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with lock.acquire(timeout=5):
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(str(tmp), str(path))


def record_job_id(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return str(metadata.get("job_id") or "").strip()


def find_queue_record_index(
    payload: List[Dict[str, Any]],
    record: Dict[str, Any],
    fallback_index: int,
) -> int:
    """Find a queue record by stable job_id, falling back to the original index."""
    job_id = record_job_id(record)
    if job_id:
        for idx, current in enumerate(payload):
            if record_job_id(current) == job_id:
                return idx
        if 0 <= fallback_index < len(payload) and not record_job_id(payload[fallback_index]):
            return fallback_index
        return -1
    if 0 <= fallback_index < len(payload):
        return fallback_index
    return -1


def merge_daemon_record(current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge daemon-owned fields without discarding GUI-side queue edits."""
    merged = dict(current)
    new_status = str(update.get("status", current.get("status", "")))
    current_status = str(current.get("status", ""))
    if not (current_status == "Stopping" and new_status not in {"Stopped", "Failed", "Completed"}):
        merged["status"] = update.get("status", current.get("status", "Queued"))
    if "progress" in update:
        merged["progress"] = update.get("progress", current.get("progress", 0))
    if "log" in update:
        merged["log"] = update.get("log", current.get("log", ""))

    current_metadata = current.get("metadata") if isinstance(current.get("metadata"), dict) else {}
    update_metadata = update.get("metadata") if isinstance(update.get("metadata"), dict) else {}
    metadata = dict(current_metadata)
    metadata.update(update_metadata)
    merged["metadata"] = metadata
    return merged


def write_queue_record(path: Path, lock: FileLock, index: int, record: Dict[str, Any]) -> bool:
    """Persist one record by stable job_id, preserving concurrent changes to other records."""
    tmp = path.with_suffix(".json.tmp")
    with lock.acquire(timeout=5):
        current = _load_queue_unlocked(path)
        target_index = find_queue_record_index(current, record, index)
        if target_index < 0:
            return False
        current[target_index] = merge_daemon_record(current[target_index], record)
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2)
        os.replace(str(tmp), str(path))
        return True


def sanitize_job_name(name: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip())
    return cleaned or "qe_job"


def ensure_job_id(record: Dict[str, Any], index: int, sanitized_name: str) -> str:
    metadata = record.setdefault("metadata", {})
    existing = str(metadata.get("job_id") or "").strip()
    if existing:
        job_id = sanitize_job_name(existing)
        metadata["job_id"] = job_id
        return job_id

    created_at = str(metadata.get("created_at") or "").strip()
    try:
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else datetime.now()
    except Exception:
        created_dt = datetime.now()
    metadata["created_at"] = created_dt.isoformat(timespec="microseconds")
    timestamp_slug = created_dt.strftime("%Y%m%d_%H%M%S_%f")
    job_id = sanitize_job_name(f"{sanitized_name}_{timestamp_slug}_{index + 1}")
    metadata["job_id"] = job_id
    return job_id


def resolve_log_path(
    payload: List[Dict[str, Any]],
    index: int,
    record: Dict[str, Any],
    working_dir: Path,
    sanitized_name: str,
) -> Path:
    metadata = record.setdefault("metadata", {})
    existing_log = str(metadata.get("log_file") or "").strip()
    if existing_log:
        return Path(existing_log).expanduser()

    job_id = ensure_job_id(record, index, sanitized_name)

    used_logs = {
        str((other.get("metadata") or {}).get("log_file") or "")
        for i, other in enumerate(payload)
        if i != index and isinstance(other, dict)
    }
    candidate = working_dir / f"{job_id}.log"
    suffix = 2
    while str(candidate) in used_logs:
        candidate = working_dir / f"{job_id}_{suffix}.log"
        suffix += 1
    metadata["log_file"] = str(candidate)
    return candidate


def current_queue_record(
    path: Path,
    lock: FileLock,
    record: Dict[str, Any],
    fallback_index: int,
) -> Dict[str, Any] | None:
    payload = read_queue(path, lock)
    index = find_queue_record_index(payload, record, fallback_index)
    if index < 0:
        return None
    return payload[index]


def stop_requested(path: Path, lock: FileLock, record: Dict[str, Any], fallback_index: int) -> tuple[bool, str]:
    current = current_queue_record(path, lock, record, fallback_index)
    if current is None:
        return True, "queue record disappeared"
    if str(current.get("status", "")) == "Stopping":
        return True, "user"
    return False, ""


def terminate_process(
    proc: subprocess.Popen[str],
    pgid: Any,
    *,
    log,
) -> None:
    try:
        if pgid is not None and sys.platform != "win32":
            os.killpg(int(pgid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        proc.wait(timeout=STOP_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        log(f"--- Process did not stop after {STOP_GRACE_SECONDS:g}s; killing it\n")
    except Exception:
        return

    try:
        if pgid is not None and sys.platform != "win32":
            os.killpg(int(pgid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=1)
    except Exception:
        pass


def mark_job_stopped(
    queue_path: Path,
    lock: FileLock,
    index: int,
    record: Dict[str, Any],
    *,
    log,
    reason: str,
    proc: subprocess.Popen[str] | None = None,
) -> None:
    metadata = record.setdefault("metadata", {})
    if proc is not None:
        terminate_process(proc, metadata.get("proc_pgid"), log=log)
    log(f"--- Job stopped by {reason}\n")
    record["status"] = "Stopped"
    metadata["proc_pid"] = None
    metadata["proc_pgid"] = None
    write_queue_record(queue_path, lock, index, record)


def run_job(payload: List[Dict[str, Any]], index: int, queue_path: Path, lock: FileLock) -> None:
    record = payload[index]
    config = record.get("config") or {}
    commands = list(record.get("commands") or [])
    metadata = record.setdefault("metadata", {})
    last_completed = int(metadata.get("last_completed_index", -1))
    total = len(commands)

    # Resolve working directory
    working_dir = Path(config.get("working_directory") or PROJECT_ROOT).expanduser()
    working_dir.mkdir(parents=True, exist_ok=True)

    job_name = config.get("name") or sanitize_job_name(config.get("name", ""))
    sanitized = sanitize_job_name(job_name)
    ensure_job_id(record, index, sanitized)
    log_path = resolve_log_path(payload, index, record, working_dir, sanitized)
    metadata["log_file"] = str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_queue_record(queue_path, lock, index, record)
    except Exception:
        pass

    # Open log file for append
    with log_path.open("a", encoding="utf-8") as logfh:
        def log(text: str) -> None:
            logfh.write(text)
            logfh.flush()

        # Execute remaining commands
        for idx in range(last_completed + 1, total):
            # Check if job should be stopped
            try:
                should_stop, reason = stop_requested(queue_path, lock, record, index)
                if should_stop:
                    mark_job_stopped(queue_path, lock, index, record, log=log, reason=reason or "user")
                    return
            except Exception:
                pass

            cmd = commands[idx]
            record["status"] = f"Running ({idx+1}/{total})"
            metadata["current_command_index"] = idx
            metadata["current_command"] = cmd
            try:
                write_queue_record(queue_path, lock, index, record)
            except Exception:
                pass

            log(f"--- Running command ({idx+1}/{total}): {cmd}\n")

            # Start subprocess and stream output line-by-line to the log file.
            # Use preexec_fn to create a new process group so we can kill the entire group later
            preexec = None
            if sys.platform != "win32":
                preexec = os.setsid  # Create new process group on Unix/Linux
            
            try:
                # Ensure pipeline exit status is propagated (e.g., pw.x | tee ...)
                bash_cmd = f"set -o pipefail; {cmd}"
                proc = subprocess.Popen(
                    ["/bin/bash", "-lc", bash_cmd],
                    cwd=str(working_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=preexec
                )
                # Store the process group ID so we can kill the whole group
                metadata["proc_pid"] = proc.pid
                metadata["proc_pgid"] = os.getpgid(proc.pid) if sys.platform != "win32" else None
                write_queue_record(queue_path, lock, index, record)
            except Exception as exc:
                log(f"Exception while starting command: {exc}\n")
                record["status"] = "Failed"
                metadata["proc_pid"] = None
                metadata["proc_pgid"] = None
                metadata["failed_command_index"] = idx
                metadata["failure_reason"] = str(exc)
                try:
                    write_queue_record(queue_path, lock, index, record)
                except Exception:
                    pass
                return

            # Read stdout as it arrives, append to log and periodically persist queue status
            fatal_error_found = False
            last_queue_update = 0.0
            last_stop_check = 0.0

            def handle_output_line(line: str) -> None:
                nonlocal fatal_error_found, last_queue_update
                if not line:
                    return
                log(line)
                if ("Error in routine" in line) or ("MPI_ABORT" in line):
                    fatal_error_found = True
                now = time.monotonic()
                if now - last_queue_update >= QUEUE_UPDATE_INTERVAL_SECONDS:
                    try:
                        record["progress"] = int((idx) / total * 100)
                    except Exception:
                        pass
                    try:
                        write_queue_record(queue_path, lock, index, record)
                    except Exception:
                        pass
                    last_queue_update = now

            try:
                if proc.stdout is None:
                    while proc.poll() is None:
                        now = time.monotonic()
                        if now - last_stop_check >= STOP_CHECK_INTERVAL_SECONDS:
                            should_stop, reason = stop_requested(queue_path, lock, record, index)
                            if should_stop:
                                mark_job_stopped(
                                    queue_path,
                                    lock,
                                    index,
                                    record,
                                    log=log,
                                    reason=reason or "user",
                                    proc=proc,
                                )
                                return
                            last_stop_check = now
                        time.sleep(STOP_CHECK_INTERVAL_SECONDS)
                else:
                    selector = selectors.DefaultSelector()
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    try:
                        while True:
                            now = time.monotonic()
                            if now - last_stop_check >= STOP_CHECK_INTERVAL_SECONDS:
                                should_stop, reason = stop_requested(queue_path, lock, record, index)
                                if should_stop:
                                    mark_job_stopped(
                                        queue_path,
                                        lock,
                                        index,
                                        record,
                                        log=log,
                                        reason=reason or "user",
                                        proc=proc,
                                    )
                                    return
                                last_stop_check = now

                            events = selector.select(timeout=STOP_CHECK_INTERVAL_SECONDS)
                            for key, _event in events:
                                line = key.fileobj.readline()
                                if line:
                                    handle_output_line(line)

                            if proc.poll() is not None:
                                for line in proc.stdout:
                                    handle_output_line(line)
                                break
                    finally:
                        selector.close()

                rc = proc.wait()
                metadata["proc_pid"] = None
                metadata["proc_pgid"] = None
                metadata.pop("current_command", None)
                try:
                    write_queue_record(queue_path, lock, index, record)
                except Exception:
                    pass
            except Exception as exc:
                log(f"Exception while streaming output: {exc}\n")
                try:
                    terminate_process(proc, metadata.get("proc_pgid"), log=log)
                except Exception:
                    pass
                record["status"] = "Failed"
                metadata["proc_pid"] = None
                metadata["proc_pgid"] = None
                metadata["failed_command_index"] = idx
                metadata["failure_reason"] = str(exc)
                try:
                    write_queue_record(queue_path, lock, index, record)
                except Exception:
                    pass
                return

            should_stop, reason = stop_requested(queue_path, lock, record, index)
            if should_stop:
                mark_job_stopped(
                    queue_path,
                    lock,
                    index,
                    record,
                    log=log,
                    reason=reason or "user",
                    proc=None,
                )
                return

            if rc != 0 or fatal_error_found:
                record["status"] = "Failed"
                metadata["failed_command_index"] = idx
                metadata["return_code"] = rc
                metadata["failure_reason"] = "fatal marker in output" if fatal_error_found else f"exit code {rc}"
                metadata["proc_pid"] = None
                metadata["proc_pgid"] = None
                try:
                    write_queue_record(queue_path, lock, index, record)
                except Exception:
                    pass
                return

            # Command succeeded: mark it completed
            metadata["last_completed_index"] = idx
            metadata["last_return_code"] = rc
            metadata["proc_pid"] = None
            metadata["proc_pgid"] = None
            metadata.pop("current_command", None)
            metadata.pop("failed_command_index", None)
            metadata.pop("failure_reason", None)
            metadata.pop("return_code", None)
            try:
                record["progress"] = int((idx + 1) / total * 100)
            except Exception:
                record["progress"] = 0
            try:
                write_queue_record(queue_path, lock, index, record)
            except Exception:
                pass

        # All commands finished successfully
        record["status"] = "Completed"
        record["progress"] = 100
        metadata["proc_pid"] = None
        metadata["proc_pgid"] = None
        metadata.pop("current_command", None)
        metadata.pop("current_command_index", None)
        metadata.pop("failed_command_index", None)
        metadata.pop("failure_reason", None)
        metadata.pop("return_code", None)
        try:
            write_queue_record(queue_path, lock, index, record)
        except Exception:
            pass


def find_next_job(payload: List[Dict[str, Any]]) -> int:
    for idx, record in enumerate(payload):
        status = str(record.get("status", "Queued"))
        if status == "Queued":
            return idx
    return -1


def has_active_job(payload: List[Dict[str, Any]]) -> bool:
    for record in payload:
        status = str(record.get("status", "")).strip()
        if status.startswith("Running") or status in {"Starting", "Stopping"}:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-file", default=str(DEFAULT_QUEUE_FILE))
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    args = parser.parse_args()

    queue_path = Path(args.queue_file)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(queue_path) + ".lock")

    print(f"Job daemon starting; watching {queue_path} (poll {args.poll_interval}s)")

    # Main loop
    while True:
        try:
            payload = []
            if queue_path.exists():
                payload = read_queue(queue_path, lock)

            # Basic normalization: ensure each entry is a dict
            payload = [p for p in (payload or []) if isinstance(p, dict)]

            if has_active_job(payload):
                time.sleep(max(0.1, float(args.poll_interval)))
                continue

            next_idx = find_next_job(payload)
            if next_idx >= 0:
                record = payload[next_idx]
                # Start or resume this job
                record_status = str(record.get("status", "Queued"))
                if record_status == "Queued":
                    # persist that we're starting
                    config = record.get("config") if isinstance(record.get("config"), dict) else {}
                    job_name = str(config.get("name") or "qe_job")
                    ensure_job_id(record, next_idx, sanitize_job_name(job_name))
                    record["status"] = "Starting"
                    try:
                        write_queue_record(queue_path, lock, next_idx, record)
                    except Exception:
                        pass
                    # run
                    run_job(payload, next_idx, queue_path, lock)
            time.sleep(max(0.1, float(args.poll_interval)))
        except KeyboardInterrupt:
            print("Shutting down daemon")
            break
        except Exception as exc:
            print(f"Daemon loop error: {exc}", file=sys.stderr)
            time.sleep(max(1.0, float(args.poll_interval)))
