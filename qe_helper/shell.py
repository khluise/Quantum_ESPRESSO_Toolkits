"""Shell execution helpers."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_command_on_terminal(
    command: str,
    *,
    cwd: Optional[Path] = None,
    show_output: bool = False,
    check: bool = True,
    stop_event: Optional[Callable[[], bool]] = None,
    on_start: Optional[Callable[[subprocess.Popen], None]] = None,
    interrupt_signal: int = signal.SIGINT,
) -> subprocess.CompletedProcess:
    """Run a shell command with predictable defaults."""

    working_dir = cwd if cwd is not None else PROJECT_ROOT
    if show_output:
        print(f"$ {command}", flush=True)
    preexec = os.setsid if os.name != "nt" else None
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        cwd=str(working_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        preexec_fn=preexec,
    )

    if on_start is not None:
        try:
            on_start(process)
        except Exception:
            pass

    selector = selectors.DefaultSelector()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ, (sys.stdout, stdout_chunks))
    if process.stderr is not None:
        selector.register(process.stderr, selectors.EVENT_READ, (sys.stderr, stderr_chunks))

    sent_interrupt = False

    def _send_interrupt() -> None:
        nonlocal sent_interrupt
        if sent_interrupt or process.poll() is not None:
            return
        try:
            if os.name != "nt" and preexec is not None:
                os.killpg(process.pid, interrupt_signal)
            else:
                process.send_signal(interrupt_signal)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        sent_interrupt = True

    try:
        while selector.get_map():
            events = selector.select(timeout=0.1)
            for key, _ in events:
                stream = key.fileobj
                target, chunks = key.data
                data = stream.readline()
                if data:
                    chunks.append(data)
                    if show_output:
                        target.write(data)
                        target.flush()
                else:
                    selector.unregister(stream)
            if stop_event is not None and stop_event() and process.poll() is None:
                _send_interrupt()
        process.wait()
    finally:
        selector.close()

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout_text,
            stderr=stderr_text,
        )

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )
