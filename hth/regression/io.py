"""Run-directory and provenance helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def create_run_directory(root: Path, detector: str, run_id: str | None = None) -> tuple[str, Path]:
    rid = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    run = root / detector / rid
    for sub in ("raw", "reports", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=False)
    return rid, run


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cpu_model() -> str:
    model = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return model or "unknown"


def _memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def environment_info(repo_root: Path) -> dict[str, Any]:
    github_actions = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    memory_bytes = _memory_bytes()
    return {
        "execution_target": "GitHub Hosted" if github_actions else "Local",
        "runner_name": os.environ.get("RUNNER_NAME") or socket.gethostname(),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT") or ("github-actions" if github_actions else "local"),
        "runner_os": os.environ.get("RUNNER_OS") or platform.system(),
        "runner_arch": os.environ.get("RUNNER_ARCH") or platform.machine(),
        "machine_name": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / (1024 ** 3), 2) if memory_bytes else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pipeline_commit": git_commit(repo_root),
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_runner_labels": [x for x in os.environ.get("HTH_RUNNER_LABELS", "").split(",") if x],
    }
