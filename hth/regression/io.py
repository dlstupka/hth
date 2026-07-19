"""Run-directory and provenance helpers."""
from __future__ import annotations
from datetime import datetime, timezone
import json, platform, subprocess
from pathlib import Path
from typing import Any
import cv2

def utc_now() -> str: return datetime.now(timezone.utc).isoformat()
def git_commit(path: Path) -> str | None:
    try: return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def create_run_directory(root: Path, detector: str, run_id: str | None=None) -> tuple[str,Path]:
    rid=run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    run=root/detector/rid
    for sub in ("raw","reports","logs"): (run/sub).mkdir(parents=True,exist_ok=False)
    return rid,run

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def environment_info(repo_root: Path) -> dict[str,Any]:
    return {"python_version":platform.python_version(),"platform":platform.platform(),"opencv_version":cv2.__version__,"pipeline_commit":git_commit(repo_root)}
