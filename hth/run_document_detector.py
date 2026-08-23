#!/usr/bin/env python3
"""Run the resolved preferred detector calibration over a preprocessed collection."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# This entrypoint is intentionally invoked as a file by the reusable preprocess
# workflow.  Put the repository root on sys.path so package-qualified imports
# behave identically in preprocess-test and full production preprocess.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hth.detector_lifecycle import finalize_detector, prepare_detector


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True, help="Resolved preferred detector JSON")
    p.add_argument("--lifecycle-root", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    entry = json.loads(a.selection.read_text(encoding="utf-8"))
    detector = str(entry.get("detector") or "").strip()
    parameters = entry.get("parameters")
    if not detector or not isinstance(parameters, dict):
        raise SystemExit(f"Invalid resolved detector selection: {a.selection}")

    lifecycle = "doc_ufcn_page_mask" if detector == "amsre_doc_ufcn_fusion" else detector
    prepare_detector(lifecycle, results_root=a.lifecycle_root, policy="reuse")
    try:
        with tempfile.TemporaryDirectory() as td:
            params = Path(td) / "parameters.json"
            params.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("detect_geometry_candidates.py")),
                "--manifest", str(a.manifest),
                "--analysis", str(a.analysis),
                "--image-root", str(a.image_root),
                "--output", str(a.output),
                "--detector", detector,
                "--parameters-json", str(params),
                "--overwrite",
            ]
            subprocess.run(cmd, check=True)
        payload = json.loads(a.output.read_text(encoding="utf-8"))
        payload["document_detector"] = {
            "selection_policy": entry.get("selection_policy"),
            "rank": entry.get("rank"),
            "approval_level": entry.get("approval_level"),
            "detector": detector,
            "display_name": entry.get("display_name"),
            "golden_set_id": entry.get("golden_set_id"),
            "golden_set_sha256": entry.get("golden_set_sha256"),
            "parameter_set_id": entry.get("parameter_set_id"),
            "parameter_identity_sha256": entry.get("parameter_identity_sha256"),
            "parameters": parameters,
            "best_avg_iou": entry.get("best_avg_iou"),
            "minimum_iou": entry.get("minimum_iou"),
            "stddev_iou": entry.get("stddev_iou"),
            "failure_count": entry.get("failure_count"),
            "calibration_evidence": entry.get("calibration_evidence"),
            "calibration_id": entry.get("calibration_id"),
            "build_number": entry.get("build_number"),
        }
        a.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        finalize_detector(lifecycle, results_root=a.lifecycle_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
