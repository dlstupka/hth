from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _line(label: str, value) -> None:
    print(f"docExtractor preflight {label:<19}: {value}", flush=True)


def run(*, load_model: bool = False) -> None:
    _line("python", sys.executable)
    _line("python version", sys.version.split()[0])
    _line("cwd", Path.cwd())

    model_raw = os.environ.get("HTH_DOCEXTRACTOR_PAGE_MODEL", "")
    source_raw = os.environ.get("HTH_DOCEXTRACTOR_PAGE_SOURCE", "")
    provenance_raw = os.environ.get("HTH_DOCEXTRACTOR_PAGE_PROVENANCE", "")
    if not model_raw or not source_raw or not provenance_raw:
        raise RuntimeError("docExtractor lifecycle environment is not loaded")

    model = Path(model_raw)
    source = Path(source_raw)
    provenance = Path(provenance_raw)
    _line("model env", model)
    _line("source env", source)
    if not model.is_file() or not source.is_dir() or not provenance.is_file():
        raise RuntimeError("Prepared docExtractor model/source/provenance is not visible to the worker")

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    _line("model id", payload.get("model_id"))
    _line("model sha256", str(payload.get("model_sha256") or "")[:12])

    import torch
    import gdown  # noqa: F401
    import toolz  # noqa: F401
    _line("torch", torch.__version__)
    _line("gdown", "available")
    _line("toolz", "available")

    from hth.geometry.registry import detector_names
    if "docextractor_page_mask" not in detector_names():
        raise RuntimeError("docextractor_page_mask is not registered in the worker's HTH registry")
    _line("registry", "registered")

    from hth.regression.strategies.cartesian import generate
    config = json.loads(Path("hth-pipeline/config/detectors/docextractor_page_mask.json").read_text(encoding="utf-8"))
    _line("parameter sets", len(generate(config)))

    if load_model:
        from hth.geometry.detector_docextractor_page_mask import _load_model
        _load_model()
        _line("model load", "ok")
    _line("status", "ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(load_model=args.load_model)
    except Exception as exc:
        print(f"docExtractor preflight FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
