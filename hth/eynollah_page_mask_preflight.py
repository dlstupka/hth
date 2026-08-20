from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _line(label: str, value) -> None:
    print(f"Eynollah preflight {label:<22}: {value}", flush=True)


def run(*, load_model: bool = False) -> None:
    _line("python", sys.executable)
    _line("python version", sys.version.split()[0])
    _line("cwd", Path.cwd())

    model_raw = os.environ.get("HTH_EYNOLLAH_PAGE_MODEL_DIR", "")
    provenance_raw = os.environ.get("HTH_EYNOLLAH_PAGE_PROVENANCE", "")
    _line("model env", model_raw or "<unset>")
    _line("provenance env", provenance_raw or "<unset>")
    if not model_raw or not provenance_raw:
        raise RuntimeError("Eynollah lifecycle environment is not loaded")

    model = Path(model_raw)
    provenance = Path(provenance_raw)
    required = (
        model / "saved_model.pb",
        model / "variables" / "variables.index",
        model / "variables" / "variables.data-00000-of-00001",
    )
    if not model.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError(f"Prepared Eynollah SavedModel is incomplete: {model}")
    if not provenance.is_file():
        raise RuntimeError(f"Prepared Eynollah provenance is not visible: {provenance}")

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    _line("model id", payload.get("model_id"))
    _line("saved_model sha256", str(payload.get("files", {}).get("saved_model.pb", {}).get("sha256") or "")[:12])

    import tensorflow as tf
    _line("tensorflow", tf.__version__)

    from hth.geometry.registry import detector_names
    if "eynollah_page_mask" not in detector_names():
        raise RuntimeError("eynollah_page_mask is not registered in the worker's HTH registry")
    _line("registry", "registered")

    from hth.regression.strategies.cartesian import generate
    config = json.loads(Path("hth-pipeline/config/detectors/eynollah_page_mask.json").read_text(encoding="utf-8"))
    _line("parameter sets", len(generate(config)))

    if load_model:
        loaded = tf.saved_model.load(str(model))
        signatures = dict(getattr(loaded, "signatures", {}) or {})
        if not signatures:
            raise RuntimeError("Eynollah SavedModel exposes no callable signatures")
        _line("model load", "ok")
    _line("status", "ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(load_model=args.load_model)
    except Exception as exc:
        print(f"Eynollah preflight FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
