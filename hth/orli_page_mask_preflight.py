from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


def _line(label: str, value) -> None:
    print(f"Orli preflight {label:<22}: {value}", flush=True)


def run(*, load_model: bool = False) -> None:
    _line("python", sys.executable)
    _line("python version", sys.version.split()[0])
    _line("cwd", Path.cwd())

    model_raw = os.environ.get("HTH_ORLI_PAGE_MODEL", "")
    provenance_raw = os.environ.get("HTH_ORLI_PAGE_PROVENANCE", "")
    _line("model env", model_raw or "<unset>")
    _line("provenance env", provenance_raw or "<unset>")

    if not model_raw:
        raise RuntimeError("HTH_ORLI_PAGE_MODEL is not set in the regression worker environment")
    if not provenance_raw:
        raise RuntimeError("HTH_ORLI_PAGE_PROVENANCE is not set in the regression worker environment")

    model = Path(model_raw)
    provenance = Path(provenance_raw)
    _line("model exists", model.is_file())
    _line("provenance exists", provenance.is_file())
    if not model.is_file():
        raise RuntimeError(f"Prepared Orli model is not visible to the worker: {model}")
    if not provenance.is_file():
        raise RuntimeError(f"Prepared Orli provenance is not visible to the worker: {provenance}")

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    _line("model id", payload.get("model_id"))
    _line("model sha256", str(payload.get("model_sha256") or "")[:12])

    version = importlib.metadata.version("orli")
    _line("kraken version", version)

    from orli.pred import segment
    _line("plugin API import", "ok")

    from hth.geometry.registry import detector_names
    registered = "orli_page_mask" in detector_names()
    _line("registry", "registered" if registered else "MISSING")
    if not registered:
        raise RuntimeError("orli_page_mask is not registered in the worker's HTH registry")

    from hth.regression.strategies.cartesian import generate
    config = json.loads(Path("hth-pipeline/config/detectors/orli_page_mask.json").read_text(encoding="utf-8"))
    count = len(generate(config))
    _line("parameter sets", count)

    if load_model:
        _line("model load", "starting")
        _line("model load", "deferred to orli.pred.segment")

    _line("status", "ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(load_model=args.load_model)
    except Exception as exc:
        print(
            f"Orli preflight FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
