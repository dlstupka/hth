from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


def _line(label: str, value) -> None:
    print(f"PageNet preflight {label:<23}: {value}", flush=True)


def run() -> None:
    _line("python", sys.executable)
    _line("python version", sys.version.split()[0])
    for env in (
        "HTH_LEARNED_PAGE_MASK_PROTOTXT",
        "HTH_LEARNED_PAGE_MASK_WEIGHTS",
        "HTH_LEARNED_PAGE_MASK_PROVENANCE",
    ):
        path = Path(os.environ.get(env, ""))
        _line(env, path if str(path) else "<unset>")
        if not path.is_file():
            raise RuntimeError(f"PageNet lifecycle asset missing: {env}={path}")

    provenance = json.loads(Path(os.environ["HTH_LEARNED_PAGE_MASK_PROVENANCE"]).read_text(encoding="utf-8"))
    _line("model id", provenance.get("model_id"))
    _line("weights sha256", str(provenance.get("weights_sha256") or "")[:12])

    from hth.geometry.registry import detector_names
    if "pagenet_page_mask" not in detector_names():
        raise RuntimeError("pagenet_page_mask is not registered in the worker's HTH registry")
    _line("registry", "registered")

    from hth.regression.strategies.cartesian import generate
    config = json.loads(Path("hth-pipeline/config/detectors/pagenet_page_mask.json").read_text(encoding="utf-8"))
    _line("parameter sets", len(generate(config)))
    _line("status", "ok")


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"PageNet preflight FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
