from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _line(label: str, value) -> None:
    print(f"Doc-UFCN preflight {label:<20}: {value}", flush=True)


def run(*, load_model: bool = False) -> None:
    _line("python", sys.executable)
    _line("python version", sys.version.split()[0])
    _line("cwd", Path.cwd())

    model_raw = os.environ.get("HTH_DOC_UFCN_PAGE_MODEL", "")
    provenance_raw = os.environ.get("HTH_DOC_UFCN_PAGE_PROVENANCE", "")
    _line("model env", model_raw or "<unset>")
    _line("provenance env", provenance_raw or "<unset>")
    if not model_raw or not provenance_raw:
        raise RuntimeError("Doc-UFCN lifecycle environment is not loaded")
    model = Path(model_raw)
    provenance = Path(provenance_raw)
    _line("model exists", model.is_file())
    _line("provenance exists", provenance.is_file())
    if not model.is_file() or not provenance.is_file():
        raise RuntimeError("Prepared Doc-UFCN model/provenance is not visible to the worker")

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    _line("model id", payload.get("model_id"))
    _line("model sha256", str(payload.get("model_sha256") or "")[:12])
    _line("source version", payload.get("doc_ufcn_version"))

    import torch
    from doc_ufcn.main import DocUFCN
    from hth.doc_ufcn_compat import use_modern_torch_autocast
    use_modern_torch_autocast()
    _line("torch version", torch.__version__)
    _line("autocast API", "torch.amp.autocast(cuda)")
    _line("DocUFCN import", "ok")

    from hth.geometry.registry import detector_names
    if "doc_ufcn_page_mask" not in detector_names():
        raise RuntimeError("doc_ufcn_page_mask is not registered in the worker's HTH registry")
    _line("registry", "registered")

    from hth.regression.strategies.cartesian import generate
    config = json.loads(Path("hth-pipeline/config/detectors/doc_ufcn_page_mask.json").read_text(encoding="utf-8"))
    _line("parameter sets", len(generate(config)))

    if load_model:
        instance = DocUFCN(len(payload["classes"]), int(payload["input_size"]), "cpu")
        instance.load(model, payload["mean"], payload["std"], mode="eval")
        _line("model load", "ok")
    _line("status", "ok")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(load_model=args.load_model)
    except Exception as exc:
        print(f"Doc-UFCN preflight FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
