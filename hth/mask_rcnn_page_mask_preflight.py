from __future__ import annotations
import json, os
from pathlib import Path
from hth.geometry.registry import detector_names


def main():
    if "mask_rcnn_page_mask" not in detector_names():
        raise RuntimeError("mask_rcnn_page_mask is not registered in the worker's HTH registry")
    import torch, torchvision, detectron2
    from detectron2.engine import DefaultPredictor
    config=json.loads(Path("hth-pipeline/config/detectors/mask_rcnn_page_mask.json").read_text(encoding="utf-8"))
    for env in ("HTH_MASK_RCNN_PAGE_MODEL","HTH_MASK_RCNN_PAGE_CONFIG","HTH_MASK_RCNN_PAGE_PROVENANCE"):
        path=Path(os.environ.get(env,""))
        if not path.is_file(): raise RuntimeError(f"Mask R-CNN preflight missing {env}: {path}")
    provenance=json.loads(Path(os.environ["HTH_MASK_RCNN_PAGE_PROVENANCE"]).read_text(encoding="utf-8"))
    active_variant=os.environ.get("HTH_ACTIVE_MODEL_VARIANT", "")
    if active_variant and provenance.get("model_variant") not in {None, active_variant}:
        raise RuntimeError(f"Mask R-CNN preflight model variant mismatch: env={active_variant} provenance={provenance.get('model_variant')}")
    print(f"Mask R-CNN preflight torch       : {torch.__version__}")
    print(f"Mask R-CNN preflight torchvision : {torchvision.__version__}")
    print(f"Mask R-CNN preflight model       : {provenance.get('model_id', 'unknown')}")
    print(f"Mask R-CNN preflight variant     : {active_variant or provenance.get('model_variant', 'legacy-default')}")
    print(f"Mask R-CNN preflight parameters  : {len(config['parameters'])}")
    print("Mask R-CNN preflight registry    : registered")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
