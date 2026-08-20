from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib, json, os, threading, time
import cv2
import numpy as np
from .model import Candidate

METHOD = "mask_rcnn_page_mask"
MODEL_ENV = "HTH_MASK_RCNN_PAGE_MODEL"
CONFIG_ENV = "HTH_MASK_RCNN_PAGE_CONFIG"
PROVENANCE_ENV = "HTH_MASK_RCNN_PAGE_PROVENANCE"
BASELINE_PARAMETERS = {
    "minimum_confidence": 0.50,
    "minimum_instance_area_fraction": 0.0005,
    "minimum_page_area_fraction": 0.20,
    "page_padding_fraction": 0.02,
}
_MODEL = None
_MODEL_KEY = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_EVIDENCE_CACHE: OrderedDict[str, tuple[dict, ...]] = OrderedDict()
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_CACHE_LIMIT = 16


def _parameters(parameters):
    values = dict(BASELINE_PARAMETERS)
    parameters = parameters or {}
    unknown = sorted(set(parameters) - set(values))
    if unknown:
        raise ValueError(f"Unknown Mask R-CNN Page-Mask parameters: {', '.join(unknown)}")
    values.update(parameters)
    return {key: float(value) for key, value in values.items()}


def _required_path(env_name: str) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {env_name}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} artifact does not exist: {path}")
    return path


def _load_model():
    global _MODEL, _MODEL_KEY
    model_path = _required_path(MODEL_ENV).resolve()
    config_path = _required_path(CONFIG_ENV).resolve()
    key = f"{config_path}|{model_path}"
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.DEVICE = "cpu"
        # Keep learned evidence parameter-invariant; HTH applies confidence later.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        predictor = DefaultPredictor(cfg)
        _MODEL = predictor
        _MODEL_KEY = key
        print(
            f"Mask R-CNN Page-Mask ready for inference: model={model_path.name} "
            "dataset=HJDataset backbone=R50-FPN device=cpu",
            flush=True,
        )
        return predictor


def _image_key(image_bgr: np.ndarray) -> str:
    arr = np.ascontiguousarray(image_bgr)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.shape).encode("ascii")); digest.update(str(arr.dtype).encode("ascii")); digest.update(memoryview(arr))
    return digest.hexdigest()


def _mask_polygon(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return contour.reshape(-1, 2).astype(np.float32)


def _infer_evidence(image_bgr: np.ndarray) -> tuple[dict, ...]:
    key = _image_key(image_bgr)
    with _EVIDENCE_CACHE_LOCK:
        cached = _EVIDENCE_CACHE.get(key)
        if cached is not None:
            _EVIDENCE_CACHE.move_to_end(key); return cached
    with _INFERENCE_LOCK:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_CACHE.get(key)
            if cached is not None:
                _EVIDENCE_CACHE.move_to_end(key); return cached
        outputs = _load_model()(image_bgr)
        instances = outputs["instances"].to("cpu")
        scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances), dtype=np.float32)
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.full(len(instances), -1)
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        records = []
        for i in range(len(instances)):
            polygon = _mask_polygon(masks[i]) if masks is not None else None
            if polygon is None and boxes is not None:
                x0, y0, x1, y1 = [float(v) for v in boxes[i]]
                polygon = np.asarray([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
            if polygon is None or len(polygon) < 3:
                continue
            area = abs(float(cv2.contourArea(polygon)))
            x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
            records.append({
                "confidence": float(scores[i]), "class_id": int(classes[i]), "area": area,
                "bounds": [float(x), float(y), float(x+w), float(y+h)],
                "polygon": polygon.astype(float).tolist(),
            })
        records.sort(key=lambda r: (-r["area"], -r["confidence"]))
        evidence = tuple(records)
        with _EVIDENCE_CACHE_LOCK:
            _EVIDENCE_CACHE[key] = evidence; _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT: _EVIDENCE_CACHE.popitem(last=False)
        return evidence


def precompute_golden_set_evidence(images, *, progress=None):
    keys=[]; total=len(images)
    for index, image_bgr in enumerate(images, 1):
        key=_image_key(image_bgr); started=time.perf_counter()
        if progress: progress("start", index, total, key, 0.0)
        _infer_evidence(image_bgr)
        if progress: progress("finish", index, total, key, time.perf_counter()-started)
        keys.append(key)
    return tuple(keys)


def export_precomputed_golden_set_evidence(images, output_dir, *, progress=None):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    keys=precompute_golden_set_evidence(images, progress=progress)
    with _EVIDENCE_CACHE_LOCK:
        records=[{"image_key": key, "instances": list(_EVIDENCE_CACHE[key])} for key in keys]
    payload={"schema_version":"0.1","detector":METHOD,"representation":"hjdataset-mask-rcnn-instances","page_count":len(records),"records":records}
    target=output_dir/"manifest.json"; tmp=target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8"); os.replace(tmp,target); return target


def load_precomputed_golden_set_evidence(output_dir, images):
    payload=json.loads((Path(output_dir)/"manifest.json").read_text(encoding="utf-8"))
    if payload.get("detector") != METHOD: raise ValueError("Shared Mask R-CNN evidence detector mismatch")
    records={str(r["image_key"]): tuple(r.get("instances") or []) for r in payload.get("records",[]) if isinstance(r,dict) and r.get("image_key")}
    expected=tuple(_image_key(image) for image in images); missing=[k for k in expected if k not in records]
    if missing: raise ValueError(f"Shared Mask R-CNN evidence is missing {len(missing)} Golden Set page(s)")
    with _EVIDENCE_CACHE_LOCK:
        for key in expected:
            _EVIDENCE_CACHE[key]=records[key]; _EVIDENCE_CACHE.move_to_end(key)
    return expected


def _pad(corners, width, height, fraction):
    corners=np.asarray(corners,dtype=np.float32); center=corners.mean(axis=0)
    padded=center+(corners-center)*(1.0+2.0*float(fraction))
    padded[:,0]=np.clip(padded[:,0],0,max(0,width-1)); padded[:,1]=np.clip(padded[:,1],0,max(0,height-1))
    return padded


def _bounds(record):
    x0, y0, x1, y1 = [float(v) for v in record["bounds"]]
    return x0, y0, x1, y1


def _overlap_fraction(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1.0, min(a1 - a0, b1 - b0))
    return overlap / denom


def _coherent_layout_envelope(qualifying, primary, *, width, height, image_area, minimum_instance_area_fraction):
    """Return a conservative cluster of learned instances that extends the primary frame.

    HJDataset often emits one strong page/frame instance for one leaf of a spread plus
    adjacent substantial layout instances on the other leaf.  Build a connected spatial
    cluster around the largest learned instance, but reject detached fragments and sparse
    hulls so a stray prediction cannot explode the page envelope.
    """
    minimum_area = max(
        image_area * float(minimum_instance_area_fraction),
        float(primary["area"]) * 0.08,
    )
    substantial = [r for r in qualifying if float(r["area"]) >= minimum_area]
    if primary not in substantial:
        substantial.append(primary)

    selected = [primary]
    remaining = [r for r in substantial if r is not primary]
    gap_x = max(8.0, width * 0.04)
    gap_y = max(8.0, height * 0.04)

    changed = True
    while changed and remaining:
        changed = False
        sx0 = min(_bounds(r)[0] for r in selected); sy0 = min(_bounds(r)[1] for r in selected)
        sx1 = max(_bounds(r)[2] for r in selected); sy1 = max(_bounds(r)[3] for r in selected)
        for record in list(remaining):
            x0, y0, x1, y1 = _bounds(record)
            near = not (x1 < sx0-gap_x or x0 > sx1+gap_x or y1 < sy0-gap_y or y0 > sy1+gap_y)
            vertical_support = _overlap_fraction(y0, y1, sy0, sy1) >= 0.35
            horizontal_support = _overlap_fraction(x0, x1, sx0, sx1) >= 0.35
            if near and (vertical_support or horizontal_support):
                selected.append(record); remaining.remove(record); changed = True

    if len(selected) < 2:
        return None

    points = np.concatenate([
        np.asarray(r["polygon"], dtype=np.float32).reshape(-1, 2)
        for r in selected
    ], axis=0)
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
    hull_area = abs(float(cv2.contourArea(hull)))
    primary_area = max(1.0, float(primary["area"]))
    covered_area = sum(float(r["area"]) for r in selected)
    compactness = min(1.0, covered_area / max(1.0, hull_area))

    # Require the learned cluster to add meaningful extent and remain reasonably dense.
    if hull_area < primary_area * 1.08 or compactness < 0.30:
        return None
    return selected, hull, hull_area, compactness


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values=_parameters(parameters); height,width=image_bgr.shape[:2]; image_area=float(max(1,width*height))
    evidence=_infer_evidence(image_bgr)
    qualifying=[]
    for record in evidence:
        if float(record["confidence"]) < values["minimum_confidence"]: continue
        if float(record["area"])/image_area < values["minimum_instance_area_fraction"]: continue
        qualifying.append(record)
    diagnostics={"instance_count":len(evidence),"qualifying_instance_count":len(qualifying),"parameters":values}
    if not qualifying:
        return Candidate(method=METHOD,bbox=None,corners=None,confidence=0.0,score=0.0,diagnostics={**diagnostics,"reason":"no-qualifying-instance"},status="no_candidate")

    # Anchor on the largest learned frame/instance, then allow a spatially coherent
    # cluster of substantial neighboring learned instances to enlarge it. This is
    # important for historical two-leaf spreads where one Page Frame prediction can
    # cover only one leaf even though the surrounding Mask R-CNN evidence describes
    # the complete physical document.
    primary=max(qualifying,key=lambda r:(float(r["area"]),float(r["confidence"])))
    primary_fraction=float(primary["area"])/image_area
    selected=[primary]; mode="largest-instance"
    polygon=np.asarray(primary["polygon"],dtype=np.float32); confidence=float(primary["confidence"])

    envelope = _coherent_layout_envelope(
        qualifying, primary, width=width, height=height, image_area=image_area,
        minimum_instance_area_fraction=values["minimum_instance_area_fraction"],
    )
    envelope_compactness = None
    if envelope is not None:
        envelope_selected, hull, hull_area, envelope_compactness = envelope
        if hull_area / image_area >= values["minimum_page_area_fraction"]:
            selected=envelope_selected; polygon=hull
            confidence=min(float(r["confidence"]) for r in selected)
            mode="coherent-multi-instance-envelope"

    selected_fraction=abs(float(cv2.contourArea(polygon)))/image_area
    if selected_fraction < values["minimum_page_area_fraction"]:
        return Candidate(method=METHOD,bbox=None,corners=None,confidence=confidence,score=selected_fraction,diagnostics={**diagnostics,"reason":"page-area-below-minimum","primary_area_fraction":primary_fraction,"selected_area_fraction":selected_fraction},status="no_candidate")

    rect=cv2.minAreaRect(polygon.reshape(-1,1,2)); corners=cv2.boxPoints(rect)
    corners=_pad(corners,width,height,values["page_padding_fraction"])
    x0=max(0,int(np.floor(corners[:,0].min()))); y0=max(0,int(np.floor(corners[:,1].min())))
    x1=min(width,int(np.ceil(corners[:,0].max()))+1); y1=min(height,int(np.ceil(corners[:,1].max()))+1)
    area_fraction=abs(float(cv2.contourArea(corners)))/image_area
    diagnostics.update({"selection":mode,"selected_instance_count":len(selected),"selected_class_ids":[int(r["class_id"]) for r in selected],"selected_area_fraction":area_fraction,"selected_confidence":confidence})
    if envelope_compactness is not None:
        diagnostics["envelope_compactness"] = envelope_compactness
    return Candidate(method=METHOD,bbox=[x0,y0,x1,y1],corners=corners.astype(float).tolist(),confidence=confidence,score=area_fraction,diagnostics=diagnostics)

def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del mask, verbose
    values = _parameters(parameters)
    overlay = image_bgr.copy()
    height, width = image_bgr.shape[:2]
    image_area = float(max(1, width * height))

    for record in _infer_evidence(image_bgr):
        polygon = np.rint(np.asarray(record.get("polygon") or [], dtype=np.float32)).astype(np.int32)
        if len(polygon) < 3:
            continue
        confidence = float(record.get("confidence", 0.0))
        area_fraction = float(record.get("area", 0.0)) / image_area
        qualifies = (
            confidence >= values["minimum_confidence"]
            and area_fraction >= values["minimum_instance_area_fraction"]
        )
        cv2.polylines(
            overlay,
            [polygon.reshape(-1, 1, 2)],
            True,
            (0, 255, 0) if qualifies else (0, 255, 255),
            2 if qualifies else 1,
        )

    if candidate_corners is not None:
        corners = np.rint(np.asarray(candidate_corners, dtype=np.float32)).astype(np.int32)
        if len(corners) >= 3:
            cv2.polylines(overlay, [corners.reshape(-1, 1, 2)], True, (0, 0, 255), 3)

    return {"mask-rcnn-page-instances.png": overlay}


__all__ = [
    "BASELINE_PARAMETERS",
    "METHOD",
    "MODEL_ENV",
    "CONFIG_ENV",
    "PROVENANCE_ENV",
    "debug_images",
    "detect",
    "precompute_golden_set_evidence",
    "export_precomputed_golden_set_evidence",
    "load_precomputed_golden_set_evidence",
]
