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

    # HJDataset directly models Page Frame, which normally becomes the largest
    # instance. If the frame is absent, use the convex envelope of substantial
    # learned layout instances rather than inventing geometry from the image.
    primary=max(qualifying,key=lambda r:(float(r["area"]),float(r["confidence"])))
    primary_fraction=float(primary["area"])/image_area
    selected=[primary]; mode="largest-instance"
    if primary_fraction < values["minimum_page_area_fraction"]:
        substantial=[r for r in qualifying if float(r["area"]) >= max(image_area*values["minimum_instance_area_fraction"], float(primary["area"])*0.08)]
        if substantial:
            pts=np.concatenate([np.asarray(r["polygon"],dtype=np.float32).reshape(-1,2) for r in substantial],axis=0)
            hull=cv2.convexHull(pts.reshape(-1,1,2)).reshape(-1,2); hull_area=abs(float(cv2.contourArea(hull)))
            if hull_area/image_area >= values["minimum_page_area_fraction"]:
                polygon=hull; confidence=min(float(r["confidence"]) for r in substantial); selected=substantial; mode="multi-instance-envelope"
            else:
                return Candidate(method=METHOD,bbox=None,corners=None,confidence=float(primary["confidence"]),score=primary_fraction,diagnostics={**diagnostics,"reason":"page-area-below-minimum","primary_area_fraction":primary_fraction},status="no_candidate")
        else:
            return Candidate(method=METHOD,bbox=None,corners=None,confidence=float(primary["confidence"]),score=primary_fraction,diagnostics={**diagnostics,"reason":"page-area-below-minimum","primary_area_fraction":primary_fraction},status="no_candidate")
    else:
        polygon=np.asarray(primary["polygon"],dtype=np.float32); confidence=float(primary["confidence"])

    rect=cv2.minAreaRect(polygon.reshape(-1,1,2)); corners=cv2.boxPoints(rect)
    corners=_pad(corners,width,height,values["page_padding_fraction"])
    x0=max(0,int(np.floor(corners[:,0].min()))); y0=max(0,int(np.floor(corners[:,1].min())))
    x1=min(width,int(np.ceil(corners[:,0].max()))+1); y1=min(height,int(np.ceil(corners[:,1].max()))+1)
    area_fraction=abs(float(cv2.contourArea(corners)))/image_area
    diagnostics.update({"selection":mode,"selected_instance_count":len(selected),"selected_class_ids":[int(r["class_id"]) for r in selected],"selected_area_fraction":area_fraction,"selected_confidence":confidence})
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
