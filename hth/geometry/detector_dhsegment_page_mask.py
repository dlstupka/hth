from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .model import Candidate
from hth.thread_safe_stderr import suppress_native_stderr

METHOD = "dhsegment_page_mask"
MODEL_DIR_ENV = "HTH_DHSEGMENT_PAGE_MODEL_DIR"
PROVENANCE_ENV = "HTH_DHSEGMENT_PAGE_PROVENANCE"

BASELINE_PARAMETERS = {
    "probability_threshold": -1.0,
    "minimum_page_area_fraction": 0.20,
    "close_kernel_fraction": 0.005,
    "open_kernel_fraction": 0.0,
    "contour_offset_fraction": 0.0,
    "fill_holes": 1,
}

_MODEL_CACHE: dict[str, _SavedModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_EVIDENCE_CACHE: OrderedDict[str, tuple[np.ndarray, tuple[int, int]]] = OrderedDict()
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_CACHE_LIMIT = 16


def _parameters(parameters):
    values = dict(BASELINE_PARAMETERS)
    parameters = parameters or {}
    unknown = sorted(set(parameters) - set(values))
    if unknown:
        raise ValueError(f"Unknown dhSegment Page-Mask parameters: {', '.join(unknown)}")
    values.update(parameters)
    for key in (
        "probability_threshold",
        "minimum_page_area_fraction",
        "close_kernel_fraction",
        "open_kernel_fraction",
        "contour_offset_fraction",
    ):
        values[key] = float(values[key])
    values["fill_holes"] = int(values["fill_holes"])
    if values["fill_holes"] not in (0, 1):
        raise ValueError("fill_holes must be 0 or 1")
    return values


def _asset_dir():
    raw = os.environ.get(MODEL_DIR_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {MODEL_DIR_ENV}")
    path = Path(raw)
    if not (path / "saved_model.pb").is_file():
        raise RuntimeError(f"{METHOD} SavedModel does not exist: {path}")
    return path


def _provenance():
    raw = os.environ.get(PROVENANCE_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{METHOD} lifecycle did not set {PROVENANCE_ENV}")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{METHOD} provenance does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def _suppress_native_stderr_during_tensorflow_startup():
    """Suppress TensorFlow native startup chatter via HTH's global fd2 lock.

    All native stderr redirection in detector evaluation shares one process-wide
    lock; normal parameter evaluation remains parallel after learned evidence
    preparation.
    """
    with suppress_native_stderr():
        yield


def _configure_tensorflow_runtime_environment():
    """Configure the legacy dhSegment runtime before importing TensorFlow.

    The detector runs in a fresh regression/optimizer Python process.  Set the
    import-time C++/absl logging controls here so they are guaranteed to exist
    before TensorFlow initializes; lifecycle PREPARE runs in a separate process
    and cannot export environment variables back into this one.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class _SavedModel:
    """Legacy graph/session adapter for the released dhSegment v0.2 model.

    The release is a TensorFlow 1 SavedModel.  Loading it through TensorFlow 2's
    object reconstruction path fails on stale object-graph references such as
    ``softmax:0``.  Import the MetaGraph directly into an isolated TF1-compatible
    graph/session and execute tensors named by the SavedModel SignatureDef.
    """

    def __init__(self, model_dir: Path):
        _configure_tensorflow_runtime_environment()
        try:
            with _suppress_native_stderr_during_tensorflow_startup():
                import tensorflow as tf

                self.tf = tf
                self.graph = tf.Graph()
                config = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1,
                    allow_soft_placement=True,
                    device_count={"GPU": 0},
                )
                self.session = tf.compat.v1.Session(graph=self.graph, config=config)

                tf_logger = tf.get_logger()
                previous_level = tf_logger.level
                tf_logger.setLevel(logging.ERROR)
                try:
                    with self.graph.as_default():
                        meta_graph = tf.compat.v1.saved_model.loader.load(
                            self.session,
                            [tf.compat.v1.saved_model.tag_constants.SERVING],
                            str(model_dir),
                        )
                finally:
                    tf_logger.setLevel(previous_level)
        except ImportError as exc:
            raise RuntimeError(
                "dhsegment_page_mask requires the TensorFlow runtime installed "
                "by the regression/optimizer workflow"
            ) from exc

        signatures = dict(meta_graph.signature_def)
        if not signatures:
            raise RuntimeError("dhSegment SavedModel has no serving signatures")

        default_key = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.signature_name = default_key if default_key in signatures else sorted(signatures)[0]
        signature = signatures[self.signature_name]
        self.input_key, input_info = self._select_input(dict(signature.inputs))
        self.probability_key, probability_info = self._select_probability_output(dict(signature.outputs))
        shape_selected = self._select_shape_output(dict(signature.outputs))
        self.shape_key, shape_info = shape_selected if shape_selected is not None else (None, None)

        self.input_tensor = self._resolve_tensor(input_info.name, role="input")
        self.probability_tensor = self._resolve_tensor(
            probability_info.name,
            role="probability output",
            fallback_tokens=("prob", "softmax", "prediction"),
            allow_stale_signature=True,
        )
        self.shape_tensor = None
        if shape_info is not None:
            self.shape_tensor = self._resolve_tensor(
                shape_info.name,
                role="original-shape output",
                fallback_tokens=("original_shape", "shape"),
                allow_stale_signature=True,
            )

        print(
            "dhSegment legacy execution adapter: "
            f"signature={self.signature_name} input={self.input_tensor.name} "
            f"input_shape={self.input_tensor.shape} feed_contract=scalar-filename "
            f"probability={self.probability_tensor.name}"
        )

    @staticmethod
    def _select_input(inputs):
        if not inputs:
            raise RuntimeError("dhSegment SavedModel serving signature has no inputs")
        key = next((name for name in inputs if "file" in name.lower()), None)
        if key is None:
            key = sorted(inputs)[0]
        return key, inputs[key]

    @staticmethod
    def _select_probability_output(outputs):
        if not outputs:
            raise RuntimeError("dhSegment SavedModel serving signature has no outputs")
        key = next(
            (name for name in outputs if any(token in name.lower() for token in ("prob", "softmax", "prediction"))),
            None,
        )
        if key is None:
            key = sorted(outputs)[0]
        return key, outputs[key]

    @staticmethod
    def _select_shape_output(outputs):
        key = next((name for name in outputs if "original_shape" in name.lower()), None)
        if key is None:
            key = next((name for name in outputs if "shape" in name.lower()), None)
        return (key, outputs[key]) if key is not None else None

    def _resolve_tensor(self, tensor_name, *, role, fallback_tokens=(), allow_stale_signature=False):
        try:
            return self.graph.get_tensor_by_name(tensor_name)
        except KeyError as exc:
            if not allow_stale_signature:
                raise RuntimeError(
                    f"dhSegment SavedModel signature {role} tensor {tensor_name!r} is missing"
                ) from exc

        candidates = []
        for operation in self.graph.get_operations():
            name = operation.name.lower()
            if fallback_tokens and not any(token in name for token in fallback_tokens):
                continue
            for tensor in operation.outputs:
                shape = tensor.shape
                rank = shape.rank
                if role == "probability output" and rank not in (3, 4, None):
                    continue
                if role == "original-shape output" and rank not in (1, 2, None):
                    continue
                candidates.append(tensor)

        if not candidates:
            raise RuntimeError(
                f"dhSegment SavedModel signature {role} tensor {tensor_name!r} is stale and "
                f"no compatible graph tensor matched {fallback_tokens!r}"
            )

        candidates.sort(key=lambda tensor: tensor.name)
        return candidates[-1]

    def _feed_value(self, image_path: Path):
        dtype = self.input_tensor.dtype
        if dtype == self.tf.string:
            # dhSegment's upstream filename prediction contract is a *single*
            # filename string.  Some exported SignatureDefs advertise this
            # tensor with a rank-1 shape, but the graph immediately feeds it
            # to ReadFile, which requires a scalar.  Honor the executable graph
            # contract instead of the misleading static SignatureDef shape.
            return str(image_path)
        raise RuntimeError(
            "dhSegment v0.2 page model is expected to expose a filename/string input; "
            f"got tensor {self.input_tensor.name!r} dtype={dtype.name} shape={self.input_tensor.shape}"
        )

    @staticmethod
    def _page_probability(raw):
        probs = np.asarray(raw, dtype=np.float32)
        while probs.ndim > 3 and probs.shape[0] == 1:
            probs = probs[0]
        if probs.ndim == 3:
            if probs.shape[-1] >= 2:
                probs = probs[:, :, 1]
            elif probs.shape[0] >= 2:
                probs = probs[1]
            else:
                probs = np.squeeze(probs)
        if probs.ndim != 2:
            raise RuntimeError(
                f"dhSegment probability output has unexpected shape {tuple(probs.shape)}"
            )
        maximum = float(np.max(probs)) if probs.size else 0.0
        if maximum > 0:
            probs = probs / maximum
        return np.clip(probs, 0.0, 1.0)

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image_path = Path(handle.name)
        try:
            if not cv2.imwrite(str(image_path), image_bgr):
                raise RuntimeError("could not serialize image for dhSegment inference")
            fetches = [self.probability_tensor]
            if self.shape_tensor is not None:
                fetches.append(self.shape_tensor)
            outputs = self.session.run(
                fetches,
                feed_dict={self.input_tensor: self._feed_value(image_path)},
            )
        finally:
            image_path.unlink(missing_ok=True)

        probability = self._page_probability(outputs[0])
        original_shape = tuple(int(v) for v in image_bgr.shape[:2])
        if self.shape_tensor is not None and len(outputs) > 1:
            raw_shape = np.asarray(outputs[1]).reshape(-1)
            if raw_shape.size >= 2:
                original_shape = (int(raw_shape[-2]), int(raw_shape[-1]))
        return probability, original_shape


def _model():
    model_dir = _asset_dir()
    key = str(model_dir.resolve())
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = _SavedModel(model_dir)
            _MODEL_CACHE[key] = model
    return model


def _image_key(image_bgr: np.ndarray) -> str:
    arr = np.ascontiguousarray(image_bgr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(memoryview(arr))
    return h.hexdigest()


def _infer_evidence(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    key = _image_key(image_bgr)
    with _EVIDENCE_CACHE_LOCK:
        cached = _EVIDENCE_CACHE.get(key)
        if cached is not None:
            _EVIDENCE_CACHE.move_to_end(key)
            return cached

    # Single-flight inference. The expensive TensorFlow prediction is invariant
    # across the parameter grid, so only the first cache miss for an image runs
    # the model. Waiters recheck after acquiring the lock.
    with _INFERENCE_LOCK:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_CACHE.get(key)
            if cached is not None:
                _EVIDENCE_CACHE.move_to_end(key)
                return cached

        probability, original_shape = _model().predict(image_bgr)
        probability = np.array(probability, copy=True)
        probability.setflags(write=False)
        evidence = (probability, tuple(int(v) for v in original_shape))
        with _EVIDENCE_CACHE_LOCK:
            _EVIDENCE_CACHE[key] = evidence
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
        return evidence


def precompute_golden_set_evidence(images, *, progress=None):
    """Populate immutable dhSegment probability maps before thread fan-out."""
    keys = []
    total = len(images)
    for index, image_bgr in enumerate(images, 1):
        key = _image_key(image_bgr)
        started = time.perf_counter()
        if progress is not None:
            progress("start", index, total, key, 0.0)
        _infer_evidence(image_bgr)
        elapsed = time.perf_counter() - started
        if progress is not None:
            progress("finish", index, total, key, elapsed)
        keys.append(key)
    return tuple(keys)


def export_precomputed_golden_set_evidence(images, output_dir, *, progress=None):
    """Persist one process-independent dhSegment Golden Set probability-map set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = precompute_golden_set_evidence(images, progress=progress)
    records = []
    with _EVIDENCE_CACHE_LOCK:
        for index, key in enumerate(keys):
            probability, original_shape = _EVIDENCE_CACHE[key]
            filename = f"page-{index:04d}.npy"
            target = output_dir / filename
            temporary = output_dir / f".{filename}.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, probability, allow_pickle=False)
            os.replace(temporary, target)
            records.append({
                "image_key": key,
                "probability_file": filename,
                "original_shape": list(original_shape),
            })
    payload = {
        "schema_version": "0.1",
        "detector": METHOD,
        "representation": "readonly-npy",
        "page_count": len(records),
        "records": records,
    }
    target = output_dir / "manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_precomputed_golden_set_evidence(output_dir, images):
    """Load parent-precomputed dhSegment probability maps without TensorFlow inference."""
    output_dir = Path(output_dir)
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("detector") != METHOD:
        raise ValueError(f"Shared evidence detector mismatch: {payload.get('detector')} != {METHOD}")
    records = {
        str(record["image_key"]): record
        for record in payload.get("records", [])
        if isinstance(record, dict) and record.get("image_key")
    }
    expected = tuple(_image_key(image_bgr) for image_bgr in images)
    missing = [key for key in expected if key not in records]
    if missing:
        raise ValueError(f"Shared dhSegment evidence is missing {len(missing)} Golden Set page(s)")
    with _EVIDENCE_CACHE_LOCK:
        for key in expected:
            record = records[key]
            # Windows keeps an mmap-backed .npy file locked until the array is
            # released, which breaks normal temp/artifact cleanup. Keep the
            # zero-copy read-only mmap on POSIX, but load a normal ndarray on
            # Windows and mark it immutable. Golden Set evidence is tiny enough
            # that the Windows copy is negligible compared with model inference.
            mmap_mode = None if os.name == "nt" else "r"
            probability = np.load(
                output_dir / str(record["probability_file"]),
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )
            if mmap_mode is None:
                probability.setflags(write=False)
            original_shape = tuple(int(v) for v in record["original_shape"])
            _EVIDENCE_CACHE[key] = (probability, original_shape)
            _EVIDENCE_CACHE.move_to_end(key)
            while len(_EVIDENCE_CACHE) > _EVIDENCE_CACHE_LIMIT:
                _EVIDENCE_CACHE.popitem(last=False)
    return expected


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    if not np.any(binary):
        return binary
    flood = binary.copy()
    height, width = binary.shape
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    seed = None
    for x in range(width):
        if binary[0, x] == 0:
            seed = (x, 0)
            break
        if binary[height - 1, x] == 0:
            seed = (x, height - 1)
            break
    if seed is None:
        for y in range(height):
            if binary[y, 0] == 0:
                seed = (0, y)
                break
            if binary[y, width - 1] == 0:
                seed = (width - 1, y)
                break
    if seed is None:
        return binary
    cv2.floodFill(flood, mask, seed, 255)
    return binary | cv2.bitwise_not(flood)


def _kernel_size(shape, fraction):
    if fraction <= 0:
        return 0
    return max(3, int(round(min(shape) * fraction)) | 1)


def _postprocess(probability: np.ndarray, values):
    if values["probability_threshold"] < 0:
        source = np.rint(probability * 255.0).astype(np.uint8)
        _, binary = cv2.threshold(
            source, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        binary = np.where(
            probability >= values["probability_threshold"], 255, 0
        ).astype(np.uint8)

    close_size = _kernel_size(binary.shape, values["close_kernel_fraction"])
    if close_size:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    open_size = _kernel_size(binary.shape, values["open_kernel_fraction"])
    if open_size:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, 8, cv2.CV_32S
    )
    if count <= 1:
        return binary, None

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dominant = np.where(labels == label, 255, 0).astype(np.uint8)
    if values["fill_holes"]:
        dominant = _fill_holes(dominant)

    offset = values["contour_offset_fraction"]
    if offset != 0:
        size = _kernel_size(dominant.shape, abs(offset))
        if size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            op = cv2.MORPH_DILATE if offset > 0 else cv2.MORPH_ERODE
            dominant = cv2.morphologyEx(dominant, op, kernel)

    contours, _ = cv2.findContours(
        dominant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return dominant, max(contours, key=cv2.contourArea) if contours else None


def _proposal(image_bgr, values):
    probability, original_shape = _infer_evidence(image_bgr)
    binary, contour = _postprocess(probability, values)
    return probability, binary, contour, original_shape


def _scale_points(points, probability_shape, image_shape):
    ph, pw = probability_shape
    ih, iw = image_shape
    scale = np.array([iw / float(pw), ih / float(ph)], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) * scale


def detect(*, image_bgr, mask, parameters=None):
    del mask
    values = _parameters(parameters)
    probability, binary, contour, _ = _proposal(image_bgr, values)
    height, width = image_bgr.shape[:2]
    provenance = _provenance()

    diagnostics = {
        "parameters": values,
        "model_id": provenance.get("model_id", "dhsegment-page-v0.2"),
        "model_family": "dhSegment",
        "model_archive_sha256": provenance.get("archive_sha256"),
        "model_source": provenance.get("model_url"),
        "upstream_repository": provenance.get("upstream_repository"),
        "upstream_license": provenance.get("license"),
        "inference_backend": "tensorflow-savedmodel",
        "probability_min": float(probability.min()) if probability.size else 0.0,
        "probability_max": float(probability.max()) if probability.size else 0.0,
        "probability_mean": float(probability.mean()) if probability.size else 0.0,
        "thresholded_fraction": float(np.count_nonzero(binary)) / float(binary.size),
        "postprocess_resolution": f"{probability.shape[1]}x{probability.shape[0]}",
    }

    if contour is None:
        return Candidate(
            METHOD,
            None,
            None,
            0,
            0,
            {**diagnostics, "reason": "no_dhsegment_page_region"},
            status="no_candidate",
        )

    area_fraction = float(cv2.contourArea(contour)) / float(binary.size)
    diagnostics["mask_area_fraction"] = area_fraction
    if area_fraction < values["minimum_page_area_fraction"]:
        return Candidate(
            METHOD,
            None,
            None,
            0,
            0,
            {**diagnostics, "reason": "dhsegment_mask_too_small"},
            status="no_candidate",
        )

    corners_model = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    corners = _scale_points(corners_model, probability.shape, (height, width))
    x, y, bw, bh = cv2.boundingRect(corners)
    selected = probability[binary > 0]
    mean_probability = float(selected.mean()) if selected.size else 0.0
    score = min(
        1.0,
        0.70 * mean_probability + 0.30 * min(1.0, area_fraction / 0.5),
    )
    diagnostics.update(
        {
            "mean_page_probability": mean_probability,
            "evidence": "dhsegment_resnet50_page_segmentation",
        }
    )
    return Candidate(
        METHOD,
        [int(x), int(y), int(x + bw), int(y + bh)],
        corners.astype(float).tolist(),
        score,
        score,
        diagnostics,
    )


def debug_images(
    *, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False
):
    del mask, verbose
    values = _parameters(parameters)
    probability, binary, contour, _ = _proposal(image_bgr, values)
    height, width = image_bgr.shape[:2]
    probability_full = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    )
    mask_full = cv2.resize(
        binary, (width, height), interpolation=cv2.INTER_NEAREST
    )
    overlay = image_bgr.copy()
    if contour is not None:
        scaled = _scale_points(
            contour.reshape(-1, 2), probability.shape, (height, width)
        )
        cv2.polylines(
            overlay,
            [np.rint(scaled).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 255, 255),
            2,
        )
    if candidate_corners is not None:
        cv2.polylines(
            overlay,
            [
                np.rint(np.asarray(candidate_corners))
                .astype(np.int32)
                .reshape(-1, 1, 2)
            ],
            True,
            (0, 0, 255),
            3,
        )
    return {
        "dhsegment-page-probability.png": np.rint(
            probability_full * 255
        ).astype(np.uint8),
        "dhsegment-page-mask.png": mask_full,
        "dhsegment-page-boundary.png": overlay,
    }


__all__ = [
    "BASELINE_PARAMETERS",
    "METHOD",
    "MODEL_DIR_ENV",
    "PROVENANCE_ENV",
    "debug_images",
    "detect",
    "precompute_golden_set_evidence",
    "export_precomputed_golden_set_evidence",
    "load_precomputed_golden_set_evidence",
]
