from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from .model import Candidate

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


class _SavedModel:
    """Adapter for the released dhSegment page-extraction SavedModel.

    Use TensorFlow 2's public SavedModel API and call the exported serving
    ConcreteFunction directly. The adapter supports the released filename
    signature and also tolerates numeric image signatures.
    """

    def __init__(self, model_dir: Path):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "dhsegment_page_mask requires the TensorFlow runtime installed "
                "by the regression/optimizer workflow"
            ) from exc

        self.tf = tf
        self.model = tf.saved_model.load(str(model_dir))
        signatures = dict(getattr(self.model, "signatures", {}) or {})
        if not signatures:
            raise RuntimeError("dhSegment SavedModel has no serving signatures")

        self.signature_name = (
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            if tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY in signatures
            else sorted(signatures)[0]
        )
        self.function = signatures[self.signature_name]

        positional, keyword = self.function.structured_input_signature
        self.input_mode = None
        self.input_key = None
        self.input_spec = None

        if keyword:
            if len(keyword) != 1:
                raise RuntimeError(
                    "dhSegment serving signature must expose exactly one input; "
                    f"got {sorted(keyword)}"
                )
            self.input_key, self.input_spec = next(iter(keyword.items()))
            self.input_mode = "keyword"
        elif positional:
            specs = [item for item in positional if hasattr(item, "dtype")]
            if len(specs) != 1:
                raise RuntimeError(
                    "dhSegment serving signature must expose exactly one tensor input; "
                    f"got {self.function.structured_input_signature!r}"
                )
            self.input_spec = specs[0]
            self.input_mode = "positional"
        else:
            raise RuntimeError("dhSegment serving signature exposes no tensor input")

    def _filename_value(self, image_path: Path):
        rank = self.input_spec.shape.rank
        if rank == 0:
            return self.tf.constant(str(image_path))
        if rank == 1 or rank is None:
            return self.tf.constant([str(image_path)])
        raise RuntimeError(
            "dhSegment filename serving input has unsupported rank "
            f"{rank}: {self.input_spec!r}"
        )

    def _image_value(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        value = self.tf.convert_to_tensor(rgb)
        rank = self.input_spec.shape.rank
        if rank == 4:
            value = value[None, ...]
        elif rank not in (3, None):
            raise RuntimeError(
                "dhSegment numeric serving input has unsupported rank "
                f"{rank}: {self.input_spec!r}"
            )
        target_dtype = self.input_spec.dtype
        if target_dtype.is_floating:
            value = self.tf.image.convert_image_dtype(value, target_dtype)
        else:
            value = self.tf.cast(value, target_dtype)
        return value

    def _invoke(self, input_value):
        if self.input_mode == "keyword":
            return self.function(**{self.input_key: input_value})
        return self.function(input_value)

    @staticmethod
    def _pick_output(outputs, *tokens):
        keys = list(outputs)
        for token in tokens:
            for key in keys:
                if token in str(key).lower():
                    return outputs[key]
        return None

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        dtype = self.input_spec.dtype
        image_path = None
        try:
            if dtype == self.tf.string:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                    image_path = Path(handle.name)
                if not cv2.imwrite(str(image_path), image_bgr):
                    raise RuntimeError("could not serialize image for dhSegment inference")
                input_value = self._filename_value(image_path)
            else:
                input_value = self._image_value(image_bgr)

            raw_outputs = self._invoke(input_value)
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

        outputs = raw_outputs if isinstance(raw_outputs, dict) else {"output": raw_outputs}

        probs_tensor = self._pick_output(outputs, "prob", "prediction", "output")
        if probs_tensor is None:
            raise RuntimeError(
                "dhSegment serving signature exposes no probability-like output; "
                f"available={sorted(str(key) for key in outputs)}"
            )

        probs = np.asarray(probs_tensor.numpy(), dtype=np.float32)
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
                "dhSegment probability output has unexpected shape "
                f"{tuple(probs.shape)} from outputs={sorted(str(key) for key in outputs)}"
            )

        pmin = float(np.min(probs)) if probs.size else 0.0
        pmax = float(np.max(probs)) if probs.size else 0.0
        if pmin < 0.0 or pmax > 1.0:
            probs = 1.0 / (1.0 + np.exp(-np.clip(probs, -30.0, 30.0)))
        probs = np.clip(probs, 0.0, 1.0)

        original_shape = tuple(int(v) for v in image_bgr.shape[:2])
        shape_tensor = self._pick_output(outputs, "original_shape", "shape")
        if shape_tensor is not None:
            raw_shape = np.asarray(shape_tensor.numpy()).reshape(-1)
            if raw_shape.size >= 2:
                original_shape = (int(raw_shape[-2]), int(raw_shape[-1]))
        return probs, original_shape


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
    probability, original_shape = _model().predict(image_bgr)
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
]
