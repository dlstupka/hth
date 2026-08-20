from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    key: str
    detector: str
    model_id: str
    model_url: str
    config_url: str | None
    model_repository: str
    description: str
    is_current: bool = False


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "rcnn_hjdataset_current": ModelVariant(
        key="rcnn_hjdataset_current",
        detector="mask_rcnn_page_mask",
        model_id="hjdataset-mask-rcnn-r50-fpn-3x",
        model_url="https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/model_final.pth?download=true",
        config_url="https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/config.yml?download=true",
        model_repository="https://huggingface.co/layoutparser/detectron2/tree/main/HJDataset/mask_rcnn_R_50_FPN_3x",
        description="Current HJDataset Mask R-CNN R50-FPN 3x artifact served by the LayoutParser Hugging Face mirror.",
        is_current=True,
    ),
    "rcnn_hjdataset_older": ModelVariant(
        key="rcnn_hjdataset_older",
        detector="mask_rcnn_page_mask",
        model_id="hjdataset-mask-rcnn-r50-fpn-3x-legacy",
        model_url="https://dl.dropboxusercontent.com/s/893paxpy5suvlx9/model_final.pth",
        config_url="https://dl.dropboxusercontent.com/s/4jmr3xanmxmjcf8/config.yml",
        model_repository="https://github.com/Layout-Parser/layout-parser",
        description="Original LayoutParser HJDataset Mask R-CNN R50-FPN 3x publication artifact from the legacy model catalog.",
        is_current=False,
    ),
}

DEFAULT_MODEL_VARIANTS: dict[str, str] = {
    "mask_rcnn_page_mask": "rcnn_hjdataset_current",
}


def resolve_model_variant(detector: str, requested: str | None) -> ModelVariant | None:
    detector = str(detector or "").strip().lower()
    requested = str(requested or "default").strip().lower() or "default"
    if requested in {"default", "n/a", "na", "none"}:
        key = DEFAULT_MODEL_VARIANTS.get(detector)
        return MODEL_VARIANTS.get(key) if key else None
    variant = MODEL_VARIANTS.get(requested)
    if variant is None:
        raise ValueError(f"Unknown model variant: {requested}")
    if variant.detector != detector:
        raise ValueError(
            f"Model variant {requested!r} belongs to detector {variant.detector!r}, not {detector!r}"
        )
    return variant


def model_variant_choices() -> tuple[str, ...]:
    return ("default", *sorted(MODEL_VARIANTS))
