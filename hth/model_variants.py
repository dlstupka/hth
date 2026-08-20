from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSource:
    site: str
    url: str
    reference: str | None = None


@dataclass(frozen=True)
class ModelVariant:
    key: str
    detector: str
    model_id: str
    model_sources: tuple[ModelSource, ...]
    config_sources: tuple[ModelSource, ...]
    model_repository: str
    description: str
    is_current: bool = False

    @property
    def model_url(self) -> str:
        """Backward-compatible primary model URL."""
        return self.model_sources[0].url

    @property
    def config_url(self) -> str | None:
        """Backward-compatible primary config URL."""
        return self.config_sources[0].url if self.config_sources else None


_HJ_MODEL_DROPBOX = "https://dl.dropboxusercontent.com/s/893paxpy5suvlx9/model_final.pth"
_HJ_CONFIG_DROPBOX = "https://dl.dropboxusercontent.com/s/4jmr3xanmxmjcf8/config.yml"
_HJ_IMPORT = "e5bb451bca0f4300df37f974b38ed806c6c7a266"
_HJ_INITIAL = "3f5bd482f4ffb44a66922303a65214b57a4eaf45"
_HJ_PATH = "HJDataset/mask_rcnn_R_50_FPN_3x"


def _hf(site_ref: str, filename: str, *, encoded_legacy_name: bool = False) -> ModelSource:
    suffix = f"{filename}%3Fdl%3D1" if encoded_legacy_name else filename
    return ModelSource(
        site="Hugging Face / LayoutParser",
        url=f"https://huggingface.co/layoutparser/detectron2/resolve/{site_ref}/{_HJ_PATH}/{suffix}?download=true",
        reference=site_ref,
    )


def _dropbox(url: str) -> ModelSource:
    return ModelSource(
        site="Dropbox / LayoutParser original catalog",
        url=url,
        reference="LayoutParser model catalog",
    )


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "rcnn_hjdataset_current": ModelVariant(
        key="rcnn_hjdataset_current",
        detector="mask_rcnn_page_mask",
        model_id="hjdataset-mask-rcnn-r50-fpn-3x",
        model_sources=(
            _hf("main", "model_final.pth"),
            _hf(_HJ_IMPORT, "model_final.pth"),
            _dropbox(_HJ_MODEL_DROPBOX),
        ),
        config_sources=(
            _hf("main", "config.yml"),
            _hf(_HJ_IMPORT, "config.yml"),
            _dropbox(_HJ_CONFIG_DROPBOX),
        ),
        model_repository="https://huggingface.co/layoutparser/detectron2/tree/main/HJDataset/mask_rcnn_R_50_FPN_3x",
        description="Current HJDataset Mask R-CNN R50-FPN 3x artifact served by the LayoutParser Hugging Face mirror.",
        is_current=True,
    ),
    "rcnn_hjdataset_older": ModelVariant(
        key="rcnn_hjdataset_older",
        detector="mask_rcnn_page_mask",
        model_id="hjdataset-mask-rcnn-r50-fpn-3x-legacy",
        model_sources=(
            _hf(_HJ_INITIAL, "model_final.pth", encoded_legacy_name=True),
            _hf(_HJ_IMPORT, "model_final.pth"),
            _dropbox(_HJ_MODEL_DROPBOX),
        ),
        config_sources=(
            _hf(_HJ_INITIAL, "config.yml", encoded_legacy_name=True),
            _hf(_HJ_IMPORT, "config.yml"),
            _dropbox(_HJ_CONFIG_DROPBOX),
        ),
        model_repository=f"https://huggingface.co/layoutparser/detectron2/commit/{_HJ_INITIAL}",
        description="Legacy HJDataset Mask R-CNN R50-FPN 3x publication snapshot pinned to LayoutParser's initial Hugging Face model import.",
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
