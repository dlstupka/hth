# Mask R-CNN Page-Mask Detector

`mask_rcnn_page_mask` integrates the LayoutParser/HJDataset **Mask R-CNN R50-FPN** model as a fixed historical-document instance-segmentation source. HJDataset explicitly includes a Page Frame class and is a historical-document corpus, making this a materially different learned architecture from the U-Net-family page-mask detectors already in HTH.

HTH does not fine-tune Mask R-CNN on the Golden Set. Neural inference runs once per Golden Set page before shard fan-out and persists parameter-invariant instance evidence. Calibration is limited to deterministic confidence filtering, minimum learned-instance area, minimum physical-page area, and final quadrilateral padding. The largest learned instance is preferred because HJDataset's Page Frame normally dominates the page; when no single instance is large enough, HTH may form a conservative convex envelope from substantial learned layout instances.

The lifecycle pins the HJDataset model/configuration in the results repository and records SHA-256 provenance. Inference is CPU-only through Detectron2/PyTorch. The detector is intentionally registered as `mask_rcnn_page_mask` rather than a generic Mask R-CNN abstraction so future Mask R-CNN models can coexist without silently changing detector identity.
The managed Detectron2 layer pins `iopath==0.1.9` because Detectron2 0.6 declares `iopath>=0.1.7,<0.1.10`, and installs the declared `black` dependency at `24.10.0`. HTH verifies those pins before accepting a reusable Mask R-CNN runtime and still requires a clean `pip check`; a failed incremental augmentation is rolled back rather than leaving the shared runtime partially updated.


## Targeted confidence/padding refinement

The first post-envelope exhaustive calibration showed that `minimum_confidence` dominated the measured landscape, while `minimum_instance_area_fraction` and `minimum_page_area_fraction` were effectively inert and `page_padding_fraction` had only a secondary interaction with confidence. The declared grid is therefore intentionally contracted for the next refinement: the two inert area thresholds are pinned to the previous winner, confidence is sampled densely from `0.0` through `0.25`, and padding is sampled densely from `0.0` through `0.04` around the previous `0.01` optimum. This produces a 90-set surgical grid that measures the one conspicuous untested confidence interval without spending another exhaustive run on dimensions already characterized as inactive. Historic Best and Baseline remain mandatory regression references outside the declared search grid.
