# Mask R-CNN Page-Mask Detector

`mask_rcnn_page_mask` integrates the LayoutParser/HJDataset **Mask R-CNN R50-FPN** model as a fixed historical-document instance-segmentation source. HJDataset explicitly includes a Page Frame class and is a historical-document corpus, making this a materially different learned architecture from the U-Net-family page-mask detectors already in HTH.

HTH does not fine-tune Mask R-CNN on the Golden Set. Neural inference runs once per Golden Set page before shard fan-out and persists parameter-invariant instance evidence. Calibration is limited to deterministic confidence filtering, minimum learned-instance area, minimum physical-page area, and final quadrilateral padding. The largest learned instance is preferred because HJDataset's Page Frame normally dominates the page; when no single instance is large enough, HTH may form a conservative convex envelope from substantial learned layout instances.

The lifecycle pins the HJDataset model/configuration in the results repository and records SHA-256 provenance. Inference is CPU-only through Detectron2/PyTorch. The detector is intentionally registered as `mask_rcnn_page_mask` rather than a generic Mask R-CNN abstraction so future Mask R-CNN models can coexist without silently changing detector identity.
