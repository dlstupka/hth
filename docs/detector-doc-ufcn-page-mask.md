# Doc-UFCN Page-Mask Detector

`doc_ufcn_page_mask` integrates Teklia's released **Doc-UFCN generic page detection model** as a fixed learned page-segmentation source. The upstream model predicts single-page polygons and was trained on the historical-document Horae and READ-BAD datasets at a largest image dimension of 768 pixels. HTH does not train or fine-tune the network during calibration; only deterministic post-inference filtering and page-envelope geometry are calibrated.

## Runtime and model provenance

The detector lifecycle installs the Doc-UFCN inference package into HTH's managed runtime, downloads `model.pth` and `parameters.yml` from `Teklia/doc-ufcn-generic-page`, stores them under the results repository, and records SHA-256 provenance. HTH runs the model on CPU. The upstream package currently declares an older Python/PyTorch dependency window, so HTH installs the package code without its pinned dependencies and validates the API against HTH's managed CPU PyTorch runtime rather than allowing Doc-UFCN to downgrade the shared environment.

The released model contract is two classes (`background`, `page`), input size 768, mean `[190, 182, 165]`, standard deviation `[48, 48, 45]`, and upstream minimum connected-component area 50. HTH requests polygons with `min_cc=1` once, preserving all learned components so calibration can perform its own deterministic component filtering.

## Parameter-invariant learned evidence

Neural inference does not depend on the calibration parameter set. The Golden Set page polygons are therefore precomputed once before parameter-thread fan-out when the detector is sharded, using the same shared learned-evidence mechanism as the other learned page detectors. Each parameter set subsequently filters and converts the immutable learned polygons without rerunning Doc-UFCN.

## Calibrated parameters

- `minimum_confidence` rejects low-confidence learned page polygons.
- `minimum_component_area_fraction` removes learned connected components that are too small relative to the source image.
- `minimum_page_area_fraction` rejects the selected learned page region when it is implausibly small for a physical page.
- `page_padding_fraction` expands the selected minimum-area page quadrilateral around its center before clipping to the source image.

The initial grid intentionally avoids model-specific training knobs. It asks whether the released historical page model is useful on HTH-0001 before any architecture-specific refinement is justified.

## Geometry

Qualifying class-`page` polygons are ranked by area and confidence. HTH selects the strongest surviving page region, fits an oriented minimum-area quadrilateral, applies calibrated padding, clips it to the source image, and returns the resulting physical-page proposal through the normal detector contract.

## Debug evidence

Winner/verbose artifacts include `doc-ufcn-page-polygons.png`, showing upstream page polygons in yellow and HTH's selected quadrilateral in red. Candidate diagnostics retain model identity, SHA-256, confidence, selected-area fraction, input size, and inference backend.
