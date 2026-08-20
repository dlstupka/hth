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

## Boundary-supported padding refinement

The first HTH-0001 exhaustive run showed a useful but asymmetric response: the calibrated winner improved four pages while page 5 regressed relative to the detector baseline. The learned page polygon itself remained strong, so HTH now treats unusually small calibrated padding as a proposal rather than an unconditional contraction. When requested padding is below the detector baseline, a parameter-free source-image check compares gradient support along the requested quadrilateral with the baseline-padded quadrilateral. Baseline padding is restored only when the larger envelope has materially stronger independent image-boundary support; otherwise the calibrated padding is preserved.

This refinement does not alter Doc-UFCN inference, persisted learned evidence, or the calibration parameter set. Verbose candidate diagnostics record both boundary-support measurements and the padding-arbitration decision.


## Multi-component spread envelope refinement

The initial HTH-0001 characterization showed a complementary failure mode on open-volume images: Doc-UFCN can correctly identify both physical leaves but emit them as separate class-`page` components. Selecting only the single largest component then truncates roughly half of the spread even though the learned evidence for the missing leaf is already present. HTH now evaluates the surviving learned components as a parameter-free page-envelope set before quadrilateral conversion.

A second component is joined only when it is substantial relative to the primary component, vertically coextensive, similarly tall, horizontally distinct, and materially expands the document span. Qualifying components are combined through a convex outer support and then pass through the existing minimum-page-area and padding logic. Small remote components, vertically unrelated detections, and ordinary single-page predictions remain unchanged. Diagnostics record the qualifying component count, joined-component evidence, span gain, and whether `multi-component-spread-envelope` was selected. The Doc-UFCN model, persisted learned evidence, and calibration grid are unchanged.

## Image-supported single-leaf spread completion

The multi-component spread envelope substantially improved HTH-0001 pages where Doc-UFCN returned both facing leaves, but damaged page 5 remained a single-leaf prediction: its learned polygon already matched the physical top and bottom while ending hundreds of pixels before the opposite side of the spread. HTH now treats that shape as a seed rather than guessing a second leaf. When a single learned component spans most of the physical page height, sits near one source-image side, and leaves a large horizontal region unexplained, a parameter-free vertical-gradient profile searches only the missing outer side for a persistent physical boundary.

Completion is accepted only when the independently measured source-image edge is strong relative to the surrounding profile, lies in the outer portion of the image, and materially expands the learned span. The proven boundary is then added to the learned polygon before the existing quadrilateral and padding stages. Local text blocks, vertically incomplete fragments, already-broad predictions, and seeds without an independently supported outer edge remain unchanged. Diagnostics record the attempted side, physical margins, boundary score/background/threshold, span gain, recovered boundary, and `image-supported-single-leaf-spread-completion` decision. The Doc-UFCN model, persisted learned evidence, and calibration grid remain unchanged.

## Outermost robust damaged-spread boundary selection

The damaged-spread completion path can encounter a strong interior fold or rule before the actual outer paper edge. Selecting the single strongest vertical gradient therefore risks stopping the recovered spread too early even when a farther physical edge is independently visible. HTH now groups above-threshold vertical-profile samples into robust boundary candidates and, once the damaged-spread contract has qualified, prefers the outermost robust candidate on the missing side rather than the raw profile maximum. The existing confidence, outer-region, and span-gain gates still apply.

This remains a parameter-free post-inference refinement. It changes neither Doc-UFCN model evidence nor the calibration grid. Diagnostics record the boundary-selection mode and number of robust candidate boundaries so verbose artifacts distinguish a true outer-sheet completion from the strongest-edge fallback.
