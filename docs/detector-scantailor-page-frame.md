# ScanTailor Page-Frame Detector

`scantailor_page_frame` adds a production scan-processing lineage to HTH's page-boundary research. It is an independent HTH/OpenCV implementation of the ScanTailor-style workflow concept: normalize uneven illumination, estimate the content envelope, then search outward in horizontal and vertical edge-energy projections for the physical page frame.

HTH does **not** embed, translate, or execute ScanTailor source code. The detector is intentionally clean-room and independently calibratable so ScanTailor's scan-processing ideas can be compared fairly with HTH's classical and learned page detectors without creating a runtime or licensing dependency on the ScanTailor application.

The initial 243-set exhaustive calibration produced a useful 0.9300 Avg-IoU result, but several winning controls landed on the declared grid boundary. The follow-up characterization space therefore contains 2,304 parameter sets. It preserves the original baseline and first winner while extending ink quantile downward, allowing zero content closing, extending projection smoothing upward, widening illumination normalization, and opening the minimum page-area gate. The outward boundary-search domain remains centered on the first winner because that control did not land on an edge.

Debug output includes the locally normalized contrast image, content mask, and selected page frame.
