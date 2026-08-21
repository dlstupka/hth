# ScanTailor Page-Frame Detector

`scantailor_page_frame` adds a production scan-processing lineage to HTH's page-boundary research. It is an independent HTH/OpenCV implementation of the ScanTailor-style workflow concept: normalize uneven illumination, estimate the content envelope, then search outward in horizontal and vertical edge-energy projections for the physical page frame.

HTH does **not** embed, translate, or execute ScanTailor source code. The detector is intentionally clean-room and independently calibratable so ScanTailor's scan-processing ideas can be compared fairly with HTH's classical and learned page detectors without creating a runtime or licensing dependency on the ScanTailor application.

The initial calibration space contains 243 parameter sets. It varies illumination normalization, residual/content thresholding, content morphology, projection smoothing, and the width of the outward page-edge search. The minimum page-area gate is retained as an explicit singleton dimension so complete parameter identities remain reproducible.

Debug output includes the locally normalized contrast image, content mask, and selected page frame.
