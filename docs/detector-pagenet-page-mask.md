# PageNet Page-Mask Detector

`pagenet_page_mask` gives the original PageNet Ohio Death Records checkpoint an explicit detector identity. PageNet was designed to return the main physical-page quadrilateral from historical handwritten document images. HTH already exercised this exact released checkpoint through the legacy `learned_page_mask` detector, so the explicit detector intentionally reuses that proven OpenCV-DNN/Caffe lifecycle and inference path instead of introducing a second implementation of the same network.

The separate identity lets PageNet participate explicitly in detector inventories, report tables, smoke/full regressions, optimizer runs, and future model-variant research while preserving the existing `learned_page_mask` result lineage. Its model provenance remains the released PageNet Ohio model and its calibration grid remains the already-established PageNet postprocessing domain.

Upstream: <https://github.com/ctensmeyer/pagenet>
