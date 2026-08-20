# Eynollah Page-Mask Detector

`eynollah_page_mask` integrates the SBB Eynollah page-extraction network as a fixed historical-document page segmentation source. Eynollah's released page-extraction model is specifically intended to remove scan margins by predicting page borders with pixel-wise segmentation.

The detector lifecycle downloads the released TensorFlow SavedModel into the results model cache, records each artifact SHA-256 plus the source/reference that supplied it, and exposes the prepared model to the regression worker. Golden Set inference is performed once in the parent learned-evidence stage before pipeline fan-out; calibration therefore changes only deterministic post-inference thresholding, page-area rejection, morphology, and final page-envelope padding.

The initial declared grid is intentionally broad enough to characterize the released model before any parameter contraction. Automatic and manual smoke/full regressions, execution-shape optimization, debug artifacts, calibration intelligence, and persistent shared learned-evidence reuse follow the same contracts as the other learned page-mask detectors.

Upstream: <https://github.com/qurator-spk/eynollah>  
Model: <https://huggingface.co/SBB/eynollah-page-extraction>
