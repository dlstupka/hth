# docExtractor Page-Mask Detector

`docextractor_page_mask` integrates the authors' released docExtractor ResUNet historical-document segmentation model. docExtractor predicts document element segmentation rather than a dedicated physical-page class, so HTH derives a page-support probability surface as one minus the model's background probability and fits the dominant physical-document envelope to that fixed learned evidence.

The lifecycle caches both the released model archive and the matching upstream source required to deserialize the original checkpoint. The primary ENPC model location is tried first and the authors' Google Drive release is retained as a logged fallback; model/source archive and extracted checkpoint SHA-256 identities are recorded in provenance. Inference runs once per Golden Set page in the shared learned-evidence stage before pipeline fan-out. Calibration is limited to deterministic probability thresholding, minimum-area rejection, morphology, and envelope padding.

This detector uses the existing learned-detector regression, optimizer, debug, calibration-intelligence, and persistent evidence-cache contracts rather than introducing a separate execution process.

Upstream: <https://github.com/monniert/docExtractor>
