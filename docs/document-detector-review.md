# Full-collection detector review

HTH can run one approved detector calibration over every page during a production preprocess build. This is corpus inference, not regression: pages without Golden Set truth receive detector geometry and diagnostics but no IoU.

For the current San Antonio collection, run **HTH preprocess collection** manually with **Run approved detector over every page for corpus review** set to `amsre_doc_ufcn_fusion`. The approved calibration is pinned in `config/document-detectors.json` to Golden Set `HTH-0001`, parameter set `57b3edb3ac1c`.

The production build writes the selected detector candidate and compact arbitration diagnostics into `page-analysis/page-analysis.json`. The normal results publication preserves that compact evidence. Keep **Upload the complete generated build as a temporary artifact** enabled while visually reviewing the corpus because the artifact contains both `raw/` images and the matching page-analysis data.

To review, download and expand the production build artifact, open `tools/reference-collection-editor-multidetector.html` locally, choose **Open results workspace**, and select the expanded build directory. Fusion Gen3 is available as a detector overlay and is selected independently from the Golden Set reference.

The full-document run is intended to identify pages that deserve human attention and possible inclusion in a future Golden Set. It does not create HTH-0002 automatically and detector confidence is not treated as ground truth.
