# Full-collection detector review

HTH can run one approved detector calibration over every page during a production preprocess build. This is corpus inference, not regression: pages without Golden Set truth receive detector geometry and diagnostics but no IoU.

For the current San Antonio collection, **HTH preprocess collection** resolves the strongest **Approved** authoritative calibration for Golden Set `HTH-0001` automatically. The current winner is Fusion Gen3 — AMSRE + Doc-UFCN (`amsre_doc_ufcn_fusion`), parameter set `57b3edb3ac1c`.

The production build writes the selected detector candidate, confidence, geometry, and compact arbitration diagnostics into `page-analysis/page-analysis.json`. The normal results publication preserves that evidence. Keep the complete generated build artifact while visually reviewing the corpus because it contains the source images and matching page-analysis data.

To review, download and expand the production build artifact, open `tools/reference-collection-editor-multidetector.html` locally, choose **Open results workspace**, and select the expanded build directory. Fusion Gen3 is available as a detector overlay and is selected independently from the Golden Set reference.

The first complete production run produced candidates for all 928 pages with no detector errors or missing candidates and an average detector confidence of `0.920780`. That confidence is useful for ranking review work, but it is not IoU and is not ground truth.

Use the low-confidence tail, detector-disagreement/rescue diagnostics, capture-regime changes, and representative high-confidence controls to decide what deserves human review and possible inclusion in a future Golden Set. The process does not create `HTH-GOLDEN-0002` automatically.
