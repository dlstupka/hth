# Learned Page-Mask detector

The stable method identifier is `learned_page_mask`. The detector uses the released PageNet Ohio model to predict a 256×256 page-membership probability map, thresholds that map, keeps the dominant learned region, and fits a quadrilateral back in source-image coordinates.

Configuration: `config/detectors/learned_page_mask.json`.

The Generation-4 calibration space is an exhaustive 50,000-set local refinement derived from the completed Generation-3 10,000-set run. Search density is concentrated around the observed interior optimum: mask threshold 0.204–0.250 in 0.002 increments, polygon epsilon 0.014–0.023 in 0.0005 increments, and bounding-box padding 0.020–0.043 in 0.001 increments. The original baseline values remain explicit anchors.

`minimum_mask_area_fraction` and `close_kernel_fraction` remain deliberately sparse at their prior winner/baseline anchors because the Generation-3 evidence showed little useful discrimination across those dimensions. This keeps the Cartesian space at 50,000 combinations while spending almost all additional evaluations on the three parameters that still showed local sensitivity.

Regression uses the learned detector lifecycle to prepare and reuse the PageNet network and records model provenance, output-layer identity, probability evidence, and failure diagnostics with the detector result.
