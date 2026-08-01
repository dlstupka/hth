# Cross-Edge Contour

The stable method identifier is `cross_edge_contour`. The detector uses Contour Quadrilateral as its geometry generator, then validates each side by sampling image intensity on the inside and outside of the proposed boundary. It measures cross-edge contrast and polarity consistency to reject geometrically plausible internal frames, writing contours, or weak boundaries.

Configuration: `config/detectors/cross_edge_contour.json`.
