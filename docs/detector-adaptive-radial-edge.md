# Adaptive Radial Edge Search detector

The stable method identifier is `adaptive_radial_edge`. The detector preserves the center-outward evidence model of `radial_edge` while adding a constrained second pass.

The first pass samples the full image at three-degree spacing and fits a coarse quadrilateral. It measures how many accepted radial endpoints confirm each fitted side. When one or more sides have comparatively weak support, the detector performs a one-degree angular refinement only across sectors that intersect those weak sides, combines the new evidence with the coarse measurements, and refits once.

The detector does not move the origin, recursively refine, or replace the original Radial Edge Search detector. It exists as an independently calibrated experiment so the value and cost of adaptive angular refinement can be measured directly.

Configuration: `config/detectors/adaptive_radial_edge.json`.

Verbose debug output adds weak-side support, refined-ray, and coarse-versus-refined quadrilateral visualizations.
