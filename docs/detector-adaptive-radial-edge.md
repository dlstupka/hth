# Adaptive Radial Edge Search detector

The stable method identifier is `adaptive_radial_edge`. The detector preserves the center-outward evidence model of `radial_edge` while adding a constrained second pass.

The first pass samples the full image at three-degree spacing and fits a coarse quadrilateral. Each fitted side is evaluated against its own geometrically eligible coarse-ray population: accepted endpoints near that side are divided by the number of coarse rays whose forward paths intersect it. This avoids incorrectly treating normal differences in angular span as weak support. When one or more sides fall below the configured normalized support threshold, the detector performs a one-degree angular refinement only across sectors that intersect those weak sides, combines the new evidence with the coarse measurements, and refits once.

The detector does not move the origin, recursively refine, or replace the original Radial Edge Search detector. It exists as an independently calibrated experiment so the value and cost of adaptive angular refinement can be measured directly.

Configuration: `config/detectors/adaptive_radial_edge.json`.

Verbose debug output preserves the complete two-pass narrative: Pass 1 rays, the Pass 1 fit, per-side eligible/accepted support with weak-side classification, Pass 2-only refinement rays, the Pass 2 fit, and the common final overlay. The ordered artifact sequence places `07-side-support.png` between the Pass 1 and Pass 2 evidence.
