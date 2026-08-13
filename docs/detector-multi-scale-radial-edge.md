# Multi-Scale Radial Edge Search

`multi_scale_radial_edge` extends the center-outward Radial Edge Search with scale-space evidence. It computes Sobel-gradient magnitude after several Gaussian blur scales, normalizes each scale independently, and fuses the strongest response before sampling boundary points along radial rays. A boundary that is weak or noisy at one spatial scale can therefore remain usable when it is stable at another.

Configuration: `config/detectors/multi_scale_radial_edge.json`.

The detector is intentionally independent from `adaptive_radial_edge`. Adaptive Radial Edge changes angular sampling density around weak fitted sides; Multi-Scale Radial Edge changes the spatial scale at which the same physical boundary is observed. Initial calibration explores blur scale, scale spacing/count, ray density, gradient percentile, and minimum angular support while keeping geometry plausibility and score weights fixed.

Winner debug artifacts include the fused scale-space gradient and the accepted radial edge points. Verbose debug additionally writes a side-by-side view of the blur scales used to build the fused evidence.
