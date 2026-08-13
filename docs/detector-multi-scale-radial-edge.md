# Multi-Scale Radial Edge Search

`multi_scale_radial_edge` extends the center-outward Radial Edge Search with scale-space evidence. It computes Sobel-gradient magnitude after several Gaussian blur scales, normalizes each scale independently, and fuses the strongest response before sampling boundary points along radial rays. A boundary that is weak or noisy at one spatial scale can therefore remain usable when it is stable at another.

Configuration: `config/detectors/multi_scale_radial_edge.json`.

The detector is intentionally independent from `adaptive_radial_edge`. Adaptive Radial Edge changes angular sampling density around weak fitted sides; Multi-Scale Radial Edge changes the spatial scale at which the same physical boundary is observed.

The first exhaustive calibration characterized 730 sets and raised Avg IoU from the baseline 0.6520 to 0.9457. That run identified `gradient_percentile` as the dominant parameter, `base_sigma` as the next strongest effect, and a smaller `ray_count` interaction; the winning values also sat at the tested high edge for sigma/scale spacing and the low edge for ray count. Generation 2 therefore uses a 100,000-set local-plus-boundary search: it densely resolves gradient percentile and sigma, probes lower and higher ray densities, extends scale spacing beyond the first winner, and keeps only the two useful scale counts. Parameters measured dormant in the first grid, including minimum ray support and fixed geometry/score controls, remain at their baseline values so the larger budget is spent on dimensions that carried actual calibration signal.

Winner debug artifacts include the fused scale-space gradient and the accepted radial edge points. Verbose debug additionally writes a side-by-side view of the blur scales used to build the fused evidence.
