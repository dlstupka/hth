# Chromatic normalization

Chromatic normalization follows contrast and tonal normalization and consumes the immutable tonal release. It evaluates color correction independently from illumination and luminance transforms so that paper color, ink color, annotations, stamps, and other historically meaningful chromatic evidence are not silently neutralized.

The stage is a four-step evidence chain:

1. **Assessment** measures bright-background color cast, background chroma variation, and the fraction of strongly colorful pixels on every page. Pages are classified as correction candidates, preserve, or review; every page continues downstream.
2. **Method comparison** evaluates bounded 25%, 50%, and 75% background-neutralization variants on the deterministic development partition.
3. **Held-out validation** requires every candidate to pass background-cast reduction, luminance-detail correlation, chroma-structure correlation, gamut-clipping, and median-luminance-shift gates.
4. **Integration** applies only the globally selected and held-out-validated method to candidate pages. If no method is safe, the complete collection is preserved byte-for-byte at the pixel level and still receives a durable chromatic result identity.

Each step uses Canonical Build Evidence. Immutable tonal release assets are restored through the shared local-runner and GitHub cache path, compact evidence is published to the results repository, and integration produces an immutable lossless release. CBE identities depend on effective source, configuration, implementation, and runtime contracts—not runner identity or display telemetry.

The `HTH normalize collection` workflow includes all four chromatic stages in the full regression and exposes each as a selectable starting stage for development and troubleshooting.
