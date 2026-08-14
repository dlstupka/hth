# dhSegment Page-Mask Detector

`dhsegment_page_mask` adds an independent learned page-boundary family based on the released dhSegment v0.2 page-extraction model.

The detector deliberately does **not** import the historical dhSegment Python package. Its lifecycle downloads the upstream `model.zip` release, records the archive SHA-256, locates the exported TensorFlow SavedModel, and exposes only the serving graph to HTH. The workflow installs TensorFlow only when this detector (or an all-detector run containing it) is executed, so ordinary detector jobs keep the existing lightweight dependency set.

Inference follows the released page-demo contract: the SavedModel predicts a page probability surface, HTH normalizes the page-class probabilities, converts them to a binary region, keeps the dominant page component, and fits an oriented minimum-area rectangle.

The initial calibration contains exactly **10,000 exhaustive parameter sets**. The network weights remain fixed while HTH calibrates six post-processing controls:

- probability threshold, including the upstream-style Otsu baseline;
- minimum accepted page area;
- closing morphology;
- opening morphology;
- small signed contour expansion/contraction;
- hole filling.

The baseline uses Otsu thresholding, a 20% minimum page area, light closing, no opening or contour offset, and hole filling. Debug artifacts include the probability map, cleaned page mask, and fitted page boundary.
