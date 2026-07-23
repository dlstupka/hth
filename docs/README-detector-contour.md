# Contour Detector

The Contour detector is the second black-box geometry detector admitted to the
HTH regression harness. It evaluates external contours from the shared physical
page mask, ranks plausible document envelopes, and returns the strongest
candidate bounding box and quadrilateral.

## Regression contract

The detector accepts the same callable contract as GrabCut:

```python
detect(image_bgr=image, mask=mask, parameters=parameters)
```

It returns a normal `Candidate`; the regression framework remains unaware of the
algorithm's internal meaning.

## Parameters

| Parameter | Purpose |
|---|---|
| `minimum_contour_area_fraction` | Rejects contours that occupy too little of the analysis image. |
| `polygon_epsilon_fraction` | Controls Douglas-Peucker polygon simplification. |
| `close_kernel_fraction` | Sizes an optional morphology-close kernel relative to the shorter image dimension. |
| `close_iterations` | Controls how many morphology-close passes join fragmented mask regions. |
| `rectangularity_weight` | Balances mask coverage against rectangularity in candidate ranking. |
| `bbox_padding_fraction` | Expands the selected axis-aligned envelope before evaluation. |

The production reference is `profiles.baseline` in
`config/detectors/contour.json`. Regression may explore the configured Cartesian
space exhaustively or through binary refinement.

## Run

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/contour.json \
  --golden-set config/golden_set.json \
  --image-root path/to/preprocessed \
  --output build/regression \
  --strategy binary-refine
```

Contour is evaluated independently from GrabCut. Results should be compared by
Golden Set metrics rather than by confidence scores across detector families.
