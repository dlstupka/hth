# Contour-Based Quadrilateral Detector

The HTH contour-based quadrilateral detector is the first hybrid geometry detector. It combines contour-derived document hypotheses with explicit quadrilateral geometry and independent edge support from the source image and analysis mask.

The stable method identifier is `contour_quad`.

## Algorithm

1. Normalize and optionally close the supplied document mask.
2. Find external contours and, when enabled, add a merged convex-hull hypothesis.
3. Sweep polygon-approximation epsilon values for each plausible contour.
4. Retain convex four-corner candidates that satisfy the configured rectangularity threshold.
5. Score each quadrilateral using image coverage, contour-to-quadrilateral rectangularity, right-angle consistency, and edge support.
6. Return the highest-scoring ordered quadrilateral and its bounding box.

Unlike the original `contour` detector, this detector does not fall back to a minimum-area rectangle when no four-corner polygon is found. A missing quadrilateral is reported as `no_candidate`, preserving the distinction between measured geometry and a geometric fallback.

## Configuration boundary

`config/detectors/contour_quad.json` contains detector calibration defaults and search ranges only. It contains no source-document identity, page ordinals, or collection-specific thresholds. Document-specific configuration remains separate so it can move to a source repository without changing the detector implementation.

## Regression

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/contour_quad.json \
  --golden-set config/golden_set.json \
  --image-root /path/to/preprocessed/images \
  --output regression-output
```

## Debug artifacts

Contour Quadrilateral regression uses the `winner` debug policy so every completed
run preserves comparable evidence for every Golden Set page in the winning
parameter set. Each page directory contains:

- the original page and detector input mask;
- the post-morphology mask;
- external-contour and merged-hull hypotheses;
- plausible convex quadrilateral hypotheses;
- combined source-image and mask edge evidence;
- the selected ordered quadrilateral;
- the approved-versus-predicted overlay; and
- complete detector diagnostics in JSON.

These artifacts support troubleshooting, cross-detector comparison, regression
reproducibility, and preservation of the experimental record.
