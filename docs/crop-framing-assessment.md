# Crop and framing assessment

`HTH assess crop and framing` is the diagnostic entry point for the first
normalization-stage decision. It compares three ways of applying the current
Rank #1 approved document-detector geometry to the 18 immutable
`HTH-GOLDEN-0002` images:

- `axis-aligned` crops the detector quadrilateral's enclosing rectangle;
- `rotation-crop` uses the quadrilateral's minimum-area rotated rectangle; and
- `projective` maps the detector quadrilateral directly to a rectangle with a
  four-point projective transform.

The workflow resolves the approved detector and exact parameter identity from
persisted calibration intelligence. It verifies and materializes the frozen
Golden Set image bundle, runs that detector over the images, and uploads a
diagnostic artifact. It does not publish normalized images, alter calibration
intelligence, or select a production normalization policy.

The initial assessment selected `axis-aligned` as the conservative production
policy. It had the strongest approved-box retention and avoids resampling when
the available detector geometry contains no independently established
perspective information. The implemented artifact-only normalization workflow
is documented in [Document normalization](normalization.md).

The manual dispatch menu uses the standard HTH runner controls. `Execution
runner` selects GitHub-hosted or one of the self-hosted runner classes;
`Specific self-hosted runner` can be changed to `custom` to route the job to
the exact label entered in `Custom self-hosted runner label`. This makes the
image-heavy comparison easy to move to a faster available self-hosted runner
without changing the workflow.

## Assessment artifact

The artifact contains:

```text
assessment.json
assessment.csv
summary.md
index.html
detector-selection.json
geometry-evidence.json
contact-sheets/
variants/
├── axis-aligned/
├── rotation-crop/
└── projective/
```

Open `index.html` after extracting the artifact. Each page shows the source
image with the approved Golden Set box in red and the calibrated detector
quadrilateral in blue, followed by all three framing results.

The machine-readable assessment records output dimensions, output/source area,
output SHA-256, transform matrices, transform reprojection error, and an
approved-box retention proxy.

## Golden Set truth limitation

`HTH-GOLDEN-0002` stores manually approved axis-aligned
`physical_document_bbox` geometry. It does not store four independently
approved perspective corners. The approved-box retention measurement can expose
obvious clipping, but it cannot establish that one perspective correction is
visually or geometrically superior. Contact-sheet review remains necessary for
the initial policy decision.

If HTH later needs automatic regression of perspective rectification, a future
Golden Set must add immutable four-corner truth or another independently
approved normalized-image contract. The current workflow deliberately labels
its results diagnostic rather than presenting a box-derived proxy as
perspective ground truth.
