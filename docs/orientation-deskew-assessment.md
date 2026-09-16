# Orientation and deskew assessment

`HTH prepare normalization recommendation` is the researcher-facing entry
point for the second normalization decision. One run reconstructs the review
sample, assesses orientation and deskew candidates, produces a plain-language
recommendation, and preserves the compact evidence and machine-readable policy.
It consumes the persisted canonical crop manifest rather than rerunning
document detection or treating a temporary image artifact as authoritative.

The workflow deliberately does not reproduce the current Golden Set creation
experience. Researchers do not download an artifact, copy identifiers, edit
JSON, or invoke separate assessment and recommendation jobs. Those remain
separate internal operations so they can be tested and maintained, but the
normal path exposes them as one guided preparation action.

The assessment reconstructs a bounded stratified sample directly from the
immutable source DOCX release. Each source image is checked against the
canonical preprocess manifest, cropped using the exact persisted half-open
bounds, and compared by pixel hash with the canonical normalization result
before any diagnostic transform is evaluated.

## Sampling policy

The sample combines:

- every approved `HTH-GOLDEN-0002` page;
- the first and last collection pages;
- the existing every-25th canonical crop review cadence;
- the lowest detector-confidence crops;
- the lowest retained-area crops; and
- the most asymmetric crop margins.

Reasons are retained per page in `sample-plan.json`. This gives the review a
broad chronological surface while deliberately including the pages most
likely to expose a geometric failure. It avoids rebuilding or uploading the
complete 1.6 GB normalized collection.

## Compared candidates

Each contact sheet contains:

- the canonical no-op crop;
- 90-degree, 180-degree, and 270-degree gross-orientation candidates;
- a bounded projection-profile deskew candidate; and
- a bounded Hough-line deskew candidate.

The small-angle estimators are limited to the configured correction range and
use a deadband around zero. Candidate rotations expand the canvas and use a
white border so the diagnostic comparison does not intentionally clip source
pixels. Metrics include estimator angle and confidence, estimator agreement,
expanded area, a foreground-retention proxy, foreground near the output
boundary, and a sharpness ratio.

Gross orientation remains explicitly review-only. Image geometry can distinguish
the horizontal text axis from the vertical text axis, but it cannot reliably
distinguish upright from upside down without independently accepted semantic
evidence. Every non-zero deskew correction resamples pixels, so the preparation
workflow may recommend a conservative policy but never applies it to the
canonical collection.

## Artifact

The uploaded `hth-orientation-deskew-*` artifact is intentionally compact:

```text
assessment.json
assessment.csv
summary.md
index.html
recommendation.md
normalization-policy.json
sample-plan.json
materialization-evidence.json
contact-sheets/
```

It does not contain a second lossless image corpus. Open `index.html` to review
all sampled pages. The GitHub Actions summary presents aggregate estimator
behavior and prioritizes estimator conflicts for visual review.

The workflow persists the compact assessment, recommendation, and proposed
policy in the results repository. This removes artifact hand-carrying and makes
the decision reproducible. It does not resample or publish normalized images.

The remaining human boundary is intentional and simple: review the recommendation,
then run collection normalization with either the default prepared recommendation
or `axis-aligned-only`. This is approval of a pixel-changing operation, not a
request to assemble the pipeline by hand.
