# Fusion Gen3 — AMSRE + Doc-UFCN

`amsre_doc_ufcn_fusion` is the current HTH production detector for Golden Set `HTH-0001`. It is the third fusion generation and deliberately combines two detectors with different strengths rather than asking either family to solve every page alone.

## Why this fusion exists

AMSRE provides strong explicit geometric boundary evidence and remains the primary child. Doc-UFCN contributes an independently learned historical-document page model. Their failure modes are complementary: when AMSRE is already well supported, learned evidence should not displace it; when AMSRE is weak or unavailable and Doc-UFCN has strong contradictory evidence, the learned detector can rescue the page.

That is the central HTH fusion principle: **combine independent evidence to cover one detector's weakness with another detector's strength, while preserving measurable arbitration rather than blindly averaging outputs.** Gen3 is the clearest payoff from that approach so far.

## Calibrated children

Gen3 executes fixed calibrated child winners:

- Adaptive Multi-Scale Radial Edge (`adaptive_multi_scale_radial_edge`) — parameter set `21ea516c3c5a`.
- Doc-UFCN Page Mask (`doc_ufcn_page_mask`) — parameter set `595002645fcc`.

The fusion layer then calibrates only its arbitration controls, keeping child behavior reproducible and making the fusion experiment attributable.

## Arbitration strategy

AMSRE is primary. Doc-UFCN is selected when AMSRE is unavailable, or when all calibrated rescue gates agree that learned evidence is strong enough and the two candidates disagree enough to justify replacement.

The four fusion controls are:

- `amsre_rescue_score_ceiling` — limits rescue to weaker AMSRE results.
- `doc_ufcn_minimum_confidence` — requires sufficiently strong learned evidence.
- `minimum_corner_disagreement_fraction` — requires meaningful geometric disagreement.
- `maximum_amsre_refined_support_fraction` — prevents rescue when AMSRE refinement already has strong measured support.

Diagnostics preserve both child candidates, child calibration identities, disagreement geometry, rescue-gate outcomes, the selected child, and the final decision. Verbose output renders the two child quadrilaterals and final selected geometry together.

## Current production result

The persisted authoritative calibration for `HTH-0001` selects Fusion Gen3 as Rank #1. Production preprocessing resolves that approved calibration rather than embedding a hand-picked parameter set in the workflow.

The full-collection production run on 928 pages completed with 928 candidates, zero detector errors, and average detector confidence approximately `0.921`, providing a broader operational check beyond the frozen Golden Set.

## Next step: automatic calibration

The current calibration system already records discrete parameter domains, Golden Set truth, parameter influence, exhaustive/contracted searches, optimizer results, and persisted authoritative winners. A natural next step is **automatic calibration**: allow HTH to select an appropriate search scope, execute calibration when a detector or Golden Set changes, evaluate the evidence, and promote a new authoritative calibration only when its validation contract is satisfied.

Autocalibration should build on the existing reproducibility rules rather than bypass them: frozen truth, stable parameter identities, explicit search provenance, comparable metrics, and reviewable promotion evidence remain required.
