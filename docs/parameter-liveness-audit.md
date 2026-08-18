# Detector Parameter Liveness Audit

This audit was introduced after Orli's completed 10,000-set calibration exposed two dimensions whose measured effect was effectively nil. The purpose is to keep routine exhaustive calibration focused without deleting the ability to challenge a prior conclusion later.

## Current audit result

All detector configurations in `config/detectors/` were reviewed for the new liveness contract. The audit is deliberately evidence-conservative: fixed values, baseline-only controls, and dimensions previously collapsed during a detector refinement are **not** automatically called zombies. Those controls may be intentionally pinned for architectural or generation-specific reasons, and the current source snapshot does not necessarily retain the complete historical value domain needed for an honest reanimation search.

The only detector promoted to explicit zombie metadata in this audit is `orli_page_mask`:

| Parameter | Default exhaustive behavior | Retained zombie domain | Audit evidence |
|---|---|---|---|
| `close_kernel_fraction` | pinned at baseline `0.006` | `0.0, 0.003, 0.006, 0.012, 0.024` | completed 10,000-set HTH-0001 exhaustive run; η² `0.0000`, Avg-IoU range `0.0001` |
| `fill_holes` | pinned at baseline `1` | `0, 1` | completed 10,000-set HTH-0001 exhaustive run; η² `0.0000`, Avg-IoU range `0.0000` |

The ordinary Orli live Cartesian grid is therefore 1,000 combinations with both zombie controls pinned at their baseline values. `exhaustive-with-zombies` restores the original 10,000-combination Orli space.

## Framework-wide audit observations

Several refined detectors already contain deliberately fixed dimensions that earlier calibration work described as dormant, including the fusion refinements, Page Background, and Segment-Supported Polar Voting. They remain fixed exactly as before. This audit does **not** manufacture historical value domains for them or relabel them as zombies without retained evidence. If a future refinement wants those controls to participate in `exhaustive-with-zombies`, their prior domains can be added explicitly under `zombie_parameters` with an audit scope and reason.

The audit utility checks every detector configuration for malformed or ambiguous zombie metadata: a zombie cannot also be a live parameter, must retain a non-empty value domain, must declare a pinned value present in that domain, must remain present in the baseline for reproducibility, and must record both audit scope and reason. It reports singleton and baseline-only dimensions for human review but makes no behavioral inference from them.

## Policy

Normal `exhaustive` means exhaustive over the detector's **current live declared space**, not over every parameter value ever tried in the detector's history. `exhaustive-with-zombies` is the explicit forensic/revalidation mode. This keeps default calibration honest and efficient while preserving the ability to regress on deceased controls if someone insists.
