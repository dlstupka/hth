# Orli Page-Mask Detector

`orli_page_mask` uses the managed Orli historical-document base model to produce ordered segmentation evidence and converts that fixed learned evidence into an HTH page mask. The neural model is not trained or modified by calibration. HTH calibration parameters operate only on the deterministic evidence returned by the fixed model.

The expensive neural inference stage is therefore parameter-invariant. For a given model, resized Golden Set page, and inference contract, HTH can safely reuse the same Orli evidence for every calibration parameter set and every execution shape. The regression driver still creates a run-local shared-evidence manifest for worker fan-out, but that manifest may now be hydrated from the persistent results-repository evidence store instead of rerunning Orli.

See [Orli learned-evidence persistence](orli-evidence-persistence.md) for the persistent identity, evidence index, repository layout, and invalidation rules.


## Parameter liveness

The completed 10,000-set HTH-0001 characterization classified `close_kernel_fraction` and `fill_holes` as empirical zombies for the current Golden Set/grid. Normal `exhaustive` calibration pins them at `0.0` and `0`, respectively, and searches the four live dimensions. The explicit `exhaustive-with-zombies` strategy restores their retained historical domains and reconstructs the original 10,000-set space for deliberate revalidation. See [Detector Parameter Liveness Audit](parameter-liveness-audit.md).
