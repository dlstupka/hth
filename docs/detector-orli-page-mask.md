# Orli Page-Mask Detector

`orli_page_mask` uses the managed Orli historical-document base model to produce ordered segmentation evidence and converts that fixed learned evidence into an HTH page mask. The neural model is not trained or modified by calibration. HTH calibration parameters operate only on the deterministic evidence returned by the fixed model.

The expensive neural inference stage is therefore parameter-invariant. For a given model, resized Golden Set page, and inference contract, HTH can safely reuse the same Orli evidence for every calibration parameter set and every execution shape. The regression driver still creates a run-local shared-evidence manifest for worker fan-out, but that manifest may now be hydrated from the persistent results-repository evidence store instead of rerunning Orli.

See [Orli learned-evidence persistence](orli-evidence-persistence.md) for the persistent identity, evidence index, repository layout, and invalidation rules.
