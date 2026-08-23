# HTH Reference Collection Editor

This candidate preserves the full measurement loop for every approved page: the original Stage 2 prediction, approved box, quality fields, source/pipeline commits, exact deltas, IoU, accepted-versus-modified status, review time, editor version, originating paths, and available build identity.

It exports:

- `reference_collection.json` — curated references plus predictions, corrections, provenance, and summary metrics.
- `reference-metrics.json` — accepted/modified counts, acceptance rate, mean/median IoU, mean edge corrections, and per-page details.

Before accepting editor changes, verify export/reload, unchanged and modified predictions, stable metrics, Golden Set validation, and a clean test publication.
