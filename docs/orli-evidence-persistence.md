# Orli Learned-Evidence Persistence

Orli uses the repository-wide [canonical learned-evidence cache](learned-evidence-cache.md). Its deterministic model/page evidence is resolved from the runner-local bundle cache or the collection cache release repository before inference is considered.

The original implementation stored Orli manifests beneath `learned-evidence/orli_page_mask/` in the results repository and indexed them with `indexes/orli-evidence-index.json`. Those paths are now a read-only migration compatibility source. New regression and optimizer workflows do not checkout, modify, or publish them.

`tools/migrate-results-models.py --seed-evidence-cache` validates the legacy index and manifests, packages them under their existing canonical evidence identities, publishes immutable cache releases, and verifies the round trip. `--verify-evidence-cache` provides a non-mutating audit before the legacy results-repository copies are removed.

The already-published Orli identities remain compatible because the repository-wide cache preserves the original Orli identity schema, release tags, and asset names.
