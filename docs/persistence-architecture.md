# Persistence Architecture

HTH treats persisted research and execution evidence as infrastructure. The persistence contract is intentionally shared across preprocessing, calibration, regression, execution optimization, report generation, and later automated calibration.

## Core rule

**Durable per-run evidence is authoritative; aggregate indexes are derived and rebuildable.**

All canonical results indexes live beneath `indexes/` and are registered in `hth.persistence`. Repository-relative paths stored inside an index are resolved from the results-repository root, never from the index directory itself. Legacy root-level indexes remain readable only as a migration compatibility path.

The canonical derived indexes are:

- `indexes/calibration-index.json`
- `indexes/multidetector-index.json`
- `indexes/optimizer-index.json`
- `indexes/orli-evidence-index.json`
- `indexes/parallelism-index.json`
- `indexes/parameter-provenance-index.json`
- `indexes/runtime-index.json`

JSON index replacement uses the same atomic-write primitive. Results-repository Git publication uses `tools/hardened-persistence.sh`, which refreshes the latest remote state, reapplies the caller-owned mutation, commits, and retries only confirmed concurrent-update collisions.

## Durable evidence

Calibration runs persist their exact calibration intelligence and provenance below `source-documents/.../calibrations/...`. Execution-optimizer runs persist run metadata, shape observations, shard observations, and runner metrics below `execution-optimizer/<detector>/runs/<run-id>/`. Multi-detector scheduling observations are preserved below `execution-history/multidetector/`. Learned Orli evidence remains independently addressable beneath `learned-evidence/orli_page_mask/`.

Lossless calibration page evidence is retained as deterministic gzip streams (`raw/results.csv.gz` and `raw/evidence.jsonl.gz`) so exhaustive runs remain below GitHub's per-blob limit. Workflow artifacts keep their run-local uncompressed files for convenient inspection. Historical readers accept both the compressed durable contract and legacy uncompressed records.

`python -m hth.persistence_rebuild --results-root <results-repo>` deletes the derived indexes and reconstructs them from those durable records. This is both a recovery mechanism and an architectural invariant: deleting an index must not delete research history.

## Execution-optimizer benchmark contract

Execution optimization measures execution shape, not detector calibration-space size. Each optimizer shape therefore evaluates a bounded canonical parameter-set workload (256 sets by default), while retaining the baseline and historic best configuration. The benchmark size and selected calibration domain are retained as informational search-scope provenance. They do not define execution-shape compatibility; stable detector evidence identity and runner/vCPU characteristics do.

A full detector calibration may still contain tens or hundreds of thousands of parameter sets; that does not make those sets part of every execution-shape benchmark.

## Source acquisition

Immutable source Releases are downloaded through the single verified source-release client. GitHub API calls authenticate with `HTH_SOURCE_TOKEN` when supplied and otherwise use the workflow `GITHUB_TOKEN`, with bounded retries for transient API/network failures. Every downloaded source asset is verified against the release manifest by byte size and SHA-256 before use.

## Artifact delivery

GitHub Actions artifacts are convenience/delivery copies, not the authoritative persistence layer. Workflows use the shared hardened artifact action, which retries transient artifact-service failures. Results-repository persistence and immutable source Releases remain the durable record.

## Architectural invariants

HTH tests enforce that:

- all derived index paths come from one registry;
- index writers cross the canonical persistence boundary instead of implementing local JSON writers;
- a missing derived calibration index can be rebuilt from persisted calibration evidence;
- workflow Git publication uses the shared collision-safe transaction;
- workflow artifact delivery uses the shared retry wrapper;
- source acquisition uses authenticated GitHub API access;
- execution optimization uses a bounded benchmark workload rather than the full calibration grid.

Autocalibration should consume this persistence boundary rather than introduce a parallel storage or selection mechanism.
