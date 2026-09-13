# Canonical Learned-Evidence Cache

Learned detector inference is deterministic and independent of calibration parameters. HTH therefore computes each compatible Golden Set evidence set once and reuses it across smoke tests, full regressions, optimizer shapes, shards, Golden Set coordinator lanes, and later builds.

The cache is a performance layer, never a source of scientific authority. Deleting it may make a run slower but must not change detector output, calibration selection, or persisted research evidence.

## Resolution order

Every learned detector uses the same resolver:

1. Reuse an already materialized run-local evidence directory.
2. Validate and hydrate the content-addressed runner-local bundle cache.
3. Download and validate the exact immutable GitHub Release from the collection cache repository.
4. On a miss or unavailable release, regenerate evidence from the verified model and Golden Set images.
5. Store the validated bundle in the runner-local cache and publish the missing immutable release when a cache token is available.

Self-hosted runners default to `/tmp/.ar/.hth-evidence-cache`; GitHub-hosted runners use a job-local directory beneath `RUNNER_TEMP`. `HTH_EVIDENCE_LOCAL_CACHE_ROOT` overrides that location and works independently in local-only or offline runs. `HTH_EVIDENCE_CACHE_REPOSITORY` additionally selects the collection release repository, and `HTH_CACHE_TOKEN` authorizes publication. Public release reads do not require a token.

Cache availability does not determine scientific success: HTH can regenerate valid evidence. A downloaded artifact is never consumed until its release metadata, bundle SHA-256, evidence manifest SHA-256, canonical identity, record order, and referenced sidecar files validate.

## Canonical identity

An evidence identity contains every input that can change deterministic neural output:

- canonical evidence-producing detector;
- model id, stable model artifact SHA-256 values, model variant, inference backend, and serving/input contract;
- Golden Set manifest SHA-256;
- maximum image dimension;
- ordered keys computed from the actual loaded and resized page pixels; and
- evidence representation contract.

The canonical JSON identity is SHA-256 hashed to produce `evidence_id`. Calibration parameters are deliberately excluded because they operate on immutable evidence after inference. The AMSRE + Doc-UFCN fusion and standalone Doc-UFCN detector share one Doc-UFCN evidence identity and bundle; the classical fusion branch is not cached.

## Release contract

The collection cache stores one immutable GitHub Release per evidence identity:

```text
tag:   HTH-EVIDENCE-<DETECTOR>-<EVIDENCE-ID>
asset: <detector>-<evidence-id>.zip
meta:  <detector>-<evidence-id>.zip.manifest.json
```

The deterministic ZIP contains `manifest.json` and only the sidecar files referenced by its records. JSON-only detectors therefore contain one file; probability-map detectors also include their `.npy` arrays. The attached release manifest records the repository, tag, asset name, detector, complete evidence identity, byte size, bundle SHA-256, evidence-manifest SHA-256, and non-authoritative trust role.

Release tags are write-once. A changed model, page set, resize contract, backend contract, or representation creates a new identity and a new release instead of overwriting prior evidence.

## Execution and timing

When one detector expands into several shards or lanes, the parent prepares or hydrates its evidence once before fan-out and every worker loads the same immutable directory. Single-task learned detectors resolve their cache inside their assigned detector pipeline, preventing a global pre-fan-out barrier from delaying unrelated classical detectors.

Detectors that consume the same canonical evidence are coalesced. In particular, Doc-UFCN and AMSRE + Doc-UFCN do not perform duplicate inference in an all-detector run.

Resolution/generation time remains visible as fixed learned-evidence preparation in runtime telemetry. It is included in end-to-end shape comparisons and never divided across shards or Golden Set lanes. Normal logs emit one concise `LOCAL CACHE HIT`, `COLLECTION CACHE HIT`, `COLLECTION CACHE MISS`, `LOCAL CACHE FILLED`, or publication status line; per-page inference progress appears when regeneration is required.

## Legacy Orli migration

`tools/migrate-results-models.py --seed-evidence-cache` publishes compatible historical Orli manifests from the results repository to immutable releases. Runtime resolution retains read-only fallback support for those legacy manifests during migration, but new workflows neither checkout nor publish learned evidence in the results repository.

After deployed workflows have demonstrated release hits, use `--verify-evidence-cache` before removing the legacy Orli files. The migration tool remains the audit and recovery path for those historical artifacts.
