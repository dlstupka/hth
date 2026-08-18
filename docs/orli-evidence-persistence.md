# Orli Learned-Evidence Persistence

Orli neural inference is expensive and deterministic for the same model and page input. HTH therefore persists `orli_page_mask` learned evidence in the results repository so that model inference is performed once for a compatible evidence identity and reused by later optimizer shapes, regressions, and builds.

## Execution contract

The detector has two distinct stages:

1. Orli inference converts each Golden Set image into immutable learned evidence.
2. HTH calibration parameters convert that evidence into candidate page masks and score them.

The current calibration space does not alter the Orli model, its weights, or its inference configuration. Calibration therefore must not cause repeated neural inference.

During a learned run, `tools/run-detector-regressions.sh` asks `hth.regression.learned_evidence` to prepare shared evidence before pipeline fan-out. For Orli, the preparer first computes the evidence identity and checks the results-repository store. A compatible persisted artifact is copied into the run-local `.learned-evidence/orli_page_mask/manifest.json` and no Orli model inference occurs. On a cache miss, Orli inference runs once per Golden Set page, the resulting manifest is persisted, and workers consume the same manifest.

## Persistent identity and invalidation

An Orli evidence identity includes the values that can change the deterministic neural output:

- detector id;
- Orli model id and model SHA-256;
- Orli package version and declared inference contract;
- Golden Set file SHA-256;
- maximum input dimension;
- ordered image keys computed from the actual loaded/resized Golden Set images; and
- evidence representation version.

The identity is serialized canonically and hashed to produce `evidence_id`. Any change to the model, Golden Set content, loaded page pixels, resize dimension, Orli/inference contract, or evidence representation produces a different id and therefore a cache miss. Downstream calibration parameters are intentionally absent because they do not change inference.

## Results-repository layout

Persistent evidence is ordinary inspectable JSON:

```text
orli-evidence-index.json
learned-evidence/
  orli_page_mask/
    <evidence-id>/
      manifest.json
```

No additional compression or decomposition is applied. HTH records the actual manifest byte size in the index; storage changes should be driven by observed evidence size rather than pre-optimization.

## Orli evidence index

`orli-evidence-index.json` is the repository-level inventory of reusable Orli inference artifacts. Each entry records:

- evidence id and repository-relative manifest path;
- manifest SHA-256 and byte size;
- creation timestamp;
- model id, model SHA-256, and Orli version;
- Golden Set SHA-256 and maximum dimension;
- page count; and
- ordered page image keys.

The index is reconstructible from persisted manifests. Publication rebuilds it after synchronizing the results repository so concurrent results-repository updates can be merged without treating the index itself as authoritative state.

## Publication and concurrency

New evidence is written into `results-repo/learned-evidence/orli_page_mask/` as soon as inference completes. This allows later execution-optimizer shapes in the same build to reuse it immediately. The regression and optimizer publication steps stage the evidence directory and rebuilt index with the normal results-repository commit.

Before publication, the results repository is reset to the latest remote `main`. New content-addressed evidence directories are deliberately outside the scoped cleanup list, so a newly generated local artifact survives that refresh. HTH then rebuilds the index against the combined latest-remote and local evidence set before committing. If another build has already published the same evidence id, the refreshed remote artifact becomes authoritative.

Concurrent builds that both miss the same not-yet-published evidence may each perform inference once; persistence eliminates repeated work after the first compatible artifact reaches the shared results repository without introducing a distributed lock.

## Diagnostics

The learned-evidence log explicitly reports whether Orli evidence came from persisted storage or fresh inference. Persistent reuse emits `PERSISTENT CACHE HIT`; fresh evidence emits `PERSISTED` with the short evidence id, page count, byte size, artifact path, and index path. `SHARED EVIDENCE READY` includes `source=persisted` or `source=inference`.
