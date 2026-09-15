# HTH Canonical Build Evidence

Canonical Build Evidence (CBE) is the deterministic execution contract shared by
HTH collection preprocessing and future processing stages such as normalization.
It answers two separate questions:

1. Are the complete effective inputs and process contract identical to a proven
   build?
2. Do the current canonical domain results exactly match the proven results?

The production preprocess workflow persists its strict, identity-keyed evidence
store at `metadata/canonical-build-evidence.json` in the collection results
repository. Each effective identity has its own canonical record, so validated
software/runtime and process identities do not overwrite one another.
There is intentionally no legacy adapter. A publication without complete v1 CBE
is unproven and must execute once to establish its baseline.

## Identities

The **Effective Build Identity** is a SHA-256 over canonical JSON containing:

- the operation contract, execution mode, and image limit;
- source repository, immutable release, release-manifest SHA-256, source commit,
  and the verified SHA-256 and size of every selected DOCX;
- configuration-file identities;
- implementation-file identities;
- runtime setup/lock-file identities;
- Python implementation/version/ABI and exact relevant package versions; and
- the resolved preferred-detector calibration and parameter identity.

The runner name, environment, operating system, and architecture are retained as
observational execution provenance but are deliberately excluded from the
Effective Build Identity. The same effective build is therefore reusable across
GitHub-hosted and self-hosted runners. A cross-runner `force-verify` compares the
canonical result and fails if a host difference changes it.

The pipeline commit and workflow run are also retained as execution provenance.
They are not blanket invalidators because the hashed implementation,
configuration, runtime contracts, and operation declaration identify the
effective process.

The **Canonical Result Identity** is a SHA-256 over the canonical published JSON
surfaces and page-result identities. Timestamps, timings, and duplicated source or
pipeline provenance are excluded from domain-result comparison. They remain
available as execution metadata.

Each page records its operation identity, canonical extracted-image SHA-256,
canonical analysis SHA-256, activity, and domain result. These page boundaries are
the forward-compatible handoff for normalization; this initial implementation
does not introduce a page-asset cache.

## Policies

- `auto`: validate exact evidence and its published artifacts. Reuse the proven
  result when no new complete build artifact was requested. Missing or genuinely
  changed effective inputs execute and establish a new baseline. An exact identity
  whose persisted artifacts are missing or corrupt fails closed.
- `audit`: require and validate exact persisted evidence without processing.
- `force-verify`: require exact persisted evidence, execute the complete process,
  compare canonical results, and fail on any discrepancy. The incumbent evidence
  is not replaced on failure.
- `rebuild`: execute only when effective inputs intentionally changed. It refuses
  to replace an unchanged identity; use `force-verify` for that case.

Requesting the temporary complete generated-build artifact requires execution
because the results repository intentionally does not persist full-resolution raw
and derived images. Disable that request when the authoritative metadata and
analysis publication is sufficient and `auto` reuse is desired.

## Execution vocabulary

Activity and domain result are recorded independently:

- `EXECUTED / APPLY`: processing ran and produced the canonical result.
- `REUSED / SKIP`: an exact proven result made page processing unnecessary.
- `EVALUATED / SKIP`: an audit validated the evidence without processing.
- `EXECUTED / ERROR`: a page execution did not produce a valid domain result.

Any same-identity/different-result event is a determinism or integrity failure and
requires investigation. It must never silently become a new authoritative result.
