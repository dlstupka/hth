# HTH Canonical Build Evidence

Canonical Build Evidence (CBE) is the deterministic execution contract shared by
HTH collection preprocessing and downstream stages such as normalization.
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
- the explicitly declared preprocessing implementation files;
- the selected detector implementation and its transitive local detector
  dependencies;
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

Source identities use explicit **logical source paths**. Workflows map physical
checkout or temporary paths to stable semantic names with
`--source-input LOGICAL_PATH=PHYSICAL_PATH`; the physical staging layout is not
hashed. Moving a file between checkout directories or consolidating reusable
workflow YAML therefore cannot invalidate CBE, while changing its logical role,
content, or digest still does. Directory mappings use `.` as the logical root
when retaining an established tree contract.

Fixed per-scope operation contracts live in `hth.canonical_build_evidence`, next
to the scope and artifact registries. Workflows may omit those operations or
repeat the exact registered contract for readability; any divergent declaration
fails closed as operation-contract drift. The normalization transform scope has
an explicit finite set of allowed recipe variants. Workflow refactors must not
rename logical inputs or operation contracts merely to match generic step names.

Every CBE plan and completed record also carries a `resource_utilization`
section. It records whether the identity-keyed CBE cache lookup hit or missed,
whether the build reused, restored, validated, populated, or verification-refreshed that
entry, and the exact immutable source release consumed by the build. Resource
provenance has its own contract version inside the Effective Build Identity, so
adding this lifecycle contract causes one intentional seed build instead of
silently reusing evidence that never recorded its resource use.

`hth.resource_lifecycle` scans all registered CBE stores and an optional release
inventory to produce `metadata/resource-lifecycle.json`. Authoritative cache
entries and releases are clean. Superseded entries are marked dirty, but
historical CBE and releases referenced by historical audit evidence are retained
and are not cleanup eligible. Only inventoried releases with no authoritative
or historical CBE reference are marked cleanup eligible. The report is
advisory: it never deletes a cache entry or release.

Resource labels are deliberately independent. Integrity is `good`, `bad`, or
`unknown`; lineage is `current`, `previous`, `superseded`, or `unreferenced`;
GitHub publication is `latest` or `not-latest`; and lifecycle state is `clean`
or `dirty`. Thus a release may correctly be both `previous` and `latest`, or
`current` and `not-latest`. `bad` is reserved for a failed integrity check and
is never inferred from age. `clean` requires good, current, non-draft state, so
an unknown-integrity release is dirty without being bad. Mirror releases use the same classifier, with tags
declared by current detector code treated as current roots. Release-backed
learned-evidence caches are also inventoried and labeled, but remain cleanup
ineligible until their regression/optimization utilization ledger provides the
authority to prove that no active or historical build references them.

The implementation boundary deliberately excludes Canonical Build Evidence
orchestration, reporting, regression, optimization, stage timing, and unrelated
detectors. Changes in those surfaces cannot alter the preprocessing algorithm's
identity. Changing a declared preprocessing implementation file, the selected
detector, or any local implementation imported by that detector does create a
new identity. This keeps invalidation conservative around executable domain
logic without turning every repository edit into a collection rebuild.

Normalization applies the same boundary explicitly. Its canonical engine is
declared as implementation; `hth/normalization_report.py` is not. Review cadence,
contact-sheet selection and rendering, HTML, CSV presentation, and Markdown are
non-canonical surfaces recorded outside `normalization-manifest.json`. Changes to
those surfaces therefore do not create a new Effective Build Identity or alter
the canonical normalization result.

The **Canonical Result Identity** is a SHA-256 over the canonical published JSON
surfaces and page-result identities. Timestamps, timings, and duplicated source or
pipeline provenance are excluded from domain-result comparison. They remain
available as execution metadata.

Each page records its operation identity, canonical extracted-image SHA-256,
canonical analysis SHA-256, activity, and domain result. These page boundaries
are the handoff used by artifact-only GS0002 normalization. The current
implementation does not introduce a page-asset cache.

## Runtime-variant publication snapshots

The results branch has one authoritative publication at a time, even though its
CBE store retains multiple effective identities. Preprocess and canonical
normalization preserve identity-keyed snapshots under
`cbe-cache/<scope>/<effective-build-identity>/`. Each snapshot carries a file
manifest with raw SHA-256 digests and the CBE canonical-result identity. It
contains the published metadata and analysis, not full-resolution preprocess
or normalized images. Snapshot creation is additive; an existing identity is
validated, not overwritten.

Under `auto`, a non-current exact identity with a complete snapshot takes the
`restore` path. The workflow validates the current publication and every
snapshot file, restores the saved publication, updates the authoritative CBE
pointer, and commits through hardened results persistence. It skips the
expensive preprocessing or normalization engine. A missing historical snapshot
falls back to a full rebuild checked against its saved canonical result; that
publication seeds the snapshot. A present but corrupt snapshot fails closed.
The first runtime switch after this change may therefore require one final
rebuild, while subsequent switches between seeded variants are metadata-only
restores.

## Policies

- `auto`: validate exact evidence and the current publication. Reuse the
  authoritative result, or restore a verified non-current variant snapshot,
  when no new complete build artifact was requested. Missing or genuinely
  changed effective inputs execute and establish a new baseline. A missing
  historical snapshot triggers a verified rebuild; a corrupt current
  publication or present-but-corrupt snapshot fails closed.
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
