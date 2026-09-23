# Hidden Texas History — Project Status

*Last updated: 2026-09-23*

## Mission

Build an open, reproducible historic-document research framework for preserving, analyzing, and publishing difficult primary-source collections, beginning with the San Antonio baptism registers and the research surrounding Juana Navarro Alsbury.

## Current reference collection

**HTH-0001 — San Antonio Baptisms, 1788–1824 and 1858–1898**

- legacy source edition `HTH-SOURCE-0001` retains its 11 DOCX masters and original 928-page build provenance;
- current `HTH-SOURCE-0002` source edition contains 929 images;
- the original five-page `HTH-0001` and the 18-page `HTH-GOLDEN-0002` are separate frozen Golden Sets with immutable identities;
- source, pipeline, detector, parameter-set, publication, and run provenance recorded in machine-readable outputs.

The original corpus was bootstrapped through browser/AHK capture. HTH now prefers direct authorized archival acquisition and includes a FamilySearch API acquisition adapter pending developer/API access.

## Production status

The production preprocess path is operational end to end:

```text
immutable source release
→ STAGE_PREPROCESS
→ STAGE_DETECT_CURRENT
→ authoritative approved detector inference
→ STAGE_VALIDATE_OUTPUTS
→ STAGE_PUBLISH_PRODUCTION
→ durable results + temporary full-build artifact
```

Detector selection is Golden-Set-specific. The original `HTH-0001` production
build selected **Fusion Gen3 — AMSRE + Doc-UFCN**
(`amsre_doc_ufcn_fusion`); the verified `HTH-GOLDEN-0002` production build
selected **Doc-UFCN Page-Mask** (`doc_ufcn_page_mask`) from its own approved
calibration evidence.

The first successful full production build on the original source edition
processed **928/928 pages**, produced **928/928 detector candidates**, recorded
**0 detector errors / 0 missing candidates**, and measured **0.920780 average
detector confidence**. The later verified `HTH-SOURCE-0002` production run
processed **929/929 pages** with **0 page-processing and detector errors**.
Confidence is a prioritization signal, not IoU or ground truth.

## Calibration and execution intelligence

HTH maintains separate research/calibration and production concerns:

- detector regression and parameter-search evidence are persisted separately from production outputs;
- Golden Set identities are frozen rather than edited in place;
- calibration selection resolves the strongest **Approved** authoritative detector/configuration;
- runtime/parallelism history supports execution-shape recommendations;
- shared hardened persistence protects results-repository writers from concurrent non-fast-forward races;
- smoke, calibration, optimizer, report, and preprocess writers use the same persistence contract.

Detector research has exercised the framework through hundreds of regression runs and more than one hundred optimizer runs. The goal is now to consume that evidence rather than continuously retune production without a new failure class.

## Source acquisition

Large source masters are distributed through immutable GitHub Releases rather than Git LFS. Builds verify release-manifest hashes before processing.

The current `HTH-SOURCE-0002` edition and its frozen Golden Set are distinct
from the original bootstrap edition. Further source-quality work should verify
the best available authorized FamilySearch API/image access and retain both
source editions as provenance rather than replacing either in place.

## Next technical work

### Hardening verification TODO

- [ ] Live-test CBE runtime-variant restoration for both full preprocess and
  canonical normalization with an A → B → A runtime sequence (including exact
  Python and NumPy versions). Confirm the final A run chooses `restore`,
  verifies the saved snapshot, republishes the A result without rerunning the
  expensive engine, and reports the correct CBE decision, runtime difference,
  and result identity. A first switch that must seed a missing snapshot is not
  sufficient evidence of a cache hit.

1. Finish the provenance-link coverage audit. Shared Markdown link helpers and
   report navigation now cover the main Actions summaries and reports, but some
   detailed report surfaces still render identities as plain code (for example,
   `hth/normalization_report.py`). Bring remaining human-facing commit,
   release-backed result, and evidence-identity links onto the shared helpers;
   keep raw deterministic values in machine-readable JSON/CSV.
2. Complete authorized direct-source acquisition for the reference collection.
3. Compare direct-source production inference with the bootstrap source edition.
4. Review full-collection low-confidence and detector-disagreement pages for
   genuinely new failure classes; any change to frozen `HTH-GOLDEN-0002` truth
   or membership requires a new Golden Set identity and release.
5. Continue downstream transcription, translation, indexing, citation, and historical-research stages.
6. Keep collection-specific data and immutable source truth outside the reusable HTH engine so additional collections can use the same framework.

## Deferred engineering triggers

These are intentional watch conditions, not currently justified projects:

- Revisit Results-repository publication architecture only if concurrent-write
  collisions become routine, exhaust the existing bounded retry transaction,
  or materially delay publication. The current collision-safe persistence
  contract remains appropriate at present scale.
- Revisit detector execution scheduling when detector count, parameter-space
  size, measured runner contention, or utilization demonstrates that the
  current scheduler no longer uses available capacity effectively.
- Add further persistence, checkout, or source/Golden-Set hardening in response
  to a concrete new failure class, a changed workflow contract, or a new
  source-release/Golden-Set model rather than speculative edge cases.

The earlier shared local-Git test-fixture refactor and Windows cleanup work are
complete. Commit `6fb44f5` centralized the fixtures, made read-only Git-object
cleanup explicit, removed leaked temporary repositories, and added bounded
retries for the observed Windows `device or resource busy` deletion race.

## Historical objective

Locate, document, and contextualize the baptism and life of Juana Navarro Alsbury while creating reusable tooling for transparent Texas historical research.
