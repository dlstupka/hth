# Hidden Texas History — Project Status

*Last updated: 2026-08-23*

## Mission

Build an open, reproducible historic-document research framework for preserving, analyzing, and publishing difficult primary-source collections, beginning with the San Antonio baptism registers and the research surrounding Juana Navarro Alsbury.

## Current reference collection

**HTH-0001 — San Antonio Baptisms, 1788–1824 and 1858–1898**

- 11 source DOCX masters preserved in immutable source release `HTH-SOURCE-0001`;
- 928 embedded source images processed in the current production corpus;
- frozen five-page Golden Set `HTH-0001` with immutable identity and SHA-256;
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

The current approved document detector is **Fusion Gen3 — AMSRE + Doc-UFCN** (`amsre_doc_ufcn_fusion`) resolved from authoritative calibration evidence for `HTH-0001`.

The first successful full production build processed all **928/928 pages**, produced **928/928 detector candidates**, recorded **0 detector errors / 0 missing candidates**, and measured **0.920780 average detector confidence** across the corpus. Confidence is a prioritization signal, not IoU or ground truth.

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

The next source-quality objective is to reacquire the current FamilySearch collection through authorized FamilySearch API/image access at the best available source quality, eliminating browser/AHK capture from the preferred acquisition path while retaining the original source edition as provenance.

## Next technical work

1. Complete authorized direct-source acquisition for the reference collection.
2. Compare direct-source production inference with the bootstrap source edition.
3. Use full-collection evidence—especially low-confidence and detector-disagreement pages—to decide whether and how to instantiate `HTH-0002`.
4. Continue downstream transcription, translation, indexing, citation, and historical-research stages.
5. Keep collection-specific data and immutable source truth outside the reusable HTH engine so additional collections can use the same framework.

## Historical objective

Locate, document, and contextualize the baptism and life of Juana Navarro Alsbury while creating reusable tooling for transparent Texas historical research.
