# Hidden Texas History — Project Status

*Last updated: 2026-07-18*  
*Framework release: v0.6*  
*AutoHotkey acquisition utility: v0.5*

## Mission

Build an open, reproducible digital-humanities framework for preserving,
analyzing, and publishing Texas historical records, beginning with the San
Fernando Cathedral baptism registers.

## Current collection

**HTH-0001 — San Antonio Baptisms, 1788–1824 and 1858–1898**

- 929 FamilySearch source images represented in DOCX masters;
- replacement captures for early pages;
- stable acquisition and preprocessing workflow;
- structured manifests, page analysis, geometry candidates, and validation.

## Framework status

Completed foundation:

- reusable production, test, and calibration workflow core;
- preprocess and physical-page analysis stages;
- detector registry and detector resilience;
- contour, RANSAC, and Hough candidate detectors;
- physical-geometry validation and reference tooling;
- production and test publication layouts;
- Publication Manifest and Pipeline Health;
- canonical stage banners, UTC timestamps, and elapsed-time reporting;
- unit tests and v0.6 architecture documentation.

## Current pipeline

```text
Acquisition
→ STAGE_PREPROCESS
→ STAGE_DETECT_CURRENT
→ STAGE_DETECT_CANDIDATES
→ STAGE_VALIDATE_GEOMETRY
→ STAGE_VALIDATE_OUTPUTS
→ STAGE_PUBLISH_PRODUCTION / STAGE_PUBLISH_TEST
```

## Next technical work

1. Establish representative stage-runtime expectations.
2. Expand the detector registry with LSD, GrabCut, Connected Components, and Consensus.
3. Benchmark detector quality and runtime against the approved reference collection.
4. Advance OCR, transcription, translation, record extraction, historical reasoning, and publication.

## Historical objective

Locate, document, and contextualize the baptism of Juana Navarro Alsbury while
creating reusable tools for transparent Texas historical research.
