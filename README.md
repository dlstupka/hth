# Hidden Texas History (HTH)

**Hidden Texas History (HTH)** is an open-source historic-document research framework for turning archival source material into reproducible, reviewable research evidence.

HTH is designed for difficult historical collections: photographed manuscripts, bound registers, irregular page geometry, degraded images, handwritten records, and source material that must retain a clear chain of provenance. The framework separates source preservation, processing, calibration, validation, publication, and later research stages so that every result can be traced back to the exact source material and software that produced it.

The current reference collection is **HTH-0001 — San Antonio Baptisms, 1788–1824 and 1858–1898**, but the pipeline and detector framework are intentionally collection-independent.

## What HTH does

HTH provides a reproducible pipeline for:

- acquiring or importing archival source material while preserving source identity and provenance;
- extracting and normalizing document images;
- detecting the physical page or document region in difficult captures;
- calibrating and comparing classical, learned, and fused document detectors;
- validating detector behavior against frozen Golden Sets;
- recording machine-readable evidence, diagnostics, timing, and build provenance;
- publishing curated production results separately from test and calibration history; and
- providing a foundation for transcription, translation, indexing, citation, and historical research.

The project favors **preserved evidence over destructive cleanup**. Source material remains identifiable and reproducible while derived images, geometry, analysis, and research outputs are generated as separate artifacts.

## Current pipeline

```text
authorized archival source
        ↓
source release + manifest + SHA-256 provenance
        ↓
STAGE_PREPROCESS
        ↓
STAGE_DETECT_CURRENT
        ↓
approved detector inference
        ↓
STAGE_VALIDATE_OUTPUTS
        ↓
STAGE_PUBLISH_PRODUCTION / STAGE_PUBLISH_TEST

detector research/calibration (separate path)
        ↓
STAGE_DETECT_CANDIDATES → Golden Set evaluation → persisted calibration intelligence
        ↓
transcription → translation → indexing → citation → research
```

Stage names are stable HTH concepts rather than GitHub Actions implementation details. The same pipeline architecture can therefore be executed by GitHub-hosted, self-hosted, or future execution environments without changing the research model.

## Document detection and calibration

HTH currently carries **47 registered document/page detectors** spanning classical geometry, learned historical-document segmentation, OpenCV primitives, and three generations of fusion. The authoritative inventory is the [Document Detector Catalog](docs/detector-catalog.md).

Historic document images are rarely clean scans. Pages may be skewed, cropped, shadowed, surrounded by viewer chrome or background material, partially obscured, or captured at inconsistent scale. HTH treats physical-page detection as a measurable research-engineering problem rather than a one-off image-processing step.

The detector framework supports multiple independent approaches, including classical image-processing methods, historical-document neural models, and detector fusions. Detectors expose their geometry, confidence, timing, diagnostics, and debug evidence through a common framework.

Calibration is evaluated against manually approved **Golden Sets** using reproducible parameter spaces and persisted calibration evidence. Regression reports track metrics such as average IoU, minimum IoU, standard deviation, failures, parameter influence, search coverage, and runtime. Production inference resolves the strongest compatible approved calibration rather than silently adopting an unverified result.

`HTH-0001`, the original five-page Golden Set, is frozen so historical calibration evidence retains a stable identity. Future Golden Sets can broaden the corpus without changing the meaning of earlier results.

## Reproducibility and provenance

HTH deliberately separates three responsibilities:

```text
Source repository
  Collection identity, source metadata, manifests, rights/provenance notes,
  and immutable source-release ownership.

Pipeline repository (this repository)
  Code, configuration, workflows, tests, detector implementations,
  calibration machinery, schemas, and documentation.

Results / publication repository
  Curated production outputs plus isolated test and calibration history.
```

Large immutable source masters are distributed as source-release assets rather than being embedded in normal Git history. Release manifests record asset names, sizes, SHA-256 hashes, source repository identity, release identity, and source commit provenance. Pipeline publications similarly record the exact source, pipeline commit, workflow run, detector calibration, and publication commit used to create them.

The goal is straightforward: **a historical conclusion should remain auditable back through the derived evidence to the source material that supports it.**

## Current reference collection

The active production collection contains **928 pages** derived from San Antonio baptismal records covering **1788–1824 and 1858–1898**. It is being used both for historical research and as the proving corpus for HTH's reusable document-processing architecture.

The broader historical objective that started the project is to locate, document, transcribe, translate, and contextualize primary-source evidence concerning **Juana Navarro Alsbury**, an Alamo survivor, while building tooling useful well beyond that individual research question.

HTH's source model uses stable collection and page identities so improved source renditions can replace older captures without discarding annotations, calibration history, citations, or research relationships.

## Archival access and source stewardship

HTH is source-neutral and does not assume that archival material is freely redistributable or available through unattended automation.

Acquisition must use access permitted by the source custodian: an official download mechanism, an approved API or integration, material supplied directly by a repository, or other authorized access. HTH records provenance and source identity so that acquisition method, rights status, and later source replacements can remain explicit rather than becoming hidden implementation details.

The framework is designed to benefit from direct archive-quality source images when authorized access is available. Lower-quality historical captures can remain as provenance while a better rendition becomes the preferred processing source.

## Repository map

| Path | Purpose |
|---|---|
| [`hth/`](hth/) | Core Python pipeline, detector, calibration, reporting, and provenance code |
| [`config/`](config/) | Pipeline, detector, Golden Set, and analysis configuration |
| [`.github/workflows/`](.github/workflows/) | Preprocess, regression, calibration, optimizer, review, freeze, and reporting workflows |
| [`tests/`](tests/) | Repository regression and invariant test suite |
| [`tools/`](tools/) | Research, review, source-release, and operational tooling |
| [`docs/`](docs/) | Architecture, detector, calibration, acquisition, publication, and development documentation |

## Documentation

Useful starting points:

- **[Architecture](docs/architecture.md)** — repository responsibilities, canonical stages, reporting, and durable design principles.
- **[Workflow architecture](docs/workflow-architecture.md)** — CI/execution organization and workflow responsibilities.
- **[Preprocessor](docs/preprocessor.md)** — source extraction and preprocessing behavior.
- **[Physical geometry evaluation](docs/physical-geometry-evaluation.md)** — geometry-analysis and validation model.
- **[Regression](docs/regression.md)** — detector calibration, metrics, search spaces, and reporting conventions.
- **[Calibration selection](docs/calibration-selection.md)** — how approved detector evidence is selected for inference.
- **[Golden Set](docs/golden-set.md)** — reference geometry and frozen Golden Set policy.
- **[Document detector review](docs/document-detector-review.md)** — review workflow for detector output.
- **[Source releases](docs/source-releases.md)** — immutable source assets and verification contract.
- **[Acquisition pipeline](docs/acquisition-pipeline.md)** — source-neutral acquisition architecture and authorization boundary.
- **[Publication](docs/publication.md)** — production/test publication and provenance reporting.
- **[Project status](docs/project-status.md)** — historical project status and objectives.
- **[HTH development standard](docs/README.md)** — authoritative engineering conventions for contributors.

Detector-specific documentation is also maintained under [`docs/`](docs/) for the implemented classical, learned, and fusion approaches.

## Engineering principles

HTH is built around a few durable rules:

1. **Preserve the source.** Derived processing never substitutes silently for source evidence.
2. **Make provenance executable.** Builds record enough identity to reproduce and audit their inputs and decisions.
3. **Measure detector quality.** Golden Sets, regression, parameter calibration, and persisted evidence replace visual guesswork.
4. **Keep algorithms interchangeable.** Detector-specific behavior stays behind common interfaces so better methods can be evaluated without rebuilding the pipeline around them.
5. **Separate production from experimentation.** Smoke tests, calibration, optimizer runs, historical evidence, and production publication have distinct responsibilities.
6. **Fail explicitly.** Missing candidates, invalid geometry, incompatible calibration evidence, corrupted model artifacts, and persistence failures should be visible rather than silently accepted.
7. **Design for additional collections.** Collection-specific source configuration remains separable from the reusable HTH framework.

## Development and validation

HTH uses Python's standard `unittest` framework for repository tests. The GitHub Actions workflows exercise production preprocessing, smoke/regression calibration, detector optimization, Golden Set validation, document review, and report generation.

For repository conventions and contribution requirements, read **[`docs/README.md`](docs/README.md)** before making implementation changes.

## Project direction

Physical-page detection and reproducible preprocessing are foundational stages, not the final research product. HTH is intended to carry verified archival evidence forward into:

```text
source preservation
→ page/document normalization
→ transcription
→ translation
→ structured record extraction
→ indexing and entity linkage
→ citation
→ historical analysis
→ reproducible publication
```

The long-term objective is a transparent research framework in which both people and automated tools can assist with difficult historical records **without losing the evidence trail that makes the research trustworthy**.
