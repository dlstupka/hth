# HTH Documentation

This directory contains the design, operating, and project-reference documentation for the Hidden Texas History Research Framework. Start with the repository-level [README](../README.md) for the project overview and current pipeline.

## Project and architecture

- [Architecture](architecture.md) — system boundaries and durable contracts.
- [Project status](project-status.md) — current implementation and research status.
- [Workflow architecture](workflow-architecture.md) — GitHub Actions structure and workflow responsibilities.
- [Development](development.md) — development, testing, and update-package practices.
- [Toolchain](toolchain.md) — supported execution environments and dependencies.

## Acquisition and preprocessing

- [Acquisition pipeline](acquisition-pipeline.md) — source-image capture and ingestion.
- [Preprocessor](preprocessor.md) — DOCX extraction and normalized publication inputs.
- [Physical-page analysis](analyze-pages.md) — page analysis and review-queue generation.

## Geometry and detectors

- [Multi-detector geometry](multidetector-geometry.md) — detector registry and geometry pipeline.
- [Physical geometry evaluation](physical-geometry-evaluation.md) — geometry validation and evaluation rules.
- [Detector components](README-detector-components.md) — connected-components detector.
- [Contour detector](README-detector-contour.md) — contour-based detector.
- [GrabCut detector](README-detector-grabcut.md) — GrabCut-based detector.
- [Line Segment Detector](README-detector-lsd.md) — LSD-based detector.

## Calibration and regression

- [Golden Set](golden-set.md) — approved references and evaluation inputs.
- [Calibration selection](calibration-selection.md) — selection and promotion of calibrated parameters.
- [Detector regression](regression.md) — regression execution, telemetry, reports, and debug artifacts.

## Publication and review tools

- [Publication](publication.md) — publication layout, provenance, and outputs.
- [Reference collection editor](reference-collection-editor.md) — single-detector review tool.
- [Multi-detector reference collection editor](reference-collection-editor-multidetector.md) — multi-detector review tool.
