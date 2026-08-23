# HTH Acquisition Pipeline Design

## Goal

Use the best authorized archival representation available and keep acquisition independent from downstream processing.

Preferred path:

```text
authorized archival source
        ↓
source-specific acquisition adapter
        ↓
verified immutable source images
        ↓
manifest + SHA-256 provenance
        ↓
source release / durable storage
        ↓
HTH preprocessing
```

The original bootstrap path used browser/AHK capture through DOCX masters. Those captures remain useful recovery and provenance evidence, but they are not the preferred acquisition architecture.

## Legal and custodial boundary

HTH does not bypass repository controls. Automated retrieval must use an official download mechanism, approved API/integration, or other permission granted by the source provider or record custodian. If an authenticated source exposes metadata but not an authorized image representation, the acquisition adapter stops rather than falling back to scraping.

## Preferred FamilySearch adapter

For FamilySearch collections, prefer `tools/acquire-familysearch-images.py` over AHK/browser capture. See [`familysearch-acquisition.md`](familysearch-acquisition.md).

The adapter is designed to:

- authenticate through FamilySearch-supported access;
- follow FamilySearch Historical Records image resources;
- store verified source images directly beneath `images/`;
- never overwrite an existing logical image name;
- validate downloaded image structure and compare dimensions, size, or checksums when FamilySearch metadata makes those checks possible;
- record FamilySearch image identity, persistent identifiers, SHA-256, dimensions, media type, and sanitized provenance; and
- resume safely after interruption.

HTH is currently pursuing FamilySearch developer/API access so this authorized direct-source path can replace the bootstrap AHK capture for the San Antonio baptism reference collection.

## Normalized acquisition package

A source adapter should produce a durable package equivalent to:

```text
images/
├── fs_0001.<ext>
├── fs_0002.<ext>
└── ...
source-manifest.json
checksums.sha256
acquisition-info.yaml
```

Each manifest entry should carry a stable HTH logical image identity, source system and persistent identifier, source sequence, filename, SHA-256, acquisition method/time, MIME type, dimensions, and rights/access status where known.

## Source editions and replacement renditions

Do not silently replace established source evidence. When a materially better source rendition becomes available, publish it as a new immutable source edition or rendition while retaining the prior source identity and provenance.

For the current collection, the existing DOCX-based source release remains reproducible evidence. A future FamilySearch-direct acquisition should receive a new source release identity rather than mutating `HTH-SOURCE-0001`.

Stable logical image IDs should remain independent of source-file format so annotations, Golden Set truth, citations, and downstream research can survive a source-rendition upgrade.

## Execution boundary

Acquisition and analysis are separate contracts. HTH preprocessing should consume a verified source package without caring whether its images originated from FamilySearch, another archive API, a local archival scan, or a legacy capture. This keeps source stewardship replaceable without redesigning detector, transcription, translation, indexing, or citation stages.
