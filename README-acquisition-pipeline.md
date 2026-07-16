# HTH Acquisition Pipeline Design

## Goal

Replace the fragile path:

```text
FamilySearch viewer → screenshot/snipping → clipboard → Word → local extraction → source repo
```

with a source-neutral package:

```text
authorized image source → acquisition adapter → immutable images → manifest/checksums → storage → HTH preprocessing
```

## Legal boundary

Do not automate retrieval in a way prohibited by the image provider or record custodian. Use an official download mechanism, an approved API/integration, or written permission before unattended cloud retrieval.

## Normalized acquisition package

```text
acquisition/
├── originals/image_000001.png
├── manifest.json
├── checksums.sha256
├── acquisition-info.yaml
└── capture-log.jsonl
```

Each manifest entry should carry an HTH image ID, source system and persistent identifier, source sequence, filename, SHA-256, capture time/method, MIME type, and rights status.

## Practical transition

1. Preserve the current DOCX masters as recovery evidence.
2. Modify AHK to save each capture directly as a numbered PNG as well as, temporarily, Word.
3. Run a local acquisition agent that watches the folder, hashes images, appends the manifest, and uploads completed batches.
4. Store large immutable originals in object storage or Git LFS; keep manifests, provenance, config, and rights notes in Git.
5. Trigger cloud preprocessing after a completed batch is uploaded.

AHK can still drive the authenticated browser locally. The cloud begins immediately after each local capture, avoiding clipboard/Word as the authoritative source.

## Cloud pattern

```text
Windows capture host + AHK
        ↓
local acquisition agent
        ↓
S3 / Azure Blob / Google Cloud Storage
        ↓
GitHub Actions or self-hosted runner
        ↓
HTH source manifest + results repo
```

A self-hosted runner is useful when browser authentication or local folders must remain on your machine while GitHub still orchestrates the job.

## Trivial future source replacement

Use stable logical IDs such as `HTH-0001-I000001`, independent of the source file. Map each ID to one or more renditions and mark one preferred. When archive TIFFs or direct source images arrive, add the new rendition, preserve the screen capture, switch the preferred rendition, and rerun only stages whose input hash changed. Reference annotations, record links, and citations remain attached to the stable HTH ID.
