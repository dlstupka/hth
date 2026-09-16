# Document normalization

`HTH normalize GS0002` is the artifact-only validation workflow for HTH's
first production normalization operation. It applies the selected
`axis-aligned-detector-envelope-v1` policy to the 18 immutable
`HTH-GOLDEN-0002` images.

The policy takes the enclosing axis-aligned rectangle of the preferred
document detector's stored quadrilateral. It performs a pixel slice only and
writes the result as lossless PNG. It does not rotate, warp, resize, enhance,
or binarize the source image.

## Canonical preprocess handoff

Normalization does not run a document detector. The workflow checks out the
published image manifest, page analysis, and Canonical Build Evidence from the
collection results repository. It validates the authoritative evidence record
and every canonical published JSON artifact before processing.

For every GS0002 page, the immutable release image SHA-256 must agree with all
three existing authorities:

- the approved Golden Set image identity;
- the canonical preprocess image manifest; and
- the corresponding Canonical Build Evidence page record.

The crop is rejected if any identity differs or the stored preferred detector
geometry is missing. This makes normalization a strict downstream consumer of
the proven preprocess result rather than a second inference path.

## Review artifact

The uploaded `hth-gs0002-normalized-*` artifact contains:

```text
normalization-manifest.json
normalization-manifest.csv
preprocess-evidence.json
geometry-evidence.json
summary.md
index.html
normalized/
contact-sheets/
```

The normalization manifest records the half-open crop coordinates, source and
output dimensions, byte and pixel hashes, detector identity, canonical
preprocess identities, normalization policy identity, and canonical
normalization-result identity. Each PNG is decoded after writing and compared
pixel-for-pixel with the in-memory crop.

The workflow is intentionally artifact-only. It does not publish normalized
images or mutate preprocess, calibration, or runtime intelligence. After its
review artifact is approved, the same operation can be extended to the full
collection with its own Canonical Build Evidence scope.

Deskew, perspective correction, tonal correction, and binarization remain
separate future normalization operations with independent evidence contracts.
