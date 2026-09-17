Implemented the initial GS0002 crop/framing assessment flow

It:

- Resolves the current Rank #1 approved GS0002 detector.
- Verifies and downloads the immutable 18-image bundle.
- Compares axis-aligned crop, rotation crop, and projective framing.
- Generates PNG variants, contact sheets, an HTML gallery, JSON/CSV metrics, hashes, and transform matrices.
- Clearly labels box-retention metrics as proxies because GS0002 lacks four-corner perspective truth.
- Remains diagnostic-only with no publication or persistence changes.

Key files:

- [assess-crop-framing.yml]
- [assess_crop_framing.py]
- [crop-framing-assessment.md]
- [overlay.zip]
- [COMMIT-MESSAGE.md]

Validation: 1,010 tests passed, compilation and git diff --check passed.

After committing and pushing, manually run HTH normalize crop and framing assess. Download the resulting artifact and open index.html for the page-by-page comparison.
