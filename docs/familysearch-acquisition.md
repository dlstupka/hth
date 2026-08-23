# FamilySearch direct source acquisition

`tools/acquire-familysearch-images.py` replaces browser/AHK capture when the
FamilySearch Historical Records Image API exposes a downloadable representation
to the authenticated user.

The tool intentionally uses the official FamilySearch API and OAuth bearer
token. It does **not** scrape the image viewer, automate browser clicks, reuse
browser cookies, or persist account credentials.

## Prerequisites

FamilySearch's Historical Records Image API uses an authenticated bearer token.
Obtain a production FamilySearch app key and authenticate the account with the
OAuth Authorization Code flow. Export only the resulting access token:

```bash
export FAMILYSEARCH_ACCESS_TOKEN='...'
```

Do not put the token, FamilySearch password, or signed image URLs in the
repository. Access tokens are short-lived. FamilySearch may also restrict image
download for a collection because of custodian agreements; the acquisition tool
will report that condition rather than bypass it.

## First probe

Open the first image of the collection in FamilySearch and copy its viewer URL.
The normal ARK form includes an image id beginning `3:1:`. Probe that image
before acquiring the collection:

```bash
python tools/acquire-familysearch-images.py \
  'https://www.familysearch.org/ark:/61903/3:1:YOUR-IMAGE-ID?...' \
  --probe-only \
  --collection-id HTH-SOURCE-0002
```

A viewer URL is preferable to a bare image id because `cc`, `wc`, or `from`
traversal context is preserved when FamilySearch supplies it.

If the viewer URL does not expose a `wc` context and `seek=next` cannot advance,
pass the API context explicitly:

```bash
  --collection-context '...' \
  --waypoint-context '...'
```

## Acquire this collection

After the probe succeeds, acquire the sequence into the existing source
repository's `images/` directory:

```bash
python tools/acquire-familysearch-images.py \
  'FIRST-FAMILYSEARCH-IMAGE-URL' \
  --images-dir images \
  --manifest familysearch-source-manifest.json \
  --collection-id HTH-SOURCE-0002 \
  --collection-title 'Baptisms: San Antonio. Baptism Records 1788-1824, 1858-1898' \
  --expected-count 929
```

If FamilySearch's sequence is intentionally shorter than 929, omit
`--expected-count` for the first complete traversal, inspect the final manifest,
then rerun with the authoritative count.

Files are named `images/fs_0001.<ext>`, `images/fs_0002.<ext>`, and so on. The
extension is determined from the actual downloaded image representation.

## Resume and existing-file behavior

The downloader is intentionally conservative.

- If `images/fs_NNNN.<image extension>` already exists, it is **not overwritten**.
- Existing files are opened and structurally verified as images.
- If a prior manifest entry exists, its FamilySearch image id and SHA-256 must
  still match.
- FamilySearch width, height, byte size, MD5, or SHA-256 values are compared
  whenever the metadata response actually exposes those fields.
- A corrupt or mismatching existing file aborts acquisition instead of silently
  replacing evidence.
- Downloads are written to a `.part` file and receive their final name only
  after image verification succeeds.
- The manifest is atomically updated after every image, so an interrupted
  collection can resume without repeating completed downloads.

## Verification and provenance

`familysearch-source-manifest.json` records, per image:

- source ordinal and local filename;
- FamilySearch image id and persistent ARK;
- local SHA-256 and byte size;
- pixel dimensions and decoded image format;
- the FamilySearch API media relation used;
- sanitized API/media provenance (signed query parameters are not persisted);
- every source-metadata check that FamilySearch made possible;
- acquisition time and status.

At collection completion the tool also checks for duplicate FamilySearch image
ids, duplicate filenames, missing local files, and the optional expected image
count.

This is verification against the information FamilySearch exposes to the
authenticated API client. It cannot prove equality to an inaccessible archival
master if FamilySearch does not publish a checksum/dimensions for that master.

## Rights boundary

FamilySearch states that contractual restrictions can disable historical-image
downloads. The tool therefore treats “metadata visible but no official
downloadable image representation” as a hard stop. It does not fall back to
viewer scraping or screen capture.
