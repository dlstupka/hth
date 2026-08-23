# Immutable source releases

HTH uses GitHub Releases as the durable binary layer for source-document masters. Git repositories contain code, manifests, provenance, configuration, annotations, and research metadata; large immutable DOCX masters do not live in Git history.

## Source release contract

Each collection owns its source release in the collection source repository. The default first release is `HTH-SOURCE-0001`. A source release contains:

- one Release asset for every DOCX master;
- `source-release-manifest.json`;
- immutable asset names, byte sizes, and SHA-256 hashes;
- the source repository and Release tag; and
- the source repository commit current when the release was created, when available.

Preprocess, preprocess-test, and geometry-calibration jobs download those assets directly from the Release API and verify every byte against the manifest before processing. They do not check out or hydrate source-document binaries from Git. Public source repositories need no extra credential. A private source repository can provide the optional `HTH_SOURCE_TOKEN` repository secret with read access to its Releases.

## Publish the existing collection

Run this once from a machine that already has the real DOCX masters and GitHub CLI authentication:

```powershell
python tools/publish-source-release.py `
  --repository dlstupka/hth-baptisms-san-antonio-1788-1824--1858-1898 `
  --tag HTH-SOURCE-0001 `
  --images C:\path\to\hth-baptisms-san-antonio-1788-1824--1858-1898\images `
  --source-root C:\path\to\hth-baptisms-san-antonio-1788-1824--1858-1898 `
  --collection-id HTH-BAPTISMS-SAN-ANTONIO
```

The publisher computes the manifest, creates the immutable Release, and uploads the manifest plus every DOCX as a separate Release asset. It requires `gh` and an authenticated `gh auth login`.

After the release verifies successfully, the source repository can remove the DOCX pointer files and any source-binary tracking rules from its current branch. Existing historical Git objects can remain as historical provenance; HTH execution no longer depends on them.


## Direct-source reacquisition

A better authorized source representation does not overwrite an established source release. For example, if the current FamilySearch collection is reacquired directly through approved API/image access, publish that corpus as a new immutable source edition and preserve `HTH-SOURCE-0001` as bootstrap provenance. Downstream comparison can then distinguish detector changes from source-quality changes.

## New or changed source material

Never mutate an established source Release. Any changed source master, new batch, or replacement rendition gets a new release tag such as `HTH-SOURCE-0002`. Builds persist both the Release tag and the Release-manifest SHA-256, so a source corpus remains reproducible even when the surrounding repository continues to evolve.

## Why Releases

Source masters are collection assets rather than code revisions. Separating them keeps Git repositories lightweight, avoids binary hydration during normal checkout, allows direct immutable provenance, and scales naturally to additional HTH collections while preserving the collection source repository as the owner of its source truth.
