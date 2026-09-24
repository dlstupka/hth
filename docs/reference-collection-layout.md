# HTH layout reference editor

Run `python tools/reference-collection-director.py` and open the localhost URL
it prints. Enter the public collection source-repository URL, then **Load both
tabs**. The director defaults to the Baptisms collection and HTH-GOLDEN-0002,
but each editor also has its own source-repository and Golden Set controls for
independent research. The director resolves the immutable release, downloads
its frozen Golden Set and source-image bundle, shows download/checking status,
and verifies the freeze record, bundle SHA-256, and every image SHA-256 before
sharing the images with both tabs. It does not need the results repository.
Opening the HTML directly from disk cannot perform automatic release downloads;
the ZIP and extracted-image controls remain as manual fallback.

The results repository is derived as `<source-repository>-results` and shown in
the director and both tabs; it is not another required input. **Git pull results
checkout** updates the matching sibling checkout with `--ff-only` after checking
its origin and tracked-file cleanliness, and reports whether the commit changed.
If using the detector tab's workspace picker, reopen it after the pull to read
the updated local files. The pull action does not change the frozen source release.

The initial `config/golden_sets/HTH-GOLDEN-0002-LAYOUT.draft.json` is an
**unapproved draft**. It carries the 18 frozen GS0002 page ordinals, source
image SHA-256 values, and approved page boxes, but zero layout regions. This
does not alter the approved detector Golden Set. Region truth uses source-image
pixel coordinates and cannot be applied directly to a normalized view without
the recorded geometric transform.

In the layout tab, load a source release (or use the director's shared load)
before editing a page. Draw rectangular or polygonal regions, drag a selected vertex
to correct a polygon, choose the region class, then mark the page reviewed. An
expanded Kraken layout artifact may be imported as **unapproved proposals**;
the editor checks its Golden Set and source-file identities before displaying
the source-view polygons. Adopt selected proposals only after visual review.
Export the draft JSON regularly; it is not an approved or frozen release.

This first editor handles region polygons. Line baselines and reading-order
truth require additional annotation controls and a versioned extension to the
layout contract before automated scoring of those dimensions. Do not treat
Kraken output, even when adopted, as independently reviewed truth.
